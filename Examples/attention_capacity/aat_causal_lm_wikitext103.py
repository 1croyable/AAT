from __future__ import annotations

r"""
Canonical Complex-Phase K/V Dual-Memory Prefix AAT on WikiText-103.

This is the controlled causal-language-model run for the finalized sequence AAT
architecture:

    complex_phase_key_value_binding_dual_memory_prefix_aat

It keeps the data, tokenizer, optimization, evaluation, checkpoint, generation,
and artifact protocol of the earlier WikiText-103 AAT/Transformer runs.  Only
the model core is replaced by the accepted canonical operator.

The default "paper" preset uses:

    - WikiText-103 raw (more than 100M source tokens)
    - the GPT-2 tokenizer, but no pretrained model weights
    - 12 decoder blocks, d_model=512, 8 heads, 32 rays/head
    - one fixed complex Ordered prefix memory
    - one fixed position-addressed Persistent prefix memory
    - decoder-local Prefix-Persistent self-read in every block
    - a parameter-matched 464-wide GELU FFN
    - tied input/output token embeddings
    - strict content causality at every output position

The Ordered residual recurrence is evaluated with an exact DeltaNet-style
chunkwise scan.  With fixed rays R and chunk size C, the core is linear in token
count T and never materializes a T-by-T attention matrix.  Prefix-Persistent
paths use vectorized cumulative sums.  The implementation is already expressed
as batched GPU tensor operations; a future fused CUDA/Triton kernel can remove
the remaining chunk-loop launches and large intermediate tensors without
changing the operator.

Install:

    pip install torch datasets transformers

Recommended full run:

    python canonical_aat_causal_lm_wikitext103.py --device cuda --precision auto

Resume is automatic when <output-dir>/latest.pt exists.  Useful checks:

    python canonical_aat_causal_lm_wikitext103.py --smoke-test-only
    python canonical_aat_causal_lm_wikitext103.py --preset quick --device cuda

Outputs include latest.pt, best.pt, history.csv, final_results.json,
model_config.json, run_config.json, generated_samples.txt, and a source snapshot.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint


IMPLEMENTATION_VERSION = "canonical-complex-dual-memory-chunkwise-v1.1"


# =================================================================================================
# Configuration
# =================================================================================================


@dataclass(frozen=True)
class Preset:
    name: str
    dataset_config: str
    d_model: int
    blocks: int
    heads: int
    rays_per_head: int
    ffn_hidden: int
    ordered_chunk_size: int
    sequence_length: int
    long_sequence_length: int
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    max_train_blocks: int | None = None
    max_eval_blocks: int | None = None


PRESETS: dict[str, Preset] = {
    "paper": Preset(
        name="paper",
        dataset_config="wikitext-103-raw-v1",
        d_model=512,
        blocks=12,
        heads=8,
        rays_per_head=32,
        ffn_hidden=464,
        ordered_chunk_size=32,
        sequence_length=256,
        long_sequence_length=512,
        epochs=3,
        batch_size=8,
        grad_accum_steps=8,
        learning_rate=2.0e-4,
    ),
    "medium": Preset(
        name="medium",
        dataset_config="wikitext-103-raw-v1",
        d_model=384,
        blocks=8,
        heads=8,
        rays_per_head=24,
        ffn_hidden=384,
        ordered_chunk_size=32,
        sequence_length=256,
        long_sequence_length=512,
        epochs=3,
        batch_size=8,
        grad_accum_steps=8,
        learning_rate=2.5e-4,
    ),
    "quick": Preset(
        name="quick",
        dataset_config="wikitext-2-raw-v1",
        d_model=128,
        blocks=2,
        heads=4,
        rays_per_head=8,
        ffn_hidden=64,
        ordered_chunk_size=16,
        sequence_length=64,
        long_sequence_length=128,
        epochs=1,
        batch_size=8,
        grad_accum_steps=1,
        learning_rate=5.0e-4,
        max_train_blocks=128,
        max_eval_blocks=32,
    ),
}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    blocks: int
    heads: int
    rays_per_head: int
    ffn_hidden: int
    ordered_chunk_size: int
    position_scale: float
    dropout: float = 0.10
    initial_kappa: float = 6.0
    score_clip: float = 30.0
    gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    def validate(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.d_model <= 0 or self.d_model % self.heads != 0:
            raise ValueError("d_model must be positive and divisible by heads")
        if min(self.blocks, self.rays_per_head, self.ordered_chunk_size) <= 0:
            raise ValueError("blocks, rays_per_head, and ordered_chunk_size must be positive")
        if self.ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive")
        if self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.score_clip <= 0.0:
            raise ValueError("score_clip must be positive")


@dataclass(frozen=True)
class TrainConfig:
    dataset_name: str
    dataset_config: str
    tokenizer_name: str
    sequence_length: int
    long_sequence_length: int
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    min_lr_ratio: float
    grad_clip: float
    seed: int
    num_workers: int
    preprocessing_workers: int
    log_every: int
    save_every: int
    eval_every: int
    eval_batches: int | None
    max_train_blocks: int | None
    max_eval_blocks: int | None


@dataclass
class EvalResult:
    split: str
    loss: float
    perplexity: float
    bits_per_token: float
    token_accuracy: float
    tokens: int
    seconds: float
    tokens_per_second: float


# =================================================================================================
# General utilities
# =================================================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def choose_precision(requested: str, device: torch.device) -> tuple[torch.dtype, bool, str]:
    if device.type != "cuda":
        return torch.float32, False, "fp32"
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        return torch.bfloat16, False, "bf16"
    if requested == "fp16":
        return torch.float16, True, "fp16"
    if requested == "fp32":
        return torch.float32, False, "fp32"
    raise ValueError(f"unknown precision: {requested}")


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older supported PyTorch releases
        return torch.cuda.amp.GradScaler(enabled=enabled)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def human_count(value: int | float) -> str:
    value = float(value)
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1000.0 or suffix == "T":
            return f"{value:.2f}{suffix}" if suffix else f"{int(value)}"
        value /= 1000.0
    raise AssertionError("unreachable")


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:d}m{secs:02d}s"


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def environment_info(device: torch.device) -> dict[str, Any]:
    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "datasets": package_version("datasets"),
        "transformers": package_version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type == "cuda" and dtype != torch.float32),
    )


# =================================================================================================
# Unified causal Persistent-Ordered AAT
# =================================================================================================


class AATRayResponse(nn.Module):
    """Shared radial-directional token-to-ray response table."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sqrt_head_dim = math.sqrt(cfg.head_dim)
        self.ray_base = nn.Parameter(
            torch.randn(cfg.heads, cfg.rays_per_head, cfg.head_dim)
            / math.sqrt(cfg.head_dim)
        )
        self.ray_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.rays_per_head))
        self.radial_weight = nn.Parameter(torch.zeros(cfg.heads, cfg.rays_per_head))
        self.log_kappa = nn.Parameter(
            torch.full((cfg.heads,), math.log(cfg.initial_kappa))
        )

    def score(self, heads: torch.Tensor) -> torch.Tensor:
        radius = heads.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = heads / radius
        rho = radius / self.sqrt_head_dim - 1.0
        rays = F.normalize(self.ray_base, dim=-1, eps=1e-8)
        cosine = torch.einsum("bhtd,hrd->bhtr", direction, rays)
        radial = torch.exp(
            (rho * self.radial_weight[None, :, None, :]).clamp(-8.0, 8.0)
        )
        kappa = self.log_kappa.exp().clamp(0.25, 50.0)
        return (
            kappa[None, :, None, None] * radial * cosine
            + self.ray_bias[None, :, None, :]
        )

    @torch.no_grad()
    def kappa_mean(self) -> float:
        return float(self.log_kappa.exp().clamp(0.25, 50.0).mean().item())


def causal_persistent_ordered_reads(
    values: torch.Tensor,
    scores: torch.Tensor,
    address: torch.Tensor,
    read_address: torch.Tensor,
    *,
    score_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-prefix Persistent and Ordered reads.

    Shapes:
        values       [B,H,T,Dh]
        scores       [B,H,T,R]
        address      [B,H,T,R]  (ray-axis softmax; used for Ordered writes)
        read_address [B,H,T,R]  (address times read-only ray strength)

    Persistent is the token-axis interpretation of the same response table.
    For every prefix t and ray r, it stores the score-normalized content from
    positions s <= t.  Ordered applies the residual Delta-rule write

        M_t = M_{t-1} + a_t (x_t - a_t^T M_{t-1})^T.

    The implementation exposes all prefix reads with triangular solves and
    causal matrix products.  No value at position t depends on a future token.
    """
    if values.ndim != 4 or scores.ndim != 4:
        raise ValueError("values and scores must be rank four")
    if scores.shape != address.shape or address.shape != read_address.shape:
        raise ValueError("score/address shapes do not match")
    if values.shape[:3] != scores.shape[:3]:
        raise ValueError("value and score prefix shapes do not match")

    output_dtype = values.dtype
    with torch.autocast(device_type=values.device.type, enabled=False):
        x = values.float().contiguous()
        score_f = scores.float().clamp(-float(score_clip), float(score_clip))
        write_a = address.float().contiguous()
        read_a = read_address.float().contiguous()
        token_count = x.shape[2]

        # Persistent: the prefix denominator is causal.  Contracting ray
        # memories into token-token coefficients avoids materializing [T,R,D].
        positive_score = score_f.exp()
        prefix_denominator = positive_score.cumsum(dim=2).clamp_min(1e-12)
        persistent_query = read_a / prefix_denominator
        persistent_coeff = torch.matmul(
            persistent_query, positive_score.transpose(-1, -2)
        )
        causal_mask = torch.ones(
            token_count, token_count, device=x.device, dtype=torch.bool
        ).tril()
        persistent_coeff = persistent_coeff.masked_fill(~causal_mask, 0.0)
        persistent_read = torch.matmul(persistent_coeff, x)

        # Ordered: solve all residual-write innovations exactly, then read the
        # memory at every prefix (including the current token's write).
        write_gram = torch.matmul(write_a, write_a.transpose(-1, -2))
        causal_system = torch.tril(write_gram, diagonal=-1)
        innovations = torch.linalg.solve_triangular(
            causal_system,
            x,
            upper=False,
            unitriangular=True,
        )
        ordered_coeff = torch.matmul(read_a, write_a.transpose(-1, -2))
        ordered_coeff = ordered_coeff.masked_fill(~causal_mask, 0.0)
        ordered_read = torch.matmul(ordered_coeff, innovations)

    return persistent_read.to(output_dtype), ordered_read.to(output_dtype)


class ProjectAATLayer(nn.Module):
    """Exact radial-directional additive transport layer from the AAT project."""

    def __init__(self, state_dim: int, rays: int, *, kappa: float, ray_dropout: float) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.rays = int(rays)
        self.kappa = float(kappa)
        self.ray_dropout = float(ray_dropout)
        self.base = nn.Parameter(torch.randn(rays, state_dim) / math.sqrt(state_dim))
        self.ray_bias = nn.Parameter(torch.zeros(rays))
        self.radial_weight = nn.Parameter(torch.zeros(rays))
        self.dr = nn.Parameter(torch.randn(rays) * 0.02)
        self.du = nn.Parameter(
            torch.randn(rays, state_dim) * 0.02 / math.sqrt(state_dim)
        )
        self.gate = nn.Parameter(torch.tensor(1.0))

    def _dropout(self, alpha: torch.Tensor) -> torch.Tensor:
        if not self.training or self.ray_dropout <= 0.0:
            return alpha
        mask = torch.rand_like(alpha) >= self.ray_dropout
        empty = mask.sum(dim=1, keepdim=True) == 0
        if bool(empty.any().item()):
            mask = mask.clone()
            rows = empty.squeeze(1).nonzero(as_tuple=False).squeeze(1)
            cols = alpha.index_select(0, rows).argmax(dim=1)
            mask[rows, cols] = True
        alpha = alpha * mask.to(alpha.dtype)
        return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, rho: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rays = F.normalize(self.base, dim=1, eps=1e-8)
        cosine = u @ rays.t()
        scale = torch.exp(
            (rho * self.radial_weight.view(1, -1)).clamp(-8.0, 8.0)
        )
        score = self.kappa * scale * cosine + self.ray_bias.view(1, -1)
        alpha = self._dropout(F.softmax(score, dim=1))
        rho = rho + (alpha @ self.dr[:, None]) * self.gate
        u = F.normalize(u + (alpha @ self.du) * self.gate, dim=1, eps=1e-8)
        return rho, u


class DeepProjectAATFusion(nn.Module):
    """Deep, relatively narrow AAT transport replacing most of a large FFN budget."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.sqrt_dim = math.sqrt(cfg.d_model)
        self.layers = nn.ModuleList(
            ProjectAATLayer(
                cfg.d_model,
                cfg.fusion_rays,
                kappa=cfg.fusion_kappa,
                ray_dropout=cfg.fusion_ray_dropout,
            )
            for _ in range(cfg.fusion_steps)
        )
        self.output_gate = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        z = x.reshape(-1, self.d_model)
        radius = z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        u0 = z / radius
        rho0 = radius / self.sqrt_dim - 1.0
        rho, u = rho0, u0
        for layer in self.layers:
            rho, u = layer(rho, u)
        initial = (1.0 + rho0).clamp_min(0.05) * u0 * self.sqrt_dim
        transported = (1.0 + rho).clamp_min(0.05) * u * self.sqrt_dim
        delta = (transported - initial).reshape(shape)
        return self.dropout(delta) * self.output_gate


_FIXED_PROJECTION_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


class CausalUnifiedAATMixer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.response = AATRayResponse(cfg)
        self.ray_strength = nn.Parameter(torch.ones(cfg.heads, cfg.rays_per_head))
        self.fusion = DeepProjectAATFusion(cfg)
        self.register_buffer(
            "fixed_projection",
            self._make_fixed_projection(2 * cfg.d_model, cfg.d_model, cfg.projection_seed),
            persistent=True,
        )
        self.output_dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _make_fixed_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
        key = (int(input_dim), int(output_dim), int(seed))
        cached = _FIXED_PROJECTION_CACHE.get(key)
        if cached is not None:
            return cached.clone()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        gaussian = torch.randn(
            input_dim, output_dim, generator=generator, dtype=torch.float64
        )
        q, r = torch.linalg.qr(gaussian, mode="reduced")
        diagonal = torch.diagonal(r)
        signs = torch.where(
            diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal)
        )
        projection = (q * signs.unsqueeze(0)).transpose(0, 1).contiguous().float()
        _FIXED_PROJECTION_CACHE[key] = projection
        return projection.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        heads = x.view(
            batch_size, token_count, self.cfg.heads, self.cfg.head_dim
        ).transpose(1, 2)
        scores = self.response.score(heads)
        address = F.softmax(scores, dim=-1)
        read_address = address * self.ray_strength[None, :, None, :]
        persistent_read, ordered_read = causal_persistent_ordered_reads(
            heads,
            scores,
            address,
            read_address,
            score_clip=self.cfg.score_clip,
        )
        persistent_full = persistent_read.transpose(1, 2).contiguous().view(
            batch_size, token_count, self.cfg.d_model
        )
        ordered_full = ordered_read.transpose(1, 2).contiguous().view(
            batch_size, token_count, self.cfg.d_model
        )
        direct_sum_delta = torch.cat(
            (persistent_full - x, ordered_full - x), dim=-1
        )
        projected_delta = F.linear(direct_sum_delta, self.fixed_projection)
        transported_delta = projected_delta + self.fusion(projected_delta)
        return self.output_dropout(transported_delta)

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        strength = self.ray_strength.detach().float()
        identity = torch.eye(
            self.cfg.d_model,
            device=self.fixed_projection.device,
            dtype=self.fixed_projection.dtype,
        )
        projection_error = (
            self.fixed_projection @ self.fixed_projection.transpose(0, 1) - identity
        ).abs().max()
        return {
            "kappa_mean": self.response.kappa_mean(),
            "ray_strength_mean": float(strength.mean().item()),
            "ray_strength_std": float(strength.std(unbiased=False).item()),
            "ray_strength_min": float(strength.min().item()),
            "ray_strength_max": float(strength.max().item()),
            "fusion_output_gate": float(self.fusion.output_gate.item()),
            "fusion_layer_gate_mean": sum(
                float(layer.gate.item()) for layer in self.fusion.layers
            ) / len(self.fusion.layers),
            "projection_error": float(projection_error.item()),
        }


class LightweightFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.output = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.gelu(self.input(x), approximate="tanh")))


class UnifiedAATBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mixer_norm = nn.LayerNorm(cfg.d_model)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.mixer = CausalUnifiedAATMixer(cfg)
        self.feed_forward = LightweightFeedForward(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.mixer_norm(x)))
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x


def sinusoidal_positions(
    token_count: int, dimension: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(token_count, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(token_count, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class AATCausalLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embedding_norm = nn.LayerNorm(cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(UnifiedAATBlock(cfg) for _ in range(cfg.blocks))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_standard_weights)

    @staticmethod
    def _initialize_standard_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        x = self.token_embedding(input_ids.long())
        x = x + sinusoidal_positions(
            x.shape[1], self.cfg.d_model, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        x = self.embedding_dropout(self.embedding_norm(x))
        for block in self.blocks:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.final_norm(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden_states(input_ids))

    def parameter_breakdown(self) -> dict[str, int]:
        embedding = self.token_embedding.weight.numel()
        mixers = sum(count_parameters(block.mixer) for block in self.blocks)
        ffn = sum(count_parameters(block.feed_forward) for block in self.blocks)
        norms = count_parameters(self.embedding_norm) + count_parameters(self.final_norm)
        norms += sum(
            count_parameters(block.mixer_norm) + count_parameters(block.ffn_norm)
            for block in self.blocks
        )
        return {
            "total": count_parameters(self),
            "tied_token_embedding_and_lm_head": embedding,
            "mixers": mixers,
            "lightweight_ffn": ffn,
            "normalization": norms,
        }

    @torch.no_grad()
    def architecture_stats(self) -> dict[str, float]:
        per_block = [block.mixer.stats() for block in self.blocks]
        keys = per_block[0].keys()
        return {
            key: sum(block_stats[key] for block_stats in per_block) / len(per_block)
            for key in keys
        }


# =================================================================================================
# Mathematical and implementation verification
# =================================================================================================


def sequential_reference_reads(
    values: torch.Tensor,
    scores: torch.Tensor,
    address: torch.Tensor,
    read_address: torch.Tensor,
    score_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, token_count, head_dim = values.shape
    rays = address.shape[-1]
    persistent_numerator = torch.zeros(
        batch, heads, rays, head_dim, dtype=torch.float32, device=values.device
    )
    persistent_denominator = torch.zeros(
        batch, heads, rays, dtype=torch.float32, device=values.device
    )
    ordered_memory = torch.zeros_like(persistent_numerator)
    persistent_outputs: list[torch.Tensor] = []
    ordered_outputs: list[torch.Tensor] = []
    x = values.float()
    positive_score = scores.float().clamp(-score_clip, score_clip).exp()
    write_a = address.float()
    read_a = read_address.float()
    for index in range(token_count):
        weight = positive_score[:, :, index]
        value = x[:, :, index]
        persistent_numerator = persistent_numerator + weight.unsqueeze(-1) * value.unsqueeze(-2)
        persistent_denominator = persistent_denominator + weight
        persistent_memory = persistent_numerator / persistent_denominator.clamp_min(1e-12).unsqueeze(-1)
        persistent_outputs.append(
            torch.einsum("bhr,bhrd->bhd", read_a[:, :, index], persistent_memory)
        )

        a = write_a[:, :, index]
        prediction = torch.einsum("bhr,bhrd->bhd", a, ordered_memory)
        innovation = value - prediction
        ordered_memory = ordered_memory + a.unsqueeze(-1) * innovation.unsqueeze(-2)
        ordered_outputs.append(
            torch.einsum("bhr,bhrd->bhd", read_a[:, :, index], ordered_memory)
        )
    return torch.stack(persistent_outputs, dim=2), torch.stack(ordered_outputs, dim=2)


@torch.no_grad()
def verify_causal_memory_math() -> None:
    generator = torch.Generator().manual_seed(271828)
    values = torch.randn(2, 3, 9, 5, generator=generator)
    scores = torch.randn(2, 3, 9, 4, generator=generator)
    address = scores.softmax(dim=-1)
    strength = torch.randn(1, 3, 1, 4, generator=generator) * 0.15 + 1.0
    read_address = address * strength
    actual = causal_persistent_ordered_reads(
        values, scores, address, read_address, score_clip=30.0
    )
    expected = sequential_reference_reads(
        values, scores, address, read_address, score_clip=30.0
    )
    for name, left, right in zip(("persistent", "ordered"), actual, expected):
        if not torch.allclose(left, right, atol=2e-5, rtol=2e-5):
            error = float((left - right).abs().max().item())
            raise RuntimeError(f"causal {name} verification failed: max_error={error}")


@torch.no_grad()
def verify_no_future_leak(model: AATCausalLanguageModel, device: torch.device) -> None:
    model.eval()
    generator = torch.Generator().manual_seed(161803)
    sequence_a = torch.randint(0, model.cfg.vocab_size, (2, 13), generator=generator)
    sequence_b = sequence_a.clone()
    prefix_length = 7
    sequence_b[:, prefix_length:] = torch.randint(
        0,
        model.cfg.vocab_size,
        sequence_b[:, prefix_length:].shape,
        generator=generator,
    )
    logits_a = model(sequence_a.to(device))[:, :prefix_length].float().cpu()
    logits_b = model(sequence_b.to(device))[:, :prefix_length].float().cpu()
    if not torch.allclose(logits_a, logits_b, atol=3e-5, rtol=3e-5):
        error = float((logits_a - logits_b).abs().max().item())
        raise RuntimeError(f"future-token leakage detected: max_error={error}")


def smoke_test(device: torch.device) -> None:
    print("Running causal memory, future-leakage, gradient, and checkpoint smoke tests...", flush=True)
    verify_causal_memory_math()
    cfg = ModelConfig(
        vocab_size=101,
        d_model=32,
        blocks=2,
        heads=4,
        rays_per_head=4,
        fusion_rays=5,
        fusion_steps=2,
        ffn_hidden=64,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    model = AATCausalLanguageModel(cfg).to(device)
    verify_no_future_leak(model, device)
    model.train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 12), device=device)
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
    loss.backward()
    if not math.isfinite(float(loss.item())):
        raise RuntimeError("smoke loss is not finite")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite gradient in {name}")
    if any(block.mixer.ray_strength.grad is None for block in model.blocks):
        raise RuntimeError("ray strength did not receive gradients")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        atomic_torch_save(
            {"model_config": asdict(cfg), "model_state_dict": model.state_dict()}, path
        )
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        restored = AATCausalLanguageModel(ModelConfig(**payload["model_config"])).to(device)
        restored.load_state_dict(payload["model_state_dict"])
        restored.eval()
        model.eval()
        with torch.no_grad():
            left = model(tokens[:, :-1])
            right = restored(tokens[:, :-1])
        if not torch.equal(left, right):
            raise RuntimeError("checkpoint round-trip changed model outputs")
    print("All smoke tests: OK", flush=True)


# =================================================================================================
# Final canonical Complex-Phase Dual-Memory Prefix AAT
#
# These definitions supersede the earlier Unified AAT prototype above.  The
# legacy implementation is intentionally retained in this source snapshot as a
# readable mathematical reference for the previous completed run.
# =================================================================================================


def split_heads(x: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    batch_size, token_count, _ = x.shape
    return x.view(
        batch_size,
        token_count,
        cfg.heads,
        cfg.head_dim,
    ).transpose(1, 2)


def merge_heads(x: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    batch_size, _, token_count, _ = x.shape
    return x.transpose(1, 2).contiguous().view(
        batch_size,
        token_count,
        cfg.d_model,
    )


def initialize_identity(projection: nn.Linear) -> None:
    if projection.in_features != projection.out_features:
        raise ValueError("identity initialization requires a square projection")
    with torch.no_grad():
        projection.weight.copy_(
            torch.eye(
                projection.in_features,
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
        )
        if projection.bias is not None:
            projection.bias.zero_()


def pad_token_axis(x: torch.Tensor, padded_tokens: int) -> torch.Tensor:
    """Pad tensor dimension 2 to ``padded_tokens`` with zeros."""
    token_count = x.shape[2]
    if padded_tokens < token_count:
        raise ValueError("padded_tokens cannot be smaller than token_count")
    if padded_tokens == token_count:
        return x
    pad_shape = list(x.shape)
    pad_shape[2] = padded_tokens - token_count
    return torch.cat((x, x.new_zeros(pad_shape)), dim=2)


def causal_prefix_weighted_memory(
    values: torch.Tensor,
    scores: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    score_clip: float,
) -> torch.Tensor:
    """Exact token-axis softmax memory for every causal prefix.

    ``values`` has shape [B,H,T,D], ``scores`` is [B,H,T,R], and the returned
    memory is [B,H,T,R,D].  The cumulative numerator/denominator formulation is
    the online form of applying softmax over tokens 0..t independently for
    every ray.  No T-by-T matrix is formed.
    """
    if values.ndim != 4 or scores.ndim != 4:
        raise ValueError("values and scores must be rank four")
    if values.shape[:3] != scores.shape[:3]:
        raise ValueError("value and score token axes do not match")
    if padding_mask.shape != (values.shape[0], values.shape[2]):
        raise ValueError("padding_mask must have shape [B,T]")

    output_dtype = values.dtype
    with torch.autocast(device_type=values.device.type, enabled=False):
        value_f = values.float()
        weight = scores.float().clamp(
            -float(score_clip),
            float(score_clip),
        ).exp()
        valid = (~padding_mask)[:, None, :, None].float()
        weight = weight * valid
        numerator = (
            weight.unsqueeze(-1) * value_f.unsqueeze(-2)
        ).cumsum(dim=2)
        denominator = weight.cumsum(dim=2).clamp_min(1e-12)
        memory = numerator / denominator.unsqueeze(-1)
        memory = memory.masked_fill(
            padding_mask[:, None, :, None, None],
            0.0,
        )
    return memory.to(output_dtype)


def chunkwise_ordered_prefix_state(
    address: torch.Tensor,
    payload: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an exact chunkwise representation of every Ordered prefix state.

    The recurrence is

        M_t = M_{t-1} + a_t (v_t - a_t^T M_{t-1})^T.

    For each chunk this function stores:

      - the write addresses ``a``;
      - the exact residual innovations ``e``;
      - the memory immediately before that chunk.

    A reader can then recover q_t^T M_t for every token with one batched
    boundary read plus a causal within-chunk product.  With fixed chunk size C,
    work and storage are linear in T.  The only Python loop is over T/C chunks,
    not over tokens.
    """
    if address.ndim != 4 or payload.ndim != 4:
        raise ValueError("address and payload must be rank four")
    if address.shape[:3] != payload.shape[:3]:
        raise ValueError("address and payload token axes do not match")
    if padding_mask.shape != (address.shape[0], address.shape[2]):
        raise ValueError("padding_mask must have shape [B,T]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output_dtype = payload.dtype
    batch_size, heads, token_count, rays = address.shape
    payload_dim = payload.shape[-1]
    chunk_count = math.ceil(token_count / chunk_size)
    padded_tokens = chunk_count * chunk_size

    with torch.autocast(device_type=payload.device.type, enabled=False):
        valid = (~padding_mask)[:, None, :, None].float()
        address_f = address.float() * valid
        payload_f = payload.float() * valid
        address_f = pad_token_axis(address_f, padded_tokens)
        payload_f = pad_token_axis(payload_f, padded_tokens)
        address_chunks = address_f.view(
            batch_size,
            heads,
            chunk_count,
            chunk_size,
            rays,
        )
        payload_chunks = payload_f.view(
            batch_size,
            heads,
            chunk_count,
            chunk_size,
            payload_dim,
        )

        memory = payload_f.new_zeros(
            batch_size,
            heads,
            rays,
            payload_dim,
        )
        innovations: list[torch.Tensor] = []
        boundaries: list[torch.Tensor] = []
        for chunk_index in range(chunk_count):
            chunk_address = address_chunks[:, :, chunk_index]
            chunk_payload = payload_chunks[:, :, chunk_index]
            boundaries.append(memory)

            previous_prediction = torch.einsum(
                "bhcr,bhrd->bhcd",
                chunk_address,
                memory,
            )
            right_hand_side = chunk_payload - previous_prediction
            gram = torch.matmul(
                chunk_address,
                chunk_address.transpose(-1, -2),
            )
            strictly_lower = torch.tril(gram, diagonal=-1)
            innovation = torch.linalg.solve_triangular(
                strictly_lower,
                right_hand_side,
                upper=False,
                unitriangular=True,
            )
            innovations.append(innovation)
            memory = memory + torch.einsum(
                "bhcr,bhcd->bhrd",
                chunk_address,
                innovation,
            )

        innovation_chunks = torch.stack(innovations, dim=2)
        boundary_chunks = torch.stack(boundaries, dim=2)

    return (
        address_chunks.to(output_dtype),
        innovation_chunks.to(output_dtype),
        boundary_chunks.to(output_dtype),
    )


def chunkwise_ordered_prefix_read(
    query_address: torch.Tensor,
    address_chunks: torch.Tensor,
    innovation_chunks: torch.Tensor,
    boundary_chunks: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Recover every q_t^T M_t from a chunkwise Ordered state."""
    if query_address.ndim != 4:
        raise ValueError("query_address must have shape [B,H,T,R]")
    batch_size, heads, token_count, rays = query_address.shape
    # Both tensors are chunked as [B,H,N,C,*].  ``innovation_chunks`` stores
    # one payload vector per token, so it is rank five just like
    # ``address_chunks`` (its last axis is D rather than R).
    if address_chunks.ndim != 5 or innovation_chunks.ndim != 5:
        raise ValueError("invalid chunkwise Ordered state")
    if boundary_chunks.ndim != 5:
        raise ValueError("invalid Ordered boundary state")
    if address_chunks.shape[:3] != innovation_chunks.shape[:3]:
        raise ValueError("Ordered chunk axes do not match")
    if address_chunks.shape[:3] != boundary_chunks.shape[:3]:
        raise ValueError("Ordered boundary axes do not match")
    if address_chunks.shape[-1] != rays:
        raise ValueError("query and writer ray counts do not match")

    chunk_count = address_chunks.shape[2]
    chunk_size = address_chunks.shape[3]
    padded_tokens = chunk_count * chunk_size
    valid = (~padding_mask)[:, None, :, None].to(query_address.dtype)
    query = pad_token_axis(query_address * valid, padded_tokens).view(
        batch_size,
        heads,
        chunk_count,
        chunk_size,
        rays,
    )

    boundary_read = torch.einsum(
        "bhncr,bhnrd->bhncd",
        query,
        boundary_chunks,
    )
    within_coefficients = torch.einsum(
        "bhnir,bhnjr->bhnij",
        query,
        address_chunks,
    )
    causal = torch.ones(
        chunk_size,
        chunk_size,
        device=query.device,
        dtype=torch.bool,
    ).tril()
    within_coefficients = within_coefficients.masked_fill(
        ~causal.view(1, 1, 1, chunk_size, chunk_size),
        0.0,
    )
    within_read = torch.einsum(
        "bhnij,bhnjd->bhnid",
        within_coefficients,
        innovation_chunks,
    )
    read = (boundary_read + within_read).reshape(
        batch_size,
        heads,
        padded_tokens,
        innovation_chunks.shape[-1],
    )[:, :, :token_count]
    return read.masked_fill(
        padding_mask[:, None, :, None],
        0.0,
    )


class ComplexPhaseKeyValueOrderedPrefixEncoder(nn.Module):
    """One fixed causal Ordered memory with learned K/V views and phase binding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.response = AATRayResponse(cfg)
        self.key_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.value_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.phase_scale = nn.Parameter(
            torch.full(
                (cfg.heads, cfg.head_dim),
                1.0 / math.sqrt(cfg.head_dim),
            )
        )

    def key_heads(self, x: torch.Tensor) -> torch.Tensor:
        return split_heads(self.key_projection(x), self.cfg)

    def value_heads(self, x: torch.Tensor) -> torch.Tensor:
        return split_heads(self.value_projection(x), self.cfg)

    def phase_heads_from_key(self, key: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.tanh(
            key * self.phase_scale[None, :, None, :]
        )

    def phase_heads(self, x: torch.Tensor) -> torch.Tensor:
        return self.phase_heads_from_key(self.key_heads(x))

    def address_from_key(self, key: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.response.score(key), dim=-1)

    def forward(
        self,
        content_source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.key_heads(content_source)
        value = self.value_heads(content_source)
        phase = self.phase_heads_from_key(key)
        address = self.address_from_key(key)

        # Concatenating real/imaginary payloads lets the shared Ordered
        # transition be solved once instead of twice.
        bound_payload = torch.cat(
            (value * phase.cos(), value * phase.sin()),
            dim=-1,
        )
        return chunkwise_ordered_prefix_state(
            address,
            bound_payload,
            padding_mask,
            chunk_size=self.cfg.ordered_chunk_size,
        )

    @torch.no_grad()
    def projection_diagnostics(self) -> tuple[float, float, float]:
        identity = torch.eye(
            self.cfg.d_model,
            device=self.key_projection.weight.device,
            dtype=self.key_projection.weight.dtype,
        )
        key_delta = (
            self.key_projection.weight.detach() - identity
        ).square().mean().sqrt()
        value_delta = (
            self.value_projection.weight.detach() - identity
        ).square().mean().sqrt()
        phase_rms = self.phase_scale.detach().square().mean().sqrt()
        return (
            float(key_delta.item()),
            float(value_delta.item()),
            float(phase_rms.item()),
        )


class PositionPrefixMemoryEncoder(nn.Module):
    """One stable position-addressed causal Prefix-Persistent memory.

    Position never enters the content payload.  Training positions exactly
    cover z in [0,1] when ``position_scale = train_length - 1``.  Longer
    sequences continue along the same curve instead of being renormalized by a
    future total length, which keeps the language model strictly prefix
    consistent.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sqrt_head_dim = math.sqrt(cfg.head_dim)
        self.response = AATRayResponse(cfg)

    def position_heads(
        self,
        padding_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, token_count = padding_mask.shape
        relative = (
            torch.arange(
                token_count,
                device=padding_mask.device,
                dtype=dtype,
            )
            / self.cfg.position_scale
        )
        pair_count = self.cfg.head_dim // 2
        frequencies = torch.arange(
            1,
            pair_count + 1,
            device=padding_mask.device,
            dtype=dtype,
        )
        angle = math.pi * relative[:, None] * frequencies[None, :]
        features = torch.stack(
            (angle.cos(), angle.sin()),
            dim=-1,
        ).flatten(-2)
        if self.cfg.head_dim % 2:
            features = torch.cat(
                (features, (2.0 * relative - 1.0).unsqueeze(-1)),
                dim=-1,
            )
        direction = F.normalize(features, dim=-1, eps=1e-8)
        rho = relative - 0.5
        state = (
            direction
            * (1.0 + rho).unsqueeze(-1)
            * self.sqrt_head_dim
        )
        state = state.view(1, 1, token_count, self.cfg.head_dim).expand(
            batch_size,
            self.cfg.heads,
            -1,
            -1,
        )
        return state.masked_fill(
            padding_mask[:, None, :, None],
            0.0,
        )

    def forward(
        self,
        content_source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content_heads = split_heads(content_source, self.cfg)
        position_heads = self.position_heads(
            padding_mask,
            dtype=content_source.dtype,
        )
        score = self.response.score(position_heads)
        position_address = F.softmax(score, dim=-1)
        prefix_memory = causal_prefix_weighted_memory(
            content_heads,
            score,
            padding_mask,
            score_clip=self.cfg.score_clip,
        )
        return prefix_memory, position_heads, position_address


class CausalPersistentSelfReader(nn.Module):
    """Canonical decoder-local Prefix-Persistent self-read."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.response = AATRayResponse(cfg)
        self.ray_strength = nn.Parameter(
            torch.ones(cfg.heads, cfg.rays_per_head)
        )
        self.output_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.output_gate = nn.Parameter(torch.tensor(1.0))
        self.output_dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def routing_query(
        content_heads: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = (~padding_mask)[:, None, :, None].to(content_heads.dtype)
        valid_content = content_heads * valid
        inclusive_sum = valid_content.cumsum(dim=2)
        inclusive_count = valid.cumsum(dim=2)
        prefix_sum = inclusive_sum - valid_content
        prefix_count = inclusive_count - valid
        context = prefix_sum / prefix_count.clamp_min(1.0)
        context = context * prefix_count.gt(0).to(context.dtype)
        context = context.masked_fill(
            padding_mask[:, None, :, None],
            0.0,
        )
        return content_heads + context

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        content_heads = split_heads(x, self.cfg)
        routing_query = self.routing_query(content_heads, padding_mask)
        score = self.response.score(routing_query)
        read_address = (
            F.softmax(score, dim=-1)
            * self.ray_strength[None, :, None, :]
        )
        prefix_memory = causal_prefix_weighted_memory(
            content_heads,
            score,
            padding_mask,
            score_clip=self.cfg.score_clip,
        )
        read = torch.einsum(
            "bhtr,bhtrd->bhtd",
            read_address,
            prefix_memory,
        )
        delta = self.output_projection(
            merge_heads(read, self.cfg) - x
        ) * self.output_gate
        return self.output_dropout(delta).masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )


class ComplexPhaseOrderedMemoryReader(nn.Module):
    """Content-address, read, and phase-unbind the shared Ordered prefix state."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ray_strength = nn.Parameter(
            torch.ones(cfg.heads, cfg.rays_per_head)
        )
        self.output_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.output_gate = nn.Parameter(torch.tensor(1.0))
        self.output_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        query: torch.Tensor,
        address_chunks: torch.Tensor,
        innovation_chunks: torch.Tensor,
        boundary_chunks: torch.Tensor,
        encoder: ComplexPhaseKeyValueOrderedPrefixEncoder,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_key = encoder.key_heads(query)
        query_value = encoder.value_heads(query)
        query_phase = encoder.phase_heads_from_key(query_key)
        query_address = (
            encoder.address_from_key(query_key)
            * self.ray_strength[None, :, None, :]
        )
        bound_read = chunkwise_ordered_prefix_read(
            query_address,
            address_chunks,
            innovation_chunks,
            boundary_chunks,
            padding_mask,
        )
        real_read, imag_read = bound_read.chunk(2, dim=-1)
        unbound = (
            real_read * query_phase.cos()
            + imag_read * query_phase.sin()
        )
        delta = self.output_projection(
            merge_heads(unbound - query_value, self.cfg)
        ) * self.output_gate
        return self.output_dropout(delta).masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )


class PositionMemoryReader(nn.Module):
    """Read the fixed Prefix-Persistent memory through shared position rays."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ray_strength = nn.Parameter(
            torch.ones(cfg.heads, cfg.rays_per_head)
        )
        self.output_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.output_gate = nn.Parameter(torch.tensor(1.0))
        self.output_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        query: torch.Tensor,
        prefix_memory: torch.Tensor,
        position_address: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        read_address = (
            position_address
            * self.ray_strength[None, :, None, :]
        )
        read = torch.einsum(
            "bhtr,bhtrd->bhtd",
            read_address,
            prefix_memory,
        )
        delta = self.output_projection(
            merge_heads(read, self.cfg) - query
        ) * self.output_gate
        return self.output_dropout(delta).masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )


class CanonicalFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.output = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(
            self.dropout(F.gelu(self.input(x), approximate="tanh"))
        )


class CanonicalDecoderBlock(nn.Module):
    """Prefix self-read, Ordered read, Position read, then parameter-matched MLP."""

    def __init__(
        self,
        cfg: ModelConfig,
        ordered_encoder: ComplexPhaseKeyValueOrderedPrefixEncoder,
    ) -> None:
        super().__init__()
        # Keep one shared encoder geometry without registering twelve duplicate
        # module paths in state_dict.  The encoder itself is registered once on
        # AATCausalLanguageModel and is moved/saved from there.
        object.__setattr__(self, "_ordered_encoder_ref", ordered_encoder)
        self.persistent_norm = nn.LayerNorm(cfg.d_model)
        self.ordered_memory_norm = nn.LayerNorm(cfg.d_model)
        self.position_memory_norm = nn.LayerNorm(cfg.d_model)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.persistent_reader = CausalPersistentSelfReader(cfg)
        self.ordered_memory_reader = ComplexPhaseOrderedMemoryReader(cfg)
        self.position_memory_reader = PositionMemoryReader(cfg)
        self.feed_forward = CanonicalFeedForward(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        ordered_address: torch.Tensor,
        ordered_innovations: torch.Tensor,
        ordered_boundaries: torch.Tensor,
        position_memory: torch.Tensor,
        position_address: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.persistent_reader(
                self.persistent_norm(x),
                padding_mask,
            )
        )
        x = x + self.dropout(
            self.ordered_memory_reader(
                self.ordered_memory_norm(x),
                ordered_address,
                ordered_innovations,
                ordered_boundaries,
                self._ordered_encoder_ref,
                padding_mask,
            )
        )
        x = x + self.dropout(
            self.position_memory_reader(
                self.position_memory_norm(x),
                position_memory,
                position_address,
                padding_mask,
            )
        )
        x = x + self.dropout(
            self.feed_forward(self.ffn_norm(x))
        )
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class AATCausalLanguageModel(nn.Module):
    """Finalized canonical AAT adapted to strict causal prefix language modeling."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embedding_norm = nn.LayerNorm(cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.ordered_encoder = ComplexPhaseKeyValueOrderedPrefixEncoder(cfg)
        self.position_encoder = PositionPrefixMemoryEncoder(cfg)
        self.blocks = nn.ModuleList(
            CanonicalDecoderBlock(cfg, self.ordered_encoder)
            for _ in range(cfg.blocks)
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_standard_weights)
        # The accepted K/V model begins at the established shared-content
        # operator and learns away from it.
        initialize_identity(self.ordered_encoder.key_projection)
        initialize_identity(self.ordered_encoder.value_projection)

    @staticmethod
    def _initialize_standard_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        x = self.token_embedding(input_ids.long())
        x = self.embedding_dropout(self.embedding_norm(x))
        padding_mask = torch.zeros(
            x.shape[0],
            x.shape[1],
            device=x.device,
            dtype=torch.bool,
        )

        ordered_state = self.ordered_encoder(x, padding_mask)
        position_memory, _, position_address = self.position_encoder(
            x,
            padding_mask,
        )
        for block in self.blocks:
            block_args = (
                x,
                *ordered_state,
                position_memory,
                position_address,
                padding_mask,
            )
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(
                    block,
                    *block_args,
                    use_reentrant=False,
                )
            else:
                x = block(*block_args)
        return self.final_norm(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden_states(input_ids))

    def parameter_breakdown(self) -> dict[str, int]:
        embedding = self.token_embedding.weight.numel()
        encoder_memories = (
            count_parameters(self.ordered_encoder)
            + count_parameters(self.position_encoder)
        )
        decoder_mixers = sum(
            count_parameters(block.persistent_reader)
            + count_parameters(block.ordered_memory_reader)
            + count_parameters(block.position_memory_reader)
            for block in self.blocks
        )
        ffn = sum(
            count_parameters(block.feed_forward)
            for block in self.blocks
        )
        norms = (
            count_parameters(self.embedding_norm)
            + count_parameters(self.final_norm)
            + sum(
                count_parameters(block.persistent_norm)
                + count_parameters(block.ordered_memory_norm)
                + count_parameters(block.position_memory_norm)
                + count_parameters(block.ffn_norm)
                for block in self.blocks
            )
        )
        return {
            "total": count_parameters(self),
            "tied_token_embedding_and_lm_head": embedding,
            "fixed_encoder_memories": encoder_memories,
            "decoder_memory_mixers": decoder_mixers,
            "parameter_matched_ffn": ffn,
            "normalization": norms,
        }

    @torch.no_grad()
    def architecture_stats(self) -> dict[str, float]:
        key_delta, value_delta, phase_rms = (
            self.ordered_encoder.projection_diagnostics()
        )
        kappa = [
            self.ordered_encoder.response.kappa_mean(),
            self.position_encoder.response.kappa_mean(),
            *[
                block.persistent_reader.response.kappa_mean()
                for block in self.blocks
            ],
        ]
        gates: list[float] = []
        strengths: list[torch.Tensor] = []
        for block in self.blocks:
            gates.extend(
                (
                    float(block.persistent_reader.output_gate.item()),
                    float(block.ordered_memory_reader.output_gate.item()),
                    float(block.position_memory_reader.output_gate.item()),
                )
            )
            strengths.extend(
                (
                    block.persistent_reader.ray_strength.detach().float().reshape(-1),
                    block.ordered_memory_reader.ray_strength.detach().float().reshape(-1),
                    block.position_memory_reader.ray_strength.detach().float().reshape(-1),
                )
            )
        strength = torch.cat(strengths)
        return {
            "kappa_mean": sum(kappa) / len(kappa),
            "read_gate_mean": sum(gates) / len(gates),
            "ray_strength_mean": float(strength.mean().item()),
            "ray_strength_std": float(strength.std(unbiased=False).item()),
            "ray_strength_min": float(strength.min().item()),
            "ray_strength_max": float(strength.max().item()),
            "key_projection_delta": key_delta,
            "value_projection_delta": value_delta,
            "phase_scale_rms": phase_rms,
            "ordered_chunk_size": float(self.cfg.ordered_chunk_size),
            "position_scale": float(self.cfg.position_scale),
        }


def sequential_ordered_prefix_reads(
    address: torch.Tensor,
    payload: torch.Tensor,
    query_address: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, heads, token_count, rays = address.shape
    memory = payload.new_zeros(
        batch_size,
        heads,
        rays,
        payload.shape[-1],
    )
    outputs: list[torch.Tensor] = []
    for token_index in range(token_count):
        valid = (~padding_mask[:, token_index])[:, None, None].to(payload.dtype)
        write_address = address[:, :, token_index] * valid
        value = payload[:, :, token_index] * valid
        prediction = torch.einsum(
            "bhr,bhrd->bhd",
            write_address,
            memory,
        )
        innovation = value - prediction
        memory = memory + (
            write_address.unsqueeze(-1)
            * innovation.unsqueeze(-2)
        )
        outputs.append(
            torch.einsum(
                "bhr,bhrd->bhd",
                query_address[:, :, token_index],
                memory,
            )
        )
    return torch.stack(outputs, dim=2).masked_fill(
        padding_mask[:, None, :, None],
        0.0,
    )


@torch.no_grad()
def verify_causal_memory_math() -> None:
    generator = torch.Generator().manual_seed(271828)
    address = torch.randn(2, 3, 11, 4, generator=generator).softmax(dim=-1)
    payload = torch.randn(2, 3, 11, 10, generator=generator)
    query = torch.randn(2, 3, 11, 4, generator=generator).softmax(dim=-1)
    padding_mask = torch.zeros(2, 11, dtype=torch.bool)
    padding_mask[1, 8:] = True

    state = chunkwise_ordered_prefix_state(
        address,
        payload,
        padding_mask,
        chunk_size=4,
    )
    actual_ordered = chunkwise_ordered_prefix_read(
        query,
        *state,
        padding_mask,
    )
    expected_ordered = sequential_ordered_prefix_reads(
        address,
        payload,
        query,
        padding_mask,
    )
    if not torch.allclose(
        actual_ordered,
        expected_ordered,
        atol=3e-5,
        rtol=3e-5,
    ):
        error = float(
            (actual_ordered - expected_ordered).abs().max().item()
        )
        raise RuntimeError(
            f"chunkwise Ordered verification failed: max_error={error}"
        )

    values = torch.randn(2, 3, 11, 5, generator=generator)
    scores = torch.randn(2, 3, 11, 4, generator=generator)
    actual_persistent = causal_prefix_weighted_memory(
        values,
        scores,
        padding_mask,
        score_clip=30.0,
    )
    numerator = torch.zeros(2, 3, 4, 5)
    denominator = torch.zeros(2, 3, 4)
    expected_prefixes: list[torch.Tensor] = []
    for token_index in range(11):
        valid = (~padding_mask[:, token_index])[:, None, None].float()
        weight = scores[:, :, token_index].exp() * valid
        value = values[:, :, token_index]
        numerator = numerator + weight.unsqueeze(-1) * value.unsqueeze(-2)
        denominator = denominator + weight
        expected_prefixes.append(
            numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
        )
    expected_persistent = torch.stack(expected_prefixes, dim=2).masked_fill(
        padding_mask[:, None, :, None, None],
        0.0,
    )
    if not torch.allclose(
        actual_persistent,
        expected_persistent,
        atol=2e-5,
        rtol=2e-5,
    ):
        error = float(
            (actual_persistent - expected_persistent).abs().max().item()
        )
        raise RuntimeError(
            f"Prefix-Persistent verification failed: max_error={error}"
        )


@torch.no_grad()
def verify_no_future_leak(
    model: AATCausalLanguageModel,
    device: torch.device,
) -> None:
    """Verify content causality without mistaking CUDA shape noise for leakage.

    The decisive causality test keeps the physical tensor shape fixed and
    changes only tokens after ``prefix_length``.  Prefix-versus-short execution
    is also checked, but it is a numerical-consistency diagnostic: changing the
    token count changes many CUDA GEMM shapes (including the 50k-wide tied
    language-model head), so two mathematically identical prefixes need not be
    bitwise-close at the same tolerance as the same-shape causality test.
    """
    model.eval()
    generator = torch.Generator().manual_seed(161803)
    prefix_length = 7
    sequence_long = torch.randint(
        0,
        model.cfg.vocab_size,
        (2, 13),
        generator=generator,
    )
    sequence_changed = sequence_long.clone()
    sequence_changed[:, prefix_length:] = torch.randint(
        0,
        model.cfg.vocab_size,
        sequence_changed[:, prefix_length:].shape,
        generator=generator,
    )
    sequence_short = sequence_long[:, :prefix_length].clone()

    # Keep this diagnostic in strict FP32.  Training/evaluation still use the
    # throughput-oriented global TF32/AMP configuration selected by the user.
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32: bool | None = None
    previous_cudnn_tf32: bool | None = None
    if device.type == "cuda":
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        hidden_long = model.hidden_states(sequence_long.to(device))
        hidden_changed = model.hidden_states(sequence_changed.to(device))
        hidden_short = model.hidden_states(sequence_short.to(device))

        hidden_long_prefix = hidden_long[:, :prefix_length]
        hidden_changed_prefix = hidden_changed[:, :prefix_length]

        # Apply the large tied vocabulary projection to every candidate prefix
        # in one call.  This removes an avoidable source of shape-dependent
        # cuBLAS rounding from three separately sized lm_head invocations.
        projected = model.lm_head(
            torch.cat(
                (
                    hidden_long_prefix,
                    hidden_changed_prefix,
                    hidden_short,
                ),
                dim=0,
            )
        ).float().cpu()
        batch_size = sequence_long.shape[0]
        logits_long, logits_changed, logits_short = projected.split(
            batch_size,
            dim=0,
        )

        hidden_long_prefix = hidden_long_prefix.float().cpu()
        hidden_changed_prefix = hidden_changed_prefix.float().cpu()
        hidden_short = hidden_short.float().cpu()
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        if device.type == "cuda":
            assert previous_matmul_tf32 is not None
            assert previous_cudnn_tf32 is not None
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    # This is the actual no-future-leak test: both runs have exactly the same
    # shape, and only future token identities differ.
    for label, reference, other in (
        (
            "future-token hidden-state",
            hidden_long_prefix,
            hidden_changed_prefix,
        ),
        (
            "future-token logits",
            logits_long,
            logits_changed,
        ),
    ):
        if not torch.allclose(
            reference,
            other,
            atol=5e-5,
            rtol=5e-5,
        ):
            error = float((reference - other).abs().max().item())
            raise RuntimeError(
                f"{label} causality verification failed: max_error={error}"
            )

    # Different sequence lengths can select different CUDA kernels and
    # reduction orders throughout the network.  A milliscale discrepancy here
    # is therefore not evidence of future-token dependence.  Keep a bounded
    # consistency guard to catch real implementation mistakes without turning
    # harmless floating-point path differences into a hard failure.
    prefix_hidden_error = float(
        (hidden_long_prefix - hidden_short).abs().max().item()
    )
    prefix_logit_error = float(
        (logits_long - logits_short).abs().max().item()
    )
    prefix_error = max(prefix_hidden_error, prefix_logit_error)
    prefix_consistency_limit = 2e-3
    if (
        prefix_hidden_error > prefix_consistency_limit
        or prefix_logit_error > prefix_consistency_limit
    ):
        raise RuntimeError(
            "prefix-extension numerical consistency failed: "
            f"hidden_max_error={prefix_hidden_error}, "
            f"logit_max_error={prefix_logit_error}, "
            f"limit={prefix_consistency_limit}"
        )
    if prefix_error > 5e-5:
        print(
            "Prefix-extension numerical note: "
            f"hidden_max_error={prefix_hidden_error:.3e}, "
            f"logit_max_error={prefix_logit_error:.3e}; "
            "same-shape future-token causality is exact within strict tolerance.",
            flush=True,
        )


def smoke_test(device: torch.device) -> None:
    print(
        "Running canonical scan, prefix-causality, gradient, and checkpoint tests...",
        flush=True,
    )
    verify_causal_memory_math()
    cfg = ModelConfig(
        vocab_size=101,
        d_model=32,
        blocks=2,
        heads=4,
        rays_per_head=4,
        ffn_hidden=32,
        ordered_chunk_size=4,
        position_scale=63.0,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    model = AATCausalLanguageModel(cfg).to(device)
    verify_no_future_leak(model, device)
    model.train()
    tokens = torch.randint(
        0,
        cfg.vocab_size,
        (2, 12),
        device=device,
    )
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    if not math.isfinite(float(loss.item())):
        raise RuntimeError("smoke loss is not finite")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(
            parameter.grad
        ).all():
            raise RuntimeError(f"non-finite gradient in {name}")
    required_gradients = (
        model.ordered_encoder.key_projection.weight,
        model.ordered_encoder.value_projection.weight,
        model.ordered_encoder.phase_scale,
        model.position_encoder.response.ray_base,
        model.blocks[0].persistent_reader.response.ray_base,
        model.blocks[0].ordered_memory_reader.ray_strength,
        model.blocks[0].position_memory_reader.ray_strength,
    )
    if any(parameter.grad is None for parameter in required_gradients):
        raise RuntimeError("a canonical memory path did not receive gradients")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        atomic_torch_save(
            {
                "model_config": asdict(cfg),
                "model_state_dict": model.state_dict(),
            },
            path,
        )
        try:
            payload = torch.load(
                path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(path, map_location=device)
        restored = AATCausalLanguageModel(
            ModelConfig(**payload["model_config"])
        ).to(device)
        restored.load_state_dict(payload["model_state_dict"])
        restored.eval()
        model.eval()
        with torch.no_grad():
            left = model(tokens[:, :-1])
            right = restored(tokens[:, :-1])
        if not torch.equal(left, right):
            raise RuntimeError(
                "checkpoint round-trip changed canonical model outputs"
            )
    print("All canonical smoke tests: OK", flush=True)


# =================================================================================================
# WikiText data pipeline
# =================================================================================================


def require_huggingface():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Run: pip install datasets transformers") from exc
    return load_dataset, AutoTokenizer


def prepare_dataset_split(
    *,
    raw_split: Any,
    tokenizer: Any,
    split_name: str,
    sequence_length: int,
    preprocessing_workers: int,
    max_blocks: int | None,
) -> Any:
    block_size = sequence_length + 1
    eos_token_id = int(tokenizer.eos_token_id)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [text for text in batch["text"] if text and text.strip()]
        if not texts:
            return {"input_ids": []}
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        return {"input_ids": [ids + [eos_token_id] for ids in encoded]}

    print(f"Tokenizing {split_name} with {preprocessing_workers} process(es)...", flush=True)
    tokenized = raw_split.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=preprocessing_workers,
        remove_columns=raw_split.column_names,
        desc=f"Tokenize {split_name}",
    )

    def group_batch(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
        usable = (len(concatenated) // block_size) * block_size
        return {
            "input_ids": [
                concatenated[start : start + block_size]
                for start in range(0, usable, block_size)
            ]
        }

    grouped = tokenized.map(
        group_batch,
        batched=True,
        batch_size=1000,
        num_proc=preprocessing_workers,
        desc=f"Pack {split_name} into {block_size}-token blocks",
    )
    if max_blocks is not None and len(grouped) > max_blocks:
        grouped = grouped.select(range(max_blocks))
    if len(grouped) == 0:
        raise RuntimeError(f"{split_name} produced no language-model blocks")
    grouped.set_format(type="torch", columns=["input_ids"])
    print(
        f"{split_name}: {len(grouped):,} blocks, "
        f"{len(grouped) * sequence_length:,} prediction tokens",
        flush=True,
    )
    return grouped


def collate_token_blocks(examples: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """Top-level collator so DataLoader workers also work with Windows spawn."""
    return torch.stack([example["input_ids"] for example in examples], dim=0).long()


def make_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_token_blocks,
        drop_last=shuffle,
    )


# =================================================================================================
# Optimization, evaluation, checkpointing, and generation
# =================================================================================================


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    parameter_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {
        "lr": cfg.learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    use_fused = next(model.parameters()).device.type == "cuda"
    if use_fused:
        try:
            return torch.optim.AdamW(
                parameter_groups,
                fused=True,
                **kwargs,
            )
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameter_groups, **kwargs)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_fraction: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_updates = max(1, int(round(total_updates * warmup_fraction)))

    def multiplier(step: int) -> float:
        if step < warmup_updates:
            return max(1e-8, (step + 1) / warmup_updates)
        progress = min(
            1.0,
            (step - warmup_updates) / max(1, total_updates - warmup_updates),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.inference_mode()
def evaluate(
    model: AATCausalLanguageModel,
    loader: DataLoader,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
    split_name: str,
    max_batches: int | None,
) -> EvalResult:
    model.eval()
    loss_sum = 0.0
    correct = 0
    token_count = 0
    started = time.perf_counter()
    for batch_index, tokens in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokens = tokens.to(device, non_blocking=True)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        with autocast_context(device, autocast_dtype):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
        loss_sum += float(loss.item())
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        token_count += targets.numel()
    if token_count == 0:
        raise RuntimeError(f"evaluation split {split_name} contained no tokens")
    seconds = time.perf_counter() - started
    mean_loss = loss_sum / token_count
    return EvalResult(
        split=split_name,
        loss=mean_loss,
        perplexity=math.exp(min(20.0, mean_loss)),
        bits_per_token=mean_loss / math.log(2.0),
        token_accuracy=correct / token_count,
        tokens=token_count,
        seconds=seconds,
        tokens_per_second=token_count / max(seconds, 1e-9),
    )


def print_eval(result: EvalResult) -> None:
    print(
        f"[{result.split}] loss={result.loss:.4f} ppl={result.perplexity:.2f} "
        f"bpt={result.bits_per_token:.3f} token_acc={result.token_accuracy:.4f} "
        f"tokens={result.tokens:,} tok/s={result.tokens_per_second:,.0f}",
        flush=True,
    )


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_checkpoint(
    *,
    model: AATCausalLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    epoch: int,
    batch_in_epoch: int,
    global_update: int,
    tokens_seen: int,
    best_validation_loss: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "source_sha256": source_sha256(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_update": global_update,
        "tokens_seen": tokens_seen,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "history": history,
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.inference_mode()
def generate_text(
    model: AATCausalLanguageModel,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_new_tokens: int,
    max_context: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if ids.numel() == 0:
        ids = torch.tensor([[int(tokenizer.eos_token_id)]], device=device)
    for _ in range(max_new_tokens):
        context = ids[:, -max_context:]
        with autocast_context(device, autocast_dtype):
            # Calling forward directly avoids compiling a fresh graph for every
            # autoregressive context length when --compile is enabled.
            logits = model.forward(context)[:, -1].float() / max(
                temperature,
                1e-5,
            )
        k = min(max(1, top_k), logits.shape[-1])
        values, indices = torch.topk(logits, k=k, dim=-1)
        probabilities = F.softmax(values, dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
        next_token = indices.gather(-1, sampled)
        ids = torch.cat((ids, next_token), dim=1)
        if int(next_token.item()) == int(tokenizer.eos_token_id):
            break
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def average_architecture_stats(model: AATCausalLanguageModel) -> dict[str, float]:
    return model.architecture_stats()


# =================================================================================================
# Training
# =================================================================================================


def train(
    *,
    model: AATCausalLanguageModel,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_dataset: Any,
    validation_loader: DataLoader,
    long_validation_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_scaler: bool,
    resume_path: Path | None,
) -> tuple[list[dict[str, Any]], EvalResult]:
    batches_per_epoch = math.floor(len(train_dataset) / train_cfg.batch_size)
    if batches_per_epoch <= 0:
        raise RuntimeError(
            "training set is smaller than one batch; lower --batch-size or increase --max-train-blocks"
        )
    updates_per_epoch = math.ceil(batches_per_epoch / train_cfg.grad_accum_steps)
    total_updates = max(1, updates_per_epoch * train_cfg.epochs)
    optimizer = make_optimizer(model, train_cfg)
    scheduler = make_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_fraction=train_cfg.warmup_fraction,
        min_lr_ratio=train_cfg.min_lr_ratio,
    )
    scaler = make_grad_scaler(use_scaler)

    start_epoch = 0
    resume_batch = -1
    global_update = 0
    tokens_seen = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []

    if resume_path is not None:
        payload = load_checkpoint(resume_path, device)
        if payload.get("implementation_version") != IMPLEMENTATION_VERSION:
            raise RuntimeError(
                "checkpoint implementation mismatch: "
                f"{payload.get('implementation_version')} != {IMPLEMENTATION_VERSION}"
            )
        if payload.get("model_config") != asdict(model_cfg):
            raise RuntimeError("checkpoint model_config does not match this run")
        if payload.get("train_config") != asdict(train_cfg):
            raise RuntimeError("checkpoint train_config does not match this run")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload.get("scaler_state_dict", {}))
        start_epoch = int(payload["epoch"])
        resume_batch = int(payload["batch_in_epoch"])
        global_update = int(payload["global_update"])
        tokens_seen = int(payload["tokens_seen"])
        best_validation_loss = float(payload["best_validation_loss"])
        history = list(payload.get("history", []))
        restore_rng_state(payload.get("rng_state"))
        print(
            f"Resumed {resume_path}: epoch={start_epoch + 1}, "
            f"batch={resume_batch + 1}, update={global_update:,}, "
            f"tokens_seen={tokens_seen:,}",
            flush=True,
        )

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    most_recent_validation: EvalResult | None = None

    for epoch in range(start_epoch, train_cfg.epochs):
        train_loader = make_loader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            seed=train_cfg.seed + epoch,
            num_workers=train_cfg.num_workers,
            device=device,
        )
        model.train()
        interval_loss = 0.0
        interval_correct = 0
        interval_tokens = 0
        interval_started = time.perf_counter()
        accumulated_batches = 0
        batch_count = len(train_loader)

        for batch_index, tokens in enumerate(train_loader):
            if epoch == start_epoch and batch_index <= resume_batch:
                continue
            tokens = tokens.to(device, non_blocking=True)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            with autocast_context(device, autocast_dtype):
                logits = model(inputs)
                raw_loss = F.cross_entropy(
                    logits.reshape(-1, model_cfg.vocab_size), targets.reshape(-1)
                )
                scaled_loss = raw_loss / train_cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            accumulated_batches += 1
            batch_tokens = targets.numel()
            tokens_seen += batch_tokens
            interval_loss += float(raw_loss.item()) * batch_tokens
            interval_correct += int((logits.detach().argmax(dim=-1) == targets).sum().item())
            interval_tokens += batch_tokens
            is_last_batch = batch_index + 1 == batch_count
            should_update = accumulated_batches >= train_cfg.grad_accum_steps or is_last_batch
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            if accumulated_batches < train_cfg.grad_accum_steps:
                # Losses above were divided by the nominal accumulation count.
                # Correct the final partial update so it is still an average of
                # the batches actually present, not an artificially small step.
                correction = train_cfg.grad_accum_steps / accumulated_batches
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item()
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0
            if step_was_skipped:
                print("AMP overflow: optimizer update skipped", flush=True)
                continue
            scheduler.step()
            global_update += 1

            if global_update % train_cfg.log_every == 0:
                now = time.perf_counter()
                elapsed = now - started
                interval_seconds = now - interval_started
                mean_loss = interval_loss / max(1, interval_tokens)
                completed_fraction = global_update / total_updates
                eta = elapsed * (1.0 - completed_fraction) / max(completed_fraction, 1e-9)
                print(
                    f"epoch={epoch + 1:02d}/{train_cfg.epochs} "
                    f"batch={batch_index + 1:,}/{batch_count:,} "
                    f"update={global_update:,}/{total_updates:,} "
                    f"loss={mean_loss:.4f} ppl={math.exp(min(20.0, mean_loss)):.2f} "
                    f"acc={interval_correct / max(1, interval_tokens):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} grad={grad_norm:.3f} "
                    f"tok/s={interval_tokens / max(interval_seconds, 1e-9):,.0f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                    flush=True,
                )
                interval_loss = 0.0
                interval_correct = 0
                interval_tokens = 0
                interval_started = now

            checkpoint_payload = None
            if global_update % train_cfg.save_every == 0:
                checkpoint_payload = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    epoch=epoch,
                    batch_in_epoch=batch_index,
                    global_update=global_update,
                    tokens_seen=tokens_seen,
                    best_validation_loss=best_validation_loss,
                    history=history,
                )
                atomic_torch_save(checkpoint_payload, output_dir / "latest.pt")
                print(f"Saved latest checkpoint at update {global_update:,}", flush=True)

            if global_update % train_cfg.eval_every == 0:
                validation = evaluate(
                    model,
                    validation_loader,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    split_name="validation",
                    max_batches=train_cfg.eval_batches,
                )
                print_eval(validation)
                row = {
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "update": global_update,
                    "tokens_seen": tokens_seen,
                    "validation_loss": validation.loss,
                    "validation_perplexity": validation.perplexity,
                    "validation_bits_per_token": validation.bits_per_token,
                    "validation_token_accuracy": validation.token_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(row)
                append_history(output_dir / "history.csv", row)
                most_recent_validation = validation
                improved = validation.loss < best_validation_loss
                if improved:
                    best_validation_loss = validation.loss
                checkpoint_payload = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    epoch=epoch,
                    batch_in_epoch=batch_index,
                    global_update=global_update,
                    tokens_seen=tokens_seen,
                    best_validation_loss=best_validation_loss,
                    history=history,
                )
                atomic_torch_save(checkpoint_payload, output_dir / "latest.pt")
                if improved:
                    atomic_torch_save(checkpoint_payload, output_dir / "best.pt")
                    print(
                        f"New best validation perplexity: {validation.perplexity:.2f}",
                        flush=True,
                    )
                model.train()

        # Always evaluate and checkpoint at the end of an epoch, including when
        # eval_every is larger than the epoch.
        validation = evaluate(
            model,
            validation_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            split_name="validation",
            max_batches=train_cfg.eval_batches,
        )
        long_validation = evaluate(
            model,
            long_validation_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            split_name=f"validation_long_{train_cfg.long_sequence_length}",
            max_batches=train_cfg.eval_batches,
        )
        print_eval(validation)
        print_eval(long_validation)
        most_recent_validation = validation
        row = {
            "epoch": epoch + 1,
            "batch": batch_count,
            "update": global_update,
            "tokens_seen": tokens_seen,
            "validation_loss": validation.loss,
            "validation_perplexity": validation.perplexity,
            "validation_bits_per_token": validation.bits_per_token,
            "validation_token_accuracy": validation.token_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        append_history(output_dir / "history.csv", row)
        improved = validation.loss < best_validation_loss
        if improved:
            best_validation_loss = validation.loss
        payload = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            epoch=epoch + 1,
            batch_in_epoch=-1,
            global_update=global_update,
            tokens_seen=tokens_seen,
            best_validation_loss=best_validation_loss,
            history=history,
        )
        atomic_torch_save(payload, output_dir / "latest.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
            print(f"New best validation perplexity: {validation.perplexity:.2f}", flush=True)
        resume_batch = -1

    if most_recent_validation is None:
        most_recent_validation = evaluate(
            model,
            validation_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            split_name="validation",
            max_batches=train_cfg.eval_batches,
        )
    return history, most_recent_validation


# =================================================================================================
# CLI and main
# =================================================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the canonical Complex-Phase Dual-Memory Prefix AAT "
            "as a causal LM on WikiText."
        )
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    parser.add_argument("--smoke-test-only", action="store_true")

    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--rays-per-head", type=int, default=None)
    parser.add_argument("--ordered-chunk-size", type=int, default=None)
    parser.add_argument("--ffn-hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--long-sequence-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 means the full split")
    parser.add_argument("--max-train-blocks", type=int, default=None)
    parser.add_argument("--max-eval-blocks", type=int, default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile the model with torch.compile. This is optional because "
            "backend support varies, especially on Windows."
        ),
    )
    parser.add_argument("--generate-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def resolve_resume(value: str, output_dir: Path) -> Path | None:
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        candidate = output_dir / "latest.pt"
        return candidate if candidate.exists() else None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    return path


def preset_value(override: Any, preset: Preset, field: str) -> Any:
    return getattr(preset, field) if override is None else override


def main() -> None:
    args = parse_arguments()
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    smoke_test(device)
    if args.smoke_test_only:
        return

    load_dataset, AutoTokenizer = require_huggingface()
    preset = PRESETS[args.preset]
    dataset_config = args.dataset_config or preset.dataset_config
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("the tokenizer must define eos_token_id")

    sequence_length = preset_value(
        args.sequence_length,
        preset,
        "sequence_length",
    )
    model_cfg = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=preset_value(args.d_model, preset, "d_model"),
        blocks=preset_value(args.blocks, preset, "blocks"),
        heads=preset_value(args.heads, preset, "heads"),
        rays_per_head=preset_value(args.rays_per_head, preset, "rays_per_head"),
        ffn_hidden=preset_value(args.ffn_hidden, preset, "ffn_hidden"),
        ordered_chunk_size=preset_value(
            args.ordered_chunk_size,
            preset,
            "ordered_chunk_size",
        ),
        position_scale=float(max(1, sequence_length - 1)),
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    model_cfg.validate()
    train_cfg = TrainConfig(
        dataset_name=args.dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=args.tokenizer,
        sequence_length=sequence_length,
        long_sequence_length=preset_value(
            args.long_sequence_length, preset, "long_sequence_length"
        ),
        epochs=preset_value(args.epochs, preset, "epochs"),
        batch_size=preset_value(args.batch_size, preset, "batch_size"),
        grad_accum_steps=preset_value(
            args.grad_accum_steps, preset, "grad_accum_steps"
        ),
        learning_rate=preset_value(args.learning_rate, preset, "learning_rate"),
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        seed=args.seed,
        num_workers=args.num_workers,
        preprocessing_workers=max(1, args.preprocessing_workers),
        log_every=max(1, args.log_every),
        save_every=max(1, args.save_every),
        eval_every=max(1, args.eval_every),
        eval_batches=None if args.eval_batches <= 0 else args.eval_batches,
        max_train_blocks=(
            args.max_train_blocks
            if args.max_train_blocks is not None
            else preset.max_train_blocks
        ),
        max_eval_blocks=(
            args.max_eval_blocks
            if args.max_eval_blocks is not None
            else preset.max_eval_blocks
        ),
    )
    if train_cfg.sequence_length <= 1 or train_cfg.long_sequence_length <= 1:
        raise ValueError("sequence lengths must be greater than one")
    if min(train_cfg.epochs, train_cfg.batch_size, train_cfg.grad_accum_steps) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if not 0.0 <= train_cfg.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0,1)")
    if not 0.0 < train_cfg.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0,1]")

    output_dir = args.output_dir or Path(
        f"runs/canonical_aat_wikitext_{args.preset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_resume(args.resume, output_dir)
    if resume_path is None and (output_dir / "latest.pt").exists() and args.resume.lower() == "none":
        raise RuntimeError(
            f"{output_dir}/latest.pt already exists; choose a new output directory "
            "or use --resume auto"
        )
    try:
        shutil.copy2(Path(__file__), output_dir / "source_snapshot.py")
    except OSError as exc:
        print(f"Warning: could not save source snapshot: {exc}", flush=True)
    atomic_write_json(asdict(model_cfg), output_dir / "model_config.json")
    atomic_write_json(asdict(train_cfg), output_dir / "run_config.json")
    atomic_write_json(environment_info(device), output_dir / "environment.json")

    autocast_dtype, use_scaler, precision_name = choose_precision(args.precision, device)
    model = AATCausalLanguageModel(model_cfg).to(device)
    verify_no_future_leak(model, device)
    breakdown = model.parameter_breakdown()
    compile_enabled = False
    if args.compile:
        compile_method = getattr(model, "compile", None)
        if compile_method is None:
            print(
                "Warning: this PyTorch build has no nn.Module.compile; "
                "continuing without compilation.",
                flush=True,
            )
        else:
            try:
                compile_method(mode="default", fullgraph=False, dynamic=False)
                compile_enabled = True
            except Exception as exc:
                print(
                    f"Warning: torch.compile setup failed ({exc}); "
                    "continuing with eager GPU kernels.",
                    flush=True,
                )
    print("=" * 110, flush=True)
    print(
        "Canonical Complex-Phase Dual-Memory Prefix AAT | causal language modeling",
        flush=True,
    )
    print("=" * 110, flush=True)
    print(
        f"implementation={IMPLEMENTATION_VERSION} source_sha256={source_sha256()[:12]} "
        f"device={device} precision={precision_name} compile={compile_enabled}",
        flush=True,
    )
    print(
        f"dataset={train_cfg.dataset_name}/{train_cfg.dataset_config} "
        f"tokenizer={train_cfg.tokenizer_name} vocab={model_cfg.vocab_size:,}",
        flush=True,
    )
    print(
        f"model: d={model_cfg.d_model} blocks={model_cfg.blocks} heads={model_cfg.heads} "
        f"rays/head={model_cfg.rays_per_head} ordered_chunk={model_cfg.ordered_chunk_size} "
        f"ffn={model_cfg.ffn_hidden} seq={train_cfg.sequence_length} "
        f"long_seq={train_cfg.long_sequence_length}",
        flush=True,
    )
    print(
        "parameters: " + ", ".join(
            f"{name}={value:,} ({human_count(value)})" for name, value in breakdown.items()
        ),
        flush=True,
    )
    print(
        "Causality checks passed. The memory core is chunkwise/vectorized and "
        "linear in T for fixed rays and chunk size; no T-by-T attention matrix is built.",
        flush=True,
    )

    print("Loading WikiText splits...", flush=True)
    raw = load_dataset(
        train_cfg.dataset_name,
        train_cfg.dataset_config,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    train_dataset = prepare_dataset_split(
        raw_split=raw["train"],
        tokenizer=tokenizer,
        split_name="train",
        sequence_length=train_cfg.sequence_length,
        preprocessing_workers=train_cfg.preprocessing_workers,
        max_blocks=train_cfg.max_train_blocks,
    )
    validation_dataset = prepare_dataset_split(
        raw_split=raw["validation"],
        tokenizer=tokenizer,
        split_name="validation",
        sequence_length=train_cfg.sequence_length,
        preprocessing_workers=train_cfg.preprocessing_workers,
        max_blocks=train_cfg.max_eval_blocks,
    )
    long_validation_dataset = prepare_dataset_split(
        raw_split=raw["validation"],
        tokenizer=tokenizer,
        split_name=f"validation_long_{train_cfg.long_sequence_length}",
        sequence_length=train_cfg.long_sequence_length,
        preprocessing_workers=train_cfg.preprocessing_workers,
        max_blocks=train_cfg.max_eval_blocks,
    )
    test_dataset = prepare_dataset_split(
        raw_split=raw["test"],
        tokenizer=tokenizer,
        split_name="test",
        sequence_length=train_cfg.sequence_length,
        preprocessing_workers=train_cfg.preprocessing_workers,
        max_blocks=train_cfg.max_eval_blocks,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
        device=device,
    )
    long_validation_loader = make_loader(
        long_validation_dataset,
        batch_size=max(1, train_cfg.batch_size // 2),
        shuffle=False,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
        device=device,
    )

    started = time.perf_counter()
    history, last_validation = train(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_dataset=train_dataset,
        validation_loader=validation_loader,
        long_validation_loader=long_validation_loader,
        output_dir=output_dir,
        device=device,
        autocast_dtype=autocast_dtype,
        use_scaler=use_scaler,
        resume_path=resume_path,
    )
    training_seconds = time.perf_counter() - started

    best_path = output_dir / "best.pt"
    if best_path.exists():
        best_payload = load_checkpoint(best_path, device)
        model.load_state_dict(best_payload["model_state_dict"])
        print(
            f"Loaded best checkpoint (validation ppl="
            f"{math.exp(min(20.0, float(best_payload['best_validation_loss']))):.2f})",
            flush=True,
        )
    test_result = evaluate(
        model,
        test_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        split_name="test",
        max_batches=train_cfg.eval_batches,
    )
    long_result = evaluate(
        model,
        long_validation_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        split_name=f"validation_long_{train_cfg.long_sequence_length}",
        max_batches=train_cfg.eval_batches,
    )
    print_eval(test_result)
    print_eval(long_result)

    prompts = [
        "The history of science",
        "In the city of London",
        "Researchers discovered that",
    ]
    generated: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts):
        text = generate_text(
            model,
            tokenizer,
            prompt,
            device=device,
            autocast_dtype=autocast_dtype,
            max_new_tokens=args.generate_tokens,
            max_context=train_cfg.sequence_length,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=train_cfg.seed + 1000 + index,
        )
        generated.append({"prompt": prompt, "text": text})
        print(f"\n[Generation {index + 1}]\n{text}\n", flush=True)
    (output_dir / "generated_samples.txt").write_text(
        "\n\n".join(
            f"PROMPT: {sample['prompt']}\n\n{sample['text']}" for sample in generated
        ) + "\n",
        encoding="utf-8",
    )

    results = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "source_sha256": source_sha256(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "parameter_breakdown": breakdown,
        "best_validation_loss": min(
            [row["validation_loss"] for row in history] or [last_validation.loss]
        ),
        "best_validation_perplexity": min(
            [row["validation_perplexity"] for row in history]
            or [last_validation.perplexity]
        ),
        "test": asdict(test_result),
        "long_validation": asdict(long_result),
        "architecture_stats": average_architecture_stats(model),
        "environment": environment_info(device),
        "training_seconds_this_invocation": training_seconds,
        "generated_samples": generated,
    }
    atomic_write_json(results, output_dir / "final_results.json")
    print("=" * 110, flush=True)
    print(
        f"FINAL: test_loss={test_result.loss:.4f} test_ppl={test_result.perplexity:.2f} "
        f"test_acc={test_result.token_accuracy:.4f} "
        f"long_val_ppl={long_result.perplexity:.2f} "
        f"runtime={format_duration(training_seconds)}",
        flush=True,
    )
    print(f"Artifacts: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
