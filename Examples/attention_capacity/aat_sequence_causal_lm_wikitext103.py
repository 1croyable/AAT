from __future__ import annotations

r"""
Deep-Transport AAT sequence model on WikiText-103.

This is the controlled follow-up to the earlier ~41.72M-parameter AAT and
Transformer language-model runs. It preserves the previous data, tokenizer,
optimization, evaluation, checkpoint, generation, and artifact protocol. Only
the model core is replaced by the frozen aat.sequence implementation.

The default paper preset uses:

    - WikiText-103 raw and the GPT-2 tokenizer
    - tied input/output token embeddings
    - d_model=512
    - 6 AAT encoder refinement layers
    - one final shared Ordered + Position memory write
    - 12 AAT decoder layers
    - 16 heads, 24 rays/head
    - 3 geometric transport steps before every final ray address
    - no FFN and no additional learned positional embedding
    - sequence length 256 and long-sequence evaluation length 512
    - the exact previous training protocol: 3 epochs, microbatch 8,
      gradient accumulation 8, AdamW, 2e-4 peak learning rate

Install the repository and experiment dependencies first:

    pip install -e .
    pip install datasets transformers

Recommended full run:

    python Examples/attention_capacity/aat_sequence_causal_lm_wikitext103.py \
        --device cuda --precision auto

Resume is automatic when <output-dir>/latest.pt exists. Useful checks:

    python Examples/attention_capacity/aat_sequence_causal_lm_wikitext103.py \
        --smoke-test-only
    python Examples/attention_capacity/aat_sequence_causal_lm_wikitext103.py \
        --preset quick --device cuda

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

from aat.sequence import AAT, AATConfig


IMPLEMENTATION_VERSION = "deep-transport-sequence-aat-e6-d12-v1"
REFERENCE_PAPER_PARAMETERS = 41_835_926


# =================================================================================================
# Configuration
# =================================================================================================


@dataclass(frozen=True)
class Preset:
    name: str
    dataset_config: str
    d_model: int
    encoder_layers: int
    decoder_layers: int
    heads: int
    rays: int
    chunk_size: int
    transport_steps: int
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
        encoder_layers=6,
        decoder_layers=12,
        heads=16,
        rays=24,
        chunk_size=32,
        transport_steps=3,
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
        encoder_layers=4,
        decoder_layers=8,
        heads=12,
        rays=24,
        chunk_size=32,
        transport_steps=3,
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
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        rays=8,
        chunk_size=16,
        transport_steps=3,
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
    encoder_layers: int
    decoder_layers: int
    heads: int
    rays: int
    chunk_size: int
    position_scale: float
    transport_steps: int = 3
    dropout: float = 0.10
    kappa: float = 6.0
    score_clip: float = 30.0
    gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    def to_aat_config(self) -> AATConfig:
        return AATConfig(
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            heads=self.heads,
            rays=self.rays,
            chunk_size=self.chunk_size,
            position_scale=self.position_scale,
            transport_steps=self.transport_steps,
            dropout=self.dropout,
            kappa=self.kappa,
            score_clip=self.score_clip,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def validate(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        self.to_aat_config().validate()


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
# Frozen aat.sequence language-model wrapper
# =================================================================================================


class AATCausalLanguageModel(nn.Module):
    """Tied-token language model whose only sequence core is aat.sequence.AAT."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.core = AAT(cfg.to_aat_config())
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        return self.core(self.token_embedding(input_ids.long()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden_states(input_ids))

    def parameter_breakdown(self) -> dict[str, int]:
        embedding = self.token_embedding.weight.numel()
        encoder_layers = count_parameters(self.core.encoder.layers)
        final_memory_writer = (
            count_parameters(self.core.encoder.ordered_memory)
            + count_parameters(self.core.encoder.position_memory)
        )
        decoder_layers = count_parameters(self.core.decoder)
        input_output_norms = (
            count_parameters(self.core.input_norm)
            + count_parameters(self.core.output_norm)
        )
        aat_core = (
            encoder_layers
            + final_memory_writer
            + decoder_layers
            + input_output_norms
        )
        actual_core = count_parameters(self.core)
        if aat_core != actual_core:
            raise RuntimeError(
                f"AAT parameter accounting mismatch: {aat_core} != {actual_core}"
            )
        return {
            "total": count_parameters(self),
            "tied_token_embedding_and_lm_head": embedding,
            "aat_sequence_core": actual_core,
            "encoder_refinement_layers": encoder_layers,
            "final_memory_writer": final_memory_writer,
            "decoder_layers": decoder_layers,
            "input_output_norms": input_output_norms,
        }

    @torch.no_grad()
    def architecture_stats(self) -> dict[str, float]:
        named = list(self.core.named_parameters())
        kappas = [
            parameter.detach().float().exp().clamp(0.25, 50.0).reshape(-1)
            for name, parameter in named
            if name.endswith("log_kappa")
        ]
        transport_gates = [
            parameter.detach().float().reshape(-1)
            for name, parameter in named
            if name.endswith(".gate") and ".transports." in name
        ]
        read_gates = [
            parameter.detach().float().reshape(-1)
            for name, parameter in named
            if name.endswith(".gate") and ".transports." not in name
        ]
        strengths = [
            parameter.detach().float().reshape(-1)
            for name, parameter in named
            if name.endswith(".strength")
        ]
        phase_scales = [
            parameter.detach().float().reshape(-1)
            for name, parameter in named
            if name.endswith(".phase_scale")
        ]

        def concatenate(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat(values) if values else torch.zeros(1)

        kappa = concatenate(kappas)
        transport_gate = concatenate(transport_gates)
        read_gate = concatenate(read_gates)
        strength = concatenate(strengths)
        phase_scale = concatenate(phase_scales)
        return {
            "kappa_mean": float(kappa.mean().item()),
            "transport_gate_mean": float(transport_gate.mean().item()),
            "read_gate_mean": float(read_gate.mean().item()),
            "ray_strength_mean": float(strength.mean().item()),
            "ray_strength_std": float(strength.std(unbiased=False).item()),
            "ray_strength_min": float(strength.min().item()),
            "ray_strength_max": float(strength.max().item()),
            "phase_scale_rms": float(phase_scale.square().mean().sqrt().item()),
            "encoder_layers": float(self.cfg.encoder_layers),
            "decoder_layers": float(self.cfg.decoder_layers),
            "heads": float(self.cfg.heads),
            "rays_per_head": float(self.cfg.rays),
            "transport_steps": float(self.cfg.transport_steps),
            "chunk_size": float(self.cfg.chunk_size),
            "position_scale": float(self.cfg.position_scale),
        }


@torch.no_grad()
def verify_no_future_leak(
    model: AATCausalLanguageModel,
    device: torch.device,
) -> None:
    """Change only future tokens and verify that prefix outputs are unchanged."""
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

    for label, reference, other in (
        ("future-token hidden-state", hidden_long_prefix, hidden_changed_prefix),
        ("future-token logits", logits_long, logits_changed),
    ):
        if not torch.allclose(reference, other, atol=5e-5, rtol=5e-5):
            error = float((reference - other).abs().max().item())
            raise RuntimeError(
                f"{label} causality verification failed: max_error={error}"
            )

    prefix_hidden_error = float(
        (hidden_long_prefix - hidden_short).abs().max().item()
    )
    prefix_logit_error = float(
        (logits_long - logits_short).abs().max().item()
    )
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
    if max(prefix_hidden_error, prefix_logit_error) > 5e-5:
        print(
            "Prefix-extension numerical note: "
            f"hidden_max_error={prefix_hidden_error:.3e}, "
            f"logit_max_error={prefix_logit_error:.3e}; "
            "same-shape future-token causality is exact within strict tolerance.",
            flush=True,
        )


def smoke_test(device: torch.device) -> None:
    print(
        "Running sequence AAT causality, gradient, and checkpoint tests...",
        flush=True,
    )
    cfg = ModelConfig(
        vocab_size=101,
        d_model=32,
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        rays=4,
        chunk_size=4,
        position_scale=63.0,
        transport_steps=1,
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
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite gradient in {name}")

    required_gradients = (
        model.core.encoder.layers[0].ordered_memory.key.weight,
        model.core.encoder.layers[0].ordered_memory.value.weight,
        model.core.encoder.layers[0].position_memory.response.memory_response.base,
        model.core.decoder.layers[0].prefix_reader.response.memory_response.base,
        model.core.decoder.layers[0].ordered_reader.strength,
        model.core.decoder.layers[0].position_reader.strength,
    )
    if any(parameter.grad is None for parameter in required_gradients):
        raise RuntimeError("an AAT sequence path did not receive gradients")

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
        restored.load_state_dict(payload["model_state_dict"], strict=True)
        restored.eval()
        model.eval()
        with torch.no_grad():
            left = model(tokens[:, :-1])
            right = restored(tokens[:, :-1])
        if not torch.equal(left, right):
            raise RuntimeError("checkpoint round-trip changed model outputs")
    print("All sequence AAT smoke tests: OK", flush=True)


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
            "Train the frozen deep-transport aat.sequence model "
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
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--rays", type=int, default=None, help="rays per head")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--transport-steps", type=int, default=None)
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
        encoder_layers=preset_value(
            args.encoder_layers,
            preset,
            "encoder_layers",
        ),
        decoder_layers=preset_value(
            args.decoder_layers,
            preset,
            "decoder_layers",
        ),
        heads=preset_value(args.heads, preset, "heads"),
        rays=preset_value(args.rays, preset, "rays"),
        chunk_size=preset_value(args.chunk_size, preset, "chunk_size"),
        position_scale=float(max(1, sequence_length - 1)),
        transport_steps=preset_value(
            args.transport_steps,
            preset,
            "transport_steps",
        ),
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
        f"runs/aat_sequence_wikitext_{args.preset}"
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
    reference_shape = (
        model_cfg.vocab_size,
        model_cfg.d_model,
        model_cfg.encoder_layers,
        model_cfg.decoder_layers,
        model_cfg.heads,
        model_cfg.rays,
        model_cfg.transport_steps,
    )
    if reference_shape == (50_257, 512, 6, 12, 16, 24, 3):
        if breakdown["total"] != REFERENCE_PAPER_PARAMETERS:
            raise RuntimeError(
                "reference parameter count changed: "
                f"{breakdown['total']:,} != {REFERENCE_PAPER_PARAMETERS:,}"
            )
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
        "Deep-Transport aat.sequence | causal language modeling",
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
        f"model: d={model_cfg.d_model} encoder={model_cfg.encoder_layers} "
        f"decoder={model_cfg.decoder_layers} heads={model_cfg.heads} "
        f"rays/head={model_cfg.rays} transport_steps={model_cfg.transport_steps} "
        f"chunk={model_cfg.chunk_size} seq={train_cfg.sequence_length} "
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
        "Causality checks passed. The frozen aat.sequence core uses no FFN "
        "and builds no T-by-T attention matrix.",
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
