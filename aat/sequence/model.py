from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import AATConfig
from .utils import (
    init_identity,
    make_padding_mask,
    merge_heads,
    ordered_read,
    ordered_state,
    prefix_memory,
    split_heads,
)


def polar_score(
    *,
    rho: torch.Tensor,
    direction: torch.Tensor,
    base: torch.Tensor,
    bias: torch.Tensor,
    radial_weight: torch.Tensor,
    log_kappa: torch.Tensor,
) -> torch.Tensor:
    rays = F.normalize(base, dim=-1, eps=1e-8)
    cosine = torch.einsum("bhtd,hrd->bhtr", direction, rays)
    radial = torch.exp(
        (rho * radial_weight[None, :, None, :]).clamp(-8.0, 8.0)
    )
    kappa = log_kappa.exp().clamp(0.25, 50.0)
    return (
        kappa[None, :, None, None] * radial * cosine
        + bias[None, :, None, :]
    )


class RayResponse(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.sqrt_dim = math.sqrt(cfg.head_dim)
        self.base = nn.Parameter(
            torch.randn(cfg.heads, cfg.rays, cfg.head_dim) / math.sqrt(cfg.head_dim)
        )
        self.bias = nn.Parameter(torch.zeros(cfg.heads, cfg.rays))
        self.radial_weight = nn.Parameter(torch.zeros(cfg.heads, cfg.rays))
        self.log_kappa = nn.Parameter(torch.full((cfg.heads,), math.log(cfg.kappa)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = x / radius
        rho = radius / self.sqrt_dim - 1.0
        return polar_score(
            rho=rho,
            direction=direction,
            base=self.base,
            bias=self.bias,
            radial_weight=self.radial_weight,
            log_kappa=self.log_kappa,
        )


class GeometricTransport(nn.Module):
    """One headwise polar AAT move without reading or writing memory."""

    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.base = nn.Parameter(
            torch.randn(cfg.heads, cfg.rays, cfg.head_dim)
            / math.sqrt(cfg.head_dim)
        )
        self.bias = nn.Parameter(torch.zeros(cfg.heads, cfg.rays))
        self.radial_weight = nn.Parameter(torch.zeros(cfg.heads, cfg.rays))
        self.log_kappa = nn.Parameter(
            torch.full((cfg.heads,), math.log(cfg.kappa))
        )
        self.delta_rho = nn.Parameter(
            torch.randn(cfg.heads, cfg.rays) * 0.02
        )
        self.delta_direction = nn.Parameter(
            torch.randn(cfg.heads, cfg.rays, cfg.head_dim)
            * 0.02
            / math.sqrt(cfg.head_dim)
        )
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        rho: torch.Tensor,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = polar_score(
            rho=rho,
            direction=direction,
            base=self.base,
            bias=self.bias,
            radial_weight=self.radial_weight,
            log_kappa=self.log_kappa,
        )
        address = F.softmax(scores, dim=-1)
        radial_move = torch.einsum(
            "bhtr,hr->bht",
            address,
            self.delta_rho,
        ).unsqueeze(-1)
        directional_move = torch.einsum(
            "bhtr,hrd->bhtd",
            address,
            self.delta_direction,
        )
        rho = rho + radial_move * self.gate
        direction = F.normalize(
            direction + directional_move * self.gate,
            dim=-1,
            eps=1e-8,
        )
        return rho, direction


class DeepRayResponse(nn.Module):
    """Transport geometry several times, then score the memory rays once."""

    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.sqrt_dim = math.sqrt(cfg.head_dim)
        self.transports = nn.ModuleList(
            GeometricTransport(cfg)
            for _ in range(cfg.transport_steps)
        )
        self.memory_response = RayResponse(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = x / radius
        rho = radius / self.sqrt_dim - 1.0
        for transport in self.transports:
            rho, direction = transport(rho, direction)

        response = self.memory_response
        return polar_score(
            rho=rho,
            direction=direction,
            base=response.base,
            bias=response.bias,
            radial_weight=response.radial_weight,
            log_kappa=response.log_kappa,
        )


class OrderedMemory(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.cfg = cfg
        self.response = DeepRayResponse(cfg)
        self.key = nn.Linear(cfg.d_model, cfg.d_model)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)
        self.phase_scale = nn.Parameter(
            torch.full((cfg.heads, cfg.head_dim), 1.0 / math.sqrt(cfg.head_dim))
        )

    def key_heads(self, x: torch.Tensor) -> torch.Tensor:
        return split_heads(self.key(x), self.cfg.heads)

    def value_heads(self, x: torch.Tensor) -> torch.Tensor:
        return split_heads(self.value(x), self.cfg.heads)

    def phase(self, key: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.tanh(
            key * self.phase_scale[None, :, None, :]
        )

    def address(self, key: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.response(key), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.key_heads(x)
        value = self.value_heads(x)
        phase = self.phase(key)
        payload = torch.cat((value * phase.cos(), value * phase.sin()), dim=-1)
        return ordered_state(
            self.address(key),
            payload,
            mask,
            self.cfg.chunk_size,
        )


class PositionMemory(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.cfg = cfg
        self.sqrt_dim = math.sqrt(cfg.head_dim)
        self.response = DeepRayResponse(cfg)

    def positions(
        self,
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, tokens = mask.shape
        relative = (
            torch.arange(tokens, device=mask.device, dtype=dtype)
            / self.cfg.position_scale
        )
        frequencies = torch.arange(
            1,
            self.cfg.head_dim // 2 + 1,
            device=mask.device,
            dtype=dtype,
        )
        angle = math.pi * relative[:, None] * frequencies[None, :]
        features = torch.stack((angle.cos(), angle.sin()), dim=-1).flatten(-2)
        if self.cfg.head_dim % 2:
            features = torch.cat(
                (features, (2.0 * relative - 1.0).unsqueeze(-1)),
                dim=-1,
            )
        direction = F.normalize(features, dim=-1, eps=1e-8)
        positions = (
            direction
            * (relative + 0.5).unsqueeze(-1)
            * self.sqrt_dim
        )
        return positions.view(1, 1, tokens, self.cfg.head_dim).expand(
            batch,
            self.cfg.heads,
            -1,
            -1,
        ).masked_fill(mask[:, None, :, None], 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.response(self.positions(mask, x.dtype))
        return (
            prefix_memory(
                split_heads(x, self.cfg.heads),
                scores,
                mask,
                self.cfg.score_clip,
            ),
            F.softmax(scores, dim=-1),
        )


class PrefixReader(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.cfg = cfg
        self.response = DeepRayResponse(cfg)
        self.strength = nn.Parameter(torch.ones(cfg.heads, cfg.rays))
        self.output = nn.Linear(cfg.d_model, cfg.d_model)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def query(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask)[:, None, :, None].to(x.dtype)
        content = x * valid
        prefix_sum = content.cumsum(dim=2) - content
        prefix_count = valid.cumsum(dim=2) - valid
        context = prefix_sum / prefix_count.clamp_min(1.0)
        context = context * prefix_count.gt(0).to(context.dtype)
        return (x + context).masked_fill(mask[:, None, :, None], 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        heads = split_heads(x, self.cfg.heads)
        scores = self.response(self.query(heads, mask))
        address = (
            F.softmax(scores, dim=-1)
            * self.strength[None, :, None, :]
        )
        memory = prefix_memory(
            heads,
            scores,
            mask,
            self.cfg.score_clip,
        )
        read = torch.einsum("bhtr,bhtrd->bhtd", address, memory)
        delta = self.output(merge_heads(read) - x) * self.gate
        return self.dropout(delta).masked_fill(mask.unsqueeze(-1), 0.0)


class OrderedReader(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.strength = nn.Parameter(torch.ones(cfg.heads, cfg.rays))
        self.output = nn.Linear(cfg.d_model, cfg.d_model)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        writer: OrderedMemory,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        key = writer.key_heads(x)
        value = writer.value_heads(x)
        phase = writer.phase(key)
        address = (
            writer.address(key)
            * self.strength[None, :, None, :]
        )
        bound = ordered_read(address, state, mask)
        real, imaginary = bound.chunk(2, dim=-1)
        read = real * phase.cos() + imaginary * phase.sin()
        delta = self.output(merge_heads(read - value)) * self.gate
        return self.dropout(delta).masked_fill(mask.unsqueeze(-1), 0.0)


class PositionReader(nn.Module):
    def __init__(self, cfg: AATConfig, output: nn.Linear):
        super().__init__()
        object.__setattr__(self, "_output", output)
        self.strength = nn.Parameter(torch.ones(cfg.heads, cfg.rays))
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(cfg.dropout)

    @property
    def output(self) -> nn.Linear:
        return self._output

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        address: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        read = torch.einsum(
            "bhtr,bhtrd->bhtd",
            address * self.strength[None, :, None, :],
            memory,
        )
        delta = self.output(merge_heads(read) - x) * self.gate
        return self.dropout(delta).masked_fill(mask.unsqueeze(-1), 0.0)


class AATDecoderLayer(nn.Module):
    def __init__(self, cfg: AATConfig, writer: OrderedMemory):
        super().__init__()
        object.__setattr__(self, "_writer", writer)
        self.prefix_norm = nn.LayerNorm(cfg.d_model)
        self.ordered_norm = nn.LayerNorm(cfg.d_model)
        self.position_norm = nn.LayerNorm(cfg.d_model)
        self.prefix_reader = PrefixReader(cfg)
        self.ordered_reader = OrderedReader(cfg)
        self.position_reader = PositionReader(cfg, self.prefix_reader.output)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        ordered_address: torch.Tensor,
        ordered_innovations: torch.Tensor,
        ordered_boundaries: torch.Tensor,
        position_memory: torch.Tensor,
        position_address: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.prefix_reader(self.prefix_norm(x), mask)
        )
        x = x + self.dropout(
            self.ordered_reader(
                self.ordered_norm(x),
                (
                    ordered_address,
                    ordered_innovations,
                    ordered_boundaries,
                ),
                self._writer,
                mask,
            )
        )
        x = x + self.dropout(
            self.position_reader(
                self.position_norm(x),
                position_memory,
                position_address,
                mask,
            )
        )
        return x.masked_fill(mask.unsqueeze(-1), 0.0)


class AATEncoderLayer(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.ordered_memory = OrderedMemory(cfg)
        self.position_memory = PositionMemory(cfg)
        self.reader = AATDecoderLayer(cfg, self.ordered_memory)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ordered = self.ordered_memory(x, mask)
        position, address = self.position_memory(x, mask)
        return self.reader(x, *ordered, position, address, mask)


class AATEncoder(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        self.cfg = cfg
        # Every configured encoder layer refines the token states through its
        # own temporary memory. The final shared memory is written afterwards.
        self.layers = nn.ModuleList(
            AATEncoderLayer(cfg)
            for _ in range(cfg.encoder_layers)
        )
        self.ordered_memory = OrderedMemory(cfg)
        self.position_memory = PositionMemory(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        for layer in self.layers:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        ordered = self.ordered_memory(x, mask)
        position, address = self.position_memory(x, mask)
        return x, *ordered, position, address


class AATDecoder(nn.Module):
    def __init__(self, cfg: AATConfig, encoder: AATEncoder):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            AATDecoderLayer(cfg, encoder.ordered_memory)
            for _ in range(cfg.decoder_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        ordered_address: torch.Tensor,
        ordered_innovations: torch.Tensor,
        ordered_boundaries: torch.Tensor,
        position_memory: torch.Tensor,
        position_address: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            args = (
                x,
                ordered_address,
                ordered_innovations,
                ordered_boundaries,
                position_memory,
                position_address,
                mask,
            )
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(layer, *args, use_reentrant=False)
            else:
                x = layer(*args)
        return x


class AAT(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.encoder = AATEncoder(cfg)
        self.decoder = AATDecoder(cfg, self.encoder)
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, OrderedMemory):
                init_identity(module.key)
                init_identity(module.value)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.cfg.d_model:
            raise ValueError(
                f"x must have shape [batch, tokens, {self.cfg.d_model}]."
            )
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one token.")
        if not x.is_floating_point():
            raise ValueError("x must contain floating-point token states.")

        mask = make_padding_mask(x, padding_mask)
        x = self.input_dropout(self.input_norm(x))
        encoded = self.encoder(x, mask)
        x = self.decoder(*encoded, mask)
        return self.output_norm(x).masked_fill(mask.unsqueeze(-1), 0.0)

    def config_dict(self) -> dict[str, Any]:
        return self.cfg.to_dict()

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "config": self.config_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @staticmethod
    def torch_load(path: str | Path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location=None) -> "AAT":
        checkpoint = cls.torch_load(path, map_location=map_location)
        if not isinstance(checkpoint, dict) or "config" not in checkpoint:
            raise RuntimeError("checkpoint must contain a config dictionary.")
        model = cls(AATConfig(**dict(checkpoint["config"])))
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        if state is None:
            raise RuntimeError("checkpoint contains no state_dict.")
        model.load_state_dict(state)
        return model

    def load_checkpoint(
        self,
        path: str | Path,
        map_location=None,
        strict: bool = True,
    ) -> None:
        checkpoint = self.torch_load(path, map_location=map_location)
        state = (
            checkpoint.get(
                "state_dict",
                checkpoint.get("model_state_dict", checkpoint),
            )
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        self.load_state_dict(state, strict=strict)


__all__ = ["AAT"]
