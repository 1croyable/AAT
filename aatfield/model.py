from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AATConfig
from .utils import polar_statistics, prepare_features


GATE_INIT = 1.0

class AATLayer(nn.Module):
    def __init__(self, state_dim: int, rays: int, *, kappa: float, ray_dropout: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.rays = int(rays)
        self.kappa = float(kappa)
        self.ray_dropout = float(ray_dropout)

        self.base = nn.Parameter(torch.randn(self.rays, self.state_dim) / math.sqrt(self.state_dim))
        self.ray_bias = nn.Parameter(torch.zeros(self.rays))
        self.radial_bias = nn.Parameter(torch.zeros(self.rays))
        self.dr = nn.Parameter(torch.randn(self.rays) * 0.02)
        self.du = nn.Parameter(torch.randn(self.rays, self.state_dim) * 0.02 / math.sqrt(self.state_dim))
        self.gate = nn.Parameter(torch.tensor(GATE_INIT))

    def dropout(self, alpha: torch.Tensor) -> torch.Tensor:
        p = self.ray_dropout
        if (not self.training) or p <= 0.0:
            return alpha
        mask = torch.rand_like(alpha) >= p
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
        score = self.kappa * (u @ rays.t()) + self.ray_bias + rho * self.radial_bias.view(1, -1)
        alpha = F.softmax(score, dim=1)
        alpha = self.dropout(alpha)
        rho = (rho + (alpha @ self.dr[:, None]) * self.gate)
        u = F.normalize(u + (alpha @ self.du) * self.gate, dim=1, eps=1e-8)
        return rho, u


class AAT(nn.Module):
    def __init__(self, cfg: AATConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.input_dim = int(cfg.input_dim)
        self.state_dim = int(cfg.state_dim)
        self.num_classes = int(cfg.num_classes)
        self.ray_counts = tuple(cfg.ray_counts)

        self.register_buffer("center", torch.zeros(1, self.state_dim), persistent=True)
        self.register_buffer("r_min", torch.tensor(0.0), persistent=True)
        self.register_buffer("r_max", torch.tensor(1.0), persistent=True)

        self.layers = nn.ModuleList([
            AATLayer(self.state_dim, rays, kappa=float(cfg.kappa), ray_dropout=float(cfg.ray_dropout))
            for rays in self.ray_counts
        ])
        self.head = nn.Linear(self.state_dim + 1, self.num_classes)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return prepare_features(x, self.input_dim)

    def fit_state(self, x: torch.Tensor) -> "AAT":
        center, r_min, r_max = polar_statistics(x, self.input_dim)
        self.center.copy_(center.to(device=self.center.device, dtype=self.center.dtype))
        self.r_min.copy_(r_min.to(device=self.r_min.device, dtype=self.r_min.dtype))
        self.r_max.copy_(r_max.to(device=self.r_max.device, dtype=self.r_max.dtype))
        return self

    def to_polar(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.prepare(x)
        c = self.center.to(device=z.device, dtype=z.dtype)
        r_min = self.r_min.to(device=z.device, dtype=z.dtype)
        r_max = self.r_max.to(device=z.device, dtype=z.dtype)
        xc = z - c
        r = xc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        u = xc / r
        rho = 2.0 * (r - r_min) / (r_max - r_min).clamp_min(1e-8) - 1.0
        return rho, u

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        rho, u = self.to_polar(x)
        for layer in self.layers:
            rho, u = layer(rho, u)
        return torch.cat((rho, u), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.transport(x))

    def config_dict(self) -> dict[str, Any]:
        return self.cfg.to_dict()

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint: dict[str, Any] = {
            "config": self.config_dict(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

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

    def load_checkpoint(self, path: str | Path, map_location=None, strict: bool = True) -> None:
        checkpoint = self.torch_load(path, map_location=map_location)
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        self.load_state_dict(state, strict=strict)


__all__ = ["AATLayer", "AAT"]
