from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() < 2:
        raise ValueError("x must have a batch dimension and at least one feature dimension.")
    if x.dim() > 2:
        x = x.flatten(1)
    return x.float()


def prepare_features(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    x = flatten_features(x)
    if x.shape[-1] != int(input_dim):
        raise ValueError(f"expected input_dim={int(input_dim)}, got {int(x.shape[-1])}")
    return x


@torch.no_grad()
def polar_statistics(x: torch.Tensor, input_dim: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = flatten_features(x) if input_dim is None else prepare_features(x, int(input_dim))
    center = x.mean(dim=0, keepdim=True)
    r = (x - center).norm(dim=1)
    r_min = torch.quantile(r, 0.01)
    r_max = torch.quantile(r, 0.99)
    if bool((r_max <= r_min + 1e-6).item()):
        r_min = r.min()
        r_max = r.max() + 1e-6
    return center.detach(), r_min.detach(), r_max.detach()


__all__ = ["count_parameters", "flatten_features", "prepare_features", "polar_statistics"]
