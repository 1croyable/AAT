# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import torch
import torch.nn as nn


def inv_softplus(y: float) -> float:
    y = float(y)
    if y <= 0:
        raise ValueError("inv_softplus requires y > 0")
    return math.log(math.exp(y) - 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_permutation(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randperm(int(dim), generator=gen, dtype=torch.long).contiguous()


@torch.no_grad()
def pairwise_dist2(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x = x.float()
    centers = centers.float()
    return ((x * x).sum(dim=1, keepdim=True) + (centers * centers).sum(dim=1).view(1, -1) - 2.0 * x @ centers.t()).clamp_min(0.0)


__all__ = [
    "inv_softplus",
    "count_parameters",
    "make_permutation",
    "pairwise_dist2",
]
