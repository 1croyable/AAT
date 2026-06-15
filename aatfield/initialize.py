# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .utils import pairwise_dist2


@torch.no_grad()
def boundary_weights(pts: torch.Tensor, class_id: int, parents: torch.Tensor) -> torch.Tensor:
    """
    Boundary-aware sample weights for child placement.
    """
    pts = pts.float()
    parents = parents.float()
    d = torch.sqrt(pairwise_dist2(pts, parents) + 1e-8)
    d_own = d[:, int(class_id)]

    if parents.shape[0] <= 1:
        return torch.ones_like(d_own)

    mask = torch.ones(parents.shape[0], dtype=torch.bool, device=pts.device)
    mask[int(class_id)] = False
    d_other = d[:, mask].min(dim=1).values

    margin = d_other - d_own
    tau = torch.median(torch.abs(margin)).clamp_min(1e-4)
    safe_margin = torch.clamp(margin, min=0.0)
    w = 1.0 + torch.exp(-(safe_margin * safe_margin) / (tau * tau + 1e-8))
    return w.clamp_min(1e-6)


@torch.no_grad()
def weighted_kmeans(pts: torch.Tensor, weights: torch.Tensor, k: int, iters: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic weighted k-means.
    """
    pts = pts.float()
    w = weights.float().clamp_min(1e-8)
    n, d = pts.shape
    k = int(max(1, min(int(k), n)))

    centers = torch.empty((k, d), device=pts.device, dtype=pts.dtype)
    centers[0] = (pts * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-8)

    if k > 1:
        nearest = pairwise_dist2(pts, centers[:1]).squeeze(1)
        for j in range(1, k):
            score = nearest * w
            idx = int(score.argmax().item())
            centers[j] = pts[idx]
            nearest = torch.minimum(nearest, pairwise_dist2(pts, centers[j:j + 1]).squeeze(1))

    assign = torch.zeros(n, dtype=torch.long, device=pts.device)
    for _ in range(int(iters)):
        assign = pairwise_dist2(pts, centers).argmin(dim=1)

        weighted_pts = pts * w[:, None]
        sum_w = torch.zeros(k, device=pts.device, dtype=pts.dtype)
        sum_w.scatter_add_(0, assign, w)

        sum_wp = torch.zeros(k, d, device=pts.device, dtype=pts.dtype)
        sum_wp.scatter_add_(0, assign.view(-1, 1).expand(-1, d), weighted_pts)

        updated = sum_wp / sum_w.clamp_min(1e-8).view(-1, 1)
        new_centers = torch.where((sum_w > 0).view(-1, 1), updated, centers)

        shift = (new_centers - centers).norm().item()
        centers = new_centers
        if shift < 1e-5:
            break

    assign = pairwise_dist2(pts, centers).argmin(dim=1)
    return centers, assign


@torch.no_grad()
def supervised_fisher_score(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    """Supervised ANOVA/Fisher ratio used by Auto-K scoring."""
    z = z.float()
    y = y.long().to(z.device)
    C = int(num_classes)
    if z.shape[0] == 0 or C <= 0:
        return 0.0

    N = max(int(z.shape[0]), 1)
    D = int(z.shape[1])
    global_mu = z.mean(dim=0)

    counts = torch.bincount(y, minlength=C)[:C].to(device=z.device, dtype=z.dtype)
    valid_mask = counts > 0
    valid = int(valid_mask.sum().item())
    if valid <= 1 or N <= valid:
        return 0.0

    class_sums = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    class_sums.scatter_add_(0, y.view(-1, 1).expand(-1, D), z)
    class_means = class_sums / counts.clamp_min(1.0).view(C, 1)

    sample_means = class_means.index_select(0, y)
    W = (z - sample_means).square().sum().clamp_min(1e-12)
    B = (counts[valid_mask] * (class_means[valid_mask] - global_mu).square().sum(dim=1)).sum()

    score = (B / max(valid - 1, 1)) / (W / max(N - valid, 1)).clamp_min(1e-12)
    return float(score.item())


@torch.no_grad()
def child_response_features_for_class(z: torch.Tensor, parent: torch.Tensor, centers: torch.Tensor, sigma_value: float) -> torch.Tensor:
    """
    Candidate child response features for equal-K scoring.
    """
    z = z.float()
    parent = parent.float()
    centers = centers.float()
    if centers.numel() == 0:
        return z.new_zeros((z.shape[0], 1))

    anchors = torch.cat([parent.view(1, -1), centers], dim=0)
    sigma = float(sigma_value)
    dist2 = pairwise_dist2(z, anchors)
    logits = -dist2 / (2.0 * sigma * sigma + 1e-8)
    alpha = torch.softmax(logits, dim=-1)[:, 1:]

    axis = centers - parent.view(1, -1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    s = ((z.unsqueeze(1) - centers.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=-1)
    s = s / max(sigma, 1e-6)
    return alpha * F.relu(s)


__all__ = [
    "boundary_weights",
    "weighted_kmeans",
    "supervised_fisher_score",
    "child_response_features_for_class",
]
