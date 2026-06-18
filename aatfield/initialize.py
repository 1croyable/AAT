# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

import math
import torch
import torch.nn.functional as F

from .utils import pairwise_dist2


@torch.no_grad()
def boundary_weights(pts: torch.Tensor, class_id: int, parents: torch.Tensor) -> torch.Tensor:
    """
    Centroid-silhouette sample weights for child placement.
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

    silhouette = (d_other - d_own) / torch.maximum(d_own, d_other).clamp_min(1e-8)
    w = (1.0 - silhouette).clamp_min(1e-3)
    return w / w.mean().clamp_min(1e-8)


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


@torch.no_grad()
def initialize_layer_auto_k(layer, z: torch.Tensor, y: torch.Tensor, *, min_children: int, kmeans_iters: int) -> None:
    z = z.detach().float()
    y = y.detach().long().to(z.device)

    C = layer.num_classes
    M = int(layer.cfg.max_children)
    D = layer.state_dim
    min_children = max(1, min(int(min_children), M))

    reserve_children = 5

    # Parent anchors: class centers
    parents = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    global_mean = z.mean(dim=0)

    for c in range(C):
        pts = z[y == c]
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    # Candidate child centers for each class and each K
    centers_by_class: List[Dict[int, torch.Tensor]] = []
    max_k_by_class: List[int] = []

    for c in range(C):
        pts = z[y == c]
        if pts.shape[0] == 0:
            pts = z

        w = boundary_weights(pts, c, parents)
        max_k = min(M, int(pts.shape[0]))
        max_k_by_class.append(max_k)

        centers_map: Dict[int, torch.Tensor] = {}
        for k in range(1, max_k + 1):
            centers, _ = weighted_kmeans(pts, w, k=k, iters=int(kmeans_iters))
            centers_map[int(k)] = centers.detach().clone()

        centers_by_class.append(centers_map)

    common_max_k = min(max_k_by_class) if max_k_by_class else M
    common_min_k = min(max(min_children, 1), common_max_k)

    # Sigma estimate, same style as original version
    all_centers_full = [centers_by_class[c][max_k_by_class[c]] for c in range(C)]
    full_anchors = torch.cat([parents] + all_centers_full, dim=0)

    nearest = torch.sqrt(pairwise_dist2(z, full_anchors).min(dim=1).values + 1e-8)
    sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
    sigma = max(0.05, min(3.0, sigma))

    # Global hard NMI score
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        p = p.float().clamp_min(1e-12)
        return -(p * p.log()).sum()

    def _global_hard_nmi(k: int) -> float:
        child_anchors = torch.cat(
            [centers_by_class[c][int(k)] for c in range(C)],
            dim=0,
        )

        dist2 = pairwise_dist2(z, child_anchors)
        assign = dist2.argmin(dim=1)
        K_total = int(child_anchors.shape[0])

        joint = torch.zeros(K_total, C, device=z.device, dtype=z.dtype)

        flat_index = assign * C + y
        joint_flat = torch.zeros(K_total * C, device=z.device, dtype=z.dtype)
        joint_flat.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=z.dtype))

        joint = joint_flat.view(K_total, C)
        joint = joint / max(int(z.shape[0]), 1)

        p_anchor = joint.sum(dim=1)
        p_label = joint.sum(dim=0)
        denom = p_anchor[:, None] * p_label[None, :]

        mask = joint > 1e-12
        mi = (joint[mask] * (joint[mask] / denom[mask].clamp_min(1e-12)).log()).sum()

        nmi = mi / torch.sqrt((_entropy(p_anchor) * _entropy(p_label)).clamp_min(1e-12))
        return float(nmi.item())

    # Knee selection on global hard NMI curve
    k_values = list(range(int(common_min_k), int(common_max_k) + 1))
    scores = [_global_hard_nmi(k) for k in k_values]

    if len(k_values) <= 1:
        knee_k = int(k_values[0])
    else:
        xs = torch.tensor(k_values, device=z.device, dtype=z.dtype)
        ys = torch.tensor(scores, device=z.device, dtype=z.dtype)

        x = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1e-8)
        y_norm = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1e-8)

        # distance to diagonal line: simple elbow/knee detector
        curve = torch.stack([x, y_norm], dim=1)
        line = torch.stack([x, x], dim=1)
        dist = (curve - line).norm(dim=1)

        knee_idx = int(dist.argmax().item())
        knee_k = int(k_values[knee_idx])

    best_k = int(knee_k + reserve_children)
    best_k = max(int(common_min_k), min(best_k, int(common_max_k)))

    # Materialize selected anchors
    child_centers = torch.zeros(C, best_k, D, device=z.device, dtype=z.dtype)

    for c in range(C):
        child_centers[c] = centers_by_class[c][best_k][:best_k]

    selected_anchors = torch.cat([parents, child_centers.reshape(C * best_k, D)], dim=0)

    nearest = torch.sqrt(pairwise_dist2(z, selected_anchors).min(dim=1).values + 1e-8)
    sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
    sigma = max(0.05, min(3.0, sigma))

    layer._materialize(parents, child_centers, sigma)


__all__ = [
    "boundary_weights",
    "weighted_kmeans",
    "supervised_fisher_score",
    "child_response_features_for_class",
    "initialize_layer_auto_k",
]
