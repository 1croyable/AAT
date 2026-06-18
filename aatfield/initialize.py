# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

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
def initialize_layer_auto_k(layer, z: torch.Tensor, y: torch.Tensor, *, min_children: int, kmeans_iters: int) -> None:
    z = z.detach().float()
    y = y.detach().long().to(z.device)

    C = layer.num_classes
    M = int(layer.cfg.max_children)
    D = layer.state_dim
    min_children = max(1, min(int(min_children), M))

    reserve_children = 5
    geo_ratio = 0.99

    # Parent anchors: class centers
    parents = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    global_mean = z.mean(dim=0)

    class_points: List[torch.Tensor] = []
    for c in range(C):
        pts = z[y == c]
        if pts.shape[0] == 0:
            pts = z
        class_points.append(pts)
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    # Common equal-K range.  Fixed-K runs pass min_children == max_children,
    common_max_k = min(M, min(int(pts.shape[0]) for pts in class_points))
    common_min_k = min(max(min_children, 1), common_max_k)
    fixed_single_k = int(common_min_k) == int(common_max_k)

    centers_by_class: List[Dict[int, torch.Tensor]] = []
    needed_ks = [int(common_max_k)] if fixed_single_k else list(range(1, int(common_max_k) + 1))

    for c in range(C):
        pts = class_points[c]
        w = boundary_weights(pts, c, parents)

        centers_map: Dict[int, torch.Tensor] = {}
        for k in needed_ks:
            centers, _ = weighted_kmeans(pts, w, k=int(k), iters=int(kmeans_iters))
            centers_map[int(k)] = centers.detach().clone()

        centers_by_class.append(centers_map)

    def _entropy(p: torch.Tensor) -> torch.Tensor:
        p = p.float().clamp_min(1e-12)
        return -(p * p.log()).sum()

    def _global_hard_nmi(k: int) -> float:
        child_anchors = torch.cat([centers_by_class[c][int(k)] for c in range(C)], dim=0)
        assign = pairwise_dist2(z, child_anchors).argmin(dim=1)
        k_total = int(child_anchors.shape[0])

        flat_index = assign * C + y
        joint_flat = torch.zeros(k_total * C, device=z.device, dtype=z.dtype)
        joint_flat.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=z.dtype))
        joint = joint_flat.view(k_total, C)
        joint = joint / max(int(z.shape[0]), 1)

        p_anchor = joint.sum(dim=1)
        p_label = joint.sum(dim=0)
        denom = p_anchor[:, None] * p_label[None, :]

        mask = joint > 1e-12
        mi = (joint[mask] * (joint[mask] / denom[mask].clamp_min(1e-12)).log()).sum()
        nmi = mi / torch.sqrt((_entropy(p_anchor) * _entropy(p_label)).clamp_min(1e-12))
        return float(nmi.item())

    def _anchor_macro_f1(k: int) -> float:
        child_anchors = torch.cat([centers_by_class[c][int(k)] for c in range(C)], dim=0)
        anchor_labels = torch.arange(C, device=z.device, dtype=torch.long).repeat_interleave(int(k))
        assign = pairwise_dist2(z, child_anchors).argmin(dim=1)
        pred = anchor_labels.index_select(0, assign)

        scores: List[float] = []
        for c in range(C):
            tp = ((pred == c) & (y == c)).sum().float()
            fp = ((pred == c) & (y != c)).sum().float()
            fn = ((pred != c) & (y == c)).sum().float()
            den = 2.0 * tp + fp + fn
            scores.append(float((2.0 * tp / den.clamp_min(1.0)).item()))
        return float(sum(scores) / max(len(scores), 1))

    k_values = list(range(int(common_min_k), int(common_max_k) + 1))

    if fixed_single_k or len(k_values) <= 1:
        k_info = int(k_values[0])
        k_geo = int(k_values[0])
    else:
        nmi_scores = [_global_hard_nmi(k) for k in k_values]
        anchor_f1_scores = [_anchor_macro_f1(k) for k in k_values]

        # 1) Information view: knee on hard NMI curve.
        xs = torch.tensor(k_values, device=z.device, dtype=z.dtype)
        ys = torch.tensor(nmi_scores, device=z.device, dtype=z.dtype)
        x = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1e-8)
        y_norm = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1e-8)
        curve = torch.stack([x, y_norm], dim=1)
        line = torch.stack([x, x], dim=1)
        dist = (curve - line).norm(dim=1)
        k_info = int(k_values[int(dist.argmax().item())])

        # 2) Geometry view: first K reaching 99% of the best nearest-anchor macro-F1.
        best_anchor = max(anchor_f1_scores) if anchor_f1_scores else 0.0
        target = float(geo_ratio) * float(best_anchor)
        k_geo = int(k_values[-1])
        for k, score in zip(k_values, anchor_f1_scores):
            if float(score) >= target:
                k_geo = int(k)
                break

    # Auto-K v2: average the information K and geometry K, then add small transport reserve.
    best_k = int(round((int(k_info) + int(k_geo)) / 2.0)) + int(reserve_children)
    best_k = max(int(common_min_k), min(best_k, int(common_max_k)))

    # Materialize selected anchors.
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
    "initialize_layer_auto_k",
]
