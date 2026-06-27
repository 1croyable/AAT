# -*- coding: utf-8 -*-
"""
Progressive Parentless Additive Gaussian AAT on flattened MNIST.

Core experiment:
  - Flatten MNIST: 28*28 = 784
  - Lift to R^(784 + extra_dims), default extra_dims=1
  - No dimension permutation
  - No activation / dropout / scheduler / softmax
  - Train layers progressively:
      L1 greedy -> init L2 on T1(z0) -> joint
      -> init L3 on T2(T1(z0)) -> joint
      -> init L4 ... -> joint
  - Previous layers keep learning with a smaller LR.
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def pairwise_dist2(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # Always compute distances in fp32 for stability.
    x = x.float()
    centers = centers.float()
    return (
        (x * x).sum(dim=1, keepdim=True)
        + (centers * centers).sum(dim=1).view(1, -1)
        - 2.0 * x @ centers.t()
    ).clamp_min(0.0)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def logit01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(x / (1.0 - x))


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(True)


def format_int(x: int) -> str:
    return f"{int(x):,}"


# -----------------------------------------------------------------------------
# MNIST loading and lift
# -----------------------------------------------------------------------------

def load_mnist_flat(
    *,
    data_root: str,
    n_train: int,
    n_val: int,
    seed: int,
    download: bool,
    smoke_test: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if smoke_test:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        x_train = torch.rand(int(n_train), 784, generator=gen)
        y_train = torch.randint(0, 10, (int(n_train),), generator=gen)
        x_val = torch.rand(int(n_val), 784, generator=gen)
        y_val = torch.randint(0, 10, (int(n_val),), generator=gen)
        return x_train.float(), y_train.long(), x_val.float(), y_val.long()

    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError(
            "This script needs torchvision to load MNIST. "
            "Install it or run with --smoke-test for a syntax check."
        ) from e

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(data_root), train=True, download=bool(download), transform=transform)
    val_ds = datasets.MNIST(root=str(data_root), train=False, download=bool(download), transform=transform)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    y_train = train_ds.targets.long()
    x_val = val_ds.data.float().view(-1, 784) / 255.0
    y_val = val_ds.targets.long()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    if int(n_train) > 0 and int(n_train) < x_train.shape[0]:
        idx = torch.randperm(x_train.shape[0], generator=gen)[: int(n_train)]
        x_train = x_train[idx]
        y_train = y_train[idx]

    if int(n_val) > 0 and int(n_val) < x_val.shape[0]:
        idx = torch.randperm(x_val.shape[0], generator=gen)[: int(n_val)]
        x_val = x_val[idx]
        y_val = y_val[idx]

    return x_train.float(), y_train.long(), x_val.float(), y_val.long()


def lift_flat_images(x: torch.Tensor, extra_dims: int) -> torch.Tensor:
    # Preserve image-coordinate order; no permutation.
    z = x.float().view(x.shape[0], -1) * 2.0 - 1.0
    if int(extra_dims) > 0:
        z = torch.cat([z, torch.zeros(z.shape[0], int(extra_dims), dtype=z.dtype, device=z.device)], dim=1)
    return z.contiguous()


# -----------------------------------------------------------------------------
# Parent-assisted initialization, parentless forward anchors
# -----------------------------------------------------------------------------

def boundary_weights(pts: torch.Tensor, class_id: int, parents: torch.Tensor) -> torch.Tensor:
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
def weighted_kmeans(pts: torch.Tensor, weights: torch.Tensor, k: int, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.detach().float()
    weights = weights.detach().float().clamp_min(1e-8)
    n = int(pts.shape[0])
    k = max(1, min(int(k), n))

    first = int(torch.argmax(weights).item())
    centers = [pts[first]]

    if k > 1:
        min_d = pairwise_dist2(pts, centers[0].view(1, -1)).view(-1)
        for _ in range(1, k):
            score = min_d * weights
            idx = int(torch.argmax(score).item())
            centers.append(pts[idx])
            d_new = pairwise_dist2(pts, pts[idx].view(1, -1)).view(-1)
            min_d = torch.minimum(min_d, d_new)

    centers_t = torch.stack(centers, dim=0)

    assign = torch.zeros(n, dtype=torch.long, device=pts.device)
    for _ in range(int(iters)):
        dist2 = pairwise_dist2(pts, centers_t)
        assign = dist2.argmin(dim=1)

        new_centers = centers_t.clone()
        for j in range(k):
            mask = assign == j
            if bool(mask.any()):
                ww = weights[mask].view(-1, 1)
                new_centers[j] = (pts[mask] * ww).sum(dim=0) / ww.sum().clamp_min(1e-8)
            else:
                # Re-seed an empty center at a high-weight distant point.
                nearest = dist2.min(dim=1).values
                idx = int(torch.argmax(nearest * weights).item())
                new_centers[j] = pts[idx]
        centers_t = new_centers

    return centers_t.contiguous(), assign


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum()


def build_k_candidates(min_k: int, max_k: int, candidate_text: str, full_k_scan: bool) -> List[int]:
    min_k = int(min_k)
    max_k = int(max_k)
    if max_k < min_k:
        return [max_k]

    if full_k_scan or (max_k - min_k <= 30 and candidate_text.lower() == "auto"):
        return list(range(min_k, max_k + 1))

    if candidate_text.lower() != "auto":
        vals = []
        for part in candidate_text.split(","):
            part = part.strip()
            if part:
                vals.append(int(part))
    else:
        # Practical high-dimensional Auto-K grid inside [min_k, max_k].
        vals = [2, 4, 8, 16, 32, 64, 100]

    vals = sorted({max(min_k, min(max_k, int(v))) for v in vals if min_k <= int(v) <= max_k or True})
    vals = [v for v in vals if min_k <= v <= max_k]
    if min_k not in vals:
        vals.insert(0, min_k)
    if max_k not in vals:
        vals.append(max_k)
    return sorted(set(vals))


@torch.no_grad()
def choose_supports_parentless(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    num_classes: int,
    min_per_class: int,
    max_per_class: int,
    kmeans_iters: int,
    reserve_supports: int,
    geo_ratio: float,
    fixed_k: int,
    k_candidates: str,
    full_k_scan: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Use labels only for initialization; returned supports are unlabeled field anchors."""
    z = z.detach().float()
    y = y.detach().long().to(z.device)
    C = int(num_classes)
    D = int(z.shape[1])

    parents = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    global_mean = z.mean(dim=0)
    class_points: List[torch.Tensor] = []
    for c in range(C):
        pts = z[y == c]
        if pts.shape[0] == 0:
            pts = z
        class_points.append(pts)
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    common_max_k = min(int(max_per_class), min(int(pts.shape[0]) for pts in class_points))
    common_min_k = min(max(int(min_per_class), 1), common_max_k)

    if int(fixed_k) > 0:
        best_k = max(1, min(int(fixed_k), common_max_k))
        needed_ks = [best_k]
        k_values = [best_k]
    else:
        k_values = build_k_candidates(common_min_k, common_max_k, str(k_candidates), bool(full_k_scan))
        needed_ks = list(k_values)

    centers_by_class: List[Dict[int, torch.Tensor]] = []
    for c in range(C):
        pts = class_points[c]
        w = boundary_weights(pts, c, parents)
        centers_map: Dict[int, torch.Tensor] = {}
        for k in needed_ks:
            centers, _ = weighted_kmeans(pts, w, k=int(k), iters=int(kmeans_iters))
            centers_map[int(k)] = centers.detach().clone()
        centers_by_class.append(centers_map)

    if int(fixed_k) <= 0:
        def global_hard_nmi(k: int) -> float:
            anchors = torch.cat([centers_by_class[c][int(k)] for c in range(C)], dim=0)
            assign = pairwise_dist2(z, anchors).argmin(dim=1)
            k_total = int(anchors.shape[0])
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

        def anchor_macro_f1(k: int) -> float:
            anchors = torch.cat([centers_by_class[c][int(k)] for c in range(C)], dim=0)
            anchor_labels = torch.arange(C, device=z.device, dtype=torch.long).repeat_interleave(int(k))
            assign = pairwise_dist2(z, anchors).argmin(dim=1)
            pred = anchor_labels.index_select(0, assign)
            scores: List[float] = []
            for c in range(C):
                tp = ((pred == c) & (y == c)).sum().float()
                fp = ((pred == c) & (y != c)).sum().float()
                fn = ((pred != c) & (y == c)).sum().float()
                den = 2.0 * tp + fp + fn
                scores.append(float((2.0 * tp / den.clamp_min(1.0)).item()))
            return float(sum(scores) / max(len(scores), 1))

        if len(k_values) <= 1:
            k_info = int(k_values[0])
            k_geo = int(k_values[0])
        else:
            nmi_scores = [global_hard_nmi(k) for k in k_values]
            anchor_f1_scores = [anchor_macro_f1(k) for k in k_values]

            xs = torch.tensor(k_values, device=z.device, dtype=z.dtype)
            ys = torch.tensor(nmi_scores, device=z.device, dtype=z.dtype)
            x_norm = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1e-8)
            y_norm = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1e-8)
            curve = torch.stack([x_norm, y_norm], dim=1)
            line = torch.stack([x_norm, x_norm], dim=1)
            dist = (curve - line).norm(dim=1)
            k_info = int(k_values[int(dist.argmax().item())])

            best_anchor = max(anchor_f1_scores) if anchor_f1_scores else 0.0
            target = float(geo_ratio) * float(best_anchor)
            k_geo = int(k_values[-1])
            for k, score in zip(k_values, anchor_f1_scores):
                if float(score) >= target:
                    k_geo = int(k)
                    break

        best_k = int(round((int(k_info) + int(k_geo)) / 2.0)) + int(reserve_supports)
        best_k = max(int(common_min_k), min(best_k, int(common_max_k)))
        # If best_k was not in candidate map, compute it once.
        if best_k not in centers_by_class[0]:
            for c in range(C):
                pts = class_points[c]
                w = boundary_weights(pts, c, parents)
                centers, _ = weighted_kmeans(pts, w, k=int(best_k), iters=int(kmeans_iters))
                centers_by_class[c][int(best_k)] = centers.detach().clone()
    else:
        k_info = best_k
        k_geo = best_k

    anchors = torch.cat([centers_by_class[c][int(best_k)] for c in range(C)], dim=0).contiguous()
    anchor_labels = torch.arange(C, device=z.device, dtype=torch.long).repeat_interleave(int(best_k))

    meta = {
        "selected_per_class": int(best_k),
        "total_supports": int(anchors.shape[0]),
        "k_info": int(k_info),
        "k_geo": int(k_geo),
        "k_values": ",".join(str(v) for v in k_values),
        "k_values_min": int(min(k_values)),
        "k_values_max": int(max(k_values)),
    }
    return anchors, anchor_labels, meta


@torch.no_grad()
def init_sigma_from_anchors(
    anchors: torch.Tensor,
    *,
    sigma_mult: float,
    sigma_knn: int,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    m = int(anchors.shape[0])
    if m <= 1:
        return torch.full((m,), float(sigma_min), device=anchors.device, dtype=anchors.dtype)

    dist = torch.sqrt(pairwise_dist2(anchors, anchors) + 1e-8)
    dist.fill_diagonal_(float("inf"))
    kk = max(1, min(int(sigma_knn), m - 1))
    kth = dist.kthvalue(kk, dim=1).values
    sigma = kth * float(sigma_mult)
    sigma = sigma.clamp(float(sigma_min), float(sigma_max))
    return sigma.contiguous()


# -----------------------------------------------------------------------------
# Additive Gaussian field
# -----------------------------------------------------------------------------

class GaussianFieldLayer(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        sigma_init: torch.Tensor,
        *,
        charge_init_std: float,
        sigma_min: float,
        sigma_max: float,
        step_scale: float,
        scale_mode: str,
    ):
        super().__init__()
        anchors = anchors.detach().float().contiguous()
        sigma_init = sigma_init.detach().float().contiguous()
        if anchors.dim() != 2:
            raise ValueError("anchors must have shape [M, D]")
        if sigma_init.shape != (anchors.shape[0],):
            raise ValueError("sigma_init must have shape [M]")

        self.anchors = nn.Parameter(anchors.clone())
        self.charge = nn.Parameter(torch.randn(anchors.shape[0]) * float(charge_init_std))
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        sigma01 = (sigma_init - self.sigma_min) / max(self.sigma_max - self.sigma_min, 1e-8)
        self.raw_sigma = nn.Parameter(logit01(sigma01))
        self.step_scale = float(step_scale)
        self.scale_mode = str(scale_mode)
        self.register_buffer("anchors_init", anchors.clone(), persistent=True)

    @property
    def supports_n(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.anchors.shape[1])

    def sigma(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.raw_sigma)

    def field_scale(self) -> float:
        m = max(self.supports_n, 1)
        if self.scale_mode == "none":
            return 1.0
        if self.scale_mode == "mean":
            return 1.0 / float(m)
        if self.scale_mode == "sqrt":
            return 1.0 / math.sqrt(float(m))
        raise ValueError(f"unknown scale_mode: {self.scale_mode}")

    def gaussian_response(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma()
        dist2 = pairwise_dist2(z, self.anchors)
        sigma2 = sigma.view(1, -1).square().clamp_min(1e-8)
        k = torch.exp(-dist2 / (2.0 * sigma2))
        return k, sigma2

    def field(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Efficient high-dimensional form:
        # sum_j coef_j * (a_j - z) = coef @ anchors - z * sum_j coef_j
        k, sigma2 = self.gaussian_response(z)
        coef = self.charge.view(1, -1) * k / sigma2
        weighted_anchor = coef @ self.anchors.float()
        coef_sum = coef.sum(dim=1, keepdim=True)
        f = self.field_scale() * (weighted_anchor - z.float() * coef_sum)
        move = self.step_scale * f
        info = {
            "kernel_sum": k.sum(dim=1),
            "kernel_max": k.max(dim=1).values,
            "active_005": (k > 0.05).float().sum(dim=1),
            "active_001": (k > 0.01).float().sum(dim=1),
        }
        return move.to(z.dtype), info

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        move, info = self.field(z)
        return z + move, move, info

    @torch.no_grad()
    def diagnostic_scalars(self, prefix: str = "") -> Dict[str, float]:
        sigma = self.sigma().detach()
        charge = self.charge.detach()
        drift = (self.anchors.detach() - self.anchors_init.to(self.anchors.device)).norm(dim=1)
        return {
            f"{prefix}sigma_mean": float(sigma.mean().item()),
            f"{prefix}sigma_min": float(sigma.min().item()),
            f"{prefix}sigma_max": float(sigma.max().item()),
            f"{prefix}charge_mean": float(charge.mean().item()),
            f"{prefix}charge_abs_mean": float(charge.abs().mean().item()),
            f"{prefix}charge_abs_max": float(charge.abs().max().item()),
            f"{prefix}anchor_drift_mean": float(drift.mean().item()),
            f"{prefix}anchor_drift_max": float(drift.max().item()),
        }


class GaussianAATStack(nn.Module):
    def __init__(self, layers: Sequence[GaussianFieldLayer]):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))

    @property
    def state_dim(self) -> int:
        if not self.layers:
            raise RuntimeError("empty stack")
        return self.layers[0].state_dim

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        moves: List[torch.Tensor] = []
        infos: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            z, move, info = layer(z)
            moves.append(move)
            infos.append(info)
        return z, moves, infos


@dataclass
class EvalStats:
    loss: float
    acc: float
    last_move_mean: float
    last_move_p95: float
    last_move_max: float
    total_move_mean: float
    total_move_p95: float
    total_move_max: float
    active005_mean: float
    kernel_sum_mean: float
    kernel_max_mean: float


@torch.no_grad()
def evaluate(
    stack: GaussianAATStack,
    head: nn.Linear,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> EvalStats:
    stack.eval()
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    last_norms: List[torch.Tensor] = []
    total_norms: List[torch.Tensor] = []
    active005: List[torch.Tensor] = []
    kernel_sum: List[torch.Tensor] = []
    kernel_max: List[torch.Tensor] = []

    amp_enabled = bool(use_amp and device.type == "cuda")
    for z0, y in loader:
        z0 = z0.to(device)
        y = y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            z, moves, infos = stack(z0)
            logits = head(z)
            loss = F.cross_entropy(logits, y)

        total_loss += float(loss.item()) * int(y.numel())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())

        last_move = moves[-1].detach().float()
        last_norms.append(last_move.norm(dim=1).cpu())
        total_norms.append((z.detach().float() - z0.detach().float()).norm(dim=1).cpu())
        info = infos[-1]
        active005.append(info["active_005"].detach().float().cpu())
        kernel_sum.append(info["kernel_sum"].detach().float().cpu())
        kernel_max.append(info["kernel_max"].detach().float().cpu())

    ln = torch.cat(last_norms)
    tn = torch.cat(total_norms)
    a5 = torch.cat(active005)
    ks = torch.cat(kernel_sum)
    km = torch.cat(kernel_max)
    return EvalStats(
        loss=total_loss / max(total, 1),
        acc=total_correct / max(total, 1),
        last_move_mean=float(ln.mean().item()),
        last_move_p95=float(torch.quantile(ln, 0.95).item()),
        last_move_max=float(ln.max().item()),
        total_move_mean=float(tn.mean().item()),
        total_move_p95=float(torch.quantile(tn, 0.95).item()),
        total_move_max=float(tn.max().item()),
        active005_mean=float(a5.mean().item()),
        kernel_sum_mean=float(ks.mean().item()),
        kernel_max_mean=float(km.mean().item()),
    )


def make_optimizer(
    stack: GaussianAATStack,
    head: nn.Linear,
    *,
    base_lr: float,
    anchor_lr_factor: float,
    sigma_lr_factor: float,
    old_layer_lr_factor: float,
    new_layer_index: int,
    weight_decay: float,
) -> torch.optim.Optimizer:
    groups = []
    for i, layer in enumerate(stack.layers):
        factor = 1.0 if i == int(new_layer_index) else float(old_layer_lr_factor)
        groups.append({"params": [layer.charge], "lr": float(base_lr) * factor, "weight_decay": float(weight_decay)})
        groups.append({"params": [layer.anchors], "lr": float(base_lr) * factor * float(anchor_lr_factor), "weight_decay": float(weight_decay)})
        groups.append({"params": [layer.raw_sigma], "lr": float(base_lr) * factor * float(sigma_lr_factor), "weight_decay": float(weight_decay)})
    groups.append({"params": list(head.parameters()), "lr": float(base_lr), "weight_decay": float(weight_decay)})
    return torch.optim.AdamW(groups)


@torch.no_grad()
def layer_diag(layer: GaussianFieldLayer) -> Dict[str, float]:
    return layer.diagnostic_scalars(prefix="")


def train_stage(
    *,
    name: str,
    stack: GaussianAATStack,
    head: nn.Linear,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    anchor_lr_factor: float,
    sigma_lr_factor: float,
    old_layer_lr_factor: float,
    new_layer_index: int,
    weight_decay: float,
    use_amp: bool,
    log_path: Path,
    print_every: int,
) -> Tuple[float, int]:
    stack.to(device)
    head.to(device)
    unfreeze_module(stack)
    unfreeze_module(head)

    opt = make_optimizer(
        stack,
        head,
        base_lr=base_lr,
        anchor_lr_factor=anchor_lr_factor,
        sigma_lr_factor=sigma_lr_factor,
        old_layer_lr_factor=old_layer_lr_factor,
        new_layer_index=new_layer_index,
        weight_decay=weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    amp_enabled = bool(use_amp and device.type == "cuda")

    best_acc = -1.0
    best_epoch = 0
    best_stack = copy.deepcopy(stack.state_dict())
    best_head = copy.deepcopy(head.state_dict())

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "best_epoch",
            "last_move_mean", "last_move_p95", "last_move_max",
            "total_move_mean", "total_move_p95", "total_move_max",
            "sigma_mean", "sigma_min", "sigma_max", "charge_abs_mean", "charge_abs_max",
            "anchor_drift_mean", "anchor_drift_max", "active005_mean",
            "kernel_sum_mean", "kernel_max_mean",
        ])

        for epoch in range(1, int(epochs) + 1):
            stack.train()
            head.train()
            total_loss = 0.0
            total_correct = 0
            total = 0

            for z0, y in train_loader:
                z0 = z0.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    z, _, _ = stack(z0)
                    logits = head(z)
                    loss = F.cross_entropy(logits, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.item()) * int(y.numel())
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total += int(y.numel())

            train_loss = total_loss / max(total, 1)
            train_acc = total_correct / max(total, 1)
            val = evaluate(stack, head, val_loader, device, use_amp)

            if val.acc > best_acc:
                best_acc = val.acc
                best_epoch = epoch
                best_stack = copy.deepcopy(stack.state_dict())
                best_head = copy.deepcopy(head.state_dict())

            current_layer = stack.layers[int(new_layer_index)]
            diag = layer_diag(current_layer)

            writer.writerow([
                epoch, train_loss, train_acc, val.loss, val.acc, best_acc, best_epoch,
                val.last_move_mean, val.last_move_p95, val.last_move_max,
                val.total_move_mean, val.total_move_p95, val.total_move_max,
                diag["sigma_mean"], diag["sigma_min"], diag["sigma_max"],
                diag["charge_abs_mean"], diag["charge_abs_max"],
                diag["anchor_drift_mean"], diag["anchor_drift_max"],
                val.active005_mean, val.kernel_sum_mean, val.kernel_max_mean,
            ])
            f.flush()

            if epoch == 1 or epoch % int(print_every) == 0 or epoch == int(epochs):
                print(
                    f"{name} epoch {epoch:04d}/{epochs} | "
                    f"train_acc={train_acc:.4f} loss={train_loss:.4f} | "
                    f"val_acc={val.acc:.4f} loss={val.loss:.4f} best={best_acc:.4f}@{best_epoch} | "
                    f"last_move mean/p95/max={val.last_move_mean:.4f}/{val.last_move_p95:.4f}/{val.last_move_max:.4f} | "
                    f"total_move mean/p95/max={val.total_move_mean:.4f}/{val.total_move_p95:.4f}/{val.total_move_max:.4f} | "
                    f"L{int(new_layer_index)+1} sigma={diag['sigma_mean']:.4f} "
                    f"c_abs={diag['charge_abs_mean']:.4f} drift={diag['anchor_drift_mean']:.4f} | "
                    f"active@.05={val.active005_mean:.1f}"
                )

    stack.load_state_dict(best_stack)
    head.load_state_dict(best_head)
    return float(best_acc), int(best_epoch)


@torch.no_grad()
def collect_stage_output(
    stack: GaussianAATStack,
    z0: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(z0.shape[0])
    if int(max_samples) > 0 and int(max_samples) < n:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        idx = torch.randperm(n, generator=gen)[: int(max_samples)]
        z0_sel = z0[idx]
        y_sel = y[idx]
    else:
        z0_sel = z0
        y_sel = y

    loader = DataLoader(TensorDataset(z0_sel, y_sel), batch_size=int(batch_size), shuffle=False)
    stack.eval().to(device)
    outs: List[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device)
        z, _, _ = stack(xb)
        outs.append(z.detach().cpu())
    return torch.cat(outs, dim=0).contiguous(), y_sel.detach().cpu().contiguous()


def build_layer_from_z(
    z_init: torch.Tensor,
    y_init: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    stage_name: str,
) -> Tuple[GaussianFieldLayer, Dict[str, object]]:
    z_dev = z_init.to(device)
    y_dev = y_init.to(device)

    t0 = time.time()
    anchors, _, meta = choose_supports_parentless(
        z_dev,
        y_dev,
        num_classes=10,
        min_per_class=int(args.min_support_per_class),
        max_per_class=int(args.max_support_per_class),
        kmeans_iters=int(args.kmeans_iters),
        reserve_supports=int(args.reserve_supports),
        geo_ratio=float(args.geo_ratio),
        fixed_k=int(args.fixed_k),
        k_candidates=str(args.k_candidates),
        full_k_scan=bool(args.full_k_scan),
    )
    sigma_init = init_sigma_from_anchors(
        anchors,
        sigma_mult=float(args.sigma_mult),
        sigma_knn=int(args.sigma_knn),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    layer = GaussianFieldLayer(
        anchors.detach().cpu(),
        sigma_init.detach().cpu(),
        charge_init_std=float(args.charge_init_std),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        step_scale=float(args.step_scale),
        scale_mode=str(args.scale_mode),
    )
    meta = dict(meta)
    meta.update({
        "sigma_init_mean": float(sigma_init.mean().item()),
        "sigma_init_min": float(sigma_init.min().item()),
        "sigma_init_max": float(sigma_init.max().item()),
        "init_seconds": float(time.time() - t0),
    })
    print(
        f"{stage_name} init done in {meta['init_seconds']:.2f}s | "
        f"selected_per_class={meta['selected_per_class']} total_supports={meta['total_supports']} "
        f"k_info={meta['k_info']} k_geo={meta['k_geo']} | "
        f"sigma_init mean={meta['sigma_init_mean']:.4f} "
        f"min={meta['sigma_init_min']:.4f} max={meta['sigma_init_max']:.4f}"
    )
    return layer, meta


def aat_compute_units(stack: GaussianAATStack) -> int:
    return sum(layer.supports_n * layer.state_dim for layer in stack.layers)


def aat_supports_text(stack: GaussianAATStack) -> str:
    return "+".join(str(layer.supports_n) for layer in stack.layers)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progressive additive Gaussian AAT on flattened MNIST.")

    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Use random data for a quick syntax/runtime check.")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="mnist_gaussian_aat_progressive_flat")

    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--init-batch-size", type=int, default=512)
    p.add_argument("--init-samples", type=int, default=4000)

    p.add_argument("--extra-dims", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--stage1-epochs", type=int, default=300)
    p.add_argument("--new-layer-epochs", type=int, default=300)

    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--old-layer-lr-factor", type=float, default=0.1)
    p.add_argument("--anchor-lr-factor", type=float, default=0.25)
    p.add_argument("--sigma-lr-factor", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--min-support-per-class", type=int, default=2)
    p.add_argument("--max-support-per-class", type=int, default=100)
    p.add_argument("--fixed-k", type=int, default=0)
    p.add_argument("--k-candidates", type=str, default="auto")
    p.add_argument("--full-k-scan", action="store_true")
    p.add_argument("--kmeans-iters", type=int, default=4)
    p.add_argument("--reserve-supports", type=int, default=5)
    p.add_argument("--geo-ratio", type=float, default=0.95)

    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--sigma-max", type=float, default=30.0)
    p.add_argument("--sigma-mult", type=float, default=0.35)
    p.add_argument("--sigma-knn", type=int, default=6)
    p.add_argument("--charge-init-std", type=float, default=1e-2)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--scale-mode", type=str, default="sqrt", choices=["sqrt", "mean", "none"])

    p.add_argument("--print-every", type=int, default=10)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("=" * 110)
    print("Progressive Parentless Additive Gaussian AAT | flattened MNIST")
    print("=" * 110)
    print(
        f"device={device} seed={args.seed} n_train={args.n_train} n_val={args.n_val} "
        f"extra_dims={args.extra_dims} max_depth={args.max_depth} out={out_dir}"
    )
    print("lift: flattened image -> [2*x-1, hidden zeros], no dimension permutation")
    print(
        "field: Phi=sum c_j exp(-||z-a_j||^2/(2 sigma_j^2)); "
        "move=step_scale * sqrt-scaled grad(Phi)"
    )
    print(
        "training: progressive init; after adding layer L, all previous layers continue with smaller LR; "
        "no activation, no dropout, no scheduler, no softmax"
    )
    print(
        f"Auto-K per class range: [{args.min_support_per_class}, {args.max_support_per_class}], "
        f"k_candidates={args.k_candidates}, full_k_scan={args.full_k_scan}"
    )

    x_train, y_train, x_val, y_val = load_mnist_flat(
        data_root=args.data_root,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        download=args.download,
        smoke_test=args.smoke_test,
    )

    z0_train = lift_flat_images(x_train, int(args.extra_dims))
    z0_val = lift_flat_images(x_val, int(args.extra_dims))
    state_dim = int(z0_train.shape[1])

    print(f"data: raw_dim=784 state_dim={state_dim} train={len(y_train)} val={len(y_val)}")
    print(f"label counts train={torch.bincount(y_train, minlength=10).tolist()}")

    train_loader = DataLoader(
        TensorDataset(z0_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(z0_val, y_val),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    summary_rows: List[Dict[str, object]] = []

    # Stage 1.
    layer1, meta1 = build_layer_from_z(
        z0_train if int(args.init_samples) <= 0 else z0_train[: min(int(args.init_samples), z0_train.shape[0])],
        y_train if int(args.init_samples) <= 0 else y_train[: min(int(args.init_samples), y_train.shape[0])],
        args=args,
        device=device,
        stage_name="AAT/L1",
    )
    stack = GaussianAATStack([layer1])
    head = nn.Linear(state_dim, 10)

    print(
        f"AAT depth=1 params trainable={format_int(count_parameters(nn.ModuleList([stack, head])))} "
        f"total={format_int(count_parameters(nn.ModuleList([stack, head]), trainable_only=False))} | "
        f"supports={aat_supports_text(stack)} | approx_field_units_per_sample={format_int(aat_compute_units(stack))}"
    )

    best_acc, best_epoch = train_stage(
        name="aat_L1",
        stack=stack,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(args.stage1_epochs),
        base_lr=float(args.lr),
        anchor_lr_factor=float(args.anchor_lr_factor),
        sigma_lr_factor=float(args.sigma_lr_factor),
        old_layer_lr_factor=float(args.old_layer_lr_factor),
        new_layer_index=0,
        weight_decay=float(args.weight_decay),
        use_amp=bool(args.amp),
        log_path=out_dir / "aat_L1_log.csv",
        print_every=int(args.print_every),
    )

    summary_rows.append({
        "model": "AAT-L1",
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "params": count_parameters(nn.ModuleList([stack, head]), trainable_only=False),
        "supports": aat_supports_text(stack),
        "approx_units_per_sample": aat_compute_units(stack),
    })

    # Deeper progressive stages.
    max_depth = max(1, int(args.max_depth))
    for depth in range(2, max_depth + 1):
        z_init, y_init = collect_stage_output(
            stack,
            z0_train,
            y_train,
            device=device,
            max_samples=int(args.init_samples),
            batch_size=int(args.init_batch_size),
            seed=int(args.seed) + 1000 + depth,
        )
        print(
            f"AAT/L{depth} init source: z=T_1..T_{depth-1}(z0), "
            f"samples={z_init.shape[0]}, state_dim={state_dim}, z_norm_mean={z_init.norm(dim=1).mean().item():.4f}"
        )
        new_layer, meta = build_layer_from_z(
            z_init,
            y_init,
            args=args,
            device=device,
            stage_name=f"AAT/L{depth}",
        )

        stack.layers.append(new_layer)
        head = nn.Linear(state_dim, 10)

        print(
            f"AAT depth={depth} params trainable={format_int(count_parameters(nn.ModuleList([stack, head])))} "
            f"total={format_int(count_parameters(nn.ModuleList([stack, head]), trainable_only=False))} | "
            f"supports={aat_supports_text(stack)} | approx_field_units_per_sample={format_int(aat_compute_units(stack))} | "
            f"old_layer_lr={float(args.lr) * float(args.old_layer_lr_factor):.2e}, "
            f"new_layer/head_lr={float(args.lr):.2e}"
        )

        best_acc, best_epoch = train_stage(
            name=f"aat_L{depth}_joint",
            stack=stack,
            head=head,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=int(args.new_layer_epochs),
            base_lr=float(args.lr),
            anchor_lr_factor=float(args.anchor_lr_factor),
            sigma_lr_factor=float(args.sigma_lr_factor),
            old_layer_lr_factor=float(args.old_layer_lr_factor),
            new_layer_index=depth - 1,
            weight_decay=float(args.weight_decay),
            use_amp=bool(args.amp),
            log_path=out_dir / f"aat_L{depth}_joint_log.csv",
            print_every=int(args.print_every),
        )

        summary_rows.append({
            "model": f"AAT-L{depth}",
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "params": count_parameters(nn.ModuleList([stack, head]), trainable_only=False),
            "supports": aat_supports_text(stack),
            "approx_units_per_sample": aat_compute_units(stack),
        })

    # Write summary.
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "best_acc", "best_epoch", "params", "supports", "approx_units_per_sample"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("=" * 110)
    print("FINAL SUMMARY")
    for row in summary_rows:
        print(
            f"{row['model']:>8} | best_acc={float(row['best_acc']):.4f}@{int(row['best_epoch'])} | "
            f"params={format_int(int(row['params']))} | supports={row['supports']} | "
            f"approx_units/sample={format_int(int(row['approx_units_per_sample']))}"
        )
    print(f"summary: {summary_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
