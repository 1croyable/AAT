# -*- coding: utf-8 -*-
# Progressive N-layer version: train L1 greedily, initialize each next layer on current transported outputs, then joint-train all layers with smaller LR for previous layers.
from __future__ import annotations

import argparse
import csv
import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Small math utilities
# -----------------------------------------------------------------------------

def pairwise_dist2(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
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


# -----------------------------------------------------------------------------
# 3D checkerboard data and direct lift, no permutation
# -----------------------------------------------------------------------------

def make_checkerboard_3d(n: int, grid_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    x = torch.rand(int(n), 3, generator=gen)
    cells = torch.floor(x * int(grid_size)).long().clamp(max=int(grid_size) - 1)
    y = ((cells[:, 0] + cells[:, 1] + cells[:, 2]) % 2).long()
    return x, y


def lift_checkerboard_3d(x: torch.Tensor, extra_dims: int = 0, extra_init_std: float = 0.0) -> torch.Tensor:
    """Lift x in [0, 1]^3 to z in R^(3+extra_dims), preserving coordinate order.

    No lift permutation is used. The first three coordinates keep the original
    checkerboard geometry. Extra coordinates are hidden state dimensions and are
    initialized to zero by default.
    """
    z = x.float() * 2.0 - 1.0
    extra_dims = int(extra_dims)
    if extra_dims <= 0:
        return z
    extra = z.new_zeros((z.shape[0], extra_dims))
    if float(extra_init_std) > 0:
        extra = extra + torch.randn_like(extra) * float(extra_init_std)
    return torch.cat([z, extra], dim=1)


# -----------------------------------------------------------------------------
# Initialization: parent-assisted selection, parentless materialization
# -----------------------------------------------------------------------------

@torch.no_grad()
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
    # Boundary attention: points closer to the decision boundary get larger weights.
    w = (1.0 - silhouette).clamp_min(1e-3)
    return w / w.mean().clamp_min(1e-8)


@torch.no_grad()
def weighted_kmeans(
    pts: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    iters: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic weighted k-means with farthest weighted initialization."""
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
def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(1e-12)
    return -(p * p.log()).sum()


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
    else:
        needed_ks = list(range(1, int(common_max_k) + 1))

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
        k_values = list(range(int(common_min_k), int(common_max_k) + 1))

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
    else:
        k_info = best_k
        k_geo = best_k
        k_values = [best_k]

    anchors = torch.cat([centers_by_class[c][int(best_k)] for c in range(C)], dim=0).contiguous()
    anchor_labels = torch.arange(C, device=z.device, dtype=torch.long).repeat_interleave(int(best_k))

    meta = {
        "selected_per_class": int(best_k),
        "total_supports": int(anchors.shape[0]),
        "k_info": int(k_info),
        "k_geo": int(k_geo),
        "k_values_min": int(min(k_values)),
        "k_values_max": int(max(k_values)),
    }
    return anchors, anchor_labels, meta


@torch.no_grad()
def init_sigma_from_anchors(
    anchors: torch.Tensor,
    *,
    sigma_mult: float,
    sigma_min: float,
    sigma_max: float,
    knn: int,
) -> torch.Tensor:
    M = int(anchors.shape[0])
    if M <= 1:
        return torch.full((M,), 0.5, device=anchors.device, dtype=anchors.dtype)

    d2 = pairwise_dist2(anchors, anchors)
    d2.fill_diagonal_(float("inf"))
    kth = max(1, min(int(knn), M - 1))
    d = torch.sqrt(torch.topk(d2, k=kth, largest=False, dim=1).values[:, -1].clamp_min(1e-8))
    sigma = float(sigma_mult) * d
    return sigma.clamp(float(sigma_min), float(sigma_max))


# -----------------------------------------------------------------------------
# Gaussian conservative field layer
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
        k, sigma2 = self.gaussian_response(z)
        vec = self.anchors.unsqueeze(0) - z.unsqueeze(1)
        coef = self.charge.view(1, -1) * k / sigma2
        f = self.field_scale() * (coef.unsqueeze(-1) * vec).sum(dim=1)
        move = self.step_scale * f
        info = {
            "kernel_sum": k.sum(dim=1),
            "kernel_max": k.max(dim=1).values,
            "active_005": (k > 0.05).float().sum(dim=1),
            "active_001": (k > 0.01).float().sum(dim=1),
        }
        return move, info

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
    def __init__(self, layers: List[GaussianFieldLayer], num_classes: int = 2, extra_dims: int = 0, extra_init_std: float = 0.0):
        super().__init__()
        if not layers:
            raise ValueError("at least one layer is required")
        state_dim = layers[0].state_dim
        for layer in layers:
            if layer.state_dim != state_dim:
                raise ValueError("all layers must share the same state_dim")
        self.layers = nn.ModuleList(layers)
        self.extra_dims = int(extra_dims)
        self.extra_init_std = float(extra_init_std)
        self.head = nn.Linear(state_dim, int(num_classes), bias=True)

    @property
    def state_dim(self) -> int:
        return int(self.layers[0].state_dim)

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        return lift_checkerboard_3d(x, extra_dims=self.extra_dims, extra_init_std=self.extra_init_std)

    def transport_from_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        moves: List[torch.Tensor] = []
        infos: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            z, move, info = layer(z)
            moves.append(move)
            infos.append(info)
        return z, moves, infos

    def transport(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        z0 = self.lift(x)
        z, moves, infos = self.transport_from_z(z0)
        return z0, z, moves, infos

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        z0, z, moves, infos = self.transport(x)
        logits = self.head(z)
        return logits, {"z0": z0, "z": z, "moves": moves, "infos": infos}


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

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
    active001_mean: float
    kernel_sum_mean: float
    kernel_max_mean: float


@torch.no_grad()
def evaluate(model: GaussianAATStack, loader: DataLoader, device: torch.device) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    last_move_norms: List[torch.Tensor] = []
    total_move_norms: List[torch.Tensor] = []
    active005: List[torch.Tensor] = []
    active001: List[torch.Tensor] = []
    kernel_sum: List[torch.Tensor] = []
    kernel_max: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, aux = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * int(x.shape[0])
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += int(x.shape[0])

        z0 = aux["z0"]
        z = aux["z"]
        moves: List[torch.Tensor] = aux["moves"]
        infos: List[Dict[str, torch.Tensor]] = aux["infos"]
        last_move = moves[-1]
        last_info = infos[-1]
        last_move_norms.append(last_move.norm(dim=1).detach().cpu())
        total_move_norms.append((z - z0).norm(dim=1).detach().cpu())
        active005.append(last_info["active_005"].detach().cpu())
        active001.append(last_info["active_001"].detach().cpu())
        kernel_sum.append(last_info["kernel_sum"].detach().cpu())
        kernel_max.append(last_info["kernel_max"].detach().cpu())

    lm = torch.cat(last_move_norms) if last_move_norms else torch.zeros(1)
    tm = torch.cat(total_move_norms) if total_move_norms else torch.zeros(1)
    a005 = torch.cat(active005) if active005 else torch.zeros(1)
    a001 = torch.cat(active001) if active001 else torch.zeros(1)
    ksum = torch.cat(kernel_sum) if kernel_sum else torch.zeros(1)
    kmax = torch.cat(kernel_max) if kernel_max else torch.zeros(1)
    return EvalStats(
        loss=total_loss / max(total_n, 1),
        acc=total_correct / max(total_n, 1),
        last_move_mean=float(lm.mean().item()),
        last_move_p95=float(torch.quantile(lm, 0.95).item()),
        last_move_max=float(lm.max().item()),
        total_move_mean=float(tm.mean().item()),
        total_move_p95=float(torch.quantile(tm, 0.95).item()),
        total_move_max=float(tm.max().item()),
        active005_mean=float(a005.mean().item()),
        active001_mean=float(a001.mean().item()),
        kernel_sum_mean=float(ksum.mean().item()),
        kernel_max_mean=float(kmax.mean().item()),
    )


def make_optimizer(
    model: GaussianAATStack,
    args,
    lr: float,
    layer_lr_factors: List[float] | None = None,
) -> torch.optim.Optimizer:
    """Plain Adam optimizer.

    layer_lr_factors is the only intentional control here: after L2 is initialized,
    L1 is allowed to move, but more slowly, so the learned first geometry is not
    immediately overwritten by the second-layer objective.
    """
    groups = []
    if layer_lr_factors is None:
        layer_lr_factors = [1.0 for _ in model.layers]

    for li, layer in enumerate(model.layers):
        prefix = f"L{li + 1}"
        layer_factor = float(layer_lr_factors[li]) if li < len(layer_lr_factors) else 1.0
        layer_lr = float(lr) * layer_factor
        if layer.charge.requires_grad:
            groups.append({"params": [layer.charge], "lr": layer_lr, "name": prefix + ".charge"})
        if layer.raw_sigma.requires_grad:
            groups.append({"params": [layer.raw_sigma], "lr": layer_lr * float(args.sigma_lr_factor), "name": prefix + ".sigma"})
        if layer.anchors.requires_grad:
            groups.append({"params": [layer.anchors], "lr": layer_lr * float(args.anchor_lr_factor), "name": prefix + ".anchors"})
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": float(lr), "name": "head"})
    if not groups:
        raise ValueError("no trainable parameters found")
    return torch.optim.Adam(groups)


def layer_diag(model: GaussianAATStack) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, layer in enumerate(model.layers):
        out.update(layer.diagnostic_scalars(prefix=f"L{i + 1}_"))
    return out


def train_stage(
    model: GaussianAATStack,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    args,
    epochs: int,
    lr: float,
    stage_name: str,
    log_path: Path,
    layer_lr_factors: List[float] | None = None,
) -> Tuple[float, int, Dict[str, torch.Tensor]]:
    optimizer = make_optimizer(model, args, lr=float(lr), layer_lr_factors=layer_lr_factors)
    rows: List[Dict[str, object]] = []
    best_acc = -1.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] = {}

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                try:
                    ctx = torch.amp.autocast(device_type="cuda", enabled=True)
                except Exception:
                    ctx = torch.cuda.amp.autocast(enabled=True)
                with ctx:
                    logits, aux = model(x)
                    loss = F.cross_entropy(logits, y)
                assert scaler is not None
                scaler.scale(loss).backward()
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, aux = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.grad_clip))
                optimizer.step()

            total_loss += float(loss.detach().item()) * int(x.shape[0])
            total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total_n += int(x.shape[0])

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)
        val_stats = evaluate(model, val_loader, device)
        diag = layer_diag(model)

        if val_stats.acc > best_acc:
            best_acc = float(val_stats.acc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row: Dict[str, object] = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_stats.loss,
            "val_acc": val_stats.acc,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "last_move_mean": val_stats.last_move_mean,
            "last_move_p95": val_stats.last_move_p95,
            "last_move_max": val_stats.last_move_max,
            "total_move_mean": val_stats.total_move_mean,
            "total_move_p95": val_stats.total_move_p95,
            "total_move_max": val_stats.total_move_max,
            "active005_mean": val_stats.active005_mean,
            "active001_mean": val_stats.active001_mean,
            "kernel_sum_mean": val_stats.kernel_sum_mean,
            "kernel_max_mean": val_stats.kernel_max_mean,
            **diag,
        }
        rows.append(row)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        last_layer_idx = len(model.layers)
        last_sigma = diag[f"L{last_layer_idx}_sigma_mean"]
        last_c_abs = diag[f"L{last_layer_idx}_charge_abs_mean"]
        last_drift = diag[f"L{last_layer_idx}_anchor_drift_mean"]
        print(
            f"{stage_name} epoch {epoch:04d}/{epochs} | "
            f"train_acc={train_acc:.4f} loss={train_loss:.4f} | "
            f"val_acc={val_stats.acc:.4f} loss={val_stats.loss:.4f} best={best_acc:.4f}@{best_epoch} | "
            f"last_move mean/p95/max={val_stats.last_move_mean:.4f}/{val_stats.last_move_p95:.4f}/{val_stats.last_move_max:.4f} | "
            f"total_move mean/p95/max={val_stats.total_move_mean:.4f}/{val_stats.total_move_p95:.4f}/{val_stats.total_move_max:.4f} | "
            f"L{last_layer_idx} sigma={last_sigma:.4f} c_abs={last_c_abs:.4f} drift={last_drift:.4f} | "
            f"active@.05={val_stats.active005_mean:.1f}"
        )

    return best_acc, best_epoch, best_state


@torch.no_grad()
def collect_stage_output(
    model: GaussianAATStack,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    n = int(x.shape[0])
    if int(samples) > 0 and int(samples) < n:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        idx = torch.randperm(n, generator=gen)[: int(samples)]
        x_use = x.index_select(0, idx)
        y_use = y.index_select(0, idx)
    else:
        x_use = x
        y_use = y

    outs: List[torch.Tensor] = []
    loader = DataLoader(TensorDataset(x_use, y_use), batch_size=int(batch_size), shuffle=False, drop_last=False)
    for xb, _ in loader:
        xb = xb.to(device)
        _, aux = model(xb)
        outs.append(aux["z"].detach().cpu())
    return torch.cat(outs, dim=0).to(device), y_use.to(device)


def build_layer_from_z(z: torch.Tensor, y: torch.Tensor, args, *, stage_label: str) -> Tuple[GaussianFieldLayer, Dict[str, object], torch.Tensor]:
    t0 = time.time()
    anchors, anchor_labels, init_meta = choose_supports_parentless(
        z,
        y,
        num_classes=2,
        min_per_class=int(args.min_support_per_class),
        max_per_class=int(args.max_support_per_class),
        kmeans_iters=int(args.kmeans_iters),
        reserve_supports=int(args.reserve_supports),
        geo_ratio=float(args.geo_ratio),
        fixed_k=int(args.fixed_k),
    )
    sigma_init = init_sigma_from_anchors(
        anchors,
        sigma_mult=float(args.sigma_mult),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        knn=int(args.sigma_knn),
    )
    layer = GaussianFieldLayer(
        anchors=anchors,
        sigma_init=sigma_init,
        charge_init_std=float(args.charge_init_std),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        step_scale=float(args.step_scale),
        scale_mode=str(args.scale_mode),
    )
    init_meta = dict(init_meta)
    init_meta["init_seconds"] = time.time() - t0
    print(
        f"{stage_label} init done in {init_meta['init_seconds']:.2f}s | "
        f"selected_per_class={init_meta['selected_per_class']} total_supports={init_meta['total_supports']} "
        f"k_info={init_meta['k_info']} k_geo={init_meta['k_geo']} | "
        f"sigma_init mean={sigma_init.mean().item():.4f} min={sigma_init.min().item():.4f} max={sigma_init.max().item():.4f}"
    )
    return layer, init_meta, sigma_init


@torch.no_grad()
def save_checkpoint(path: Path, model: GaussianAATStack, args, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "TwoStageJointGaussianAAT3DCheckpoint",
            "version": 1,
            "args": vars(args),
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        path,
    )


# -----------------------------------------------------------------------------
# MLP baselines
# -----------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, num_classes: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.ReLU())
            d = int(hidden_dim)
        layers.append(nn.Linear(d, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float() * 2.0 - 1.0)


@torch.no_grad()
def evaluate_mlp(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * int(x.shape[0])
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += int(x.shape[0])
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    amp: bool,
    log_path: Path,
    name: str,
) -> Tuple[float, int, Dict[str, torch.Tensor]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    rows: List[Dict[str, object]] = []
    best_acc = -1.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] = {}

    use_amp = bool(amp and device.type == "cuda")
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                try:
                    ctx = torch.amp.autocast(device_type="cuda", enabled=True)
                except Exception:
                    ctx = torch.cuda.amp.autocast(enabled=True)
                with ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * int(x.shape[0])
            total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total_n += int(x.shape[0])

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)
        val_loss, val_acc = evaluate_mlp(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = float(val_acc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        }
        rows.append(row)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"{name} epoch {epoch:04d}/{epochs} | "
            f"train_acc={train_acc:.4f} loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} loss={val_loss:.4f} best={best_acc:.4f}@{best_epoch}"
        )

    return best_acc, best_epoch, best_state


def parse_depths(text: str) -> List[int]:
    vals = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return sorted(set(vals))


def aat_compute_units(model: GaussianAATStack) -> int:
    # Dominant field distance/weighted-vector work is proportional to sum_l M_l * D.
    return int(sum(layer.supports_n * layer.state_dim for layer in model.layers))


def aat_supports_text(model: GaussianAATStack) -> str:
    return "+".join(str(layer.supports_n) for layer in model.layers)


def parse_args():
    p = argparse.ArgumentParser(description="Progressive multi-layer additive Gaussian AAT vs MLP baselines on 3D checkerboard.")
    p.add_argument("--out-dir", type=str, default="gaussian_aat3d_progressive_compare")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--extra-dims", type=int, default=2, help="Default: 2 hidden dimensions, so state_dim=5 for 3D input.")
    p.add_argument("--extra-init-std", type=float, default=0.0)
    p.add_argument("--n-train", type=int, default=12000)
    p.add_argument("--n-val", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=2048)

    # Progressive AAT stages.
    p.add_argument("--aat-depths", type=str, default="3,4", help="Which AAT depths to report. The script trains progressively up to max(depths).")
    p.add_argument("--stage1-epochs", type=int, default=300)
    p.add_argument("--new-layer-epochs", type=int, default=300, help="Epochs after each new layer is initialized.")
    p.add_argument("--init-samples", type=int, default=4000, help="Transported train samples used to initialize each new layer. Use 0 for all train samples.")

    # Initialization / support points. Defaults follow the new requested Auto-K range.
    p.add_argument("--min-support-per-class", type=int, default=2)
    p.add_argument("--max-support-per-class", type=int, default=100)
    p.add_argument("--fixed-k", type=int, default=0, help="If >0, bypass Auto-K and use this many anchors per class for every layer.")
    p.add_argument("--reserve-supports", type=int, default=5)
    p.add_argument("--geo-ratio", type=float, default=0.99)
    p.add_argument("--kmeans-iters", type=int, default=12)

    # Gaussian field.
    p.add_argument("--sigma-min", type=float, default=0.04)
    p.add_argument("--sigma-max", type=float, default=2.0)
    p.add_argument("--sigma-mult", type=float, default=0.35)
    p.add_argument("--sigma-knn", type=int, default=6)
    p.add_argument("--charge-init-std", type=float, default=1e-2)
    p.add_argument("--step-scale", type=float, default=0.3)
    p.add_argument("--scale-mode", type=str, default="sqrt", choices=["sqrt", "mean", "none"])

    # Optimization. No scheduler/dropout/activation/regularization is used.
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--new-layer-lr", type=float, default=2e-3)
    p.add_argument("--prev-layer-lr-factor", type=float, default=0.10, help="When adding a new layer, older layers use this LR fraction.")
    p.add_argument("--anchor-lr-factor", type=float, default=0.25)
    p.add_argument("--sigma-lr-factor", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0)  # present for compatibility; Adam path does not use it.
    p.add_argument("--grad-clip", type=float, default=0.0)     # present for compatibility; default disabled.
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")

    # MLP baselines.
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument("--mlp-epochs", type=int, default=300)
    p.add_argument("--mlp-lr", type=float, default=2e-3)
    p.add_argument("--mlp-small-hidden", type=int, default=64)
    p.add_argument("--mlp-small-layers", type=int, default=3)
    p.add_argument("--mlp-large-hidden", type=int, default=256)
    p.add_argument("--mlp-large-layers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aat_depths = parse_depths(args.aat_depths)
    if not aat_depths:
        raise ValueError("--aat-depths cannot be empty")
    max_aat_depth = max(aat_depths)
    if max_aat_depth < 1:
        raise ValueError("AAT depth must be >= 1")

    state_dim = 3 + max(int(args.extra_dims), 0)
    print("=" * 104)
    print("Progressive Parentless Additive Gaussian AAT vs MLP baselines | 3D checkerboard")
    print("=" * 104)
    print(
        f"device={device} seed={args.seed} grid_size={args.grid_size} extra_dims={args.extra_dims} "
        f"state_dim={state_dim} out={out_dir}"
    )
    print("lift: [2*x-1, hidden zeros], no dimension permutation")
    print(
        f"field: z in R^{state_dim}, Phi=sum c_j exp(-||z-a_j||^2/(2 sigma_j^2)), "
        f"move=step_scale * {args.scale_mode}-scaled grad(Phi)"
    )
    print("training: progressive init; after adding layer L, all previous layers continue with smaller LR")
    print("no activation, no dropout, no scheduler; Auto-K range per class: "
          f"[{args.min_support_per_class}, {args.max_support_per_class}]")

    train_x, train_y = make_checkerboard_3d(args.n_train, args.grid_size, seed=args.seed)
    val_x, val_y = make_checkerboard_3d(args.n_val, args.grid_size, seed=args.seed + 12345)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    summary_rows: List[Dict[str, object]] = []

    # -------------------------
    # Progressive AAT training.
    # -------------------------
    z0_init = lift_checkerboard_3d(
        train_x.to(device),
        extra_dims=int(args.extra_dims),
        extra_init_std=float(args.extra_init_std),
    )
    layer1, init1_meta, _ = build_layer_from_z(z0_init, train_y.to(device), args, stage_label="AAT/L1")
    model = GaussianAATStack([layer1], num_classes=2, extra_dims=int(args.extra_dims), extra_init_std=float(args.extra_init_std)).to(device)
    print(
        f"AAT depth=1 params trainable={count_parameters(model, True):,} total={count_parameters(model, False):,} | "
        f"supports={aat_supports_text(model)} | approx_field_units_per_sample={aat_compute_units(model):,}"
    )
    best_acc, best_epoch, best_state = train_stage(
        model,
        train_loader,
        val_loader,
        device=device,
        args=args,
        epochs=int(args.stage1_epochs),
        lr=float(args.lr),
        stage_name="aat_L1",
        log_path=out_dir / "aat_L1_log.csv",
        layer_lr_factors=[1.0],
    )
    if best_state:
        model.load_state_dict(best_state)
    stage_metas: Dict[int, Dict[str, object]] = {1: init1_meta}
    if 1 in aat_depths:
        summary_rows.append({
            "model": "AAT-L1",
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "params": count_parameters(model, False),
            "supports": aat_supports_text(model),
            "approx_field_units_per_sample": aat_compute_units(model),
        })

    for depth in range(2, max_aat_depth + 1):
        # Use the current best lower-depth model as a fixed coordinate map for the next layer's initialization.
        model.eval()
        z_init, y_init = collect_stage_output(
            model,
            train_x,
            train_y,
            device=device,
            samples=int(args.init_samples),
            batch_size=int(args.eval_batch_size),
            seed=int(args.seed) + 1000 + depth,
        )
        print(
            f"AAT/L{depth} init source: z{depth-1}=T_1..T_{depth-1}(z0), "
            f"samples={z_init.shape[0]}, state_dim={z_init.shape[1]}, z_norm_mean={z_init.norm(dim=1).mean().item():.4f}"
        )
        new_layer, init_meta, _ = build_layer_from_z(z_init, y_init, args, stage_label=f"AAT/L{depth}")
        stage_metas[depth] = init_meta

        old_layers = [layer for layer in model.layers]
        model = GaussianAATStack(
            old_layers + [new_layer],
            num_classes=2,
            extra_dims=int(args.extra_dims),
            extra_init_std=float(args.extra_init_std),
        ).to(device)
        unfreeze_module(model)
        lr_factors = [float(args.prev_layer_lr_factor) for _ in range(depth - 1)] + [1.0]
        print(
            f"AAT depth={depth} params trainable={count_parameters(model, True):,} total={count_parameters(model, False):,} | "
            f"supports={aat_supports_text(model)} | approx_field_units_per_sample={aat_compute_units(model):,} | "
            f"old_layer_lr={float(args.new_layer_lr) * float(args.prev_layer_lr_factor):.2e}, "
            f"new_layer/head_lr={float(args.new_layer_lr):.2e}"
        )
        best_acc, best_epoch, best_state = train_stage(
            model,
            train_loader,
            val_loader,
            device=device,
            args=args,
            epochs=int(args.new_layer_epochs),
            lr=float(args.new_layer_lr),
            stage_name=f"aat_L{depth}_joint",
            log_path=out_dir / f"aat_L{depth}_joint_log.csv",
            layer_lr_factors=lr_factors,
        )
        if best_state:
            model.load_state_dict(best_state)

        if depth in aat_depths:
            summary_rows.append({
                "model": f"AAT-L{depth}",
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "params": count_parameters(model, False),
                "supports": aat_supports_text(model),
                "approx_field_units_per_sample": aat_compute_units(model),
            })

    if bool(args.save_checkpoint):
        save_checkpoint(
            out_dir / f"aat_L{max_aat_depth}_best.pt",
            model,
            args,
            {
                "depth": max_aat_depth,
                "stage_metas": stage_metas,
                "summary_so_far": summary_rows,
            },
        )

    # -------------------------
    # MLP baselines.
    # -------------------------
    if not bool(args.skip_mlp):
        mlp_configs = [
            ("MLP-small", int(args.mlp_small_hidden), int(args.mlp_small_layers)),
            ("MLP-large", int(args.mlp_large_hidden), int(args.mlp_large_layers)),
        ]
        for name, hidden, hidden_layers in mlp_configs:
            mlp = MLPBaseline(input_dim=3, hidden_dim=hidden, hidden_layers=hidden_layers, num_classes=2).to(device)
            params = count_parameters(mlp, False)
            approx_dense_units = 3 * hidden + max(hidden_layers - 1, 0) * hidden * hidden + hidden * 2
            print(
                f"{name} params={params:,} | hidden={hidden} layers={hidden_layers} | "
                f"approx_dense_mults_per_sample={approx_dense_units:,}"
            )
            mlp_best, mlp_epoch, mlp_state = train_mlp(
                mlp,
                train_loader,
                val_loader,
                device=device,
                epochs=int(args.mlp_epochs),
                lr=float(args.mlp_lr),
                amp=bool(args.amp),
                log_path=out_dir / f"{name.lower().replace('-', '_')}_log.csv",
                name=name,
            )
            if mlp_state:
                mlp.load_state_dict(mlp_state)
            summary_rows.append({
                "model": name,
                "best_acc": mlp_best,
                "best_epoch": mlp_epoch,
                "params": params,
                "supports": "",
                "approx_field_units_per_sample": approx_dense_units,
            })

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "best_acc", "best_epoch", "params", "supports", "approx_field_units_per_sample"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 104)
    print("FINAL SUMMARY")
    for row in summary_rows:
        print(
            f"{row['model']:>10s} | best_acc={float(row['best_acc']):.4f}@{int(row['best_epoch'])} | "
            f"params={int(row['params']):,} | supports={row['supports']} | "
            f"approx_units/sample={int(row['approx_field_units_per_sample']):,}"
        )
    print(f"summary: {summary_path}")
    if bool(args.save_checkpoint):
        print(f"checkpoint: {out_dir / f'aat_L{max_aat_depth}_best.pt'}")
    print("=" * 104)


if __name__ == "__main__":
    main()
