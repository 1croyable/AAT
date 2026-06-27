# -*- coding: utf-8 -*-
r"""
Minimal MNIST experiment for AAT 2.0 Gaussian terrain transport.

This script runs ONE model only:
  local_spin

Transport is unchanged from the previous AAT 2.0 Gaussian field experiment.
Only activation is changed:
  - old response-center activation is replaced by a local activation field.
  - each transport layer has an independent activation field initialized from the
    same anchors/sigma as the transport field.
  - at activation time, the activation field provides:
      1) local center r(x)
      2) potential theta(x), used as a rotation angle
      3) tangent direction u(x), computed as a cheap fixed-skew rotation of the
         activation field gradient
  - local displacement d = z_mid - r is rotated in the plane (u, extra_height),
    ReLU is applied, then the displacement is rotated back.

No scans. No output files. Everything prints to console.

Run from:
  C:\Projets\AATField\Experiments\research-capacity

Example:
  python experiment_mnist_aat_2p0_local_spin_activation.py
  python experiment_mnist_aat_2p0_local_spin_activation.py --K 16 --depth 3 --epochs 80
"""
from __future__ import annotations

import argparse
import gzip
import math
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Defaults
# -----------------------------

SEED = 42
MNIST_RAW_DIR = r"C:\Projets\AATField\Experiments\data\MNIST\raw"
BASE_DIM = 28 * 28
N_TRAIN = 10000
N_VAL = 10000
BATCH_SIZE = 256
LR = 2e-3
SIGMA_LR = 5e-4
USE_AMP = True
C_INIT_STD = 0.04
ACT_C_INIT_STD = 0.01
EPS = 1e-12


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_quantile(x: torch.Tensor, q: float) -> float:
    x = x.detach().float().flatten().cpu()
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, torch.tensor(float(q))).item())


def fixed_pairwise_skew(v: torch.Tensor) -> torch.Tensor:
    """
    Cheap high-dimensional analogue of 2D J*v = (-y, x).
    Applies 90-degree rotations on coordinate pairs:
      (x0, x1) -> (-x1, x0), (x2, x3) -> (-x3, x2), ...
    If dimension is odd, the last coordinate is set to zero.
    """
    out = torch.zeros_like(v)
    d_pair = (v.shape[1] // 2) * 2
    if d_pair > 0:
        even = v[:, 0:d_pair:2]
        odd = v[:, 1:d_pair:2]
        out[:, 0:d_pair:2] = -odd
        out[:, 1:d_pair:2] = even
    return out


# -----------------------------
# MNIST loader
# -----------------------------

def _open_idx(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def _find_file(root: Path, names: Tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing MNIST file in {root}. Tried: {names}")


def read_idx_images(path: Path) -> torch.Tensor:
    with _open_idx(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image idx file {path}: magic={magic}")
        data = f.read(rows * cols * n)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).float().view(n, rows * cols) / 255.0


def read_idx_labels(path: Path) -> torch.Tensor:
    with _open_idx(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label idx file {path}: magic={magic}")
        data = f.read(n)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def load_mnist_raw(root: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_path = Path(root)
    train_images = _find_file(root_path, ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"))
    train_labels = _find_file(root_path, ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"))
    test_images = _find_file(root_path, ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"))
    test_labels = _find_file(root_path, ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"))
    print(f"Using MNIST raw dir: {root_path}")
    return read_idx_images(train_images), read_idx_labels(train_labels), read_idx_images(test_images), read_idx_labels(test_labels)


def normalize_to_margin_field(x: torch.Tensor) -> torch.Tensor:
    return x - 0.5


def add_extra_dim(x: torch.Tensor, extra_dims: int) -> torch.Tensor:
    extra_dims = int(extra_dims)
    if extra_dims <= 0:
        return x
    return torch.cat([x, x.new_zeros((x.shape[0], extra_dims))], dim=1)


def make_loaders(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int):
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# -----------------------------
# 2.0 anchor initialization
# -----------------------------

def sample_data_anchors(x: torch.Tensor, k_points: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    n = x.size(0)
    if int(k_points) <= n:
        idx = torch.randperm(n, generator=g)[: int(k_points)]
    else:
        idx = torch.randint(0, n, (int(k_points),), generator=g)
    return x[idx].float().clone()


@torch.no_grad()
def hard_kmeans_anchors(
    x: torch.Tensor,
    k_points: int,
    *,
    seed: int,
    device: torch.device,
    iters: int = 25,
    batch_size: int = 2048,
) -> torch.Tensor:
    centers = sample_data_anchors(x, k_points, seed).to(device)
    x_cpu = x.float().cpu()
    for it in range(int(iters)):
        sums = torch.zeros(k_points, x.size(1), device=device)
        counts = torch.zeros(k_points, device=device)
        total_inertia = 0.0
        for start in range(0, x_cpu.size(0), int(batch_size)):
            xb = x_cpu[start:start + int(batch_size)].to(device)
            d2 = torch.cdist(xb, centers).square()
            vals, idx = d2.min(dim=1)
            total_inertia += float(vals.sum().item())
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        if (~nonempty).any():
            refill = sample_data_anchors(x_cpu, int((~nonempty).sum().item()), seed + 10000 + it).to(device)
            centers[~nonempty] = refill
        if it in {0, int(iters) - 1}:
            print(f"  kmeans iter {it+1:02d}/{iters}: inertia={total_inertia / x_cpu.size(0):.4f}, empty={(~nonempty).sum().item()}")
    return centers.detach().cpu()


def anchor_knn_sigma(anchors: torch.Tensor, knn: int = 8, scale: float = 1.0) -> Tuple[torch.Tensor, float]:
    with torch.no_grad():
        k = anchors.size(0)
        kk = max(1, min(int(knn), k - 1))
        d = torch.cdist(anchors.float(), anchors.float())
        d.fill_diagonal_(float("inf"))
        kth = torch.topk(d, k=kk, largest=False, dim=1).values[:, -1]
        per = (kth * float(scale)).clamp_min(1e-3)
        shared = float(torch.median(per).item())
    return per, shared


@torch.no_grad()
def build_anchors_and_sigmas(
    x_train: torch.Tensor,
    *,
    depth: int,
    k_points: int,
    seed: int,
    device: torch.device,
    kmeans_iters: int,
    sigma_knn: int,
    sigma_scale: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    anchors_by_layer: List[torch.Tensor] = []
    sigmas_by_layer: List[torch.Tensor] = []
    for li in range(int(depth)):
        layer_seed = int(seed) + 1000 * (li + 1)
        print(f"  init layer {li+1}/{depth}: kmeans, seed={layer_seed}")
        anchors = hard_kmeans_anchors(
            x_train,
            int(k_points),
            seed=layer_seed,
            device=device,
            iters=int(kmeans_iters),
        )
        per_sigma, shared_sigma = anchor_knn_sigma(anchors, knn=int(sigma_knn), scale=float(sigma_scale))
        print(
            f"    anchors mean/std={anchors.mean().item():.4f}/{anchors.std().item():.4f}, "
            f"sigma q50={tensor_quantile(per_sigma,0.50):.4f}, "
            f"q05/q95={tensor_quantile(per_sigma,0.05):.4f}/{tensor_quantile(per_sigma,0.95):.4f}"
        )
        anchors_by_layer.append(anchors)
        sigmas_by_layer.append(torch.tensor([shared_sigma], dtype=torch.float32))
    return anchors_by_layer, sigmas_by_layer


# -----------------------------
# 2.0 Gaussian terrain model + local spin activation
# -----------------------------

@dataclass
class ModelSpec:
    eta: float = 1.0
    zero_mean_c: bool = True
    trainable_anchors: bool = True


class GaussianLocalSpinLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        k_points: int,
        anchors: torch.Tensor,
        sigma_init: torch.Tensor,
        spec: ModelSpec,
        use_activation: bool,
    ):
        super().__init__()
        self.dim = int(dim)
        self.base_dim = int(dim) - 1
        if self.base_dim < 1:
            raise ValueError("local spin activation needs dim >= 2 and one extra height dimension.")
        self.k_points = int(k_points)
        self.spec = spec
        self.use_activation = bool(use_activation)
        self.force_denom = math.sqrt(float(k_points))

        a_init = anchors.float().contiguous()
        if bool(spec.trainable_anchors):
            self.a = nn.Parameter(a_init.clone())
        else:
            self.register_buffer("a", a_init)

        shared = float(torch.median(sigma_init.float()).item()) if sigma_init.numel() > 1 else float(sigma_init.item())
        self.log_sigma = nn.Parameter(torch.tensor([math.log(max(shared, 1e-3))], dtype=torch.float32))
        self.c = nn.Parameter(torch.randn(k_points, dtype=torch.float32) * C_INIT_STD)

        # Independent activation field. Same initial anchor positions/sigma as the transport field,
        # but trained separately after initialization.
        if self.use_activation:
            self.act_a = nn.Parameter(a_init.clone())
            self.act_log_sigma = nn.Parameter(torch.tensor([math.log(max(shared, 1e-3))], dtype=torch.float32))
            self.act_c = nn.Parameter(torch.randn(k_points, dtype=torch.float32) * ACT_C_INIT_STD)

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().expand(self.k_points)

    def act_sigma(self) -> torch.Tensor:
        return self.act_log_sigma.exp().expand(self.k_points)

    def effective_c(self) -> torch.Tensor:
        c = self.c
        if self.spec.zero_mean_c:
            c = c - c.mean()
        return c

    def effective_act_c(self) -> torch.Tensor:
        c = self.act_c
        if self.spec.zero_mean_c:
            c = c - c.mean()
        return c

    @staticmethod
    def _dist2(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return (
            z.square().sum(dim=1, keepdim=True)
            + a.square().sum(dim=1).view(1, -1)
            - 2.0 * z @ a.t()
        ).clamp_min(0.0)

    def transport_kernel_terms(self, z: torch.Tensor):
        a = self.a.to(dtype=z.dtype, device=z.device)
        dist2 = self._dist2(z, a)
        sigma = self.sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        kernel = torch.exp(-dist2 / (2.0 * sigma2))
        c_vec = self.effective_c().to(dtype=z.dtype, device=z.device).view(1, -1)
        weighted = kernel * c_vec / sigma2
        return weighted

    def force_from_weighted(self, z: torch.Tensor, weighted: torch.Tensor) -> torch.Tensor:
        a = self.a.to(dtype=z.dtype, device=z.device)
        wa = weighted @ a
        wsum = weighted.sum(dim=1, keepdim=True)
        return float(self.spec.eta) * (wa - wsum * z) / self.force_denom

    def force(self, z: torch.Tensor) -> torch.Tensor:
        return self.force_from_weighted(z, self.transport_kernel_terms(z))

    def _act_field_response_center(self, z: torch.Tensor) -> torch.Tensor:
        a = self.act_a.to(dtype=z.dtype, device=z.device)
        dist2 = self._dist2(z, a)
        sigma = self.act_sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        c_vec = self.effective_act_c().to(dtype=z.dtype, device=z.device).view(1, -1)
        score = c_vec - dist2 / (2.0 * sigma2)
        alpha = torch.softmax(score, dim=1)
        return alpha @ a

    def _act_field_potential_and_gradient(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return scalar potential theta(z) and gradient-like direction material.
        theta is used directly as the local rotation angle.
        gradient is then converted into a tangent by a fixed skew operator.
        """
        a = self.act_a.to(dtype=z.dtype, device=z.device)
        dist2 = self._dist2(z, a)
        sigma = self.act_sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        kernel = torch.exp(-dist2 / (2.0 * sigma2))
        c_vec = self.effective_act_c().to(dtype=z.dtype, device=z.device).view(1, -1)

        theta = (kernel * c_vec).sum(dim=1, keepdim=True)

        # ∇ sum c_j K_j(x) = sum c_j K_j(x) * (a_j - x) / sigma_j^2
        weighted = kernel * c_vec / sigma2
        wa = weighted @ a
        wsum = weighted.sum(dim=1, keepdim=True)
        grad = wa - wsum * z
        return theta, grad

    def _rotate_displacement_in_local_plane(self, d: torch.Tensor, u: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rotate displacement d in the 2D plane spanned by:
          - local base-space direction u
          - the final extra height dimension
        """
        base = d[:, :self.base_dim]
        h = d[:, self.base_dim:self.base_dim + 1]
        p = (base * u).sum(dim=1, keepdim=True)
        perp = base - p * u

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        p_new = p * cos_t - h * sin_t
        h_new = p * sin_t + h * cos_t
        base_new = perp + p_new * u
        return torch.cat([base_new, h_new], dim=1)

    def activate(self, z_before: torch.Tensor, z_mid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if not self.use_activation:
            return z_mid, torch.zeros_like(z_mid), {"theta_abs": 0.0, "u_norm": 0.0}

        # Local center is read from the independent activation field.
        r = self._act_field_response_center(z_before)

        # Field state at the local center controls angle and local tangent direction.
        theta, grad = self._act_field_potential_and_gradient(r)
        tangent_base = fixed_pairwise_skew(grad[:, :self.base_dim])
        # Fallback: if skew tangent is near zero, use gradient base direction.
        grad_base = grad[:, :self.base_dim]
        t_norm = tangent_base.norm(dim=1, keepdim=True)
        g_norm = grad_base.norm(dim=1, keepdim=True)
        tangent_base = torch.where(t_norm > 1e-8, tangent_base, grad_base)
        u = tangent_base / tangent_base.norm(dim=1, keepdim=True).clamp_min(1e-8)

        d = z_mid - r
        d_rot = self._rotate_displacement_in_local_plane(d, u, theta)
        d_act = F.relu(d_rot)
        d_back = self._rotate_displacement_in_local_plane(d_act, u, -theta)
        z_after = r + d_back

        debug = {
            "theta_abs": float(theta.detach().abs().mean().item()),
            "u_norm": float(u.detach().norm(dim=1).mean().item()),
        }
        return z_after, z_after - z_mid, debug


class GaussianAATLocalSpinModel(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        k_points: int,
        anchors_by_layer: List[torch.Tensor],
        sigmas_by_layer: List[torch.Tensor],
        spec: ModelSpec,
    ):
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.k_points = int(k_points)
        self.spec = spec
        last_idx = int(depth) - 1
        self.layers = nn.ModuleList([
            GaussianLocalSpinLayer(
                dim=dim,
                k_points=k_points,
                anchors=anchors,
                sigma_init=sigma,
                spec=spec,
                use_activation=(i != last_idx),
            )
            for i, (anchors, sigma) in enumerate(zip(anchors_by_layer, sigmas_by_layer))
        ])
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor):
        z = x
        metrics = []
        for layer in self.layers:
            z_before = z
            f = layer.force(z)
            z_mid = z + f
            z_after, act_delta, debug = layer.activate(z_before, z_mid)
            metrics.append((
                float(f.detach().abs().mean().item()),
                float(f.detach().norm(dim=1).mean().item()),
                float(act_delta.detach().abs().mean().item()),
                float((z_after - z_before).detach().norm(dim=1).mean().item()),
                debug["theta_abs"],
            ))
            z = z_after
        return self.head(z), metrics


# -----------------------------
# Train / eval
# -----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def split_params(model: nn.Module):
    sigma_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "log_sigma" in name or "act_log_sigma" in name:
            sigma_params.append(p)
        else:
            base_params.append(p)
    return base_params, sigma_params


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    print_every: int,
) -> Dict[str, float]:
    base_params, sigma_params = split_params(model)
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": LR, "weight_decay": 0.0},
        {"params": sigma_params, "lr": SIGMA_LR, "weight_decay": 0.0},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = -1

    print("metric format per layer: force_abs / force_norm / act_abs / delta_norm / theta_abs")
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        metric_accum = None
        metric_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits, metrics = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += bs

            if metric_accum is None:
                metric_accum = [[0.0 for _ in range(5)] for _ in metrics]
            for li, m in enumerate(metrics):
                for mi in range(5):
                    metric_accum[li][mi] += m[mi]
            metric_count += 1

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch

        if epoch % int(print_every) == 0 or epoch == int(epochs) - 1:
            parts = []
            if metric_accum is not None and metric_count > 0:
                for li, vals in enumerate(metric_accum):
                    avg = [v / metric_count for v in vals]
                    parts.append(f"L{li+1}:{avg[0]:.4g}/{avg[1]:.4g}/{avg[2]:.4g}/{avg[3]:.4g}/{avg[4]:.4g}")
            print(
                f"Epoch {epoch:03d}/{int(epochs)-1:03d} | "
                f"train loss {total_loss/max(total,1):.4f} acc {total_correct/max(total,1):.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"best {best_acc:.4f}@{best_epoch} | metrics " + "; ".join(parts)
            )

    return {"best_acc": float(best_acc), "best_epoch": int(best_epoch), "best_loss": float(best_loss)}


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-root", type=str, default=MNIST_RAW_DIR)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--extra-dims", type=int, default=1, help="Use exactly 1 for local spin height axis.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--sigma-knn", type=int, default=8)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    args = parser.parse_args()

    if int(args.extra_dims) != 1:
        raise ValueError("This experiment is intentionally minimal: use --extra-dims 1.")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Experiment: AAT 2.0 transport unchanged; activation is local independent spin field.")
    print("Model: local_spin only. Last layer has no activation.")
    print(f"Config: K={args.K}, depth={args.depth}, extra_dims={args.extra_dims}, epochs={args.epochs}")

    x_train_raw, y_train_all, x_test_raw, y_test_all = load_mnist_raw(args.mnist_root)
    x_train = normalize_to_margin_field(x_train_raw[:N_TRAIN]).contiguous()
    y_train = y_train_all[:N_TRAIN].contiguous()
    x_val = normalize_to_margin_field(x_test_raw[:N_VAL]).contiguous()
    y_val = y_test_all[:N_VAL].contiguous()

    x_train = add_extra_dim(x_train, int(args.extra_dims)).contiguous()
    x_val = add_extra_dim(x_val, int(args.extra_dims)).contiguous()
    dim = int(x_train.shape[1])

    print(f"Loaded train {tuple(x_train.shape)}, val {tuple(x_val.shape)}")
    print(f"x_train range [{x_train.min().item():.3f}, {x_train.max().item():.3f}], mean/std {x_train.mean().item():.4f}/{x_train.std().item():.4f}")

    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, int(args.batch_size))

    print("Initializing shared k-means anchors once...")
    anchors_by_layer, sigmas_by_layer = build_anchors_and_sigmas(
        x_train,
        depth=int(args.depth),
        k_points=int(args.K),
        seed=SEED,
        device=device,
        kmeans_iters=int(args.kmeans_iters),
        sigma_knn=int(args.sigma_knn),
        sigma_scale=float(args.sigma_scale),
    )

    print("\n" + "=" * 88)
    print("local_spin")
    print("=" * 88)
    set_seed(SEED)
    spec = ModelSpec(eta=1.0, zero_mean_c=True, trainable_anchors=True)
    model = GaussianAATLocalSpinModel(
        dim=dim,
        depth=int(args.depth),
        k_points=int(args.K),
        anchors_by_layer=[a.clone() for a in anchors_by_layer],
        sigmas_by_layer=[s.clone() for s in sigmas_by_layer],
        spec=spec,
    ).to(device)
    print(f"params={count_params(model):,}")
    t0 = time.time()
    info = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=int(args.epochs),
        print_every=int(args.print_every),
    )
    dt = time.time() - t0

    with torch.no_grad():
        active_layers = model.layers[:-1]
        act_sigma = [float(layer.act_log_sigma.exp().detach().cpu().item()) for layer in active_layers]
        act_c_std = [float(layer.act_c.detach().float().std().cpu().item()) for layer in active_layers]
    print(f"act_sigma by active layer: {[round(v, 6) for v in act_sigma]}")
    print(f"act_c_std by active layer: {[round(v, 6) for v in act_c_std]}")
    print(f"DONE local_spin: best_acc={info['best_acc']:.4f}@{info['best_epoch']}, best_loss={info['best_loss']:.4f}, time={dt:.1f}s")


if __name__ == "__main__":
    main()
