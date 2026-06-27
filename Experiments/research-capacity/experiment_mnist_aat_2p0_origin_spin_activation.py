# -*- coding: utf-8 -*-
r"""
Minimal MNIST experiment for AAT 2.0 Gaussian terrain transport.

Only compares two models, one config:
  1) old_response: original 2.0 response-center activation.
  2) origin_spin:  same 2.0 transport, but activation is replaced by an origin one-Gaussian spin activation.

No scan, no output files. Everything prints to console.

Run from:
  C:\Projets\AATField\Experiments\research-capacity

Example:
  python experiment_mnist_aat_2p0_origin_spin_activation.py
  python experiment_mnist_aat_2p0_origin_spin_activation.py --K 16 --depth 3 --epochs 80
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
    dim: int,
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
# 2.0 Gaussian terrain model + activation variants
# -----------------------------

@dataclass
class ModelSpec:
    activation: str                 # "old_response" or "origin_spin"
    eta: float = 1.0
    zero_mean_c: bool = True
    trainable_anchors: bool = True


class GaussianFieldLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        k_points: int,
        anchors: torch.Tensor,
        sigma_init: torch.Tensor,
        spec: ModelSpec,
        enable_spin: bool,
    ):
        super().__init__()
        self.dim = int(dim)
        self.k_points = int(k_points)
        self.spec = spec
        self.force_denom = math.sqrt(float(k_points))

        a_init = anchors.float().contiguous()
        if bool(spec.trainable_anchors):
            self.a = nn.Parameter(a_init.clone())
        else:
            self.register_buffer("a", a_init)

        shared = float(torch.median(sigma_init.float()).item()) if sigma_init.numel() > 1 else float(sigma_init.item())
        self.log_sigma = nn.Parameter(torch.tensor([math.log(max(shared, 1e-3))], dtype=torch.float32))
        self.c = nn.Parameter(torch.randn(k_points, dtype=torch.float32) * C_INIT_STD)

        self.enable_spin = bool(enable_spin)
        if self.enable_spin:
            self.spin_charge = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
            self.log_spin_sigma = nn.Parameter(torch.tensor(math.log(max(shared, 1e-3)), dtype=torch.float32))

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().expand(self.k_points)

    def effective_c(self) -> torch.Tensor:
        c = self.c
        if self.spec.zero_mean_c:
            c = c - c.mean()
        return c

    def _dist2(self, z: torch.Tensor) -> torch.Tensor:
        a = self.a.to(dtype=z.dtype, device=z.device)
        return (
            z.square().sum(dim=1, keepdim=True)
            + a.square().sum(dim=1).view(1, -1)
            - 2.0 * z @ a.t()
        ).clamp_min(0.0)

    def kernel_terms(self, z: torch.Tensor):
        dist2 = self._dist2(z)
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
        return self.force_from_weighted(z, self.kernel_terms(z))

    def response_center(self, z: torch.Tensor) -> torch.Tensor:
        a = self.a.to(dtype=z.dtype, device=z.device)
        dist2 = self._dist2(z)
        sigma = self.sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        c_vec = self.effective_c().to(dtype=z.dtype, device=z.device).view(1, -1)
        score = c_vec - dist2 / (2.0 * sigma2)
        alpha = torch.softmax(score, dim=1)
        return alpha @ a

    def _origin_spin_angle(self, z: torch.Tensor) -> torch.Tensor:
        sigma = self.log_spin_sigma.exp().to(dtype=z.dtype, device=z.device).clamp_min(1e-6)
        dist2 = z.square().sum(dim=1, keepdim=True)
        kernel = torch.exp(-dist2 / (2.0 * sigma.square()))
        return self.spin_charge.to(dtype=z.dtype, device=z.device).view(1, 1) * kernel

    @staticmethod
    def _radial_height_rotate(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # Rotate the 2D plane: (radius of all dimensions except last, last coordinate).
        # This uses the last appended extra dimension as the height axis.
        base = z[:, :-1]
        h = z[:, -1:]
        r = base.norm(dim=1, keepdim=True)
        unit = base / r.clamp_min(1e-8)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        r_new = r * cos_t - h * sin_t
        h_new = r * sin_t + h * cos_t
        base_new = unit * r_new
        base_new = torch.where((r > 1e-8).expand_as(base_new), base_new, torch.zeros_like(base_new))
        return torch.cat([base_new, h_new], dim=1)

    def activate(self, z_before: torch.Tensor, z_mid: torch.Tensor, *, is_last: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_last:
            return z_mid, torch.zeros_like(z_mid)

        if self.spec.activation == "old_response":
            r = self.response_center(z_before)
            z_after = r + F.relu(z_mid - r)
        elif self.spec.activation == "origin_spin":
            if not self.enable_spin:
                raise RuntimeError("origin_spin activation requested but spin parameters are disabled.")
            theta = self._origin_spin_angle(z_mid)
            z_rot = self._radial_height_rotate(z_mid, theta)
            z_act = F.relu(z_rot)
            z_after = self._radial_height_rotate(z_act, -theta)
        else:
            raise ValueError(f"Unknown activation: {self.spec.activation}")

        return z_after, z_after - z_mid


class GaussianAATModel(nn.Module):
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
        enable_spin = spec.activation == "origin_spin"
        self.layers = nn.ModuleList([
            GaussianFieldLayer(
                dim=dim,
                k_points=k_points,
                anchors=anchors,
                sigma_init=sigma,
                spec=spec,
                enable_spin=enable_spin,
            )
            for anchors, sigma in zip(anchors_by_layer, sigmas_by_layer)
        ])
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor):
        z = x
        metrics = []
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            z_before = z
            f = layer.force(z)
            z_mid = z + f
            z_after, act_delta = layer.activate(z_before, z_mid, is_last=(i == last_idx))
            metrics.append((
                float(f.detach().abs().mean().item()),
                float(f.detach().norm(dim=1).mean().item()),
                float(act_delta.detach().abs().mean().item()),
                float((z_after - z_before).detach().norm(dim=1).mean().item()),
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
        if "log_sigma" in name or "log_spin_sigma" in name:
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

    print("metric format per layer: force_abs / force_norm / act_abs / delta_norm")
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
                metric_accum = [[0.0, 0.0, 0.0, 0.0] for _ in metrics]
            for li, m in enumerate(metrics):
                for mi in range(4):
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
                    parts.append(f"L{li+1}:{avg[0]:.4g}/{avg[1]:.4g}/{avg[2]:.4g}/{avg[3]:.4g}")
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
    parser.add_argument("--extra-dims", type=int, default=1, help="Use 1 for origin_spin radial-height rotation.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--sigma-knn", type=int, default=8)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--only", type=str, default="both", choices=["both", "old_response", "origin_spin"])
    args = parser.parse_args()

    if int(args.extra_dims) < 1:
        raise ValueError("origin_spin needs at least one extra dimension; use --extra-dims 1.")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Experiment: AAT 2.0 Gaussian transport unchanged; only activation is changed.")
    print("Models: old_response vs origin_spin. Last layer has no activation in both models.")
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
        dim=dim,
        depth=int(args.depth),
        k_points=int(args.K),
        seed=SEED,
        device=device,
        kmeans_iters=int(args.kmeans_iters),
        sigma_knn=int(args.sigma_knn),
        sigma_scale=float(args.sigma_scale),
    )

    activations = ["old_response", "origin_spin"] if args.only == "both" else [args.only]
    results = []
    for ai, act in enumerate(activations):
        print("\n" + "=" * 88)
        print(act)
        print("=" * 88)
        set_seed(SEED + 1000 * ai)
        spec = ModelSpec(activation=act, eta=1.0, zero_mean_c=True, trainable_anchors=True)
        model = GaussianAATModel(
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
        if act == "origin_spin":
            with torch.no_grad():
                charges = [float(layer.spin_charge.detach().cpu().item()) for layer in model.layers[:-1]]
                sigmas = [float(layer.log_spin_sigma.exp().detach().cpu().item()) for layer in model.layers[:-1]]
            print(f"spin_charge by active layer: {[round(v, 6) for v in charges]}")
            print(f"spin_sigma  by active layer: {[round(v, 6) for v in sigmas]}")
        print(f"DONE {act}: best_acc={info['best_acc']:.4f}@{info['best_epoch']}, time={dt:.1f}s")
        results.append((act, info, count_params(model)))

    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    for act, info, params in results:
        print(f"{act:<14} | params {params:>9,} | best_val_acc {info['best_acc']:.4f}@{info['best_epoch']} | best_loss {info['best_loss']:.4f}")


if __name__ == "__main__":
    main()
