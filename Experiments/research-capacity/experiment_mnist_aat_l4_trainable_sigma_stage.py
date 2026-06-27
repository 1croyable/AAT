# -*- coding: utf-8 -*-
r"""
experiment_mnist_aat_l4_trainable_sigma_stage.py

AATField 2.0 MNIST first test.

Core structure:
  - Continuous integral Gaussian terrain layer
  - One shared sigma per layer, initialized by DoG+floor and then trained after warmup
  - Staged schedule: first train c/head with staged trainable sigma, then unfreeze sigma
  - Hidden layers use local activation
  - Last layer is plain transport
  - Final linear classifier

This script intentionally avoids torchvision and reads MNIST IDX files directly from:
  C:\Projets\AATField\Experiments\data\MNIST\raw

Run:
  python experiment_mnist_aat_l4_trainable_sigma_stage.py
"""

import gzip
import math
import os
import random
import struct
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Basic config
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MNIST_RAW_DIR = r"C:\Projets\AATField\Experiments\data\MNIST\raw"

# First MNIST test: use a moderate subset so the experiment finishes quickly.
# Set N_TRAIN = 60000 if you want the full training set later.
N_TRAIN = 10000
N_VAL = 10000

BATCH_SIZE = 256
EPOCHS = 80
LR = 2e-3
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 5
USE_AMP = True

# Staged sigma training.
# Warmup lets c/head become task-aligned first; then sigma receives meaningful gradients.
SIGMA_UNFREEZE_EPOCH = 10
SIGMA_LR = 3e-4
SIGMA_REG = 1e-4
SIGMA_MIN = 0.50
SIGMA_MAX = 20.0

BASE_DIM = 28 * 28
DEPTH = 4
EXTRA_DIMS_LIST = [1]      # +1 was slightly better in the fixed-sigma MNIST test

N_INTEGRAL_POINTS = 512
C_INIT_STD = 0.04
EXTRA_ANCHOR_STD = 0.35

# Sigma initializer settings.
INIT_SAMPLE_SIZE = 2048
SIGMA_GRID_SIZE = 120
DOG_KAPPA = 1.60
SIGMA_MAX_PAIR_MULT = 1.50
SIGMA_FLOOR_NN_MULT = 1.00
SIGMA_FLOOR_PAIR_FRAC = 0.05
PROFILE_TOP_FRAC = 0.70
MIN_PROFILE_LOG_SPAN = 0.55

# For MNIST high-dimensional vectors, using real data points as base anchors is
# much more reasonable than sampling uniformly in a 784D box.
ANCHOR_NOISE_STD = 0.01


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ============================================================
# MNIST IDX loader
# ============================================================


def _open_idx(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _find_file(root: Path, names: List[str]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of {names} under {root}")


def read_idx_images(path: Path) -> torch.Tensor:
    with _open_idx(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic {magic} in {path}")
        buf = f.read(rows * cols * n)
    x = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(n, rows * cols)
    return x.float() / 255.0


def read_idx_labels(path: Path) -> torch.Tensor:
    with _open_idx(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic {magic} in {path}")
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).long()


def load_mnist_raw(root: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_path = Path(root)
    if not root_path.exists():
        # Convenience fallback for running elsewhere.
        fallback = Path("data") / "MNIST" / "raw"
        if fallback.exists():
            root_path = fallback
        else:
            raise FileNotFoundError(
                f"MNIST raw directory not found: {root_path}\n"
                f"Expected files like train-images-idx3-ubyte(.gz) under this folder."
            )

    train_images = _find_file(root_path, ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"])
    train_labels = _find_file(root_path, ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"])
    test_images = _find_file(root_path, ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"])
    test_labels = _find_file(root_path, ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"])

    x_train = read_idx_images(train_images)
    y_train = read_idx_labels(train_labels)
    x_val = read_idx_images(test_images)
    y_val = read_idx_labels(test_labels)
    return x_train, y_train, x_val, y_val


# ============================================================
# Data preprocessing / lift
# ============================================================


def make_permutation(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randperm(dim, generator=g)


def expand_inputs(x: torch.Tensor, extra_dims: int, perm: torch.Tensor) -> torch.Tensor:
    if extra_dims > 0:
        pad = torch.zeros(x.size(0), extra_dims, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    return x[:, perm]


# ============================================================
# DoG visible-structure sigma initializer
# ============================================================


@torch.no_grad()
def dog_floor_sigma_initializer(x: torch.Tensor, y: torch.Tensor, depth: int) -> Tuple[List[float], dict]:
    """
    DoG / center-surround visible-structure scan.

    This version uses sample positions as eye centers:
        B_sigma(x_m) = sum_i [G_sigma(x_m-x_i)-G_{k sigma}(x_m-x_i)] s_i
        J(sigma) = mean_m ||B_sigma(x_m)||^2

    It avoids normalized Gaussian constants, which are numerically unsuitable in
    hundreds of dimensions, but keeps the center-surround idea.
    """
    n = min(INIT_SAMPLE_SIZE, x.size(0))
    x = x[:n].to(DEVICE).float().contiguous()
    y = y[:n].to(DEVICE).long().contiguous()

    num_classes = int(y.max().item()) + 1
    s = F.one_hot(y, num_classes=num_classes).float()
    s = s - s.mean(dim=0, keepdim=True)

    dist = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float("inf"))
    nn = dist.min(dim=1).values
    median_nn = float(nn.median().item())

    finite_dist = dist[torch.isfinite(dist)]
    median_pair = float(finite_dist.median().item())

    sigma_floor = max(SIGMA_FLOOR_NN_MULT * median_nn, SIGMA_FLOOR_PAIR_FRAC * median_pair, 1e-4)
    sigma_max = max(sigma_floor * 1.25, SIGMA_MAX_PAIR_MULT * median_pair)

    # restore diagonal distance for kernels
    dist.fill_diagonal_(0.0)
    dist2 = dist.square()

    sigma_grid = torch.exp(
        torch.linspace(math.log(sigma_floor), math.log(sigma_max), SIGMA_GRID_SIZE, device=DEVICE)
    )

    scores = []
    kappa = DOG_KAPPA
    for sigma in sigma_grid:
        g_center = torch.exp(-dist2 / (2.0 * sigma * sigma))
        g_surround = torch.exp(-dist2 / (2.0 * (kappa * sigma) * (kappa * sigma)))
        dog = g_center - g_surround
        # The diagonal is naturally zero because 1-1=0.
        response = dog @ s / float(n)
        score = response.square().sum(dim=1).mean()
        scores.append(score.detach())
    scores = torch.stack(scores)

    peak_idx = int(scores.argmax().item())
    peak_sigma = float(sigma_grid[peak_idx].item())
    peak_score = float(scores[peak_idx].item())

    if depth == 1:
        profile = [peak_sigma]
    else:
        threshold = PROFILE_TOP_FRAC * peak_score
        high = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
        if high.numel() >= depth:
            log_high = torch.log(sigma_grid[high])
            if float((log_high[-1] - log_high[0]).item()) < MIN_PROFILE_LOG_SPAN:
                mid = float(torch.log(sigma_grid[peak_idx]).item())
                lo = max(math.log(sigma_floor), mid - 0.5 * MIN_PROFILE_LOG_SPAN)
                hi = min(math.log(sigma_max), lo + MIN_PROFILE_LOG_SPAN)
                log_profile = torch.linspace(lo, hi, depth, device=DEVICE)
            else:
                q = torch.linspace(0.0, 1.0, depth, device=DEVICE)
                pos = q * (log_high.numel() - 1)
                idx0 = torch.floor(pos).long()
                idx1 = torch.clamp(idx0 + 1, max=log_high.numel() - 1)
                frac = pos - idx0.float()
                log_profile = log_high[idx0] * (1.0 - frac) + log_high[idx1] * frac
        else:
            mid = float(torch.log(sigma_grid[peak_idx]).item())
            lo = max(math.log(sigma_floor), mid - 0.5 * MIN_PROFILE_LOG_SPAN)
            hi = min(math.log(sigma_max), lo + MIN_PROFILE_LOG_SPAN)
            log_profile = torch.linspace(lo, hi, depth, device=DEVICE)
        profile = [float(v.exp().item()) for v in log_profile]
        profile.sort()

    meta = {
        "median_nn": median_nn,
        "median_pair": median_pair,
        "sigma_floor": sigma_floor,
        "sigma_max": sigma_max,
        "peak_sigma": peak_sigma,
        "peak_score": peak_score,
    }
    return profile, meta


# ============================================================
# AAT integral terrain model, efficient high-D implementation
# ============================================================


class IntegralTerrainLayer(nn.Module):
    def __init__(self, dim: int, sigma_init: float, anchor_source: torch.Tensor, extra_dims: int, perm: torch.Tensor):
        super().__init__()
        self.dim = int(dim)
        self.extra_dims = int(extra_dims)

        idx = torch.randint(0, anchor_source.size(0), (N_INTEGRAL_POINTS,))
        base = anchor_source[idx].clone().float()

        if extra_dims > 0:
            extra = torch.randn(N_INTEGRAL_POINTS, extra_dims) * EXTRA_ANCHOR_STD
            anchors = torch.cat([base, extra], dim=1)
        else:
            anchors = base

        anchors = anchors[:, perm]
        anchors = anchors + torch.randn_like(anchors) * ANCHOR_NOISE_STD

        self.register_buffer("a", anchors.contiguous())

        init_log_sigma = math.log(float(sigma_init))
        self.register_buffer("init_log_sigma", torch.tensor(init_log_sigma, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=torch.float32))
        # Sigma is frozen during the warmup phase and unfrozen later by train_one().
        self.log_sigma.requires_grad_(False)

        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(min=SIGMA_MIN, max=SIGMA_MAX)

    def _dist2(self, z: torch.Tensor) -> torch.Tensor:
        # [B,D] x [K,D] -> [B,K], without creating [B,K,D]
        return (
            z.square().sum(dim=1, keepdim=True)
            + self.a.square().sum(dim=1).view(1, -1)
            - 2.0 * z @ self.a.t()
        ).clamp_min(0.0)

    def force(self, z: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma().to(z.dtype)
        dist2 = self._dist2(z)
        k = torch.exp(-dist2 / (2.0 * sigma * sigma))
        w = k * self.c.to(z.dtype).view(1, -1)
        # mean_j c_j k_j (a_j - z) / sigma^2
        wa = w @ self.a.to(z.dtype)
        wsum = w.sum(dim=1, keepdim=True)
        move = (wa - wsum * z) / (float(N_INTEGRAL_POINTS) * sigma * sigma)
        return move

    def response_center(self, z: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma().to(z.dtype)
        dist2 = self._dist2(z)
        score = self.c.to(z.dtype).view(1, -1) - dist2 / (2.0 * sigma * sigma)
        alpha = torch.softmax(score, dim=1)
        return alpha @ self.a.to(z.dtype)


class AATMNIST(nn.Module):
    def __init__(self, base_train_x: torch.Tensor, extra_dims: int, sigmas: List[float], perm: torch.Tensor):
        super().__init__()
        self.extra_dims = int(extra_dims)
        self.dim = BASE_DIM + self.extra_dims
        self.register_buffer("perm", perm.clone().long())

        self.layers = nn.ModuleList([
            IntegralTerrainLayer(
                dim=self.dim,
                sigma_init=sigmas[i],
                anchor_source=base_train_x,
                extra_dims=self.extra_dims,
                perm=self.perm,
            )
            for i in range(len(sigmas))
        ])
        self.head = nn.Linear(self.dim, 10)

    def local_activation(self, z: torch.Tensor, z_mid: torch.Tensor, layer: IntegralTerrainLayer) -> torch.Tensor:
        r = layer.response_center(z)
        return r + F.relu(z_mid - r)

    def forward(self, x: torch.Tensor):
        z = x
        if self.extra_dims > 0:
            pad = torch.zeros(z.size(0), self.extra_dims, dtype=z.dtype, device=z.device)
            z = torch.cat([z, pad], dim=1)
        z = z[:, self.perm]

        forces = []
        for i, layer in enumerate(self.layers):
            f = layer.force(z)
            z_mid = z + f
            is_last = (i == len(self.layers) - 1)
            if not is_last:
                z = self.local_activation(z, z_mid, layer)
            else:
                z = z_mid
            forces.append(float(f.detach().abs().mean().item()))
        return self.head(z), forces

    def sigma_list(self) -> List[float]:
        return [float(layer.sigma().detach().cpu().item()) for layer in self.layers]


# ============================================================
# Training helpers
# ============================================================


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss_acc(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * yb.numel()
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one(extra_dims: int, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
    dim = BASE_DIM + extra_dims
    perm = make_permutation(dim, SEED + 1000 + extra_dims)

    # Sigma initializer sees the actual input manifold after zero-padding and permutation.
    x_init = expand_inputs(x_train[:INIT_SAMPLE_SIZE], extra_dims, perm)
    sigmas, meta = dog_floor_sigma_initializer(x_init, y_train[:INIT_SAMPLE_SIZE], DEPTH)

    model_name = f"AAT_L{DEPTH}_MNIST_D{dim}_extra{extra_dims}_dog_floor_trainable_sigma_stage"
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print(
        f"dim {dim} | extra_dims {extra_dims} | layers {DEPTH} | "
        f"integral points/layer {N_INTEGRAL_POINTS} | staged trainable sigmas {[round(s, 4) for s in sigmas]}"
    )
    print(
        "initializer: "
        f"median_nn {meta['median_nn']:.4f} | median_pair {meta['median_pair']:.4f} | "
        f"sigma_floor {meta['sigma_floor']:.4f} | sigma_max {meta['sigma_max']:.4f} | "
        f"peak_sigma {meta['peak_sigma']:.4f} | peak_score {meta['peak_score']:.6e}"
    )

    model = AATMNIST(x_train, extra_dims=extra_dims, sigmas=sigmas, perm=perm).to(DEVICE)
    print(f"params: {count_parameters(model)}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    sigma_params = [p for n, p in model.named_parameters() if "log_sigma" in n]
    base_params = [p for n, p in model.named_parameters() if "log_sigma" not in n]
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": sigma_params, "lr": SIGMA_LR, "weight_decay": 0.0},
        ]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    best_acc = 0.0
    best_epoch = 0
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        if epoch == SIGMA_UNFREEZE_EPOCH:
            for p in sigma_params:
                p.requires_grad_(True)
            print(f"-- Unfreeze sigma at epoch {epoch}; sigma lr={SIGMA_LR}, sigma_reg={SIGMA_REG} --")

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        last_forces = None

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits, forces = model(xb)
                ce_loss = F.cross_entropy(logits, yb)
                if epoch >= SIGMA_UNFREEZE_EPOCH:
                    sigma_reg = sum(
                        (layer.log_sigma - layer.init_log_sigma.to(layer.log_sigma.device)).square()
                        for layer in model.layers
                    )
                    loss = ce_loss + SIGMA_REG * sigma_reg
                else:
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item()) * yb.numel()
            pred = logits.detach().argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            last_forces = forces

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            val_loss, val_acc = evaluate_loss_acc(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_loss = val_loss
            force_text = ""
            if last_forces is not None:
                force_text = " | force " + ", ".join(f"L{i+1}:{v:.4f}" for i, v in enumerate(last_forces))
            sigma_text = ", ".join(f"{s:.3f}" for s in model.sigma_list())
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"sigma [{sigma_text}] | best {best_acc:.4f}@{best_epoch}"
                f"{force_text}"
            )

    return {
        "name": model_name,
        "params": count_parameters(model),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "sigmas": model.sigma_list(),
    }


# ============================================================
# Main
# ============================================================


def main():
    print(f"Device: {DEVICE}")
    print(f"MNIST raw dir: {MNIST_RAW_DIR}")
    print("AAT: L4 MNIST, DoG+floor init, staged trainable sigma, hidden local activation, last plain")
    print(f"N_TRAIN={N_TRAIN}, N_VAL={N_VAL}, epochs={EPOCHS}, batch={BATCH_SIZE}")

    x_train_all, y_train_all, x_val_all, y_val_all = load_mnist_raw(MNIST_RAW_DIR)
    x_train = x_train_all[:N_TRAIN].contiguous()
    y_train = y_train_all[:N_TRAIN].contiguous()
    x_val = x_val_all[:N_VAL].contiguous()
    y_val = y_val_all[:N_VAL].contiguous()

    print(f"Loaded train {x_train.shape}, val {x_val.shape}")

    results = []
    for extra_dims in EXTRA_DIMS_LIST:
        results.append(train_one(extra_dims, x_train, y_train, x_val, y_val))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        print(
            f"{r['name']:<60} | params {r['params']:7d} | "
            f"best_val_acc {r['best_acc']:.4f}@{r['best_epoch']} | "
            f"val_loss {r['best_loss']:.4f} | sigmas {[round(s, 4) for s in r['sigmas']]}"
        )


if __name__ == "__main__":
    main()
