# -*- coding: utf-8 -*-
r"""
experiment_mnist_aat_deep_depth_scan.py

MNIST deeper depth scan for AATField 2.0.

Purpose:
  - Check whether the weak MNIST result is mainly a depth problem.
  - Continue the previous depth scan with deeper AAT depths L8/L12/L16/L24.
  - Use +1 extra dimension, because it was slightly better than +2 in the first MNIST test.
  - Use the learned MNIST sigma profile as a warm start, with sigma trainable from epoch 0.
  - Print both force magnitude and actual layer displacement, because hidden local activation can change z
    even when the raw force is numerically small.

Data path is fixed to:
  C:/Projets/AATField/Experiments/data/MNIST/raw

Run:
  python experiment_mnist_aat_deep_depth_scan.py
"""

import gzip
import math
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

N_TRAIN = 10000
N_VAL = 10000

BATCH_SIZE = 256
EPOCHS = 80
LR = 2e-3
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 5
USE_AMP = True

# Deeper depth scan. Previous scan showed L8 > L4 > L2 > Linear, so continue upward.
DEPTH_LIST = [8, 12, 16, 24]
RUN_LINEAR_BASELINE = False
EXTRA_DIMS = 1
BASE_DIM = 28 * 28

# Integral terrain settings.
N_INTEGRAL_POINTS = 512
C_INIT_STD = 0.04
EXTRA_ANCHOR_STD = 0.35
ANCHOR_NOISE_STD = 0.01

# Sigma training.
SIGMA_LR = 5e-4
SIGMA_REG = 0.0
SIGMA_MIN = 0.30
SIGMA_MAX = 20.0

# Learned from the previous best L4 MNIST run.
# Previous best: sigmas around [1.7264, 1.6826, 1.7449, 6.6205], best_val_acc ~= 0.9358.
L4_LEARNED_PROFILE = [1.7264, 1.6826, 1.7449, 6.6205]


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
        fallback = Path("data") / "MNIST" / "raw"
        if fallback.exists():
            root_path = fallback
        else:
            raise FileNotFoundError(
                f"MNIST raw directory not found: {root_path}\n"
                f"Expected IDX files under this folder."
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
# Helpers
# ============================================================


def make_permutation(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randperm(dim, generator=g)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_depth_sigma_profile(depth: int) -> List[float]:
    """
    Warm-start profile for different depths.

    Interpretation:
      - Hidden layers use the useful MNIST fine scale found by the previous L4 run, around 1.7~2.0.
      - The last plain layer keeps a coarser global scale, around 6.6.

    This is intentionally not a final theory. It is a diagnostic warm start to test whether depth helps.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth == 1:
        # Single plain transport layer: use a useful local scale rather than the old very-large DoG scale.
        return [1.75]
    if depth == 2:
        return [1.72, 6.62]
    if depth == 3:
        return [1.72, 1.82, 6.62]
    if depth == 4:
        return list(L4_LEARNED_PROFILE)

    # For deeper models, spread hidden layers mildly around the learned fine-scale region.
    hidden_count = depth - 1
    hidden = torch.linspace(1.60, 2.10, hidden_count).tolist()
    return [float(v) for v in hidden] + [6.62]


# ============================================================
# Models
# ============================================================


class LinearMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(BASE_DIM, 10)

    def forward(self, x: torch.Tensor):
        return self.head(x)


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
        self.log_sigma.requires_grad_(True)

        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(min=SIGMA_MIN, max=SIGMA_MAX)

    def _dist2(self, z: torch.Tensor) -> torch.Tensor:
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

        metrics = []
        for i, layer in enumerate(self.layers):
            z_before = z
            f = layer.force(z)
            z_mid = z + f
            is_last = (i == len(self.layers) - 1)
            if not is_last:
                z_after = self.local_activation(z, z_mid, layer)
                act_delta = z_after - z_mid
            else:
                z_after = z_mid
                act_delta = torch.zeros_like(z_after)

            layer_delta = z_after - z_before
            metrics.append((
                float(f.detach().abs().mean().item()),
                float(f.detach().norm(dim=1).mean().item()),
                float(act_delta.detach().abs().mean().item()),
                float(layer_delta.detach().norm(dim=1).mean().item()),
            ))
            z = z_after
        return self.head(z), metrics

    def sigma_list(self) -> List[float]:
        return [float(layer.sigma().detach().cpu().item()) for layer in self.layers]


# ============================================================
# Training / evaluation
# ============================================================


@torch.no_grad()
def evaluate_linear(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * yb.numel()
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_aat(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
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


def make_loaders(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
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
    return train_loader, val_loader


def train_linear_baseline(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
    print("\n" + "=" * 80)
    print("Linear_MNIST_baseline")
    print("=" * 80)
    torch.manual_seed(SEED + 777)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 777)

    model = LinearMNIST().to(DEVICE)
    print(f"params: {count_parameters(model)}")
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    best_acc, best_epoch, best_loss = 0.0, 0, float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item()) * yb.numel()
            pred = logits.detach().argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            val_loss, val_acc = evaluate_linear(model, val_loader)
            if val_acc > best_acc:
                best_acc, best_epoch, best_loss = val_acc, epoch, val_loss
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train loss {total_loss/max(total,1):.4f} acc {correct/max(total,1):.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | best {best_acc:.4f}@{best_epoch}"
            )

    return {"name": "Linear_MNIST_baseline", "params": count_parameters(model), "best_acc": best_acc, "best_epoch": best_epoch, "best_loss": best_loss, "sigmas": []}


def train_aat_depth(depth: int, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
    torch.manual_seed(SEED + 100 * depth)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 100 * depth)

    dim = BASE_DIM + EXTRA_DIMS
    perm = make_permutation(dim, SEED + 1000 + EXTRA_DIMS)
    sigmas = make_depth_sigma_profile(depth)

    model_name = f"AAT_L{depth}_MNIST_D{dim}_extra{EXTRA_DIMS}_depth_scan_trainable_sigma"
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print(
        f"dim {dim} | depth {depth} | extra_dims {EXTRA_DIMS} | integral points/layer {N_INTEGRAL_POINTS} | "
        f"init sigmas {[round(s, 4) for s in sigmas]}"
    )

    model = AATMNIST(x_train, extra_dims=EXTRA_DIMS, sigmas=sigmas, perm=perm).to(DEVICE)
    print(f"params: {count_parameters(model)}")

    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)
    sigma_params = [p for n, p in model.named_parameters() if "log_sigma" in n]
    base_params = [p for n, p in model.named_parameters() if "log_sigma" not in n]
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": sigma_params, "lr": SIGMA_LR, "weight_decay": 0.0},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    best_acc, best_epoch, best_loss = 0.0, 0, float("inf")
    print(f"-- Sigma trainable from epoch 0; sigma lr={SIGMA_LR}, sigma_reg={SIGMA_REG} --")
    print("metric format per layer: force_abs / force_norm / act_abs / delta_norm")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        last_metrics = None
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits, metrics = model(xb)
                ce_loss = F.cross_entropy(logits, yb)
                if SIGMA_REG > 0.0:
                    sigma_reg = sum((layer.log_sigma - layer.init_log_sigma.to(layer.log_sigma.device)).square() for layer in model.layers)
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
            last_metrics = metrics

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            val_loss, val_acc = evaluate_aat(model, val_loader)
            if val_acc > best_acc:
                best_acc, best_epoch, best_loss = val_acc, epoch, val_loss
            sigma_text = ", ".join(f"{s:.3f}" for s in model.sigma_list())
            metric_text = ""
            if last_metrics is not None:
                metric_parts = []
                for i, (f_abs, f_norm, act_abs, delta_norm) in enumerate(last_metrics):
                    metric_parts.append(f"L{i+1}:{f_abs:.4g}/{f_norm:.4g}/{act_abs:.4g}/{delta_norm:.4g}")
                metric_text = " | metrics " + "; ".join(metric_parts)
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train loss {total_loss/max(total,1):.4f} acc {correct/max(total,1):.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"sigma [{sigma_text}] | best {best_acc:.4f}@{best_epoch}"
                f"{metric_text}"
            )

    return {"name": model_name, "params": count_parameters(model), "best_acc": best_acc, "best_epoch": best_epoch, "best_loss": best_loss, "sigmas": model.sigma_list()}


# ============================================================
# Main
# ============================================================


def main():
    print(f"Device: {DEVICE}")
    print(f"MNIST raw dir: {MNIST_RAW_DIR}")
    print("AAT MNIST deeper depth scan: +1 extra dim, learned-profile sigma warm start, trainable sigma from epoch 0")
    print(f"N_TRAIN={N_TRAIN}, N_VAL={N_VAL}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"DEPTH_LIST={DEPTH_LIST}")

    x_train_all, y_train_all, x_val_all, y_val_all = load_mnist_raw(MNIST_RAW_DIR)
    x_train = x_train_all[:N_TRAIN].contiguous()
    y_train = y_train_all[:N_TRAIN].contiguous()
    x_val = x_val_all[:N_VAL].contiguous()
    y_val = y_val_all[:N_VAL].contiguous()
    print(f"Loaded train {x_train.shape}, val {x_val.shape}")

    results = []
    if RUN_LINEAR_BASELINE:
        results.append(train_linear_baseline(x_train, y_train, x_val, y_val))
    for depth in DEPTH_LIST:
        results.append(train_aat_depth(depth, x_train, y_train, x_val, y_val))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        print(
            f"{r['name']:<64} | params {r['params']:7d} | "
            f"best_val_acc {r['best_acc']:.4f}@{r['best_epoch']} | "
            f"val_loss {r['best_loss']:.4f} | sigmas {[round(s, 4) for s in r['sigmas']]}"
        )


if __name__ == "__main__":
    main()
