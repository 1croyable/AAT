# -*- coding: utf-8 -*-
r"""
experiment_mnist_aat_l8_ablation_no_k.py

MNIST L8 AATField 2.0 mechanism ablation.

Purpose:
  Test whether the weak raw force in previous MNIST runs was caused by:
    1) dividing force by the number of integral points K,
    2) hidden activation using trainable charge c as an attention bias,
    3) missing trainable transport gain.

This script runs three L8 models:
  A) force_only_no_k
       z_{l+1} = z_l + v_l(z_l)
       No hidden local activation.
       Force does NOT divide by K.

  B) geom_activation_no_k
       Hidden layers use local activation, but response center is pure geometry:
         alpha_j = softmax(-||z-a_j||^2 / (2 sigma^2))
         r = sum_j alpha_j a_j
         z_out = r + ReLU(z_mid - r)
       c is used only by force, not by activation.
       Force does NOT divide by K.

  C) geom_activation_no_k_gain
       Same as B, but with trainable positive per-layer transport gain gamma_l:
         z_mid = z + gamma_l * v(z)
       Force does NOT divide by K.

Data path is fixed to:
  C:/Projets/AATField/Experiments/data/MNIST/raw

Run:
  python experiment_mnist_aat_l8_ablation_no_k.py
"""

import gzip
import math
import random
import struct
from pathlib import Path
from typing import List, Tuple, Optional

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

DEPTH = 8
EXTRA_DIMS = 1
BASE_DIM = 28 * 28

N_INTEGRAL_POINTS = 512
C_INIT_STD = 0.04
EXTRA_ANCHOR_STD = 0.35
ANCHOR_NOISE_STD = 0.01

SIGMA_LR = 5e-4
SIGMA_REG = 0.0
SIGMA_MIN = 0.30
SIGMA_MAX = 20.0

# Previous L8 MNIST profile was stable around this region.
L8_SIGMA_PROFILE = [1.7281, 1.6662, 1.8395, 1.9370, 1.7940, 1.8489, 1.9949, 6.6992]

# Gain is only used in the gain variant.
GAIN_LR = 1e-3
GAIN_MIN = 0.05
GAIN_MAX = 20.0
GAIN_INIT = 1.0

VARIANTS = [
    {
        "name": "AAT_L8_MNIST_force_only_no_k",
        "activation": "none",
        "trainable_gain": False,
    },
    {
        "name": "AAT_L8_MNIST_geom_activation_no_k",
        "activation": "geom",
        "trainable_gain": False,
    },
    {
        "name": "AAT_L8_MNIST_geom_activation_no_k_gain",
        "activation": "geom",
        "trainable_gain": True,
    },
]


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


# ============================================================
# AAT model
# ============================================================


class IntegralTerrainLayerNoK(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma_init: float,
        anchor_source: torch.Tensor,
        extra_dims: int,
        perm: torch.Tensor,
        trainable_gain: bool,
    ):
        super().__init__()
        self.dim = int(dim)
        self.extra_dims = int(extra_dims)
        self.trainable_gain = bool(trainable_gain)

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

        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)

        if self.trainable_gain:
            self.log_gain = nn.Parameter(torch.tensor(math.log(float(GAIN_INIT)), dtype=torch.float32))
        else:
            self.register_buffer("log_gain", torch.tensor(math.log(float(GAIN_INIT)), dtype=torch.float32))

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(min=SIGMA_MIN, max=SIGMA_MAX)

    def gain(self) -> torch.Tensor:
        return self.log_gain.exp().clamp(min=GAIN_MIN, max=GAIN_MAX)

    def _dist2(self, z: torch.Tensor) -> torch.Tensor:
        return (
            z.square().sum(dim=1, keepdim=True)
            + self.a.square().sum(dim=1).view(1, -1)
            - 2.0 * z @ self.a.t()
        ).clamp_min(0.0)

    def force(self, z: torch.Tensor) -> torch.Tensor:
        """
        Important: this intentionally does NOT divide by K=N_INTEGRAL_POINTS.

        Old code used:
            move = (wa - wsum * z) / (K * sigma^2)

        This experiment uses:
            move = (wa - wsum * z) / sigma^2
        """
        sigma = self.sigma().to(z.dtype)
        dist2 = self._dist2(z)
        k = torch.exp(-dist2 / (2.0 * sigma * sigma))
        w = k * self.c.to(z.dtype).view(1, -1)
        wa = w @ self.a.to(z.dtype)
        wsum = w.sum(dim=1, keepdim=True)
        move = (wa - wsum * z) / (sigma * sigma)
        return move

    def response_center_geom(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pure geometric response center.
        c is NOT used here, so activation cannot learn an attention shortcut through c.
        """
        sigma = self.sigma().to(z.dtype)
        dist2 = self._dist2(z)
        score = -dist2 / (2.0 * sigma * sigma)
        alpha = torch.softmax(score, dim=1)
        return alpha @ self.a.to(z.dtype)


class AATMNISTAblation(nn.Module):
    def __init__(self, base_train_x: torch.Tensor, variant: dict):
        super().__init__()
        self.variant = dict(variant)
        self.activation = str(variant["activation"])
        self.trainable_gain = bool(variant["trainable_gain"])
        self.extra_dims = EXTRA_DIMS
        self.dim = BASE_DIM + self.extra_dims
        self.register_buffer("perm", make_permutation(self.dim, SEED + 1000 + self.extra_dims).long())

        self.layers = nn.ModuleList([
            IntegralTerrainLayerNoK(
                dim=self.dim,
                sigma_init=L8_SIGMA_PROFILE[i],
                anchor_source=base_train_x,
                extra_dims=self.extra_dims,
                perm=self.perm,
                trainable_gain=self.trainable_gain,
            )
            for i in range(DEPTH)
        ])
        self.head = nn.Linear(self.dim, 10)

    def local_activation_geom(self, z: torch.Tensor, z_mid: torch.Tensor, layer: IntegralTerrainLayerNoK) -> torch.Tensor:
        r = layer.response_center_geom(z)
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
            f_raw = layer.force(z)
            gain = layer.gain().to(z.dtype)
            f = gain * f_raw
            z_mid = z + f

            is_last = (i == len(self.layers) - 1)
            if self.activation == "geom" and not is_last:
                z_after = self.local_activation_geom(z, z_mid, layer)
                act_delta = z_after - z_mid
            elif self.activation == "none":
                z_after = z_mid
                act_delta = torch.zeros_like(z_after)
            else:
                z_after = z_mid
                act_delta = torch.zeros_like(z_after)

            layer_delta = z_after - z_before
            metrics.append((
                float(f_raw.detach().abs().mean().item()),
                float(f_raw.detach().norm(dim=1).mean().item()),
                float(gain.detach().cpu().item()),
                float(f.detach().norm(dim=1).mean().item()),
                float(act_delta.detach().abs().mean().item()),
                float(layer_delta.detach().norm(dim=1).mean().item()),
            ))
            z = z_after

        return self.head(z), metrics

    def sigma_list(self) -> List[float]:
        return [float(layer.sigma().detach().cpu().item()) for layer in self.layers]

    def gain_list(self) -> List[float]:
        return [float(layer.gain().detach().cpu().item()) for layer in self.layers]


# ============================================================
# Training / eval
# ============================================================


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


def train_variant(variant: dict, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
    model_name = variant["name"]
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    torch.manual_seed(SEED + 2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 2026)

    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)
    model = AATMNISTAblation(x_train, variant).to(DEVICE)

    print(
        f"dim {BASE_DIM + EXTRA_DIMS} | depth {DEPTH} | extra_dims {EXTRA_DIMS} | "
        f"integral points/layer {N_INTEGRAL_POINTS} | activation {variant['activation']} | "
        f"trainable_gain {variant['trainable_gain']} | NO /K in force"
    )
    print(f"init sigmas {[round(s, 4) for s in L8_SIGMA_PROFILE]}")
    print(f"params: {count_parameters(model)}")

    sigma_params = [p for n, p in model.named_parameters() if "log_sigma" in n]
    gain_params = [p for n, p in model.named_parameters() if "log_gain" in n and p.requires_grad]
    base_params = [p for n, p in model.named_parameters() if "log_sigma" not in n and "log_gain" not in n]

    param_groups = [
        {"params": base_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": sigma_params, "lr": SIGMA_LR, "weight_decay": 0.0},
    ]
    if len(gain_params) > 0:
        param_groups.append({"params": gain_params, "lr": GAIN_LR, "weight_decay": 0.0})

    opt = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    best_acc, best_epoch, best_loss = 0.0, 0, float("inf")
    print(
        f"-- sigma lr={SIGMA_LR}, sigma_reg={SIGMA_REG}, "
        f"gain lr={GAIN_LR if len(gain_params) > 0 else 'N/A'} --"
    )
    print("metric format per layer: raw_force_abs / raw_force_norm / gain / gained_force_norm / act_abs / delta_norm")

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
            last_metrics = metrics

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            val_loss, val_acc = evaluate_aat(model, val_loader)
            if val_acc > best_acc:
                best_acc, best_epoch, best_loss = val_acc, epoch, val_loss

            sigma_text = ", ".join(f"{s:.3f}" for s in model.sigma_list())
            gain_text = ", ".join(f"{g:.3f}" for g in model.gain_list())
            metric_text = ""
            if last_metrics is not None:
                metric_parts = []
                for i, (raw_abs, raw_norm, gain, gained_norm, act_abs, delta_norm) in enumerate(last_metrics):
                    metric_parts.append(
                        f"L{i+1}:{raw_abs:.4g}/{raw_norm:.4g}/{gain:.3g}/{gained_norm:.4g}/{act_abs:.4g}/{delta_norm:.4g}"
                    )
                metric_text = " | metrics " + "; ".join(metric_parts)

            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train loss {total_loss/max(total,1):.4f} acc {correct/max(total,1):.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"sigma [{sigma_text}] | gain [{gain_text}] | best {best_acc:.4f}@{best_epoch}"
                f"{metric_text}"
            )

    return {
        "name": model_name,
        "params": count_parameters(model),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "sigmas": model.sigma_list(),
        "gains": model.gain_list(),
    }


# ============================================================
# Main
# ============================================================


def main():
    print(f"Device: {DEVICE}")
    print(f"MNIST raw dir: {MNIST_RAW_DIR}")
    print("AAT MNIST L8 ablation: remove /K, compare force-only / geometric activation / gain")
    print(f"N_TRAIN={N_TRAIN}, N_VAL={N_VAL}, epochs={EPOCHS}, batch={BATCH_SIZE}")

    x_train_all, y_train_all, x_val_all, y_val_all = load_mnist_raw(MNIST_RAW_DIR)
    x_train = x_train_all[:N_TRAIN].contiguous()
    y_train = y_train_all[:N_TRAIN].contiguous()
    x_val = x_val_all[:N_VAL].contiguous()
    y_val = y_val_all[:N_VAL].contiguous()
    print(f"Loaded train {x_train.shape}, val {x_val.shape}")

    results = []
    for variant in VARIANTS:
        results.append(train_variant(variant, x_train, y_train, x_val, y_val))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        sig = [round(s, 4) for s in r["sigmas"]]
        gains = [round(g, 4) for g in r["gains"]]
        print(
            f"{r['name']:<58} | params {r['params']:7d} | "
            f"best_val_acc {r['best_acc']:.4f}@{r['best_epoch']} | "
            f"val_loss {r['best_loss']:.4f} | sigmas {sig} | gains {gains}"
        )


if __name__ == "__main__":
    main()
