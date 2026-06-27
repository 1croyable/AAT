# -*- coding: utf-8 -*-
"""
Minimal MNIST experiment for AATField v1-style transport.

Only compares two models:
  1) old_activation: original response-center activation, but no activation on the last layer.
  2) origin_spin_activation: same transport and initialization, but activation is replaced by
     a one-center origin Gaussian spin field.

Run from:
  C:\Projets\AATField\Experiments\research-capacity

Example:
  python experiment_mnist_aat_v1_origin_spin_activation.py
"""
from __future__ import annotations

import argparse
import gzip
import math
import struct
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Make "import aatfield" work when this script is placed in
# C:\Projets\AATField\Experiments\research-capacity.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aatfield.config import AATFieldConfig, AATFieldLayerConfig  # noqa: E402
from aatfield.model import AATFieldLayer  # noqa: E402
from aatfield.initialize import initialize_layer_auto_k  # noqa: E402
from aatfield.utils import count_parameters, inv_softplus, make_permutation  # noqa: E402


# -----------------------------
# Data
# -----------------------------

def _read_idx_images(path: Path) -> torch.Tensor:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError(f"Invalid image IDX magic in {path}: {magic}")
        data = torch.frombuffer(f.read(n * rows * cols), dtype=torch.uint8).clone()
    return data.view(n, rows * cols).float() / 255.0


def _read_idx_labels(path: Path) -> torch.Tensor:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise RuntimeError(f"Invalid label IDX magic in {path}: {magic}")
        data = torch.frombuffer(f.read(n), dtype=torch.uint8).clone()
    return data.long()


def _find_file(root: Path, names: Tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of {names} under {root}")


def load_mnist_raw(raw_dir: Path, n_train: int, n_val: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_images = _find_file(raw_dir, ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"))
    train_labels = _find_file(raw_dir, ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"))
    test_images = _find_file(raw_dir, ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"))
    test_labels = _find_file(raw_dir, ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"))

    x_train = _read_idx_images(train_images)[: int(n_train)]
    y_train = _read_idx_labels(train_labels)[: int(n_train)]
    x_val = _read_idx_images(test_images)[: int(n_val)]
    y_val = _read_idx_labels(test_labels)[: int(n_val)]
    return x_train, y_train, x_val, y_val


# -----------------------------
# Layers
# -----------------------------

class OldActivationLayer(AATFieldLayer):
    """Original transport, original response-center activation, optional last-layer skip."""

    def forward_with_last_flag(self, z: torch.Tensor, *, is_last: bool) -> torch.Tensor:
        move, field_target = self._potential_response(z)
        z_mid = z + move
        if is_last:
            return z_mid
        # Same activation formula as the uploaded v1 model, except not used on the last layer.
        return torch.relu(field_target + torch.relu(z_mid - field_target))


class OriginSpinActivationLayer(AATFieldLayer):
    """
    Same transport as v1, but activation is replaced by an origin Gaussian spin activation.

    The spin field has a single trainable Gaussian centered at the origin:
        theta(z) = spin_charge * exp(-||z||^2 / (2 * spin_sigma^2))

    Because high-dimensional "clockwise tangent" is not unique, we use the requested extra
    dimension as a radial-height spin plane:
        base = all dimensions except the last extra dimension
        h    = last extra dimension
        rotate the 2D pair (||base||, h) by theta(z), preserving base direction.

    Activation is the rotated-axis ReLU analogue:
        z_rot = Spin(theta, z_mid)
        z_act = ReLU(z_rot)
        z_out = Spin(-theta, z_act)
    """

    def __init__(self, cfg: AATFieldLayerConfig):
        super().__init__(cfg)
        self.spin_charge = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.raw_spin_sigma = nn.Parameter(torch.tensor(inv_softplus(float(cfg.sigma_init)), dtype=torch.float32))

    def spin_sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_spin_sigma) + 1e-4

    def _origin_spin_angle(self, z: torch.Tensor) -> torch.Tensor:
        sigma = self.spin_sigma().clamp_min(1e-6)
        dist2 = (z * z).sum(dim=-1, keepdim=True)
        kernel = torch.exp(-dist2 / (2.0 * sigma.square()))
        return self.spin_charge.view(1, 1) * kernel

    @staticmethod
    def _radial_height_rotate(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] < 2:
            raise RuntimeError("origin spin activation needs state_dim >= 2")

        base = z[:, :-1]
        h = z[:, -1:]
        r = base.norm(dim=-1, keepdim=True)
        unit = base / r.clamp_min(1e-8)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        r_new = r * cos_t - h * sin_t
        h_new = r * sin_t + h * cos_t

        base_new = unit * r_new
        # If r is exactly zero, keep base at zero and only rotate height.
        base_new = torch.where((r > 1e-8).expand_as(base_new), base_new, torch.zeros_like(base_new))
        return torch.cat([base_new, h_new], dim=-1)

    def forward_with_last_flag(self, z: torch.Tensor, *, is_last: bool) -> torch.Tensor:
        move, _field_target = self._potential_response(z)
        z_mid = z + move
        if is_last:
            return z_mid

        theta = self._origin_spin_angle(z_mid)
        z_rot = self._radial_height_rotate(z_mid, theta)
        z_act = torch.relu(z_rot)
        z_out = self._radial_height_rotate(z_act, -theta)
        return z_out


class ExperimentAATField(nn.Module):
    def __init__(self, cfg: AATFieldConfig, *, activation: str):
        super().__init__()
        if int(cfg.extra_dims) != 1 and activation == "origin_spin":
            raise ValueError("origin_spin activation expects cfg.extra_dims == 1.")

        self.cfg = cfg
        self.activation = str(activation)
        self.input_dim = int(cfg.input_dim)
        self.extra_dims = int(cfg.extra_dims)
        self.state_dim = int(cfg.state_dim)
        self.num_classes = int(cfg.num_classes)

        layer_cfg = AATFieldLayerConfig(
            state_dim=self.state_dim,
            max_children=int(cfg.max_children),
            num_classes=int(cfg.num_classes),
            sigma_init=float(cfg.sigma_init),
            charge_init=float(cfg.charge_init),
            step_cap=float(cfg.step_cap),
            gate_bias=bool(cfg.gate_bias),
        )
        layer_cls = OldActivationLayer if activation == "old" else OriginSpinActivationLayer
        self.layers = nn.ModuleList([layer_cls(layer_cfg) for _ in range(int(cfg.layers))])
        self.head = nn.Linear(self.state_dim, self.num_classes, bias=bool(cfg.head_bias))

        if self.extra_dims > 0:
            lift_perm = make_permutation(self.state_dim, int(cfg.lift_seed))
        else:
            lift_perm = torch.arange(self.state_dim, dtype=torch.long)
        self.register_buffer("lift_perm", lift_perm.long(), persistent=True)

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        x = x.float()
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")
        z = x * 2.0 - 1.0
        if self.extra_dims > 0:
            z = torch.cat([z, z.new_zeros((z.shape[0], self.extra_dims))], dim=-1)
        return z.index_select(dim=-1, index=self.lift_perm.to(z.device))

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lift(x)
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            z = layer.forward_with_last_flag(z, is_last=(i == last_idx))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.transport(x))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, y: torch.Tensor, *, samples: int, min_children: int, kmeans_iters: int) -> None:
        if int(min_children) < 2:
            raise ValueError("min_children must be >= 2 for AATField initialization.")

        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device=device, dtype=torch.long)

        if int(samples) > 0 and x.shape[0] > int(samples):
            idx = torch.randperm(x.shape[0], device=device)[: int(samples)]
            x = x[idx]
            y = y[idx]

        z = self.lift(x)
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            initialize_layer_auto_k(layer, z, y, min_children=int(min_children), kmeans_iters=int(kmeans_iters))
            z = layer.forward_with_last_flag(z, is_last=(i == last_idx))

        if was_training:
            self.train()


# -----------------------------
# Train / eval
# -----------------------------

def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            total_loss += float(loss.item()) * xb.shape[0]
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(xb.shape[0])
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one(name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, args) -> None:
    model.to(device)
    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)
    print(f"params={count_parameters(model):,}")
    print(f"selected children before init: {[getattr(l, 'selected_counts', None) for l in model.layers]}")

    print("initializing anchors...")
    x_init = train_loader.dataset.tensors[0]
    y_init = train_loader.dataset.tensors[1]
    model.initialize(x_init, y_init, samples=args.init_samples, min_children=args.min_children, kmeans_iters=args.kmeans_iters)
    print(f"selected children after init: {[getattr(l, 'selected_counts', None) for l in model.layers]}")
    print(f"params after init={count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            train_loss += float(loss.item()) * xb.shape[0]
            train_correct += int((logits.argmax(dim=1) == yb).sum().item())
            train_total += int(xb.shape[0])

        val_loss, val_acc = accuracy(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == int(args.epochs):
            print(
                f"Epoch {epoch:03d}/{int(args.epochs)-1:03d} | "
                f"train loss {train_loss / max(train_total, 1):.4f} acc {train_correct / max(train_total, 1):.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"best {best_acc:.4f}@{best_epoch}"
            )

    print(f"DONE {name}: best_acc={best_acc:.4f}@{best_epoch}, time={(time.time() - t0):.1f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=r"C:\Projets\AATField\Experiments\data\MNIST\raw")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--extra-dims", type=int, default=1)
    parser.add_argument("--max-children", type=int, default=100)
    parser.add_argument("--min-children", type=int, default=2)
    parser.add_argument("--init-samples", type=int, default=10000)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {args.data_root}")
    print("Experiment: v1 transport unchanged; only activation is changed.")
    print("Models: old_activation vs origin_spin_activation. Last layer has no activation in both models.")

    x_train, y_train, x_val, y_val = load_mnist_raw(Path(args.data_root), args.n_train, args.n_val)
    print(f"Loaded train {tuple(x_train.shape)}, val {tuple(x_val.shape)}")
    print(f"x range [{float(x_train.min()):.3f}, {float(x_train.max()):.3f}], mean/std {float(x_train.mean()):.4f}/{float(x_train.std()):.4f}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    cfg = AATFieldConfig(
        input_dim=28 * 28,
        extra_dims=int(args.extra_dims),
        num_classes=10,
        layers=int(args.layers),
        max_children=int(args.max_children),
        sigma_init=0.75,
        charge_init=0.08,
        step_cap=1.0,
        gate_bias=True,
        head_bias=True,
        lift_seed=1234,
    )

    for model_name, activation in [
        ("old_activation", "old"),
        ("origin_spin_activation", "origin_spin"),
    ]:
        torch.manual_seed(int(args.seed))
        model = ExperimentAATField(cfg, activation=activation)
        train_one(model_name, model, train_loader, val_loader, device, args)


if __name__ == "__main__":
    main()
