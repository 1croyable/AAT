#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
AAT-Lite + one extra zero dimension | 3 datasets x 3 seeds.

Purpose
-------
Test the new simplified AAT-Lite model with an added empty coordinate:

  original x in R^D  ->  padded x' = [x, 0] in R^(D+1)

Then the usual center/radius/polar state is computed in the padded space.
The added coordinate starts at zero for every sample, but AAT transport can write
into it through du. This tests whether a small latent geometric dimension gives
points extra room to move, as earlier anchor-response experiments suggested.

Model
-----
This uses AAT-Lite only:
  - fixed angular rays:          v_i = normalize(B_i)
  - fixed transport values:      dr_i = dr0_i, du_i = du0_i
  - no ray bending S
  - no rho-dependent dr1/du1

Kept unchanged:
  - angular ray response
  - ray bias beta
  - scalar gate init = 1.0 and trainable
  - ray dropout
  - u renormalization
  - final linear head on (rho, u)

Default tasks/runs
------------------
Datasets: airline, occupancy, mnist
Seeds   : 0,1,2
Total   : 3 x 3 = 9 runs

Default model sizes:
  Airline:   L=4,  R=32, epochs=60
  Occupancy: L=4,  R=16, epochs=60
  MNIST:     L=12, R=48, train=50000, val=10000, test=10000, epochs=150

Recommended command from research-capacity:
  python .\aat_lite_extra1d_3tasks_3seeds.py --device cuda --amp --resume --download ^
    --airline-root "..\data\AirlineSatisfaction" ^
    --occupancy-root ".\data\occupancy+detection" ^
    --mnist-root ".\data"

PowerShell with backticks:
  python .\aat_lite_extra1d_3tasks_3seeds.py --device cuda --amp --resume --download `
    --airline-root "..\data\AirlineSatisfaction" `
    --occupancy-root ".\data\occupancy+detection" `
    --mnist-root ".\data"

Quick smoke test:
  python .\aat_lite_extra1d_3tasks_3seeds.py --device cuda --datasets occupancy --seeds 0 --epochs-occupancy 2
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Defaults
# -----------------------------

DEFAULT_SEEDS = "0,1,2"
DEFAULT_DATASETS = "airline,occupancy,mnist"
LR = 3e-3
WD = 1e-4
KAPPA = 6.0
STEP_SCALE = 0.25
SLOPE_INIT = 0.05
GATE_INIT = 1.0
RAY_DROPOUT = 0.15
MNIST_RAY_DROPOUT = 0.20
EXTRA_DIMS = 1
EVAL_EVERY = 5

# Manageable defaults for a 12-run comparison.
AIRLINE_LAYERS = 4
AIRLINE_RAYS = 32
AIRLINE_EPOCHS = 60
AIRLINE_PATIENCE = 15

OCC_LAYERS = 4
OCC_RAYS = 16
OCC_EPOCHS = 60
OCC_PATIENCE = 15

MNIST_LAYERS = 12
MNIST_RAYS = 48
MNIST_EPOCHS = 150
MNIST_PATIENCE = 150
MNIST_N_TRAIN = 50_000
MNIST_N_VAL = 10_000
MNIST_N_TEST = 10_000

TAB_BATCH = 512
TAB_EVAL_BATCH = 8192
MNIST_BATCH = 256
MNIST_EVAL_BATCH = 4096


# -----------------------------
# Utilities
# -----------------------------


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    vals: List[float] = []
    for c in range(num_classes):
        tp = np.sum((pred_np == c) & (y_np == c))
        fp = np.sum((pred_np == c) & (y_np != c))
        fn = np.sum((pred_np != c) & (y_np == c))
        den = 2 * tp + fp + fn
        if den > 0:
            vals.append(float(2 * tp / den))
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class DataBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_dim: int
    num_classes: int
    label_names: List[str]


def make_loaders(data: DataBundle, device: torch.device, *, batch_size: int, eval_batch_size: int) -> Dict[str, DataLoader]:
    pin = device.type == "cuda"

    def ds(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).long())

    return {
        "train": DataLoader(ds(data.x_train, data.y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin),
        "train_eval": DataLoader(ds(data.x_train, data.y_train), batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin),
        "val": DataLoader(ds(data.x_val, data.y_val), batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin),
        "test": DataLoader(ds(data.x_test, data.y_test), batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin),
    }


def train_radii(x_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    center = x_train.mean(axis=0).astype(np.float32)
    r = np.linalg.norm(x_train - center[None, :], axis=1)
    r_min = float(np.quantile(r, 0.01))
    r_max = float(np.quantile(r, 0.99))
    if r_max <= r_min + 1e-6:
        r_min = float(r.min())
        r_max = float(r.max() + 1e-6)
    return center, r_min, r_max


# -----------------------------
# Airline loader
# -----------------------------


def fit_transform_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    drop_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for df in (train_df, val_df, test_df):
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    labels = sorted(train_df[target_col].astype(str).unique().tolist())
    label_to_id = {v: i for i, v in enumerate(labels)}

    def encode_y(df: pd.DataFrame) -> np.ndarray:
        return df[target_col].astype(str).map(label_to_id).astype(np.int64).to_numpy()

    y_train = encode_y(train_df)
    y_val = encode_y(val_df)
    y_test = encode_y(test_df)

    feature_cols = [c for c in train_df.columns if c != target_col]
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    if numeric_cols:
        num_train = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_val = val_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_test = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        means = num_train.mean(axis=0)
        num_train = num_train.fillna(means)
        num_val = num_val.fillna(means)
        num_test = num_test.fillna(means)
        x_num_train = num_train.to_numpy(dtype=np.float32)
        x_num_val = num_val.to_numpy(dtype=np.float32)
        x_num_test = num_test.to_numpy(dtype=np.float32)
    else:
        x_num_train = np.zeros((len(train_df), 0), dtype=np.float32)
        x_num_val = np.zeros((len(val_df), 0), dtype=np.float32)
        x_num_test = np.zeros((len(test_df), 0), dtype=np.float32)

    cat_train_parts: List[np.ndarray] = []
    cat_val_parts: List[np.ndarray] = []
    cat_test_parts: List[np.ndarray] = []
    for c in categorical_cols:
        tr = train_df[c].astype(str).fillna("__NA__")
        va = val_df[c].astype(str).fillna("__NA__")
        te = test_df[c].astype(str).fillna("__NA__")
        cats = sorted(tr.unique().tolist())
        cat_to_id = {v: i for i, v in enumerate(cats)}

        def onehot(s: pd.Series) -> np.ndarray:
            arr = np.zeros((len(s), len(cats)), dtype=np.float32)
            for row, val in enumerate(s.tolist()):
                j = cat_to_id.get(val)
                if j is not None:
                    arr[row, j] = 1.0
            return arr

        cat_train_parts.append(onehot(tr))
        cat_val_parts.append(onehot(va))
        cat_test_parts.append(onehot(te))

    if cat_train_parts:
        x_cat_train = np.concatenate(cat_train_parts, axis=1)
        x_cat_val = np.concatenate(cat_val_parts, axis=1)
        x_cat_test = np.concatenate(cat_test_parts, axis=1)
    else:
        x_cat_train = np.zeros((len(train_df), 0), dtype=np.float32)
        x_cat_val = np.zeros((len(val_df), 0), dtype=np.float32)
        x_cat_test = np.zeros((len(test_df), 0), dtype=np.float32)

    x_train = np.concatenate([x_num_train, x_cat_train], axis=1).astype(np.float32)
    x_val = np.concatenate([x_num_val, x_cat_val], axis=1).astype(np.float32)
    x_test = np.concatenate([x_num_test, x_cat_test], axis=1).astype(np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val = ((x_val - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test, labels


def load_airline(root: Path, seed: int, val_frac: float = 0.15) -> DataBundle:
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Could not find train.csv/test.csv under {root}")

    train_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_full = train_full.dropna(subset=["satisfaction"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["satisfaction"]).reset_index(drop=True)

    y_str = train_full["satisfaction"].astype(str).to_numpy()
    labels = sorted(np.unique(y_str).tolist())
    label_to_id = {v: i for i, v in enumerate(labels)}
    y_all = np.array([label_to_id[v] for v in y_str], dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in np.unique(y_all):
        idx = np.where(y_all == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_full.iloc[val_idx].reset_index(drop=True)

    xtr, ytr, xva, yva, xte, yte, label_names = fit_transform_tabular(
        train_df, val_df, test_df, target_col="satisfaction", drop_cols=["Unnamed: 0", "id"]
    )
    return DataBundle(xtr, ytr, xva, yva, xte, yte, xtr.shape[1], len(label_names), label_names)


# -----------------------------
# Occupancy loader
# -----------------------------


def read_occupancy_txt(path: Path) -> pd.DataFrame:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().strip('"') for h in header]
        for raw in reader:
            if not raw:
                continue
            raw = [x.strip().strip('"') for x in raw]
            if len(raw) == len(header) + 1:
                raw = raw[1:]
            if len(raw) != len(header):
                continue
            rows.append(raw)
    return pd.DataFrame(rows, columns=header)


def load_occupancy(root: Path) -> DataBundle:
    train_df = read_occupancy_txt(root / "datatraining.txt")
    val_df = read_occupancy_txt(root / "datatest.txt")
    test_df = read_occupancy_txt(root / "datatest2.txt")

    target_col = "Occupancy"
    drop_cols = ["date"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols + [target_col]]

    for df in (train_df, val_df, test_df):
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(int)

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0.0, 1.0).fillna(1.0)

    def make_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        x = ((df[feature_cols].fillna(mean) - mean) / std).to_numpy(dtype=np.float32).copy()
        y = df[target_col].to_numpy(dtype=np.int64).copy()
        return x, y

    x_train, y_train = make_xy(train_df)
    x_val, y_val = make_xy(val_df)
    x_test, y_test = make_xy(test_df)
    return DataBundle(x_train, y_train, x_val, y_val, x_test, y_test, x_train.shape[1], 2, ["0", "1"])


# -----------------------------
# MNIST loader
# -----------------------------


def load_mnist(root: Path, seed: int, *, download: bool, n_train: int, n_val: int, n_test: int) -> DataBundle:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for MNIST") from e

    tr = datasets.MNIST(str(root), train=True, download=download, transform=transforms.ToTensor())
    te = datasets.MNIST(str(root), train=False, download=download, transform=transforms.ToTensor())
    x_all = (tr.data.float().view(-1, 784) / 255.0).numpy().astype(np.float32)
    y_all = tr.targets.long().numpy().astype(np.int64)
    x_test = (te.data.float().view(-1, 784) / 255.0).numpy().astype(np.float32)
    y_test = te.targets.long().numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x_all.shape[0])
    if n_train <= 0:
        n_train = x_all.shape[0] - n_val
    n_train = min(int(n_train), x_all.shape[0])
    n_val = min(int(n_val), x_all.shape[0] - n_train)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    if n_test > 0:
        test_idx = rng.permutation(x_test.shape[0])[:min(int(n_test), x_test.shape[0])]
        x_test = x_test[test_idx]
        y_test = y_test[test_idx]

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    return DataBundle(x_train, y_train, x_val, y_val, x_test, y_test, 784, 10, [str(i) for i in range(10)])



# -----------------------------
# AAT-Lite + extra dimension model
# -----------------------------


def add_extra_dims(data: DataBundle, extra_dims: int) -> DataBundle:
    """Append zero-valued coordinates to train/val/test features."""
    extra_dims = int(extra_dims)
    if extra_dims <= 0:
        return data

    def pad(x: np.ndarray) -> np.ndarray:
        z = np.zeros((x.shape[0], extra_dims), dtype=np.float32)
        return np.concatenate([x.astype(np.float32), z], axis=1).astype(np.float32)

    return DataBundle(
        pad(data.x_train), data.y_train,
        pad(data.x_val), data.y_val,
        pad(data.x_test), data.y_test,
        data.input_dim + extra_dims,
        data.num_classes,
        data.label_names,
    )


class AngularAATLiteExtra(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        layers: int,
        rays: int,
        center: torch.Tensor,
        r_min: float,
        r_max: float,
        ray_dropout: float = RAY_DROPOUT,
        kappa: float = KAPPA,
        step_scale: float = STEP_SCALE,
        gate_init: float = GATE_INIT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.layers_n = int(layers)
        self.rays = int(rays)
        self.ray_dropout = float(ray_dropout)
        self.kappa = float(kappa)
        self.step_scale = float(step_scale)
        self.register_buffer("center", center.float().view(1, -1).clone())
        self.register_buffer("r_min", torch.tensor(float(r_min)))
        self.register_buffer("r_max", torch.tensor(float(r_max)))

        # AAT-Lite parameters only.
        # Response: fixed angular ray directions + ray bias.
        self.base = nn.Parameter(torch.randn(self.layers_n, self.rays, self.input_dim) / math.sqrt(self.input_dim))
        self.ray_bias = nn.Parameter(torch.zeros(self.layers_n, self.rays))

        # Transport values: fixed per ray, no rho-dependent slope.
        self.dr0 = nn.Parameter(torch.randn(self.layers_n, self.rays) * 0.02)
        self.du0 = nn.Parameter(torch.randn(self.layers_n, self.rays, self.input_dim) * 0.02 / math.sqrt(self.input_dim))

        self.alpha_gate = nn.Parameter(torch.full((self.layers_n,), float(gate_init)))
        self.head = nn.Linear(self.input_dim + 1, self.num_classes)

    def _to_polar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xc = x.float() - self.center.to(x.device)
        r = xc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        u = xc / r
        denom = (self.r_max - self.r_min).to(x.device).clamp_min(1e-8)
        rho = 2.0 * (r - self.r_min.to(x.device)) / denom - 1.0
        return rho.clamp(-1.5, 1.5), u

    def _ray_dropout(self, alpha: torch.Tensor) -> torch.Tensor:
        p = self.ray_dropout
        if (not self.training) or p <= 0.0:
            return alpha
        keep = max(1e-6, 1.0 - p)
        mask = (torch.rand_like(alpha) < keep).to(alpha.dtype)
        empty = mask.sum(dim=1, keepdim=True) < 0.5
        if bool(empty.any()):
            idx = alpha.argmax(dim=1, keepdim=True)
            mask.scatter_(1, idx, torch.ones_like(idx, dtype=mask.dtype))
        alpha = alpha * mask
        return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor, *, return_stats: bool = False):
        rho, u = self._to_polar(x)
        if return_stats:
            entropies: List[torch.Tensor] = []
            maxes: List[torch.Tensor] = []
            moves: List[torch.Tensor] = []

        for li in range(self.layers_n):
            # Fixed angular rays: v_i = normalize(B_i).
            # Use GEMM instead of materializing [B, R, D].
            v = F.normalize(self.base[li], dim=-1, eps=1e-8)  # [R, D]
            scores = self.kappa * (u @ v.t()) + self.ray_bias[li].view(1, -1)
            alpha_raw = F.softmax(scores, dim=1)
            alpha = self._ray_dropout(alpha_raw)

            # Fixed transport values: dr_i = dr0_i, du_i = du0_i.
            # Again use GEMM for alpha-weighted transport.
            rho_move = (alpha @ self.dr0[li].view(-1, 1)) * self.step_scale * self.alpha_gate[li]
            u_move = (alpha @ self.du0[li]) * self.step_scale * self.alpha_gate[li]

            if return_stats:
                safe = alpha_raw.clamp_min(1e-12)
                entropies.append((-(safe * safe.log()).sum(dim=1)).mean().detach())
                maxes.append(alpha_raw.max(dim=1).values.mean().detach())
                moves.append(torch.cat([rho_move, u_move], dim=1).norm(dim=1).mean().detach())

            rho = (rho + rho_move).clamp(-2.0, 2.0)
            u = F.normalize(u + u_move, dim=1, eps=1e-8)

        logits = self.head(torch.cat([rho, u], dim=1))
        if not return_stats:
            return logits
        return logits, {
            "gate_mean": float(self.alpha_gate.detach().mean().item()),
            "alpha_entropy": float(torch.stack(entropies).mean().item()),
            "alpha_max": float(torch.stack(maxes).mean().item()),
            "move_norm": float(torch.stack(moves).mean().item()),
        }

# -----------------------------
# Training / evaluation
# -----------------------------



@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float]:
    model.eval()
    preds: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).detach().cpu())
        ys.append(yb.detach().cpu())
    pred = torch.cat(preds)
    y = torch.cat(ys)
    return float((pred == y).float().mean().item()), macro_f1(pred, y, num_classes)


@torch.no_grad()
def get_stats(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    acc: Dict[str, List[float]] = {"gate_mean": [], "alpha_entropy": [], "alpha_max": [], "move_norm": []}
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        _, stats = model(xb, return_stats=True)
        for k, v in stats.items():
            acc[k].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def train_one(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    *,
    amp: bool,
    epochs: int,
    patience: int,
    num_classes: int,
) -> Dict[str, object]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and device.type == "cuda"))
    best: Dict[str, object] = {
        "epoch": 0,
        "train_acc": 0.0,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "test_f1": 0.0,
        "state": None,
    }
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and device.type == "cuda")):
                loss = F.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.detach().item()) * xb.shape[0]
            total_n += xb.shape[0]

        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            train_acc, _ = evaluate(model, loaders["train_eval"], device, num_classes)
            val_acc, _ = evaluate(model, loaders["val"], device, num_classes)
            test_acc, test_f1 = evaluate(model, loaders["test"], device, num_classes)
            if val_acc > float(best["val_acc"]):
                best.update(
                    epoch=epoch,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    test_acc=test_acc,
                    test_f1=test_f1,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            print(
                f"epoch={epoch:04d} loss={total_loss/max(1,total_n):.4f} train={train_acc:.4f} "
                f"val={val_acc:.4f} test={test_acc:.4f} f1={test_f1:.4f} "
                f"best_val={best['val_acc']:.4f} best_test={best['test_acc']:.4f}",
                flush=True,
            )
            if int(best["epoch"]) > 0 and (epoch - int(best["epoch"])) >= patience:
                print(f"early_stop epoch={epoch} best_epoch={best['epoch']} patience={patience}", flush=True)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    best.update(get_stats(model, loaders["test"], device))
    best["seconds"] = time.time() - t0
    best.pop("state", None)
    return best



# -----------------------------
# Experiment orchestration
# -----------------------------


def load_dataset(name: str, args: argparse.Namespace, seed: int) -> Tuple[DataBundle, int, int, int, int, int, int, float]:
    if name == "airline":
        data = load_airline(Path(args.airline_root), seed=seed)
        data = add_extra_dims(data, args.extra_dims)
        return data, args.airline_layers, args.airline_rays, args.epochs_airline, args.patience_airline, TAB_BATCH, TAB_EVAL_BATCH, args.ray_dropout_airline
    if name == "occupancy":
        data = load_occupancy(Path(args.occupancy_root))
        data = add_extra_dims(data, args.extra_dims)
        return data, args.occupancy_layers, args.occupancy_rays, args.epochs_occupancy, args.patience_occupancy, TAB_BATCH, TAB_EVAL_BATCH, args.ray_dropout_occupancy
    if name == "mnist":
        data = load_mnist(
            Path(args.mnist_root),
            seed=seed,
            download=bool(args.download),
            n_train=args.mnist_n_train,
            n_val=args.mnist_n_val,
            n_test=args.mnist_n_test,
        )
        data = add_extra_dims(data, args.extra_dims)
        return data, args.mnist_layers, args.mnist_rays, args.epochs_mnist, args.patience_mnist, MNIST_BATCH, MNIST_EVAL_BATCH, args.ray_dropout_mnist
    raise ValueError(f"Unknown dataset: {name}")


def append_row(path: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        "dataset", "seed", "extra_dims", "layers", "rays", "params", "best_epoch",
        "train_acc", "val_acc", "test_acc", "test_f1",
        "gate_mean", "alpha_entropy", "alpha_max", "move_norm", "seconds",
        "ray_dropout", "kappa", "step_scale",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_completed(path: Path) -> set[Tuple[str, int, int]]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    done: set[Tuple[str, int, int]] = set()
    for _, r in df.iterrows():
        try:
            done.add((str(r["dataset"]), int(r["seed"]), int(r.get("extra_dims", 1))))
        except Exception:
            pass
    return done


def print_summary(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    print("\n" + "=" * 120)
    print("RESULTS")
    print("=" * 120)
    cols = ["dataset", "seed", "extra_dims", "params", "best_epoch", "val_acc", "test_acc", "test_f1", "gate_mean", "alpha_entropy", "seconds"]
    print(df[cols].sort_values(["dataset", "seed"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 120)
    print("MEAN BY DATASET")
    print("=" * 120)
    g = df.groupby(["dataset"], as_index=False).agg(
        runs=("test_acc", "count"),
        params=("params", "mean"),
        val_mean=("val_acc", "mean"),
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", "std"),
        f1_mean=("test_f1", "mean"),
        seconds_mean=("seconds", "mean"),
    )
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download", action="store_true", help="download MNIST if needed")
    parser.add_argument("--out", type=str, default="aat_lite_extra1d_3tasks_3seeds_results.csv")

    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS, help="comma list: airline,occupancy,mnist")
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--extra-dims", type=int, default=EXTRA_DIMS)

    parser.add_argument("--airline-root", type=str, default="../data/AirlineSatisfaction")
    parser.add_argument("--occupancy-root", type=str, default="./data/occupancy+detection")
    parser.add_argument("--mnist-root", type=str, default="./data")

    parser.add_argument("--airline-layers", type=int, default=AIRLINE_LAYERS)
    parser.add_argument("--airline-rays", type=int, default=AIRLINE_RAYS)
    parser.add_argument("--epochs-airline", type=int, default=AIRLINE_EPOCHS)
    parser.add_argument("--patience-airline", type=int, default=AIRLINE_PATIENCE)
    parser.add_argument("--ray-dropout-airline", type=float, default=RAY_DROPOUT)

    parser.add_argument("--occupancy-layers", type=int, default=OCC_LAYERS)
    parser.add_argument("--occupancy-rays", type=int, default=OCC_RAYS)
    parser.add_argument("--epochs-occupancy", type=int, default=OCC_EPOCHS)
    parser.add_argument("--patience-occupancy", type=int, default=OCC_PATIENCE)
    parser.add_argument("--ray-dropout-occupancy", type=float, default=RAY_DROPOUT)

    parser.add_argument("--mnist-layers", type=int, default=MNIST_LAYERS)
    parser.add_argument("--mnist-rays", type=int, default=MNIST_RAYS)
    parser.add_argument("--epochs-mnist", type=int, default=MNIST_EPOCHS)
    parser.add_argument("--patience-mnist", type=int, default=MNIST_PATIENCE)
    parser.add_argument("--mnist-n-train", type=int, default=MNIST_N_TRAIN)
    parser.add_argument("--mnist-n-val", type=int, default=MNIST_N_VAL)
    parser.add_argument("--mnist-n-test", type=int, default=MNIST_N_TEST)
    parser.add_argument("--ray-dropout-mnist", type=float, default=MNIST_RAY_DROPOUT)

    args = parser.parse_args()

    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    valid_datasets = {"airline", "occupancy", "mnist"}
    for d in datasets:
        if d not in valid_datasets:
            raise ValueError(f"bad dataset {d}; valid={sorted(valid_datasets)}")

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_path = Path(args.out)
    done = read_completed(out_path) if args.resume else set()

    print("=" * 132)
    print("AAT-Lite + extra zero dimension | fixed rays + fixed transport values")
    print("=" * 132)
    print(f"device={device} amp={args.amp} resume={args.resume} out={out_path}")
    print(f"datasets={datasets} seeds={seeds} extra_dims={args.extra_dims}")
    print("Input is padded as x' = [x, 0, ..., 0], then centered/polarized in the padded space.")
    print("Model is Lite only: no slope S, no dr1, no du1.")

    total = len(datasets) * len(seeds)
    run_idx = 0
    for dataset in datasets:
        for seed in seeds:
            run_idx += 1
            key = (dataset, seed, int(args.extra_dims))
            if key in done:
                print(f"\n[{run_idx}/{total}] skip dataset={dataset} seed={seed} extra_dims={args.extra_dims}", flush=True)
                continue

            set_seed(seed)
            data, layers, rays, epochs, patience, batch_size, eval_batch_size, ray_dropout = load_dataset(dataset, args, seed)
            loaders = make_loaders(data, device, batch_size=batch_size, eval_batch_size=eval_batch_size)
            center_np, r_min, r_max = train_radii(data.x_train)
            print(
                f"\nDataset={dataset} seed={seed} train={data.x_train.shape} val={data.x_val.shape} "
                f"test={data.x_test.shape} D={data.input_dim} C={data.num_classes} L={layers} R={rays} "
                f"epochs={epochs} patience={patience} ray_dropout={ray_dropout} r_min={r_min:.5f} r_max={r_max:.5f}",
                flush=True,
            )
            print("-" * 132)
            print(f"[{run_idx}/{total}] dataset={dataset} seed={seed} extra_dims={args.extra_dims}", flush=True)
            center = torch.tensor(center_np, dtype=torch.float32, device=device)
            model = AngularAATLiteExtra(
                data.input_dim,
                data.num_classes,
                layers=layers,
                rays=rays,
                center=center,
                r_min=r_min,
                r_max=r_max,
                ray_dropout=ray_dropout,
                kappa=KAPPA,
                step_scale=STEP_SCALE,
            ).to(device)
            params = count_params(model)
            print(f"params={params}", flush=True)

            best = train_one(
                model,
                loaders,
                device,
                amp=bool(args.amp),
                epochs=epochs,
                patience=patience,
                num_classes=data.num_classes,
            )
            row = {
                "dataset": dataset,
                "seed": seed,
                "extra_dims": int(args.extra_dims),
                "layers": layers,
                "rays": rays,
                "params": params,
                "best_epoch": int(best["epoch"]),
                "train_acc": float(best["train_acc"]),
                "val_acc": float(best["val_acc"]),
                "test_acc": float(best["test_acc"]),
                "test_f1": float(best["test_f1"]),
                "gate_mean": float(best.get("gate_mean", 0.0)),
                "alpha_entropy": float(best.get("alpha_entropy", 0.0)),
                "alpha_max": float(best.get("alpha_max", 0.0)),
                "move_norm": float(best.get("move_norm", 0.0)),
                "seconds": float(best["seconds"]),
                "ray_dropout": ray_dropout,
                "kappa": KAPPA,
                "step_scale": STEP_SCALE,
            }
            append_row(out_path, row)
            print(
                f"BEST dataset={dataset} seed={seed} epoch={row['best_epoch']} "
                f"train={row['train_acc']:.4f} val={row['val_acc']:.4f} test={row['test_acc']:.4f} "
                f"f1={row['test_f1']:.4f} gate={row['gate_mean']:.3f} entropy={row['alpha_entropy']:.3f} "
                f"alpha_max={row['alpha_max']:.3f} move={row['move_norm']:.3f} sec={row['seconds']:.1f}",
                flush=True,
            )

    print_summary(out_path)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
