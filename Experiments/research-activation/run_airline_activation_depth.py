# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.parent.name.lower() == "experiments" else SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aatfield import AATField, AATFieldConfig  # noqa: E402
from aatfield.utils import count_parameters  # noqa: E402


# ============================================================
# Data
# ============================================================


@dataclass
class DataBundle:
    name: str
    input_dim: int
    num_classes: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def label_to_int(v: Any) -> int:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "satisfied"}:
        return 1
    return 0


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def airline_bundle(train_csv: Path, test_csv: Path, *, val_ratio: float = 0.15, seed: int = 123) -> DataBundle:
    train_rows = read_csv_rows(train_csv)
    test_rows = read_csv_rows(test_csv)
    if not train_rows or not test_rows:
        raise RuntimeError("empty train/test csv")

    columns = list(train_rows[0].keys())
    target_col = "satisfaction" if "satisfaction" in columns else columns[-1]
    drop_cols = {target_col, "id", "ID", "Id"}
    drop_cols.update(c for c in columns if c.lower().startswith("unnamed"))
    feature_cols = [c for c in columns if c not in drop_cols]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in feature_cols:
        vals = [str(r.get(c, "")).strip() for r in train_rows]
        non_empty = [v for v in vals if v != "" and v.lower() not in {"nan", "none", "null"}]
        if non_empty and all(parse_float_or_none(v) is not None for v in non_empty):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for c in numeric_cols:
        vals = [parse_float_or_none(r.get(c, "")) for r in train_rows]
        clean = [float(v) for v in vals if v is not None]
        mean = sum(clean) / max(len(clean), 1)
        var = sum((v - mean) ** 2 for v in clean) / max(len(clean), 1)
        means[c] = float(mean)
        stds[c] = float(math.sqrt(var) if var > 1e-12 else 1.0)

    vocabs: Dict[str, List[str]] = {}
    for c in categorical_cols:
        values = sorted({str(r.get(c, "")).strip() for r in train_rows if str(r.get(c, "")).strip() != ""})
        vocabs[c] = values

    def encode_rows(rows: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        xs: List[List[float]] = []
        ys: List[int] = []
        for r in rows:
            feat: List[float] = []
            for c in numeric_cols:
                v = parse_float_or_none(r.get(c, ""))
                value = means[c] if v is None else float(v)
                feat.append((value - means[c]) / stds[c])
            for c in categorical_cols:
                s = str(r.get(c, "")).strip()
                vocab = vocabs[c]
                vec = [0.0] * (len(vocab) + 1)
                try:
                    j = vocab.index(s)
                except ValueError:
                    j = len(vocab)
                vec[j] = 1.0
                feat.extend(vec)
            xs.append(feat)
            ys.append(label_to_int(r.get(target_col, "")))
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(len(train_rows), generator=gen).tolist()
    val_n = max(1, int(round(len(idx) * float(val_ratio))))
    val_idx = set(idx[:val_n])
    val_rows = [r for i, r in enumerate(train_rows) if i in val_idx]
    tr_rows = [r for i, r in enumerate(train_rows) if i not in val_idx]

    x_train, y_train = encode_rows(tr_rows)
    x_val, y_val = encode_rows(val_rows)
    x_test, y_test = encode_rows(test_rows)

    print(
        f"Airline features: raw_cols={len(feature_cols)} numeric={len(numeric_cols)} "
        f"categorical={len(categorical_cols)} encoded_dim={x_train.shape[1]} "
        f"train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}",
        flush=True,
    )
    return DataBundle("airline_satisfaction", int(x_train.shape[1]), 2, x_train, y_train, x_val, y_val, x_test, y_test)


def make_loaders(data: DataBundle, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(TensorDataset(data.x_train, data.y_train), batch_size=int(batch_size), shuffle=True),
        DataLoader(TensorDataset(data.x_val, data.y_val), batch_size=int(batch_size) * 2, shuffle=False),
        DataLoader(TensorDataset(data.x_test, data.y_test), batch_size=int(batch_size) * 2, shuffle=False),
    )


# ============================================================
# Metrics
# ============================================================


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1).long().cpu()
    y = y.view(-1).long().cpu()
    scores: List[float] = []
    for c in range(int(num_classes)):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        den = 2 * tp + fp + fn
        scores.append(0.0 if den == 0 else 2 * tp / den)
    return float(sum(scores) / max(len(scores), 1))


@torch.no_grad()
def accuracy_and_f1(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float]:
    model.eval()
    preds, ys = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        ys.append(yb.cpu())
    pred = torch.cat(preds)
    y = torch.cat(ys)
    return float((pred == y).float().mean().item()), macro_f1(pred, y, num_classes)


def clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_to_device(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


# ============================================================
# Activation wrapper around current AATField
# ============================================================


class ActivationAATField(AATField):
    VALID_MODES = {"none", "post_relu", "parent_relu"}

    def __init__(self, cfg: AATFieldConfig, activation_mode: str = "none", init_uses_activation: bool = True):
        super().__init__(cfg)
        activation_mode = str(activation_mode).strip().lower()
        if activation_mode not in self.VALID_MODES:
            raise ValueError(f"activation_mode must be one of {sorted(self.VALID_MODES)}, got {activation_mode!r}")
        self.activation_mode = activation_mode
        self.init_uses_activation = bool(init_uses_activation)

    def _nearest_parent(self, layer, z_mid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parents = layer.parents.to(device=z_mid.device, dtype=z_mid.dtype)
        dist2 = (
            (z_mid * z_mid).sum(dim=-1, keepdim=True)
            + (parents * parents).sum(dim=-1).view(1, -1)
            - 2.0 * (z_mid @ parents.t())
        ).clamp_min(0.0)
        parent_idx = dist2.argmin(dim=1)
        p = parents.index_select(0, parent_idx)
        return p, parent_idx

    def _apply_activation(self, layer, z_mid: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.activation_mode == "none":
            return z_mid, None, None
        if self.activation_mode == "post_relu":
            origin = torch.zeros_like(z_mid)
            return F.relu(z_mid), origin, None
        if self.activation_mode == "parent_relu":
            origin, parent_idx = self._nearest_parent(layer, z_mid)
            return origin + F.relu(z_mid - origin), origin, parent_idx
        raise RuntimeError(f"unknown activation_mode={self.activation_mode!r}")

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lift(x)
        for layer in self.layers:
            z_mid = layer(z)
            z, _origin, _parent_idx = self._apply_activation(layer, z_mid)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.transport(x))

    @torch.no_grad()
    def initialize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        samples: int = 8192,
        min_children: int = 2,
        kmeans_iters: int = 8,
    ) -> None:
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
        for layer in self.layers:
            layer.auto_k_init(z, y, min_children=int(min_children), kmeans_iters=int(kmeans_iters))
            z_mid = layer(z)
            if self.init_uses_activation:
                z, _origin, _parent_idx = self._apply_activation(layer, z_mid)
            else:
                z = z_mid

        if was_training:
            self.train()


@torch.no_grad()
def layer_mid_and_info(layer, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    C = layer.num_classes
    child_flat = layer.child_anchors().reshape(layer.child_n, layer.state_dim)
    anchors = torch.cat([layer.parents, child_flat], dim=0)
    sigma = layer.sigma()

    dist2 = (
        (z * z).sum(dim=-1, keepdim=True)
        + (anchors * anchors).sum(dim=-1).view(1, -1)
        - 2.0 * (z @ anchors.t())
    ).clamp_min(0.0)
    logits = -dist2 / (2.0 * sigma.view(1, -1).square() + 1e-8)
    alpha = torch.softmax(logits, dim=-1)

    dist = torch.sqrt(dist2 + 1e-8)
    base = alpha * layer.charge.view(1, -1)
    gate = layer._child_gate(z, child_flat, sigma[C:])
    strength = torch.cat([base[:, :C], base[:, C:] * sigma[C:].view(1, -1) * gate], dim=1)

    beta = strength / dist.clamp_min(1e-6)
    move = beta @ anchors - z * beta.sum(dim=1, keepdim=True)

    cap = float(layer.cfg.step_cap)
    raw_move_norm = move.norm(dim=-1, keepdim=True)
    if cap > 0:
        capped = cap * torch.tanh(raw_move_norm / cap)
        move = move * (capped / raw_move_norm.clamp_min(1e-6))
        move_norm = capped.squeeze(-1)
    else:
        move_norm = raw_move_norm.squeeze(-1)

    entropy = -(alpha.detach().float() * alpha.detach().float().clamp_min(1e-8).log()).sum(dim=-1)
    z_mid = z + move
    info = {
        "move": move.detach(),
        "move_norm": move_norm.detach(),
        "gate_mean": gate.detach().mean(dim=1),
        "gate_active": (gate.detach() > 0).float().mean(dim=1),
        "alpha_entropy": entropy.detach(),
        "alpha_eff": torch.exp(entropy.detach()),
    }
    return z_mid, info


@torch.no_grad()
def collect_activation_diagnostics(
    model: ActivationAATField,
    x: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    eps: float = 1e-8,
) -> Dict[str, object]:
    model.eval()
    n_layers = len(model.layers)
    sums: List[Dict[str, float]] = []
    for _ in range(n_layers):
        sums.append({
            "n": 0.0,
            "state_norm": 0.0,
            "pre_state_norm": 0.0,
            "mid_state_norm": 0.0,
            "move_norm": 0.0,
            "move_state_ratio": 0.0,
            "state_zero_rate": 0.0,
            "active_dim_rate": 0.0,
            "activation_zero_rate": 0.0,
            "origin_norm": 0.0,
            "gate_mean": 0.0,
            "gate_active": 0.0,
            "alpha_entropy": 0.0,
            "alpha_eff": 0.0,
        })

    final_norm_sum = 0.0
    final_zero_sum = 0.0
    total_n = 0.0

    for start in range(0, x.shape[0], int(batch_size)):
        xb = x[start:start + int(batch_size)].to(device)
        n = float(xb.shape[0])
        z = model.lift(xb)
        for li, layer in enumerate(model.layers):
            z_pre = z
            z_mid, info = layer_mid_and_info(layer, z_pre)
            z_next, origin, _parent_idx = model._apply_activation(layer, z_mid)

            pre_norm = z_pre.norm(dim=-1)
            mid_norm = z_mid.norm(dim=-1)
            state_norm = z_next.norm(dim=-1)
            move_norm = info["move_norm"]
            ratio = move_norm / pre_norm.clamp_min(1e-6)

            if origin is None:
                act_zero = torch.zeros_like(state_norm)
                origin_norm = torch.zeros_like(state_norm)
            else:
                act_zero = ((z_next - origin).abs() <= float(eps)).float().mean(dim=1)
                origin_norm = origin.norm(dim=-1)

            d = sums[li]
            d["n"] += n
            d["pre_state_norm"] += float(pre_norm.mean().item()) * n
            d["mid_state_norm"] += float(mid_norm.mean().item()) * n
            d["state_norm"] += float(state_norm.mean().item()) * n
            d["move_norm"] += float(move_norm.mean().item()) * n
            d["move_state_ratio"] += float(ratio.mean().item()) * n
            d["state_zero_rate"] += float((z_next.abs() <= float(eps)).float().mean().item()) * n
            d["active_dim_rate"] += float((z_next.abs() > float(eps)).float().mean().item()) * n
            d["activation_zero_rate"] += float(act_zero.mean().item()) * n
            d["origin_norm"] += float(origin_norm.mean().item()) * n
            d["gate_mean"] += float(info["gate_mean"].mean().item()) * n
            d["gate_active"] += float(info["gate_active"].mean().item()) * n
            d["alpha_entropy"] += float(info["alpha_entropy"].mean().item()) * n
            d["alpha_eff"] += float(info["alpha_eff"].mean().item()) * n

            z = z_next

        final_norm_sum += float(z.norm(dim=-1).mean().item()) * n
        final_zero_sum += float((z.abs() <= float(eps)).float().mean().item()) * n
        total_n += n

    per_layer: List[Dict[str, float]] = []
    for d in sums:
        n = max(float(d.pop("n")), 1.0)
        per_layer.append({k: float(v) / n for k, v in d.items()})

    def mean_key(name: str) -> float:
        vals = [float(d[name]) for d in per_layer]
        return float(sum(vals) / max(len(vals), 1))

    return {
        "per_layer": per_layer,
        "final_state_norm": float(final_norm_sum / max(total_n, 1.0)),
        "final_state_zero_rate": float(final_zero_sum / max(total_n, 1.0)),
        "mean_state_norm": mean_key("state_norm"),
        "mean_move_norm": mean_key("move_norm"),
        "mean_move_state_ratio": mean_key("move_state_ratio"),
        "mean_state_zero_rate": mean_key("state_zero_rate"),
        "mean_active_dim_rate": mean_key("active_dim_rate"),
        "mean_activation_zero_rate": mean_key("activation_zero_rate"),
    }


# ============================================================
# CSV schema
# ============================================================


BASE_FIELDS = [
    "status", "run_id", "seed", "activation_mode", "layers",
    "extra_dims", "max_children", "min_children", "input_dim", "state_dim", "num_classes",
    "params", "epochs", "lr", "batch_size", "init_samples", "kmeans_iters", "init_uses_activation",
    "best_epoch", "best_val_acc", "best_val_f1", "final_val_acc", "final_val_f1", "test_acc", "test_f1",
    "init_time_sec", "train_time_sec", "total_time_sec",
    "selected_k_by_layer", "total_children", "error",
    "diag_mean_state_norm", "diag_final_state_norm", "diag_mean_move_norm", "diag_mean_move_state_ratio",
    "diag_mean_state_zero_rate", "diag_final_state_zero_rate", "diag_mean_active_dim_rate", "diag_mean_activation_zero_rate",
]

LAYER_DIAG_KEYS = [
    "selected_k",
    "pre_state_norm",
    "mid_state_norm",
    "state_norm",
    "move_norm",
    "move_state_ratio",
    "state_zero_rate",
    "active_dim_rate",
    "activation_zero_rate",
    "origin_norm",
    "gate_mean",
    "gate_active",
    "alpha_entropy",
    "alpha_eff",
]

MAX_REPORTED_LAYERS = 8
RESULT_FIELDS = BASE_FIELDS + [f"layer{i}_{k}" for i in range(1, MAX_REPORTED_LAYERS + 1) for k in LAYER_DIAG_KEYS]

SUMMARY_FIELDS = [
    "activation_mode", "layers", "best_val_acc", "best_val_f1", "test_acc", "test_f1",
    "params", "total_children", "selected_k_by_layer",
    "diag_mean_move_norm", "diag_mean_move_state_ratio", "diag_mean_state_zero_rate", "diag_mean_activation_zero_rate",
]


def append_csv(path: Path, row: Dict[str, object], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != list(fields):
            raise RuntimeError(
                f"Existing CSV header does not match this experiment schema: {path}. "
                "Rename the old file, delete it, or choose another --out-dir."
            )
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("status") == "ok"}


def write_summary(results_csv: Path, summary_csv: Path, best_csv: Path) -> None:
    if not results_csv.exists():
        return

    rows: List[Dict[str, str]] = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                rows.append(row)

    rows.sort(key=lambda r: (r.get("activation_mode", ""), int(r.get("layers") or 0)))
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

    best_rows: List[Dict[str, str]] = []
    for mode in sorted({r.get("activation_mode", "") for r in rows}):
        candidates = [r for r in rows if r.get("activation_mode") == mode]
        if candidates:
            best_rows.append(max(candidates, key=lambda r: float(r.get("best_val_f1") or 0.0)))
    with best_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in best_rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


# ============================================================
# Experiment runner
# ============================================================


def make_model(
    data: DataBundle,
    *,
    layers: int,
    activation_mode: str,
    extra_dims: int,
    max_children: int,
    sigma_init: float,
    charge_init: float,
    step_cap: float,
    init_uses_activation: bool,
) -> ActivationAATField:
    cfg = AATFieldConfig(
        input_dim=int(data.input_dim),
        extra_dims=int(extra_dims),
        num_classes=int(data.num_classes),
        layers=int(layers),
        max_children=int(max_children),
        sigma_init=float(sigma_init),
        charge_init=float(charge_init),
        step_cap=float(step_cap),
    )
    return ActivationAATField(cfg, activation_mode=activation_mode, init_uses_activation=init_uses_activation)


def run_id_for(mode: str, layers: int, seed: int) -> str:
    return f"airline__act_{mode}__L{int(layers)}__seed{int(seed)}"


def fit_one(
    *,
    data: DataBundle,
    activation_mode: str,
    layers: int,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    init_samples: int,
    min_children: int,
    kmeans_iters: int,
    extra_dims: int,
    max_children: int,
    sigma_init: float,
    charge_init: float,
    step_cap: float,
    weight_decay: float,
    best_metric: str,
    diag_samples: int,
    save_checkpoints: bool,
    checkpoint_dir: Path,
    init_uses_activation: bool,
) -> Dict[str, object]:
    set_seed(seed)
    total_start = time.time()
    model = make_model(
        data,
        layers=layers,
        activation_mode=activation_mode,
        extra_dims=extra_dims,
        max_children=max_children,
        sigma_init=sigma_init,
        charge_init=charge_init,
        step_cap=step_cap,
        init_uses_activation=init_uses_activation,
    ).to(device)

    init_start = time.time()
    model.initialize(
        data.x_train.to(device),
        data.y_train.to(device),
        samples=int(init_samples),
        min_children=int(min_children),
        kmeans_iters=int(kmeans_iters),
    )
    init_time = time.time() - init_start

    train_loader, val_loader, test_loader = make_loaders(data, batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    best_state = clone_state_dict_cpu(model)
    best_val_acc, best_val_f1, best_epoch = -1.0, -1.0, 0
    train_start = time.time()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_acc, val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)
        score = val_f1 if str(best_metric).lower() == "val_f1" else val_acc
        best_score = best_val_f1 if str(best_metric).lower() == "val_f1" else best_val_acc
        if score > best_score:
            best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epoch
            best_state = clone_state_dict_cpu(model)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
            print(
                f"{run_id_for(activation_mode, layers, seed)} epoch={epoch:03d} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                f"best_acc={best_val_acc:.4f} best_f1={best_val_f1:.4f}@{best_epoch}",
                flush=True,
            )

    train_time = time.time() - train_start
    final_val_acc, final_val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)
    load_state_dict_to_device(model, best_state, device)
    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)

    diag_x = data.x_val
    if int(diag_samples) > 0 and data.x_val.shape[0] > int(diag_samples):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 2026)
        idx = torch.randperm(data.x_val.shape[0], generator=gen)[: int(diag_samples)]
        diag_x = data.x_val[idx]
    diag = collect_activation_diagnostics(model, diag_x, device=device, batch_size=int(batch_size) * 2)

    selected = model.selected_children_by_layer()
    selected_flat = [int(counts[0]) if counts else 0 for counts in selected]

    row: Dict[str, object] = {
        "status": "ok",
        "run_id": run_id_for(activation_mode, layers, seed),
        "seed": int(seed),
        "activation_mode": str(activation_mode),
        "layers": int(layers),
        "extra_dims": int(extra_dims),
        "max_children": int(max_children),
        "min_children": int(min_children),
        "input_dim": int(data.input_dim),
        "state_dim": int(data.input_dim + extra_dims),
        "num_classes": int(data.num_classes),
        "params": int(count_parameters(model)),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "init_samples": int(init_samples),
        "kmeans_iters": int(kmeans_iters),
        "init_uses_activation": bool(init_uses_activation),
        "best_epoch": int(best_epoch),
        "best_val_acc": round(float(best_val_acc), 6),
        "best_val_f1": round(float(best_val_f1), 6),
        "final_val_acc": round(float(final_val_acc), 6),
        "final_val_f1": round(float(final_val_f1), 6),
        "test_acc": round(float(test_acc), 6),
        "test_f1": round(float(test_f1), 6),
        "init_time_sec": round(float(init_time), 3),
        "train_time_sec": round(float(train_time), 3),
        "total_time_sec": round(float(time.time() - total_start), 3),
        "selected_k_by_layer": json.dumps(selected_flat),
        "total_children": int(model.total_children()),
        "error": "",
        "diag_mean_state_norm": round(float(diag["mean_state_norm"]), 6),
        "diag_final_state_norm": round(float(diag["final_state_norm"]), 6),
        "diag_mean_move_norm": round(float(diag["mean_move_norm"]), 6),
        "diag_mean_move_state_ratio": round(float(diag["mean_move_state_ratio"]), 6),
        "diag_mean_state_zero_rate": round(float(diag["mean_state_zero_rate"]), 6),
        "diag_final_state_zero_rate": round(float(diag["final_state_zero_rate"]), 6),
        "diag_mean_active_dim_rate": round(float(diag["mean_active_dim_rate"]), 6),
        "diag_mean_activation_zero_rate": round(float(diag["mean_activation_zero_rate"]), 6),
    }

    per_layer = diag["per_layer"]
    assert isinstance(per_layer, list)
    for i in range(1, MAX_REPORTED_LAYERS + 1):
        if i <= len(per_layer):
            d = per_layer[i - 1]
            row[f"layer{i}_selected_k"] = selected_flat[i - 1]
            for key in LAYER_DIAG_KEYS:
                if key == "selected_k":
                    continue
                row[f"layer{i}_{key}"] = round(float(d.get(key, 0.0)), 6)
        else:
            for key in LAYER_DIAG_KEYS:
                row[f"layer{i}_{key}"] = ""

    if save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{row['run_id']}.pt"
        torch.save(
            {
                "format": "AATFieldActivationExperimentCheckpoint",
                "activation_mode": str(activation_mode),
                "init_uses_activation": bool(init_uses_activation),
                "config": model.config_dict(),
                "state_dict": model.state_dict(),
                "selected_children": model.selected_children_by_layer(),
                "metadata": {k: row.get(k) for k in BASE_FIELDS if k in row},
            },
            ckpt_path,
        )
        row["checkpoint"] = str(ckpt_path)

    return {k: row.get(k, "") for k in RESULT_FIELDS}


def parse_layers(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(dict.fromkeys(out))
    if not out:
        raise ValueError("empty --layers")
    return out


def parse_modes(s: str) -> List[str]:
    modes = [p.strip().lower() for p in str(s).split(",") if p.strip()]
    for m in modes:
        if m not in ActivationAATField.VALID_MODES:
            raise ValueError(f"unknown activation mode {m!r}; valid={sorted(ActivationAATField.VALID_MODES)}")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/AirlineSatisfaction")
    parser.add_argument("--out-dir", default="./airline_activation_depth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Core experiment grid.
    parser.add_argument("--activation-modes", default="none,post_relu,parent_relu")
    parser.add_argument("--layers", default="1-12", help="Examples: '1-8' or '1,2,4,6,8'.")
    parser.add_argument("--seed", type=int, default=0)

    # Keep these defaults aligned with the previous Airline script unless changed explicitly.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--extra-dims", type=int, default=2)
    parser.add_argument("--max-children", type=int, default=100)
    parser.add_argument("--min-children", type=int, default=2)
    parser.add_argument("--init-samples", type=int, default=8192)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--sigma-init", type=float, default=0.75)
    parser.add_argument("--charge-init", type=float, default=0.08)
    parser.add_argument("--step-cap", type=float, default=1.0)

    # Diagnostics / execution.
    parser.add_argument("--best-metric", choices=["val_acc", "val_f1"], default="val_f1")
    parser.add_argument("--diag-samples", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N specs after grid construction.")
    parser.add_argument("--fresh", action="store_true", help="Do not skip runs already present in results.csv.")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--init-without-activation", action="store_true", help="Debug only: initialize deeper layers on raw z+move states.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    best_csv = out_dir / "best_by_activation.csv"
    checkpoint_dir = out_dir / "checkpoints"

    device = torch.device(args.device)
    data = airline_bundle(train_csv, test_csv, val_ratio=float(args.val_ratio), seed=123)

    modes = parse_modes(args.activation_modes)
    layer_values = parse_layers(args.layers)
    specs = [(m, L) for m in modes for L in layer_values]
    if int(args.limit) > 0:
        specs = specs[: int(args.limit)]

    done = set() if bool(args.fresh) else load_done(results_csv)
    print(
        f"device={device} runs={len(specs)} done={len(done)} out={out_dir.resolve()} "
        f"best_metric={args.best_metric}",
        flush=True,
    )

    for idx, (mode, L) in enumerate(specs, 1):
        rid = run_id_for(mode, L, args.seed)
        if rid in done:
            print(f"[{idx}/{len(specs)}] skip {rid}", flush=True)
            continue

        print(f"[{idx}/{len(specs)}] run {rid}", flush=True)
        try:
            row = fit_one(
                data=data,
                activation_mode=mode,
                layers=int(L),
                device=device,
                seed=int(args.seed),
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                init_samples=int(args.init_samples),
                min_children=int(args.min_children),
                kmeans_iters=int(args.kmeans_iters),
                extra_dims=int(args.extra_dims),
                max_children=int(args.max_children),
                sigma_init=float(args.sigma_init),
                charge_init=float(args.charge_init),
                step_cap=float(args.step_cap),
                weight_decay=float(args.weight_decay),
                best_metric=str(args.best_metric),
                diag_samples=int(args.diag_samples),
                save_checkpoints=bool(args.save_checkpoints),
                checkpoint_dir=checkpoint_dir,
                init_uses_activation=not bool(args.init_without_activation),
            )
        except Exception as e:
            row = {k: "" for k in RESULT_FIELDS}
            row.update({
                "status": "failed",
                "run_id": rid,
                "seed": int(args.seed),
                "activation_mode": mode,
                "layers": int(L),
                "extra_dims": int(args.extra_dims),
                "max_children": int(args.max_children),
                "min_children": int(args.min_children),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "init_samples": int(args.init_samples),
                "kmeans_iters": int(args.kmeans_iters),
                "init_uses_activation": not bool(args.init_without_activation),
                "error": repr(e),
            })
            print(f"FAILED {rid}: {e!r}", flush=True)

        append_csv(results_csv, row, RESULT_FIELDS)
        write_summary(results_csv, summary_csv, best_csv)

    write_summary(results_csv, summary_csv, best_csv)
    print(f"done: {results_csv}", flush=True)
    print(f"summary: {summary_csv}", flush=True)
    print(f"best_by_activation: {best_csv}", flush=True)


if __name__ == "__main__":
    main()
