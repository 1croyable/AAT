# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import csv
import itertools
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
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.parent.name.lower() == "experiments" else SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aatfield import AATField, AATFieldConfig  # noqa: E402
from aatfield.initialize import boundary_weights, supervised_fisher_score, weighted_kmeans  # noqa: E402
from aatfield.utils import count_parameters, pairwise_dist2  # noqa: E402


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


RESULT_FIELDS = [
    "status", "run_id", "kind", "seed", "k_combo", "k1", "k2", "k3",
    "layers", "extra_dims", "max_children", "input_dim", "state_dim", "num_classes",
    "params", "epochs", "lr", "batch_size", "init_samples", "kmeans_iters",
    "best_epoch", "best_val_acc", "best_val_f1", "final_val_acc", "final_val_f1",
    "test_acc", "test_f1", "init_time_sec", "train_time_sec", "total_time_sec",
    "selected_children", "total_children", "error",
    "train_fisher_l0", "train_fisher_l1", "train_fisher_l2", "train_fisher_l3",
    "val_fisher_l0", "val_fisher_l1", "val_fisher_l2", "val_fisher_l3",
    "test_fisher_l0", "test_fisher_l1", "test_fisher_l2", "test_fisher_l3",
    "head_val_acc_l0", "head_val_acc_l1", "head_val_acc_l2", "head_val_acc_l3",
    "head_val_f1_l0", "head_val_f1_l1", "head_val_f1_l2", "head_val_f1_l3",
    "head_test_acc_l0", "head_test_acc_l1", "head_test_acc_l2", "head_test_acc_l3",
    "head_test_f1_l0", "head_test_f1_l1", "head_test_f1_l2", "head_test_f1_l3",
    "layer1_children", "layer2_children", "layer3_children",
    "layer1_sigma_mean", "layer2_sigma_mean", "layer3_sigma_mean",
    "layer1_charge_abs_mean", "layer2_charge_abs_mean", "layer3_charge_abs_mean",
    "layer1_parent_norm", "layer2_parent_norm", "layer3_parent_norm",
    "layer1_child_offset_norm", "layer2_child_offset_norm", "layer3_child_offset_norm",
    "layer1_val_move_norm", "layer2_val_move_norm", "layer3_val_move_norm",
    "layer1_val_gate_mean", "layer2_val_gate_mean", "layer3_val_gate_mean",
    "layer1_val_gate_active", "layer2_val_gate_active", "layer3_val_gate_active",
    "layer1_val_alpha_entropy", "layer2_val_alpha_entropy", "layer3_val_alpha_entropy",
    "layer1_val_alpha_eff", "layer2_val_alpha_eff", "layer3_val_alpha_eff",
]

SUMMARY_FIELDS = [
    "rank", "run_id", "kind", "k_combo", "best_val_acc", "best_val_f1", "test_acc", "test_f1",
    "best_epoch", "params", "total_children", "train_time_sec",
    "val_fisher_l0", "val_fisher_l1", "val_fisher_l2", "val_fisher_l3",
    "head_val_acc_l0", "head_val_acc_l1", "head_val_acc_l2", "head_val_acc_l3",
]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1).long().cpu()
    y = y.view(-1).long().cpu()
    scores = []
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


def make_loaders(data: DataBundle, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(TensorDataset(data.x_train, data.y_train), batch_size=int(batch_size), shuffle=True),
        DataLoader(TensorDataset(data.x_val, data.y_val), batch_size=int(batch_size) * 2, shuffle=False),
        DataLoader(TensorDataset(data.x_test, data.y_test), batch_size=int(batch_size) * 2, shuffle=False),
    )


@torch.no_grad()
def initialize_layer_fixed_k(layer, z: torch.Tensor, y: torch.Tensor, *, k: int, kmeans_iters: int) -> None:
    z = z.detach().float()
    y = y.detach().long().to(z.device)
    C = int(layer.num_classes)
    D = int(layer.state_dim)
    k = int(k)
    if k < 1:
        raise ValueError("fixed k must be >= 1")

    parents = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    global_mean = z.mean(dim=0)
    for c in range(C):
        pts = z[y == c]
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    child_centers = torch.zeros(C, k, D, device=z.device, dtype=z.dtype)
    for c in range(C):
        pts = z[y == c]
        if pts.shape[0] == 0:
            pts = z
        w = boundary_weights(pts, c, parents)
        centers, _ = weighted_kmeans(pts, w, k=min(k, int(pts.shape[0])), iters=int(kmeans_iters))
        if centers.shape[0] < k:
            repeat = torch.arange(k, device=z.device) % centers.shape[0]
            centers = centers.index_select(0, repeat)
        child_centers[c] = centers[:k]

    anchors = torch.cat([parents, child_centers.reshape(C * k, D)], dim=0)
    nearest = torch.sqrt(pairwise_dist2(z, anchors).min(dim=1).values + 1e-8)
    sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
    sigma = max(0.05, min(3.0, sigma))
    layer._materialize(parents, child_centers, sigma)


@torch.no_grad()
def initialize_fixed_k_stack(
    model: AATField,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    k_combo: Sequence[int],
    samples: int,
    kmeans_iters: int,
    seed: int,
) -> None:
    if len(k_combo) != len(model.layers):
        raise ValueError(f"k_combo length {len(k_combo)} != model layers {len(model.layers)}")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device=device, dtype=torch.long)

    if int(samples) > 0 and x.shape[0] > int(samples):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        idx = torch.randperm(x.shape[0], generator=gen, device=device)[: int(samples)]
        x = x[idx]
        y = y[idx]

    z = model.lift(x)
    for layer, k in zip(model.layers, k_combo):
        initialize_layer_fixed_k(layer, z, y, k=int(k), kmeans_iters=int(kmeans_iters))
        out = layer(z)
        z = out[0] if isinstance(out, tuple) else out

    if was_training:
        model.train()


@torch.no_grad()
def layer_forward_with_info(layer, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    C = int(layer.num_classes)
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
    gate_out = layer._child_gate(z, child_flat, sigma[C:])
    gate = gate_out[0] if isinstance(gate_out, tuple) else gate_out
    strength = torch.cat([base[:, :C], base[:, C:] * sigma[C:].view(1, -1) * gate], dim=1)

    diff = anchors.unsqueeze(0) - z.unsqueeze(1)
    move = ((strength / dist.clamp_min(1e-6)).unsqueeze(-1) * diff).sum(dim=1)

    raw_norm = move.norm(dim=-1, keepdim=True)
    cap = float(layer.cfg.step_cap)
    if cap > 0:
        capped = cap * torch.tanh(raw_norm / cap)
        move = move * (capped / raw_norm.clamp_min(1e-6))
        move_norm = capped.squeeze(-1)
    else:
        move_norm = raw_norm.squeeze(-1)

    entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=-1)
    z_next = z + move
    info = {
        "move_norm": float(move_norm.mean().item()),
        "gate_mean": float(gate.mean().item()) if gate.numel() else 0.0,
        "gate_active": float((gate > 0).float().mean().item()) if gate.numel() else 0.0,
        "alpha_entropy": float(entropy.mean().item()),
        "alpha_eff": float(torch.exp(entropy.mean()).item()),
    }
    return z_next, info


@torch.no_grad()
def collect_states_and_layer_info(
    model: AATField,
    x: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
    model.eval()
    depth = len(model.layers) + 1
    states: List[List[torch.Tensor]] = [[] for _ in range(depth)]
    info_sums: List[Dict[str, float]] = [
        {"move_norm": 0.0, "gate_mean": 0.0, "gate_active": 0.0, "alpha_entropy": 0.0, "alpha_eff": 0.0, "n": 0.0}
        for _ in model.layers
    ]

    for start in range(0, x.shape[0], int(batch_size)):
        xb = x[start:start + int(batch_size)].to(device)
        n = int(xb.shape[0])
        z = model.lift(xb)
        states[0].append(z.detach().cpu())
        for li, layer in enumerate(model.layers):
            z, info = layer_forward_with_info(layer, z)
            states[li + 1].append(z.detach().cpu())
            for key in ["move_norm", "gate_mean", "gate_active", "alpha_entropy", "alpha_eff"]:
                info_sums[li][key] += float(info[key]) * n
            info_sums[li]["n"] += n

    merged_states = [torch.cat(parts, dim=0) for parts in states]
    merged_infos: List[Dict[str, float]] = []
    for d in info_sums:
        n = max(float(d.pop("n")), 1.0)
        merged_infos.append({k: float(v) / n for k, v in d.items()})
    return merged_states, merged_infos


@torch.no_grad()
def fisher_by_depth(states: Sequence[torch.Tensor], y: torch.Tensor, num_classes: int) -> List[float]:
    return [float(supervised_fisher_score(s, y, num_classes)) for s in states]


@torch.no_grad()
def head_metrics_by_depth(
    model: AATField,
    states: Sequence[torch.Tensor],
    y: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Tuple[List[float], List[float]]:
    accs: List[float] = []
    f1s: List[float] = []
    model.eval()
    for state in states:
        preds: List[torch.Tensor] = []
        for start in range(0, state.shape[0], int(batch_size)):
            z = state[start:start + int(batch_size)].to(device)
            logits = model.head(z)
            preds.append(logits.argmax(dim=1).cpu())
        pred = torch.cat(preds, dim=0)
        accs.append(float((pred == y.cpu()).float().mean().item()))
        f1s.append(float(macro_f1(pred, y.cpu(), num_classes)))
    return accs, f1s


def clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_to_device(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def make_model(data: DataBundle, *, layers: int, extra_dims: int, max_children: int) -> AATField:
    cfg = AATFieldConfig(
        input_dim=int(data.input_dim),
        extra_dims=int(extra_dims),
        num_classes=int(data.num_classes),
        layers=int(layers),
        max_children=int(max_children),
    )
    return AATField(cfg)


def run_name(kind: str, k_combo: Optional[Sequence[int]], seed: int) -> str:
    if kind == "autok":
        return f"airline__autok_L3_extra2__seed{int(seed)}"
    assert k_combo is not None
    return f"airline__fixed_k_{'-'.join(map(str, k_combo))}__seed{int(seed)}"


def layer_geometry_row(model: AATField) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for i, layer in enumerate(model.layers, 1):
        with torch.no_grad():
            row[f"layer{i}_children"] = int(layer.children_per_class)
            row[f"layer{i}_sigma_mean"] = round(float(layer.sigma().detach().mean().item()), 6)
            row[f"layer{i}_charge_abs_mean"] = round(float(layer.charge.detach().abs().mean().item()), 6)
            row[f"layer{i}_parent_norm"] = round(float(layer.parents.detach().norm(dim=-1).mean().item()), 6)
            row[f"layer{i}_child_offset_norm"] = round(float(layer.child_offsets.detach().norm(dim=-1).mean().item()), 6)
    return row


def fit_one(
    *,
    data: DataBundle,
    kind: str,
    k_combo: Optional[Tuple[int, int, int]],
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    init_samples: int,
    kmeans_iters: int,
    extra_dims: int,
    max_children: int,
    train_probe_samples: int,
) -> Dict[str, object]:
    set_seed(seed)
    total_start = time.time()
    model = make_model(data, layers=3, extra_dims=extra_dims, max_children=max_children).to(device)

    init_start = time.time()
    if kind == "autok":
        model.initialize(
            data.x_train.to(device),
            data.y_train.to(device),
            samples=int(init_samples),
            min_children=2,
            kmeans_iters=int(kmeans_iters),
            seed=int(seed),
        )
    else:
        assert k_combo is not None
        initialize_fixed_k_stack(
            model,
            data.x_train,
            data.y_train,
            k_combo=k_combo,
            samples=int(init_samples),
            kmeans_iters=int(kmeans_iters),
            seed=int(seed),
        )
    init_time = time.time() - init_start

    train_loader, val_loader, test_loader = make_loaders(data, batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_state = clone_state_dict_cpu(model)
    best_val, best_f1, best_epoch = -1.0, 0.0, 0
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
        if val_acc > best_val:
            best_val, best_f1, best_epoch = val_acc, val_f1, epoch
            best_state = clone_state_dict_cpu(model)

        if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
            print(
                f"{run_name(kind, k_combo, seed)} epoch={epoch:03d} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} best={best_val:.4f}@{best_epoch}",
                flush=True,
            )

    train_time = time.time() - train_start
    final_val_acc, final_val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)
    load_state_dict_to_device(model, best_state, device)
    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)

    probe_x_train = data.x_train
    probe_y_train = data.y_train
    if int(train_probe_samples) > 0 and data.x_train.shape[0] > int(train_probe_samples):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 909)
        idx = torch.randperm(data.x_train.shape[0], generator=gen)[: int(train_probe_samples)]
        probe_x_train = data.x_train[idx]
        probe_y_train = data.y_train[idx]

    train_states, _ = collect_states_and_layer_info(model, probe_x_train, device=device, batch_size=batch_size * 2)
    val_states, val_layer_infos = collect_states_and_layer_info(model, data.x_val, device=device, batch_size=batch_size * 2)
    test_states, _ = collect_states_and_layer_info(model, data.x_test, device=device, batch_size=batch_size * 2)

    train_fisher = fisher_by_depth(train_states, probe_y_train, data.num_classes)
    val_fisher = fisher_by_depth(val_states, data.y_val, data.num_classes)
    test_fisher = fisher_by_depth(test_states, data.y_test, data.num_classes)
    head_val_acc, head_val_f1 = head_metrics_by_depth(
        model, val_states, data.y_val, device=device, batch_size=batch_size * 2, num_classes=data.num_classes
    )
    head_test_acc, head_test_f1 = head_metrics_by_depth(
        model, test_states, data.y_test, device=device, batch_size=batch_size * 2, num_classes=data.num_classes
    )

    row: Dict[str, object] = {
        "status": "ok",
        "run_id": run_name(kind, k_combo, seed),
        "kind": kind,
        "seed": int(seed),
        "k_combo": "auto" if kind == "autok" else "-".join(map(str, k_combo or [])),
        "k1": "" if kind == "autok" else int(k_combo[0]),
        "k2": "" if kind == "autok" else int(k_combo[1]),
        "k3": "" if kind == "autok" else int(k_combo[2]),
        "layers": 3,
        "extra_dims": int(extra_dims),
        "max_children": int(max_children),
        "input_dim": int(data.input_dim),
        "state_dim": int(data.input_dim + extra_dims),
        "num_classes": int(data.num_classes),
        "params": int(count_parameters(model)),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "init_samples": int(init_samples),
        "kmeans_iters": int(kmeans_iters),
        "best_epoch": int(best_epoch),
        "best_val_acc": round(float(best_val), 6),
        "best_val_f1": round(float(best_f1), 6),
        "final_val_acc": round(float(final_val_acc), 6),
        "final_val_f1": round(float(final_val_f1), 6),
        "test_acc": round(float(test_acc), 6),
        "test_f1": round(float(test_f1), 6),
        "init_time_sec": round(float(init_time), 3),
        "train_time_sec": round(float(train_time), 3),
        "total_time_sec": round(float(time.time() - total_start), 3),
        "selected_children": json.dumps(model.selected_children_by_layer()),
        "total_children": int(model.total_children()),
        "error": "",
    }

    for i in range(4):
        row[f"train_fisher_l{i}"] = round(float(train_fisher[i]), 6)
        row[f"val_fisher_l{i}"] = round(float(val_fisher[i]), 6)
        row[f"test_fisher_l{i}"] = round(float(test_fisher[i]), 6)
        row[f"head_val_acc_l{i}"] = round(float(head_val_acc[i]), 6)
        row[f"head_val_f1_l{i}"] = round(float(head_val_f1[i]), 6)
        row[f"head_test_acc_l{i}"] = round(float(head_test_acc[i]), 6)
        row[f"head_test_f1_l{i}"] = round(float(head_test_f1[i]), 6)

    row.update(layer_geometry_row(model))
    for i, info in enumerate(val_layer_infos, 1):
        row[f"layer{i}_val_move_norm"] = round(float(info["move_norm"]), 6)
        row[f"layer{i}_val_gate_mean"] = round(float(info["gate_mean"]), 6)
        row[f"layer{i}_val_gate_active"] = round(float(info["gate_active"]), 6)
        row[f"layer{i}_val_alpha_entropy"] = round(float(info["alpha_entropy"]), 6)
        row[f"layer{i}_val_alpha_eff"] = round(float(info["alpha_eff"]), 6)

    return {k: row.get(k, "") for k in RESULT_FIELDS}


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != RESULT_FIELDS:
            raise RuntimeError(
                f"Existing CSV header does not match this experiment schema: {path}. "
                "Rename the old file or choose another --out-dir."
            )
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("status") == "ok"}


def write_summary(results_csv: Path, summary_csv: Path) -> None:
    if not results_csv.exists():
        return
    rows: List[Dict[str, str]] = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                rows.append(row)
    rows.sort(key=lambda r: float(r.get("best_val_acc") or 0.0), reverse=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            out = {k: row.get(k, "") for k in SUMMARY_FIELDS}
            out["rank"] = rank
            writer.writerow(out)


def make_specs(k_min: int, k_max: int, include_autok: bool) -> List[Tuple[str, Optional[Tuple[int, int, int]]]]:
    specs: List[Tuple[str, Optional[Tuple[int, int, int]]]] = []
    if include_autok:
        specs.append(("autok", None))
    for combo in itertools.product(range(int(k_min), int(k_max) + 1), repeat=3):
        specs.append(("fixed", tuple(map(int, combo))))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/AirlineSatisfaction")
    parser.add_argument("--out-dir", default="./airline")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--init-samples", type=int, default=8192)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--train-probe-samples", type=int, default=8192)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--extra-dims", type=int, default=2)
    parser.add_argument("--max-children", type=int, default=8)
    parser.add_argument("--no-autok", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
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
    device = torch.device(args.device)

    data = airline_bundle(train_csv, test_csv, val_ratio=float(args.val_ratio), seed=123)
    specs = make_specs(args.k_min, args.k_max, include_autok=not bool(args.no_autok))
    if int(args.limit) > 0:
        specs = specs[: int(args.limit)]

    done = set() if bool(args.fresh) else load_done(results_csv)
    print(f"device={device} runs={len(specs)} done={len(done)} out={out_dir.resolve()}", flush=True)

    for idx, (kind, combo) in enumerate(specs, 1):
        rid = run_name(kind, combo, args.seed)
        if rid in done:
            print(f"[{idx}/{len(specs)}] skip {rid}", flush=True)
            continue
        print(f"[{idx}/{len(specs)}] run {rid}", flush=True)
        try:
            row = fit_one(
                data=data,
                kind=kind,
                k_combo=combo,
                device=device,
                seed=int(args.seed),
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                init_samples=int(args.init_samples),
                kmeans_iters=int(args.kmeans_iters),
                extra_dims=int(args.extra_dims),
                max_children=int(args.max_children),
                train_probe_samples=int(args.train_probe_samples),
            )
        except Exception as e:
            row = {k: "" for k in RESULT_FIELDS}
            row.update({
                "status": "failed",
                "run_id": rid,
                "kind": kind,
                "seed": int(args.seed),
                "k_combo": "auto" if combo is None else "-".join(map(str, combo)),
                "k1": "" if combo is None else int(combo[0]),
                "k2": "" if combo is None else int(combo[1]),
                "k3": "" if combo is None else int(combo[2]),
                "layers": 3,
                "extra_dims": int(args.extra_dims),
                "max_children": int(args.max_children),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "init_samples": int(args.init_samples),
                "kmeans_iters": int(args.kmeans_iters),
                "error": repr(e),
            })
            print(f"FAILED {rid}: {e!r}", flush=True)
        append_csv(results_csv, row)
        write_summary(results_csv, summary_csv)

    write_summary(results_csv, summary_csv)
    print(f"done: {results_csv}", flush=True)
    print(f"summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
