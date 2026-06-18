# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aatfield import AATField, AATFieldConfig
from aatfield.initialize import boundary_weights, weighted_kmeans
from aatfield.utils import count_parameters, pairwise_dist2


# ============================================================
# Fixed experiment config
# ============================================================

OUT_DIR = Path("./checkerboard3d_l2_child_grid")
GRID_SIZE = 4
N_TRAIN = 4096
N_VAL = 2048
N_TEST = 2048

LAYERS = 2
EXTRA_DIMS = 3
NUM_CLASSES = 2

K_LIST = [34, 37, 40, 43, 46, 31, 28, 25, 22, 19, 16, 13, 10]
AUTO_MAX_CHILDREN = 100

EPOCHS = 300
LR = 2e-3
BATCH_SIZE = 256
INIT_SAMPLES = 4096
KMEANS_ITERS = 8
SEED = 0


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


@dataclass
class RunSpec:
    model: str
    mode: str
    seed: int
    layers: int
    k_combo: Tuple[int, ...]
    max_children: int
    extra_dims: int
    epochs: int
    lr: float
    batch_size: int
    init_samples: int
    kmeans_iters: int


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_checkerboard_3d(n: int, grid_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    x = torch.rand(int(n), 3, generator=gen)
    cells = torch.floor(x * int(grid_size)).long().clamp(max=int(grid_size) - 1)
    y = (cells.sum(dim=1) % 2).long()

    idx = torch.randperm(int(n), generator=gen)
    return x[idx], y[idx]


def checkerboard_bundle(seed: int) -> DataBundle:
    x, y = make_checkerboard_3d(N_TRAIN + N_VAL + N_TEST, GRID_SIZE, seed)
    return DataBundle(
        name=f"checkerboard3d_g{GRID_SIZE}",
        input_dim=3,
        num_classes=NUM_CLASSES,
        x_train=x[:N_TRAIN],
        y_train=y[:N_TRAIN],
        x_val=x[N_TRAIN:N_TRAIN + N_VAL],
        y_val=y[N_TRAIN:N_TRAIN + N_VAL],
        x_test=x[N_TRAIN + N_VAL:],
        y_test=y[N_TRAIN + N_VAL:],
    )


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


@torch.no_grad()
def supervised_fisher(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    z = z.detach().float()
    y = y.detach().long().to(z.device)

    if z.numel() == 0:
        return 0.0

    global_mu = z.mean(dim=0)
    W = z.new_tensor(0.0)
    B = z.new_tensor(0.0)
    valid = 0

    for c in range(int(num_classes)):
        pts = z[y == c]
        if pts.shape[0] == 0:
            continue
        valid += 1
        mu = pts.mean(dim=0)
        W = W + (pts - mu).square().sum()
        B = B + pts.shape[0] * (mu - global_mu).square().sum()

    n = max(int(z.shape[0]), 1)
    if valid <= 1 or n <= valid:
        return 0.0

    score = (B / max(valid - 1, 1)) / (W / max(n - valid, 1)).clamp_min(1e-12)
    return float(score.item())


@torch.no_grad()
def initialize_layer_fixed_k(layer, z: torch.Tensor, y: torch.Tensor, k: int, kmeans_iters: int) -> None:
    z = z.detach().float()
    y = y.detach().long().to(z.device)

    C = layer.num_classes
    D = layer.state_dim
    k = int(max(1, k))

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
        centers, _ = weighted_kmeans(pts, w, k=k, iters=int(kmeans_iters))
        child_centers[c] = centers[:k]

    full_anchors = torch.cat([parents, child_centers.reshape(C * k, D)], dim=0)
    nearest = torch.sqrt(pairwise_dist2(z, full_anchors).min(dim=1).values + 1e-8)
    sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
    sigma = max(0.05, min(3.0, sigma))

    layer._materialize(parents, child_centers, sigma)


@torch.no_grad()
def initialize_fixed_stack(model: AATField, x: torch.Tensor, y: torch.Tensor, k_combo: Tuple[int, ...], kmeans_iters: int) -> None:
    z = model.lift(x)

    for i, k in enumerate(k_combo):
        initialize_layer_fixed_k(model.layers[i], z, y, int(k), kmeans_iters)
        z = model.layers[i](z)


@torch.no_grad()
def layer_states(model: AATField, x: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
    model.eval()
    z = model.lift(x.to(device))
    states = [z.detach()]

    for layer in model.layers:
        z = layer(z)
        states.append(z.detach())

    return states


@torch.no_grad()
def layer_diagnostics(model: AATField, x: torch.Tensor, device: torch.device, max_samples: int = 4096) -> Dict[str, float]:
    model.eval()
    xb = x[: int(max_samples)].to(device)
    z = model.lift(xb)

    out: Dict[str, float] = {}

    for li, layer in enumerate(model.layers, 1):
        C = layer.num_classes
        K = layer.children_per_class
        child_flat = layer.child_anchors().reshape(layer.child_n, layer.state_dim)
        anchors = torch.cat([layer.parents, child_flat], dim=0)
        sigma = layer.sigma()

        dist2 = pairwise_dist2(z, anchors)
        logits = -dist2 / (2.0 * sigma.view(1, -1).square() + 1e-8)
        alpha = torch.softmax(logits, dim=-1)
        entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()

        axis = layer.child_offsets.reshape(C * K, layer.state_dim)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        s = ((z.unsqueeze(1) - child_flat.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=-1)
        s = s / sigma[C:].view(1, -1).clamp_min(1e-6)

        if layer.child_gate_bias is not None:
            s = s + layer.child_gate_bias.view(1, -1)

        gate = F.relu(s)

        z_next = layer(z)
        move = z_next - z

        top_idx = alpha.argmax(dim=1)
        parent_top_rate = (top_idx < C).float().mean()
        child_top_rate = 1.0 - parent_top_rate

        prefix = f"layer{li}"
        out[f"{prefix}_children"] = float(K)
        out[f"{prefix}_sigma_mean"] = float(sigma.mean().item())
        out[f"{prefix}_gate_active"] = float((gate > 0).float().mean().item())
        out[f"{prefix}_move_norm"] = float(move.norm(dim=-1).mean().item())
        out[f"{prefix}_alpha_eff"] = float(torch.exp(entropy).item())
        out[f"{prefix}_parent_top_rate"] = float(parent_top_rate.item())
        out[f"{prefix}_child_top_rate"] = float(child_top_rate.item())

        z = z_next

    return out


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())

    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            old = next(csv.reader(f), None)
        if old:
            fields = old
            row = {k: row.get(k, "") for k in fields}

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_done(path: Path) -> set:
    if not path.exists():
        return set()

    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("status") == "ok"}


def write_summary(results_csv: Path, summary_csv: Path) -> None:
    if not results_csv.exists():
        return

    rows = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                rows.append(row)

    rows.sort(key=lambda r: float(r.get("best_val_acc", 0.0)), reverse=True)

    fields = [
        "rank", "run_id", "model", "mode", "k_combo", "selected_children", "params",
        "best_epoch", "best_val_acc", "best_val_f1", "test_acc", "test_f1",
        "final_val_acc", "train_time_sec",
        "val_fisher_l0", "val_fisher_l1", "val_fisher_l2",
        "layer1_children", "layer2_children",
        "layer1_gate_active", "layer2_gate_active",
        "layer1_move_norm", "layer2_move_norm",
    ]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            out = {k: row.get(k, "") for k in fields}
            out["rank"] = i
            writer.writerow(out)


def run_id(data: DataBundle, spec: RunSpec) -> str:
    if spec.mode == "auto":
        return f"{data.name}__L2_extra{spec.extra_dims}__autok_max{spec.max_children}__seed{spec.seed}"

    k_text = "-".join(str(k) for k in spec.k_combo)
    return f"{data.name}__L2_extra{spec.extra_dims}__fixed_k_{k_text}__seed{spec.seed}"


def make_model(data: DataBundle, spec: RunSpec) -> AATField:
    cfg = AATFieldConfig(
        input_dim=data.input_dim,
        extra_dims=int(spec.extra_dims),
        num_classes=data.num_classes,
        layers=int(spec.layers),
        max_children=int(spec.max_children),
    )
    return AATField(cfg)


def train_one(spec: RunSpec, data: DataBundle, device: torch.device) -> Dict[str, object]:
    set_seed(spec.seed)
    model = make_model(data, spec).to(device)

    if spec.mode == "auto":
        model.initialize(
            data.x_train.to(device),
            data.y_train.to(device),
            samples=spec.init_samples,
            min_children=2,
            kmeans_iters=spec.kmeans_iters,
            seed=spec.seed,
        )
    else:
        initialize_fixed_stack(
            model,
            data.x_train.to(device),
            data.y_train.to(device),
            spec.k_combo,
            spec.kmeans_iters,
        )

    train_loader = DataLoader(TensorDataset(data.x_train, data.y_train), batch_size=spec.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data.x_val, data.y_val), batch_size=spec.batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(data.x_test, data.y_test), batch_size=spec.batch_size * 2, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(spec.lr), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_state = deepcopy(model.state_dict())
    best_val = -1.0
    best_f1 = 0.0
    best_epoch = 0
    rid = run_id(data, spec)
    start = time.time()

    for epoch in range(1, int(spec.epochs) + 1):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_acc, val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)

        if val_acc > best_val:
            best_val = val_acc
            best_f1 = val_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if epoch == 1 or epoch % 20 == 0 or epoch == spec.epochs:
            print(
                f"{rid} epoch={epoch:03d} val_acc={val_acc:.4f} "
                f"val_f1={val_f1:.4f} best={best_val:.4f}@{best_epoch}",
                flush=True,
            )

    final_val_acc, final_val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)

    model.load_state_dict(best_state)
    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)

    train_states = layer_states(model, data.x_train, device)
    val_states = layer_states(model, data.x_val, device)
    test_states = layer_states(model, data.x_test, device)

    diag = layer_diagnostics(model, data.x_val, device)

    selected_children = model.selected_children_by_layer()
    k_combo_text = "auto" if spec.mode == "auto" else "-".join(str(k) for k in spec.k_combo)

    row: Dict[str, object] = {
        "status": "ok",
        "run_id": rid,
        "dataset": data.name,
        "model": spec.model,
        "mode": spec.mode,
        "seed": spec.seed,
        "layers": spec.layers,
        "k_combo": k_combo_text,
        "max_children": spec.max_children,
        "extra_dims": spec.extra_dims,
        "input_dim": data.input_dim,
        "state_dim": data.input_dim + spec.extra_dims,
        "num_classes": data.num_classes,
        "epochs": spec.epochs,
        "lr": spec.lr,
        "batch_size": spec.batch_size,
        "init_samples": spec.init_samples,
        "kmeans_iters": spec.kmeans_iters,
        "params": int(count_parameters(model)),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 6),
        "best_val_f1": round(best_f1, 6),
        "final_val_acc": round(final_val_acc, 6),
        "final_val_f1": round(final_val_f1, 6),
        "test_acc": round(test_acc, 6),
        "test_f1": round(test_f1, 6),
        "train_time_sec": round(time.time() - start, 3),
        "selected_children": json.dumps(selected_children),
        "total_children": model.total_children(),
    }

    for i, z in enumerate(train_states):
        row[f"train_fisher_l{i}"] = round(supervised_fisher(z, data.y_train.to(device), data.num_classes), 6)
    for i, z in enumerate(val_states):
        row[f"val_fisher_l{i}"] = round(supervised_fisher(z, data.y_val.to(device), data.num_classes), 6)
    for i, z in enumerate(test_states):
        row[f"test_fisher_l{i}"] = round(supervised_fisher(z, data.y_test.to(device), data.num_classes), 6)

    row.update({k: round(v, 6) for k, v in diag.items()})
    return row


def build_specs() -> List[RunSpec]:
    specs: List[RunSpec] = []

    specs.append(RunSpec(
        model=f"AAT-L2-extra{EXTRA_DIMS}-AutoK-max{AUTO_MAX_CHILDREN}",
        mode="auto",
        seed=SEED,
        layers=LAYERS,
        k_combo=(),
        max_children=AUTO_MAX_CHILDREN,
        extra_dims=EXTRA_DIMS,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        init_samples=INIT_SAMPLES,
        kmeans_iters=KMEANS_ITERS,
    ))

    for k1 in K_LIST:
        for k2 in K_LIST:
            specs.append(RunSpec(
                model=f"AAT-L2-extra{EXTRA_DIMS}-K{k1}-{k2}",
                mode="fixed",
                seed=SEED,
                layers=LAYERS,
                k_combo=(int(k1), int(k2)),
                max_children=max(int(k1), int(k2)),
                extra_dims=EXTRA_DIMS,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                init_samples=INIT_SAMPLES,
                kmeans_iters=KMEANS_ITERS,
            ))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = OUT_DIR
    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"

    device = torch.device(args.device)
    data = checkerboard_bundle(SEED)

    specs = build_specs()
    if args.limit > 0:
        specs = specs[: int(args.limit)]

    done = load_done(results_csv)

    print(
        f"dataset={data.name} train={len(data.y_train)} val={len(data.y_val)} test={len(data.y_test)} "
        f"device={device} layers={LAYERS} K_LIST={K_LIST} runs={len(specs)} done={len(done)} out={out_dir}",
        flush=True,
    )

    for i, spec in enumerate(specs, 1):
        rid = run_id(data, spec)

        if rid in done:
            print(f"[{i}/{len(specs)}] skip {rid}", flush=True)
            continue

        print(f"[{i}/{len(specs)}] run {rid}", flush=True)

        try:
            row = train_one(spec, data, device)
        except Exception as e:
            row = {
                "status": "failed",
                "run_id": rid,
                "dataset": data.name,
                "model": spec.model,
                "mode": spec.mode,
                "seed": spec.seed,
                "k_combo": "auto" if spec.mode == "auto" else "-".join(str(k) for k in spec.k_combo),
                "max_children": spec.max_children,
                "error": repr(e),
            }
            print(f"FAILED {rid}: {e}", flush=True)

        append_csv(results_csv, row)
        write_summary(results_csv, summary_csv)

    write_summary(results_csv, summary_csv)
    print(f"done: {results_csv}", flush=True)
    print(f"summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
