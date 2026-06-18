# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aatfield import AATField, AATFieldConfig
from aatfield.utils import count_parameters, pairwise_dist2


# ============================================================
# Fixed experiment config
# ============================================================

OUT_DIR = Path("./mnist_l1_x1p5_child_sweep_autok")

# Edit these two like the airline script.
FIXED_K_MIN = 10
FIXED_K_MAX = 40
AUTO_MAX_CHILDREN = 100

LAYERS = 1
STATE_MODE = "x1p5"       # x1p5 means state_dim = 1.5 * input_dim
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
INIT_SAMPLES = 8192
KMEANS_ITERS = 6
SEED = 0
TRAIN_LIMIT = 60000
VAL_SIZE = 10000


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
    fixed_k: int
    max_children: int
    state_mode: str
    extra_dims: int
    epochs: int
    lr: float
    batch_size: int
    eval_batch_size: int
    init_samples: int
    kmeans_iters: int


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def mnist_bundle(data_root: Path, *, train_limit: int, val_size: int) -> DataBundle:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("MNIST requires torchvision.") from e

    tr = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transforms.ToTensor())
    te = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transforms.ToTensor())

    x_all = torch.stack([tr[i][0] for i in range(len(tr))]).flatten(1)
    y_all = torch.tensor([int(tr[i][1]) for i in range(len(tr))], dtype=torch.long)
    x_test = torch.stack([te[i][0] for i in range(len(te))]).flatten(1)
    y_test = torch.tensor([int(te[i][1]) for i in range(len(te))], dtype=torch.long)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)

    limit = min(int(train_limit), x_all.shape[0])
    val_n = min(int(val_size), max(limit // 6, 1))
    idx = torch.randperm(x_all.shape[0], generator=gen)[:limit]

    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    x_train = x_all[train_idx].contiguous()
    y_train = y_all[train_idx].contiguous()
    x_val = x_all[val_idx].contiguous()
    y_val = y_all[val_idx].contiguous()

    print(
        f"MNIST: input_dim=784 train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}",
        flush=True,
    )

    return DataBundle(
        name="mnist",
        input_dim=784,
        num_classes=10,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test.contiguous(),
        y_test=y_test.contiguous(),
    )


def extra_dims_for_state(state_mode: str, input_dim: int) -> int:
    if state_mode == "x1":
        return 0
    if state_mode == "x1p5":
        return int(round(int(input_dim) * 0.5))
    if state_mode == "x2":
        return int(input_dim)
    if state_mode == "x3":
        return int(input_dim) * 2
    if state_mode == "x4":
        return int(input_dim) * 3
    raise ValueError(f"unknown state_mode: {state_mode}")


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
        logits = model(xb.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu())
        ys.append(yb.cpu())

    pred = torch.cat(preds)
    y = torch.cat(ys)
    return float((pred == y).float().mean().item()), macro_f1(pred, y, num_classes)


@torch.no_grad()
def layer_diagnostics(model: AATField, x: torch.Tensor, device: torch.device, max_samples: int = 1024) -> Dict[str, float]:
    model.eval()
    layer = model.layers[0]
    xb = x[: int(max_samples)].to(device)
    z0 = model.lift(xb)

    C = layer.num_classes
    K = layer.children_per_class
    child_flat = layer.child_anchors().reshape(layer.child_n, layer.state_dim)
    anchors = torch.cat([layer.parents, child_flat], dim=0)
    sigma = layer.sigma()

    dist2 = pairwise_dist2(z0, anchors)
    logits = -dist2 / (2.0 * sigma.view(1, -1).square() + 1e-8)
    alpha = torch.softmax(logits, dim=-1)
    entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()

    axis = layer.child_offsets.reshape(C * K, layer.state_dim)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    s = ((z0.unsqueeze(1) - child_flat.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=-1)
    s = s / sigma[C:].view(1, -1).clamp_min(1e-6)
    if layer.child_gate_bias is not None:
        s = s + layer.child_gate_bias.view(1, -1)
    gate = torch.relu(s)

    # Avoid doing full huge diagnostics on MNIST; only use a small probe.
    z1 = layer(z0)
    move = z1 - z0

    return {
        "layer1_children": float(K),
        "layer1_sigma_mean": float(sigma.mean().item()),
        "layer1_gate_active": float((gate > 0).float().mean().item()),
        "layer1_move_norm": float(move.norm(dim=-1).mean().item()),
        "layer1_alpha_eff": float(torch.exp(entropy).item()),
    }


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
        "rank", "run_id", "model", "mode", "fixed_k", "selected_k", "selected_children",
        "total_children", "params", "best_epoch", "best_val_acc", "best_val_f1",
        "test_acc", "test_f1", "final_val_acc", "train_time_sec",
        "layer1_children", "layer1_sigma_mean", "layer1_gate_active", "layer1_move_norm", "layer1_alpha_eff",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            out = {k: row.get(k, "") for k in fields}
            out["rank"] = rank
            writer.writerow(out)


def run_id(data: DataBundle, spec: RunSpec) -> str:
    if spec.mode == "auto":
        return f"{data.name}__L1_{spec.state_mode}__autok_max{spec.max_children}__seed{spec.seed}"
    return f"{data.name}__L1_{spec.state_mode}__fixed_k{spec.fixed_k:02d}__seed{spec.seed}"


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

    model.initialize(
        data.x_train.to(device),
        data.y_train.to(device),
        samples=spec.init_samples,
        min_children=2 if spec.mode == "auto" else int(spec.fixed_k),
        kmeans_iters=spec.kmeans_iters,
        seed=spec.seed,
    )

    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=spec.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(data.x_val, data.y_val),
        batch_size=spec.eval_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        TensorDataset(data.x_test, data.y_test),
        batch_size=spec.eval_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

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
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
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

        if epoch == 1 or epoch % 5 == 0 or epoch == spec.epochs:
            print(
                f"{rid} epoch={epoch:03d} val_acc={val_acc:.4f} "
                f"val_f1={val_f1:.4f} best={best_val:.4f}@{best_epoch}",
                flush=True,
            )

    final_val_acc, final_val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)
    model.load_state_dict(best_state)
    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)

    diag = layer_diagnostics(model, data.x_val, device)
    selected_children = model.selected_children_by_layer()
    selected_k = int(selected_children[0][0]) if selected_children and selected_children[0] else 0

    row: Dict[str, object] = {
        "status": "ok",
        "run_id": rid,
        "dataset": data.name,
        "model": spec.model,
        "mode": spec.mode,
        "seed": spec.seed,
        "layers": spec.layers,
        "fixed_k": spec.fixed_k if spec.mode == "fixed" else "",
        "selected_k": selected_k,
        "max_children": spec.max_children,
        "state_mode": spec.state_mode,
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
    row.update({k: round(v, 6) for k, v in diag.items()})
    return row


def build_specs(data: DataBundle) -> List[RunSpec]:
    specs: List[RunSpec] = []
    extra_dims = extra_dims_for_state(STATE_MODE, data.input_dim)

    specs.append(RunSpec(
        model=f"AAT-L1-{STATE_MODE}-AutoK-max{AUTO_MAX_CHILDREN}",
        mode="auto",
        seed=SEED,
        layers=LAYERS,
        fixed_k=0,
        max_children=AUTO_MAX_CHILDREN,
        state_mode=STATE_MODE,
        extra_dims=extra_dims,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        init_samples=INIT_SAMPLES,
        kmeans_iters=KMEANS_ITERS,
    ))

    for k in range(int(FIXED_K_MIN), int(FIXED_K_MAX) + 1):
        specs.append(RunSpec(
            model=f"AAT-L1-{STATE_MODE}-K{k}",
            mode="fixed",
            seed=SEED,
            layers=LAYERS,
            fixed_k=int(k),
            max_children=int(k),
            state_mode=STATE_MODE,
            extra_dims=extra_dims,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            init_samples=INIT_SAMPLES,
            kmeans_iters=KMEANS_ITERS,
        ))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=TRAIN_LIMIT)
    parser.add_argument("--val-size", type=int, default=VAL_SIZE)
    args = parser.parse_args()

    out_dir = OUT_DIR
    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    device = torch.device(args.device)

    data = mnist_bundle(Path(args.data_root), train_limit=args.train_limit, val_size=args.val_size)
    specs = build_specs(data)
    if int(args.limit) > 0:
        specs = specs[: int(args.limit)]

    done = load_done(results_csv)

    print(
        f"dataset={data.name} input_dim={data.input_dim} state_dim={data.input_dim + extra_dims_for_state(STATE_MODE, data.input_dim)} "
        f"train={len(data.y_train)} val={len(data.y_val)} test={len(data.y_test)} "
        f"device={device} layers={LAYERS} state={STATE_MODE} "
        f"fixed_k={FIXED_K_MIN}..{FIXED_K_MAX} autok_max={AUTO_MAX_CHILDREN} "
        f"runs={len(specs)} done={len(done)} out={out_dir}",
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
                "fixed_k": spec.fixed_k if spec.mode == "fixed" else "",
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
