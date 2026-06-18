# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aatfield import AATField, AATFieldConfig
from aatfield.utils import count_parameters


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


def make_checkerboard_3d(
    n: int,
    grid_size: int,
    seed: int,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    x = torch.rand(int(n), 3, generator=gen)
    if float(noise_std) > 0.0:
        x = (x + torch.randn(x.shape, generator=gen) * float(noise_std)).clamp(0.0, 1.0)
    cells = torch.floor(x * int(grid_size)).long().clamp(max=int(grid_size) - 1)
    y = (cells.sum(dim=1) % 2).long()
    idx = torch.randperm(int(n), generator=gen)
    return x[idx], y[idx]


def checkerboard_bundle(
    *,
    grid_size: int,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    noise_std: float = 0.0,
) -> DataBundle:
    x, y = make_checkerboard_3d(n_train + n_val + n_test, grid_size, seed, noise_std)
    return DataBundle(
        name=f"checkerboard3d_g{grid_size}",
        input_dim=3,
        num_classes=2,
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_val=x[n_train:n_train + n_val],
        y_val=y[n_train:n_train + n_val],
        x_test=x[n_train + n_val:],
        y_test=y[n_train + n_val:],
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
def layer_states(model: AATField, x: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    xb = x.to(device)
    z0 = model.lift(xb)
    z1 = model.layers[0](z0)
    return z0.detach(), z1.detach()


@torch.no_grad()
def layer_diagnostics(model: AATField, x: torch.Tensor, device: torch.device, max_samples: int = 4096) -> Dict[str, float]:
    model.eval()
    layer = model.layers[0]
    xb = x[: int(max_samples)].to(device)
    z0 = model.lift(xb)

    C = layer.num_classes
    K = layer.children_per_class
    child_flat = layer.child_anchors().reshape(layer.child_n, layer.state_dim)
    anchors = torch.cat([layer.parents, child_flat], dim=0)
    sigma = layer.sigma()

    dist2 = (
        (z0 * z0).sum(dim=-1, keepdim=True)
        + (anchors * anchors).sum(dim=-1).view(1, -1)
        - 2.0 * (z0 @ anchors.t())
    ).clamp_min(0.0)
    logits = -dist2 / (2.0 * sigma.view(1, -1).square() + 1e-8)
    alpha = torch.softmax(logits, dim=-1)
    entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()

    axis = layer.child_offsets.reshape(C * K, layer.state_dim)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    s = ((z0.unsqueeze(1) - child_flat.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=-1)
    s = s / sigma[C:].view(1, -1).clamp_min(1e-6)
    if layer.child_gate_bias is not None:
        s = s + layer.child_gate_bias.view(1, -1)
    gate = F.relu(s)

    z1 = layer(z0)
    move = z1 - z0
    top_idx = alpha.argmax(dim=1)
    parent_top_rate = (top_idx < C).float().mean()
    child_top_rate = 1.0 - parent_top_rate

    return {
        "diag_sigma_mean": float(sigma.mean().item()),
        "diag_sigma_min": float(sigma.min().item()),
        "diag_sigma_max": float(sigma.max().item()),
        "diag_charge_abs_mean": float(layer.charge.detach().abs().mean().item()),
        "diag_parent_norm": float(layer.parents.detach().norm(dim=-1).mean().item()),
        "diag_child_offset_norm": float(layer.child_offsets.detach().norm(dim=-1).mean().item()),
        "diag_anchor_alpha_entropy": float(entropy.item()),
        "diag_anchor_alpha_eff": float(torch.exp(entropy).item()),
        "diag_parent_top_rate": float(parent_top_rate.item()),
        "diag_child_top_rate": float(child_top_rate.item()),
        "diag_gate_mean": float(gate.mean().item()),
        "diag_gate_active": float((gate > 0).float().mean().item()),
        "diag_move_norm": float(move.norm(dim=-1).mean().item()),
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
            if row.get("status") != "ok":
                continue
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("best_val_acc", 0.0)), reverse=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "rank", "run_id", "model", "mode", "fixed_k", "selected_k", "max_children",
            "params", "best_epoch", "best_val_acc", "best_val_f1", "test_acc", "test_f1",
            "final_val_acc", "train_time_sec", "val_fisher_after", "diag_gate_active", "diag_move_norm",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({k: row.get(k, "") for k in fields} | {"rank": i})


def run_id(data: DataBundle, spec: RunSpec) -> str:
    if spec.mode == "auto":
        return f"{data.name}__L1_extra{spec.extra_dims}__autok_max{spec.max_children}__seed{spec.seed}"
    return f"{data.name}__L1_extra{spec.extra_dims}__fixed_k{spec.fixed_k:02d}__seed{spec.seed}"


def make_model(data: DataBundle, spec: RunSpec) -> AATField:
    cfg = AATFieldConfig(
        input_dim=data.input_dim,
        extra_dims=int(spec.extra_dims),
        num_classes=data.num_classes,
        layers=int(spec.layers),
        max_children=int(spec.max_children),
    )
    return AATField(cfg)


def train_one(spec: RunSpec, data: DataBundle, out_dir: Path, device: torch.device) -> Dict[str, object]:
    set_seed(spec.seed)
    model = make_model(data, spec).to(device)

    if spec.mode == "auto":
        model.initialize(
            data.x_train.to(device), data.y_train.to(device),
            samples=spec.init_samples,
            min_children=2,
            kmeans_iters=spec.kmeans_iters,
            seed=spec.seed,
        )
    else:
        model.initialize(
            data.x_train.to(device), data.y_train.to(device),
            samples=spec.init_samples,
            min_children=spec.fixed_k,
            kmeans_iters=spec.kmeans_iters,
            seed=spec.seed,
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
    start = time.time()
    rid = run_id(data, spec)

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
            print(f"{rid} epoch={epoch:03d} val_acc={val_acc:.4f} best={best_val:.4f}@{best_epoch}", flush=True)

    final_val_acc, final_val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)
    model.load_state_dict(best_state)
    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)

    ztr0, ztr1 = layer_states(model, data.x_train, device)
    zv0, zv1 = layer_states(model, data.x_val, device)
    zte0, zte1 = layer_states(model, data.x_test, device)
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
        "train_fisher_before": round(supervised_fisher(ztr0, data.y_train.to(device), data.num_classes), 6),
        "train_fisher_after": round(supervised_fisher(ztr1, data.y_train.to(device), data.num_classes), 6),
        "val_fisher_before": round(supervised_fisher(zv0, data.y_val.to(device), data.num_classes), 6),
        "val_fisher_after": round(supervised_fisher(zv1, data.y_val.to(device), data.num_classes), 6),
        "test_fisher_before": round(supervised_fisher(zte0, data.y_test.to(device), data.num_classes), 6),
        "test_fisher_after": round(supervised_fisher(zte1, data.y_test.to(device), data.num_classes), 6),
    }
    row.update({k: round(v, 6) for k, v in diag.items()})
    return row


def build_specs(args) -> List[RunSpec]:
    specs: List[RunSpec] = []
    specs.append(RunSpec(
        model=f"AAT-L1-extra{args.extra_dims}-AutoK-max{args.max_k}",
        mode="auto",
        seed=args.seed,
        layers=1,
        fixed_k=0,
        max_children=args.max_k,
        extra_dims=args.extra_dims,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        init_samples=args.init_samples,
        kmeans_iters=args.kmeans_iters,
    ))
    for k in range(int(args.min_k), int(args.max_k) + 1):
        specs.append(RunSpec(
            model=f"AAT-L1-extra{args.extra_dims}-K{k}",
            mode="fixed",
            seed=args.seed,
            layers=1,
            fixed_k=k,
            max_children=k,
            extra_dims=args.extra_dims,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            init_samples=args.init_samples,
            kmeans_iters=args.kmeans_iters,
        ))
    if args.limit > 0:
        specs = specs[: int(args.limit)]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./checkerboard3d_l1_child_sweep")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--extra-dims", type=int, default=3)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--init-samples", type=int, default=4096)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    device = torch.device(args.device)
    data = checkerboard_bundle(
        grid_size=args.grid_size,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        noise_std=args.noise_std,
    )
    specs = build_specs(args)
    done = load_done(results_csv)

    print(
        f"dataset={data.name} train={len(data.y_train)} val={len(data.y_val)} test={len(data.y_test)} "
        f"device={device} runs={len(specs)} done={len(done)} out={out_dir}",
        flush=True,
    )
    for i, spec in enumerate(specs, 1):
        rid = run_id(data, spec)
        if rid in done:
            print(f"[{i}/{len(specs)}] skip {rid}", flush=True)
            continue
        print(f"[{i}/{len(specs)}] run {rid}", flush=True)
        try:
            row = train_one(spec, data, out_dir, device)
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
