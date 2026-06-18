# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from aatfield import AATField, AATFieldConfig
from aatfield.initialize import boundary_weights, weighted_kmeans
from aatfield.utils import pairwise_dist2


# ============================================================
# Config
# ============================================================

OUT_DIR = Path("./init_selector_probe_mixed")

MAX_K = 100
MIN_K = 2
KMEANS_ITERS = 8
SEED = 0

CHECKERBOARD_N = 4096
CHECKERBOARD_GRID = 4
CHECKERBOARD_EXTRA_DIMS = 3

AIRLINE_DATA_DIR = Path("../data/AirlineSatisfaction")
AIRLINE_INIT_SAMPLES = 8192
AIRLINE_STATE_MODE = "x2"


@dataclass
class ProbeDataset:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    input_dim: int
    extra_dims: int
    num_classes: int


# ============================================================
# Generic helpers
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1).long().cpu()
    y = y.view(-1).long().cpu()

    vals: List[float] = []
    for c in range(int(num_classes)):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        den = 2 * tp + fp + fn
        vals.append(0.0 if den == 0 else 2.0 * tp / den)

    return float(sum(vals) / max(len(vals), 1))


def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(1e-12)
    return -(p * p.log()).sum()


def knee_select(k_values: List[int], scores: List[float], *, larger_is_better: bool = True) -> int:
    if len(k_values) <= 1:
        return int(k_values[0])

    xs = torch.tensor(k_values, dtype=torch.float32)
    ys = torch.tensor(scores, dtype=torch.float32)

    if not larger_is_better:
        ys = -ys

    x = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1e-8)
    y = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1e-8)

    curve = torch.stack([x, y], dim=1)
    line = torch.stack([x, x], dim=1)
    dist = (curve - line).norm(dim=1)

    return int(k_values[int(dist.argmax().item())])


def first_fraction_k(k_values: List[int], scores: List[float], frac: float) -> int:
    best = max(scores)
    target = best * float(frac)

    for k, s in zip(k_values, scores):
        if s >= target:
            return int(k)

    return int(k_values[-1])


def clamp_k(k: int, min_k: int, max_k: int) -> int:
    return int(max(int(min_k), min(int(k), int(max_k))))


# ============================================================
# Dataset loading
# ============================================================

def make_checkerboard3d(seed: int) -> ProbeDataset:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    x = torch.rand(CHECKERBOARD_N, 3, generator=gen)
    cells = torch.floor(x * CHECKERBOARD_GRID).long().clamp(max=CHECKERBOARD_GRID - 1)
    y = (cells.sum(dim=1) % 2).long()

    idx = torch.randperm(CHECKERBOARD_N, generator=gen)
    x = x[idx]
    y = y[idx]

    return ProbeDataset(
        name=f"checkerboard3d_g{CHECKERBOARD_GRID}",
        x=x,
        y=y,
        input_dim=3,
        extra_dims=CHECKERBOARD_EXTRA_DIMS,
        num_classes=2,
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


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


def extra_dims_for_state_mode(input_dim: int, state_mode: str) -> int:
    if state_mode == "x1":
        return 0
    if state_mode == "x2":
        return int(input_dim)
    if state_mode == "x3":
        return int(input_dim) * 2
    if state_mode == "x4":
        return int(input_dim) * 3
    raise ValueError(f"unknown state_mode={state_mode}")


def make_airline(data_dir: Path, seed: int) -> ProbeDataset:
    train_csv = data_dir / "train.csv"
    rows = read_csv_rows(train_csv)
    if not rows:
        raise RuntimeError(f"empty CSV: {train_csv}")

    columns = list(rows[0].keys())
    target_col = "satisfaction" if "satisfaction" in columns else columns[-1]

    drop_cols = {target_col, "id", "ID", "Id"}
    drop_cols.update(c for c in columns if c.lower().startswith("unnamed"))

    feature_cols = [c for c in columns if c not in drop_cols]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in feature_cols:
        vals = [str(r.get(c, "")).strip() for r in rows]
        non_empty = [v for v in vals if v != "" and v.lower() not in {"nan", "none", "null"}]
        if non_empty and all(parse_float_or_none(v) is not None for v in non_empty):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for c in numeric_cols:
        vals = [parse_float_or_none(r.get(c, "")) for r in rows]
        clean = [float(v) for v in vals if v is not None]
        mean = sum(clean) / max(len(clean), 1)
        var = sum((v - mean) ** 2 for v in clean) / max(len(clean), 1)
        means[c] = float(mean)
        stds[c] = float(math.sqrt(var) if var > 1e-12 else 1.0)

    vocabs: Dict[str, List[str]] = {}
    for c in categorical_cols:
        values = sorted({
            str(r.get(c, "")).strip()
            for r in rows
            if str(r.get(c, "")).strip() != ""
        })
        vocabs[c] = values

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

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    if x.shape[0] > AIRLINE_INIT_SAMPLES:
        idx = torch.randperm(x.shape[0], generator=gen)[:AIRLINE_INIT_SAMPLES]
        x = x[idx]
        y = y[idx]

    input_dim = int(x.shape[1])
    extra_dims = extra_dims_for_state_mode(input_dim, AIRLINE_STATE_MODE)

    print(
        f"Airline: raw_cols={len(feature_cols)} numeric={len(numeric_cols)} "
        f"categorical={len(categorical_cols)} encoded_dim={input_dim} samples={x.shape[0]} "
        f"state_dim={input_dim + extra_dims}",
        flush=True,
    )

    return ProbeDataset(
        name=f"airline_{AIRLINE_STATE_MODE}_sample{x.shape[0]}",
        x=x,
        y=y,
        input_dim=input_dim,
        extra_dims=extra_dims,
        num_classes=2,
    )


# ============================================================
# Initialization probe
# ============================================================

@torch.no_grad()
def lift_dataset(ds: ProbeDataset, device: torch.device) -> torch.Tensor:
    cfg = AATFieldConfig(
        input_dim=ds.input_dim,
        extra_dims=ds.extra_dims,
        num_classes=ds.num_classes,
        layers=1,
        max_children=MAX_K,
    )
    model = AATField(cfg).to(device)
    return model.lift(ds.x.to(device)).detach().float()


@torch.no_grad()
def build_parents(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    C = int(num_classes)
    D = int(z.shape[1])
    parents = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    global_mean = z.mean(dim=0)

    for c in range(C):
        pts = z[y == c]
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    return parents


@torch.no_grad()
def build_centers_by_class(
    z: torch.Tensor,
    y: torch.Tensor,
    parents: torch.Tensor,
    *,
    num_classes: int,
    max_k: int,
    kmeans_iters: int,
) -> Tuple[List[Dict[int, torch.Tensor]], int]:
    centers_by_class: List[Dict[int, torch.Tensor]] = []
    max_k_by_class: List[int] = []

    for c in range(int(num_classes)):
        pts = z[y == c]
        if pts.shape[0] == 0:
            pts = z

        w = boundary_weights(pts, c, parents)
        local_max = min(int(max_k), int(pts.shape[0]))
        max_k_by_class.append(local_max)

        centers_map: Dict[int, torch.Tensor] = {}
        for k in range(1, local_max + 1):
            centers, _ = weighted_kmeans(pts, w, k=int(k), iters=int(kmeans_iters))
            centers_map[int(k)] = centers.detach().clone()

        centers_by_class.append(centers_map)

    return centers_by_class, min(max_k_by_class)


@torch.no_grad()
def probe_curves(
    z: torch.Tensor,
    y: torch.Tensor,
    parents: torch.Tensor,
    centers_by_class: List[Dict[int, torch.Tensor]],
    *,
    num_classes: int,
    min_k: int,
    max_k: int,
) -> List[Dict[str, float]]:
    C = int(num_classes)
    y = y.long().to(z.device)

    parent_dist = torch.sqrt(pairwise_dist2(z, parents) + 1e-8)
    own_parent_dist = parent_dist.gather(1, y.view(-1, 1)).squeeze(1)

    other_parent_dist = parent_dist.clone()
    other_parent_dist[torch.arange(z.shape[0], device=z.device), y] = float("inf")
    other_parent_dist = other_parent_dist.min(dim=1).values

    boundary_score = 1.0 - (other_parent_dist - own_parent_dist).abs() / torch.maximum(
        other_parent_dist,
        own_parent_dist,
    ).clamp_min(1e-8)
    boundary_score = boundary_score.clamp(0.0, 1.0)

    boundary_thr = torch.quantile(boundary_score, 0.60)
    boundary_mask = boundary_score >= boundary_thr

    out: List[Dict[str, float]] = []

    H_y = entropy(torch.bincount(y, minlength=C).float() / max(int(y.numel()), 1)).clamp_min(1e-12)

    for k in range(int(min_k), int(max_k) + 1):
        child_anchors = torch.cat([centers_by_class[c][int(k)] for c in range(C)], dim=0)
        anchor_labels = torch.arange(C, device=z.device).repeat_interleave(int(k))

        dist2 = pairwise_dist2(z, child_anchors)
        assign = dist2.argmin(dim=1)
        pred = anchor_labels[assign]

        anchor_acc = float((pred == y).float().mean().item())
        anchor_f1 = macro_f1(pred.cpu(), y.cpu(), C)

        boundary_anchor_acc = float((pred[boundary_mask] == y[boundary_mask]).float().mean().item())

        K_total = int(child_anchors.shape[0])
        flat_index = assign * C + y

        joint_flat = torch.zeros(K_total * C, device=z.device, dtype=z.dtype)
        joint_flat.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=z.dtype))
        joint = joint_flat.view(K_total, C)
        joint = joint / max(int(z.shape[0]), 1)

        p_anchor = joint.sum(dim=1)
        p_label = joint.sum(dim=0)

        denom = p_anchor[:, None] * p_label[None, :]
        mask = joint > 1e-12
        mi = (joint[mask] * (joint[mask] / denom[mask].clamp_min(1e-12)).log()).sum()
        hard_nmi = float((mi / torch.sqrt((entropy(p_anchor) * entropy(p_label)).clamp_min(1e-12))).item())

        purity = float(joint.max(dim=1).values.sum().item())
        cond_entropy = float((entropy(p_label) - mi).clamp_min(0.0).item())
        cond_entropy_norm = float(cond_entropy / float(H_y.item()))

        boundary_score_metric = 0.50 * anchor_f1 + 0.50 * boundary_anchor_acc

        out.append({
            "k": float(k),
            "hard_nmi": hard_nmi,
            "purity": purity,
            "cond_entropy_norm": cond_entropy_norm,
            "anchor_acc": anchor_acc,
            "anchor_f1": anchor_f1,
            "boundary_anchor_acc": boundary_anchor_acc,
            "boundary_score": float(boundary_score_metric),
        })

    return out


def selectors_from_curves(curves: List[Dict[str, float]], min_k: int, max_k: int) -> Dict[str, int]:
    k_values = [int(r["k"]) for r in curves]

    hard_nmi = [float(r["hard_nmi"]) for r in curves]
    anchor_f1 = [float(r["anchor_f1"]) for r in curves]
    boundary_acc = [float(r["boundary_anchor_acc"]) for r in curves]
    boundary_score = [float(r["boundary_score"]) for r in curves]

    nmi_knee = knee_select(k_values, hard_nmi, larger_is_better=True)
    nmi95 = first_fraction_k(k_values, hard_nmi, 0.95)
    nmi99 = first_fraction_k(k_values, hard_nmi, 0.99)

    anchor_f1_99 = first_fraction_k(k_values, anchor_f1, 0.99)
    anchor_f1_995 = first_fraction_k(k_values, anchor_f1, 0.995)

    boundary_acc_99 = first_fraction_k(k_values, boundary_acc, 0.99)
    boundary_score_99 = first_fraction_k(k_values, boundary_score, 0.99)

    current = clamp_k(nmi_knee + 5, min_k, max_k)

    hybrid_anchor99 = clamp_k(max(current, anchor_f1_99), min_k, max_k)
    hybrid_anchor995 = clamp_k(max(current, anchor_f1_995), min_k, max_k)
    hybrid_boundary99 = clamp_k(max(current, boundary_score_99), min_k, max_k)

    return {
        "nmi_knee": int(nmi_knee),
        "nmi95": int(nmi95),
        "nmi99": int(nmi99),
        "current_nmi_knee_plus5": int(current),
        "anchor_f1_99": int(anchor_f1_99),
        "anchor_f1_995": int(anchor_f1_995),
        "boundary_acc_99": int(boundary_acc_99),
        "boundary_score_99": int(boundary_score_99),
        "hybrid_anchor99": int(hybrid_anchor99),
        "hybrid_anchor995": int(hybrid_anchor995),
        "hybrid_boundary99": int(hybrid_boundary99),
        "hard_nmi_argmax": int(k_values[int(torch.tensor(hard_nmi).argmax().item())]),
        "anchor_f1_argmax": int(k_values[int(torch.tensor(anchor_f1).argmax().item())]),
        "boundary_score_argmax": int(k_values[int(torch.tensor(boundary_score).argmax().item())]),
    }


def write_curves_csv(path: Path, curves: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not curves:
        return

    fields = list(curves[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in curves:
            writer.writerow({k: round(float(v), 8) for k, v in r.items()})


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "n",
        "input_dim",
        "state_dim",
        "max_k",
        "nmi_knee",
        "current_nmi_knee_plus5",
        "anchor_f1_99",
        "anchor_f1_995",
        "boundary_score_99",
        "hybrid_anchor99",
        "hybrid_anchor995",
        "hybrid_boundary99",
        "hard_nmi_argmax",
        "anchor_f1_argmax",
        "boundary_score_argmax",
        "selectors_json",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


@torch.no_grad()
def run_probe(ds: ProbeDataset, device: torch.device, out_dir: Path) -> Dict[str, Any]:
    print(f"\n=== Probe {ds.name} ===", flush=True)

    z = lift_dataset(ds, device)
    y = ds.y.to(device=device, dtype=torch.long)

    parents = build_parents(z, y, ds.num_classes)
    centers_by_class, common_max = build_centers_by_class(
        z,
        y,
        parents,
        num_classes=ds.num_classes,
        max_k=MAX_K,
        kmeans_iters=KMEANS_ITERS,
    )

    max_k = min(int(common_max), int(MAX_K))
    min_k = min(int(MIN_K), max_k)

    curves = probe_curves(
        z,
        y,
        parents,
        centers_by_class,
        num_classes=ds.num_classes,
        min_k=min_k,
        max_k=max_k,
    )

    selectors = selectors_from_curves(curves, min_k=min_k, max_k=max_k)

    curve_path = out_dir / f"{ds.name}_curves.csv"
    write_curves_csv(curve_path, curves)

    print("selectors:")
    for k, v in selectors.items():
        print(f"  {k:24s}: {v}", flush=True)
    print(f"curves: {curve_path}", flush=True)

    return {
        "dataset": ds.name,
        "n": int(ds.x.shape[0]),
        "input_dim": int(ds.input_dim),
        "state_dim": int(ds.input_dim + ds.extra_dims),
        "max_k": int(max_k),
        **selectors,
        "selectors_json": json.dumps(selectors, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--airline-data-dir", default=str(AIRLINE_DATA_DIR))
    parser.add_argument("--only", choices=["all", "checkerboard", "airline"], default="all")
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(args.device)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[ProbeDataset] = []

    if args.only in {"all", "checkerboard"}:
        datasets.append(make_checkerboard3d(SEED))

    if args.only in {"all", "airline"}:
        datasets.append(make_airline(Path(args.airline_data_dir), SEED))

    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        rows.append(run_probe(ds, device, out_dir))

    summary_path = out_dir / "summary.csv"
    write_summary_csv(summary_path, rows)

    print(f"\nsummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
