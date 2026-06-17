# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    dataset: str
    group: str
    model: str
    family: str
    seed: int
    layers: int = 0
    max_children: int = 0
    state: str = ""
    widths: str = ""
    centers: int = 0
    epochs: int = 0
    lr: float = 0.0
    batch_size: int = 512
    init_samples: int = 8192
    kmeans_iters: int = 8


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, widths: List[int]):
        super().__init__()
        parts: List[nn.Module] = []
        last = int(input_dim)
        for width in widths:
            parts += [nn.Linear(last, int(width)), nn.GELU()]
            last = int(width)
        parts.append(nn.Linear(last, int(num_classes)))
        self.net = nn.Sequential(*parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x.float())


class RBFClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, centers: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(int(centers), int(input_dim)) * 0.1)
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Linear(int(centers), int(num_classes))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, seed: int) -> None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        if x.shape[0] >= self.centers.shape[0]:
            idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[: self.centers.shape[0]]
        else:
            idx = torch.randint(0, x.shape[0], (self.centers.shape[0],), generator=gen, device=x.device)
        self.centers.copy_(x[idx].float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        x = x.float()
        dist2 = (
            (x * x).sum(dim=1, keepdim=True)
            + (self.centers * self.centers).sum(dim=1).view(1, -1)
            - 2.0 * x @ self.centers.t()
        ).clamp_min(0.0)
        gamma = F.softplus(self.raw_gamma) + 1e-4
        return self.head(torch.exp(-gamma * dist2))


class SmallTabularCNN(nn.Module):
    """A very small 1D-CNN baseline over tabular feature order.

    This is only a weak engineering baseline, not a claim that tabular features
    have a natural spatial ordering.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.GELU(),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x.float().unsqueeze(1))


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_widths(widths: str) -> List[int]:
    return [] if not widths else [int(x) for x in widths.split("-") if x]


def extra_dims_for_state(state: str, input_dim: int) -> int:
    if state == "x1":
        return 0
    if state == "x2":
        return int(input_dim)
    if state == "x3":
        return int(input_dim) * 2
    if state == "x4":
        return int(input_dim) * 3
    if state == "x8":
        return int(input_dim) * 7
    if state == "p64":
        return 64
    if state == "p128":
        return 128
    if state == "p256":
        return 256
    if state == "p512":
        return 512
    if state == "":
        return 0
    raise ValueError(f"unknown state mode: {state}")


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
    groups: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            groups.setdefault((row["dataset"], row["model"]), []).append(
                {k: float(row[k]) for k in ["best_val_acc", "test_acc", "test_f1", "params", "train_time_sec"]}
            )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "dataset", "model", "runs", "params_mean",
            "best_val_acc_mean", "best_val_acc_std", "test_acc_mean", "test_f1_mean", "train_time_sec_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (dataset, model), rows in sorted(groups.items()):
            vals = [r["best_val_acc"] for r in rows]
            mean = sum(vals) / len(vals)
            std = 0.0 if len(vals) == 1 else math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            writer.writerow({
                "dataset": dataset,
                "model": model,
                "runs": len(rows),
                "params_mean": round(sum(r["params"] for r in rows) / len(rows), 2),
                "best_val_acc_mean": round(mean, 6),
                "best_val_acc_std": round(std, 6),
                "test_acc_mean": round(sum(r["test_acc"] for r in rows) / len(rows), 6),
                "test_f1_mean": round(sum(r["test_f1"] for r in rows) / len(rows), 6),
                "train_time_sec_mean": round(sum(r["train_time_sec"] for r in rows) / len(rows), 3),
            })


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

    # Detect numeric columns using train rows only.
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in feature_cols:
        vals = [str(r.get(c, "")).strip() for r in train_rows]
        non_empty = [v for v in vals if v != "" and v.lower() not in {"nan", "none", "null"}]
        if non_empty and all(parse_float_or_none(v) is not None for v in non_empty):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    # Numeric statistics on train rows.
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for c in numeric_cols:
        vals = [parse_float_or_none(r.get(c, "")) for r in train_rows]
        clean = [float(v) for v in vals if v is not None]
        mean = sum(clean) / max(len(clean), 1)
        var = sum((v - mean) ** 2 for v in clean) / max(len(clean), 1)
        means[c] = float(mean)
        stds[c] = float(math.sqrt(var) if var > 1e-12 else 1.0)

    # One-hot vocabularies on train rows, plus an unknown bucket.
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

    return DataBundle(
        "airline_satisfaction", int(x_train.shape[1]), 2,
        x_train, y_train, x_val, y_val, x_test, y_test,
    )


def make_model(spec: RunSpec, data: DataBundle) -> nn.Module:
    if spec.family == "aat":
        cfg = AATFieldConfig(
            input_dim=data.input_dim,
            extra_dims=extra_dims_for_state(spec.state, data.input_dim),
            num_classes=data.num_classes,
            layers=spec.layers,
            max_children=spec.max_children,
        )
        return AATField(cfg)
    if spec.family == "mlp":
        return MLPClassifier(data.input_dim, data.num_classes, parse_widths(spec.widths))
    if spec.family == "rbf":
        return RBFClassifier(data.input_dim, data.num_classes, spec.centers)
    if spec.family == "cnn":
        return SmallTabularCNN(data.input_dim, data.num_classes)
    raise ValueError(f"unknown model family: {spec.family}")


def loaders_for_spec(data: DataBundle, spec: RunSpec) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(data.x_train, data.y_train)
    val_ds = TensorDataset(data.x_val, data.y_val)
    test_ds = TensorDataset(data.x_test, data.y_test)
    return (
        DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=spec.batch_size * 2, shuffle=False),
        DataLoader(test_ds, batch_size=spec.batch_size * 2, shuffle=False),
    )


def run_id(spec: RunSpec) -> str:
    return f"{spec.dataset}__{spec.model}__seed{spec.seed}".replace("/", "_").replace(" ", "_")


def safe_torch_save(obj: object, path: Path, retries: int = 5, sleep: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_error: Optional[BaseException] = None
    for _ in range(int(retries)):
        try:
            torch.save(obj, tmp)
            tmp.replace(path)
            return
        except OSError as e:
            last_error = e
            time.sleep(float(sleep))
    if last_error is not None:
        raise last_error


def save_best(model: nn.Module, ckpt_path: Path, spec: RunSpec, epoch: int, val_acc: float, val_f1: float) -> None:
    meta = {
        **asdict(spec),
        "best_epoch": int(epoch),
        "best_val_acc": float(val_acc),
        "best_val_f1": float(val_f1),
        "params": int(count_parameters(model)),
    }
    if isinstance(model, AATField):
        meta["selected_children"] = model.selected_children_by_layer()
        meta["total_children"] = model.total_children()
        model.save_checkpoint(str(ckpt_path), metadata=meta)
    else:
        safe_torch_save({"format": "AirlineBenchmarkCheckpoint", "state_dict": model.state_dict(), "metadata": meta}, ckpt_path)


def fit_one(spec: RunSpec, data: DataBundle, out_dir: Path, device: torch.device) -> Dict[str, object]:
    set_seed(spec.seed)
    model = make_model(spec, data).to(device)
    if spec.family == "aat":
        model.initialize(
            data.x_train.to(device), data.y_train.to(device),
            samples=spec.init_samples, min_children=2, kmeans_iters=spec.kmeans_iters, seed=spec.seed,
        )
    if spec.family == "rbf":
        model.initialize(data.x_train.to(device), spec.seed)

    train_loader, val_loader, test_loader = loaders_for_spec(data, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_dir = out_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"{run_id(spec)}.pt"

    best_val, best_f1, best_epoch = -1.0, 0.0, 0
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
            best_val, best_f1, best_epoch = val_acc, val_f1, epoch
            save_best(model, ckpt_path, spec, epoch, val_acc, val_f1)
        if epoch == 1 or epoch % 5 == 0 or epoch == spec.epochs:
            print(f"{run_id(spec)} epoch={epoch} val_acc={val_acc:.4f} val_f1={val_f1:.4f} best={best_val:.4f}@{best_epoch}", flush=True)

    if isinstance(model, AATField):
        model = AATField.from_checkpoint(str(ckpt_path), map_location=device).to(device)
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)
    return {
        "status": "ok",
        "run_id": run_id(spec),
        "dataset": spec.dataset,
        "group": spec.group,
        "model": spec.model,
        "family": spec.family,
        "seed": spec.seed,
        "layers": spec.layers,
        "max_children": spec.max_children,
        "state": spec.state,
        "widths": spec.widths,
        "centers": spec.centers,
        "input_dim": data.input_dim,
        "state_dim": data.input_dim + extra_dims_for_state(spec.state, data.input_dim),
        "num_classes": data.num_classes,
        "params": int(count_parameters(model)),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 6),
        "best_val_f1": round(best_f1, 6),
        "test_acc": round(test_acc, 6),
        "test_f1": round(test_f1, 6),
        "train_time_sec": round(time.time() - start, 3),
        "checkpoint": str(ckpt_path),
        "selected_children": json.dumps(model.selected_children_by_layer()) if isinstance(model, AATField) else "",
        "total_children": model.total_children() if isinstance(model, AATField) else "",
    }


def airline_specs() -> List[RunSpec]:
    # 12 runs: 7 AAT variants + 3 MLP + 1 RBF + 1 small tabular CNN.
    models = [
        ("AAT-Tab-S-x1", "aat", 4, 8, "x1", "", 0, 80, 1e-3),
        ("AAT-Tab-S-x2", "aat", 4, 8, "x2", "", 0, 80, 1e-3),
        ("AAT-Tab-S-x4", "aat", 4, 8, "x4", "", 0, 80, 1e-3),
        ("AAT-Tab-M-x2", "aat", 8, 12, "x2", "", 0, 80, 8e-4),
        ("AAT-Tab-M-x4", "aat", 8, 12, "x4", "", 0, 80, 8e-4),
        ("AAT-Tab-L-x4", "aat", 16, 16, "x4", "", 0, 80, 6e-4),
        ("AAT-Tab-XL-x4", "aat", 24, 24, "x4", "", 0, 80, 5e-4),
        ("MLP-small", "mlp", 0, 0, "", "64-64", 0, 80, 1e-3),
        ("MLP-medium", "mlp", 0, 0, "", "128-128", 0, 80, 1e-3),
        ("MLP-large", "mlp", 0, 0, "", "256-256-128", 0, 80, 1e-3),
        ("RBF-match", "rbf", 0, 0, "", "", 512, 80, 1e-3),
        ("CNN-small", "cnn", 0, 0, "", "", 0, 80, 1e-3),
    ]
    return [
        RunSpec("airline_satisfaction", "tabular", name, fam, 0, L, K, state, widths, centers, epochs, lr, 512, 8192, 8)
        for name, fam, L, K, state, widths, centers, epochs, lr in models
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/AirlineSatisfaction")
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--out-dir", default="./airline_runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = Path(args.train_csv) if args.train_csv else data_dir / "train.csv"
    test_csv = Path(args.test_csv) if args.test_csv else data_dir / "test.csv"
    out_dir = Path(args.out_dir)
    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    device = torch.device(args.device)

    data = airline_bundle(train_csv, test_csv, val_ratio=args.val_ratio, seed=123)
    specs = airline_specs()
    done = load_done(results_csv) if args.resume else set()
    print(f"device={device} runs={len(specs)} resume_done={len(done)} out={out_dir}", flush=True)

    for i, spec in enumerate(specs, 1):
        rid = run_id(spec)
        if rid in done:
            print(f"[{i}/{len(specs)}] skip {rid}", flush=True)
            continue
        print(f"[{i}/{len(specs)}] run {rid}", flush=True)
        try:
            row = fit_one(spec, data, out_dir, device)
        except Exception as e:
            row = {
                "status": "failed", "run_id": rid, "dataset": spec.dataset, "group": spec.group,
                "model": spec.model, "family": spec.family, "seed": spec.seed, "error": repr(e),
            }
            print(f"FAILED {rid}: {e}", flush=True)
        append_csv(results_csv, row)
        write_summary(results_csv, summary_csv)

    write_summary(results_csv, summary_csv)
    print(f"done: {results_csv}")
    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
