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
from typing import Dict, List, Optional, Tuple

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
    img_train: Optional[torch.Tensor] = None
    img_val: Optional[torch.Tensor] = None
    img_test: Optional[torch.Tensor] = None


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
    batch_size: int = 256
    init_samples: int = 8192
    kmeans_iters: int = 8
    input_mode: str = "flat"


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
        idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[: self.centers.shape[0]]
        if idx.numel() < self.centers.shape[0]:
            idx = torch.randint(0, x.shape[0], (self.centers.shape[0],), generator=gen, device=x.device)
        self.centers.copy_(x[idx].float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        x = x.float()
        dist2 = (x * x).sum(dim=1, keepdim=True) + (self.centers * self.centers).sum(dim=1).view(1, -1) - 2.0 * x @ self.centers.t()
        gamma = F.softplus(self.raw_gamma) + 1e-4
        return self.head(torch.exp(-gamma * dist2.clamp_min(0.0)))


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.GELU(), nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


_DATA_CACHE: Dict[str, DataBundle] = {}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1).long().cpu()
    y = y.view(-1).long().cpu()
    out = []
    for c in range(int(num_classes)):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        den = 2 * tp + fp + fn
        out.append(0.0 if den == 0 else 2 * tp / den)
    return float(sum(out) / max(len(out), 1))


def accuracy_and_f1(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float]:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
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
            groups.setdefault((row["dataset"], row["model"]), []).append({k: float(row[k]) for k in ["best_val_acc", "test_acc", "test_f1", "params"]})
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "model", "runs", "params_mean", "best_val_acc_mean", "best_val_acc_std", "test_acc_mean", "test_f1_mean"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (dataset, model), rows in sorted(groups.items()):
            vals = [r["best_val_acc"] for r in rows]
            mean = sum(vals) / len(vals)
            std = 0.0 if len(vals) == 1 else math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            writer.writerow({"dataset": dataset, "model": model, "runs": len(rows), "params_mean": round(sum(r["params"] for r in rows) / len(rows), 2), "best_val_acc_mean": round(mean, 6), "best_val_acc_std": round(std, 6), "test_acc_mean": round(sum(r["test_acc"] for r in rows) / len(rows), 6), "test_f1_mean": round(sum(r["test_f1"] for r in rows) / len(rows), 6)})


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    return ((x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values).clamp_min(1e-6)).clamp(0.0, 1.0)


def make_checkerboard(n: int, dim: int, grid_size: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, dim, generator=gen)
    cells = torch.floor(x * grid_size).long().clamp(max=grid_size - 1)
    return x, (cells.sum(dim=1) % 2).long()


def make_moons(n: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    n0 = n // 2
    t0 = torch.rand(n0, generator=gen) * math.pi
    t1 = torch.rand(n - n0, generator=gen) * math.pi
    x0 = torch.stack([torch.cos(t0), torch.sin(t0)], dim=1)
    x1 = torch.stack([1.0 - torch.cos(t1), 0.55 - torch.sin(t1)], dim=1)
    x = torch.cat([x0, x1], dim=0) + 0.08 * torch.randn(n, 2, generator=gen)
    y = torch.cat([torch.zeros(n0), torch.ones(n - n0)]).long()
    return normalize_01(x), y


def make_circles(n: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    n0 = n // 2
    t0 = torch.rand(n0, generator=gen) * 2 * math.pi
    t1 = torch.rand(n - n0, generator=gen) * 2 * math.pi
    r0 = 0.35 + 0.03 * torch.randn(n0, generator=gen)
    r1 = 0.75 + 0.04 * torch.randn(n - n0, generator=gen)
    x = torch.cat([torch.stack([r0 * torch.cos(t0), r0 * torch.sin(t0)], dim=1), torch.stack([r1 * torch.cos(t1), r1 * torch.sin(t1)], dim=1)], dim=0)
    y = torch.cat([torch.zeros(n0), torch.ones(n - n0)]).long()
    return normalize_01(x), y


def make_spiral(n: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    n0 = n // 2
    t0 = torch.rand(n0, generator=gen) * 3.7 * math.pi
    t1 = torch.rand(n - n0, generator=gen) * 3.7 * math.pi + math.pi
    r0 = t0 / (3.7 * math.pi)
    r1 = (t1 - math.pi) / (3.7 * math.pi)
    x = torch.cat([torch.stack([r0 * torch.cos(t0), r0 * torch.sin(t0)], dim=1), torch.stack([r1 * torch.cos(t1), r1 * torch.sin(t1)], dim=1)], dim=0) + 0.03 * torch.randn(n, 2, generator=gen)
    y = torch.cat([torch.zeros(n0), torch.ones(n - n0)]).long()
    return normalize_01(x), y


def synthetic_bundle(name: str, seed: int, n_train: int = 4096, n_val: int = 2048, n_test: int = 2048) -> DataBundle:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    n = n_train + n_val + n_test
    if name.startswith("checkerboard"):
        x, y = make_checkerboard(n, int(name.split("_")[1].replace("d", "")), 4, gen)
    elif name == "moons_2d":
        x, y = make_moons(n, gen)
    elif name == "circles_2d":
        x, y = make_circles(n, gen)
    elif name == "spiral_2d":
        x, y = make_spiral(n, gen)
    else:
        raise ValueError(f"unknown synthetic dataset: {name}")
    return DataBundle(name, x.shape[1], 2, x[:n_train], y[:n_train], x[n_train:n_train+n_val], y[n_train:n_train+n_val], x[n_train+n_val:], y[n_train+n_val:])


def image_bundle(name: str, data_root: Path, train_limit: int, val_size: int) -> DataBundle:
    key = f"{name}:{train_limit}:{val_size}"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("MNIST/Fashion-MNIST requires torchvision.") from e
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    tr = cls(root=str(data_root), train=True, download=True, transform=transforms.ToTensor())
    te = cls(root=str(data_root), train=False, download=True, transform=transforms.ToTensor())
    x_all = torch.stack([tr[i][0] for i in range(len(tr))])
    y_all = torch.tensor([int(tr[i][1]) for i in range(len(tr))], dtype=torch.long)
    x_test = torch.stack([te[i][0] for i in range(len(te))])
    y_test = torch.tensor([int(te[i][1]) for i in range(len(te))], dtype=torch.long)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    limit = min(int(train_limit), x_all.shape[0])
    val = min(int(val_size), max(limit // 6, 1))
    idx = torch.randperm(x_all.shape[0], generator=gen)[:limit]
    val_idx, train_idx = idx[:val], idx[val:]
    img_train, img_val = x_all[train_idx], x_all[val_idx]
    bundle = DataBundle(name, 784, 10, img_train.flatten(1), y_all[train_idx], img_val.flatten(1), y_all[val_idx], x_test.flatten(1), y_test, img_train, img_val, x_test)
    _DATA_CACHE[key] = bundle
    return bundle


def extra_dims_for_state(state: str, input_dim: int) -> int:
    if state == "x1": return 0
    if state == "x2": return int(input_dim)
    if state == "x4": return int(input_dim) * 3
    if state == "x8": return int(input_dim) * 7
    if state == "p256": return 256
    if state == "p512": return 512
    if state == "": return 0
    raise ValueError(f"unknown state mode: {state}")


def parse_widths(widths: str) -> List[int]:
    return [] if not widths else [int(x) for x in widths.split("-") if x]


def make_model(spec: RunSpec, data: DataBundle) -> nn.Module:
    if spec.family == "aat":
        cfg = AATFieldConfig(input_dim=data.input_dim, extra_dims=extra_dims_for_state(spec.state, data.input_dim), num_classes=data.num_classes, layers=spec.layers, max_children=spec.max_children)
        return AATField(cfg)
    if spec.family == "mlp": return MLPClassifier(data.input_dim, data.num_classes, parse_widths(spec.widths))
    if spec.family == "rbf": return RBFClassifier(data.input_dim, data.num_classes, spec.centers)
    if spec.family == "cnn": return SmallCNN(data.num_classes)
    raise ValueError(f"unknown model family: {spec.family}")


def loaders_for_spec(data: DataBundle, spec: RunSpec) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if spec.input_mode == "image":
        train_ds, val_ds, test_ds = TensorDataset(data.img_train, data.y_train), TensorDataset(data.img_val, data.y_val), TensorDataset(data.img_test, data.y_test)
    else:
        train_ds, val_ds, test_ds = TensorDataset(data.x_train, data.y_train), TensorDataset(data.x_val, data.y_val), TensorDataset(data.x_test, data.y_test)
    return DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True), DataLoader(val_ds, batch_size=spec.batch_size * 2, shuffle=False), DataLoader(test_ds, batch_size=spec.batch_size * 2, shuffle=False)


def run_id(spec: RunSpec) -> str:
    return f"{spec.dataset}__{spec.model}__seed{spec.seed}".replace("/", "_").replace(" ", "_")


def save_best(model: nn.Module, ckpt_path: Path, spec: RunSpec, epoch: int, val_acc: float, val_f1: float) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**asdict(spec), "best_epoch": int(epoch), "best_val_acc": float(val_acc), "best_val_f1": float(val_f1), "params": int(count_parameters(model))}
    if isinstance(model, AATField):
        meta["selected_children"] = model.selected_children_by_layer()
        meta["total_children"] = model.total_children()
        model.save_checkpoint(str(ckpt_path), metadata=meta)
    else:
        torch.save({"format": "BenchmarkCheckpoint", "state_dict": model.state_dict(), "metadata": meta}, ckpt_path)


def fit_one(spec: RunSpec, data: DataBundle, out_dir: Path, device: torch.device) -> Dict[str, object]:
    set_seed(spec.seed)
    model = make_model(spec, data).to(device)
    if spec.family == "aat": model.initialize(data.x_train.to(device), data.y_train.to(device), samples=spec.init_samples, min_children=2, kmeans_iters=spec.kmeans_iters, seed=spec.seed)
    if spec.family == "rbf": model.initialize(data.x_train.to(device), spec.seed)
    train_loader, val_loader, test_loader = loaders_for_spec(data, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ckpt_path = out_dir / "checkpoints" / f"{run_id(spec)}.pt"
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
        if epoch == 1 or epoch % 10 == 0 or epoch == spec.epochs:
            print(f"{run_id(spec)} epoch={epoch} val_acc={val_acc:.4f} best={best_val:.4f}@{best_epoch}", flush=True)
    if isinstance(model, AATField):
        model = AATField.from_checkpoint(str(ckpt_path), map_location=device).to(device)
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)
    return {"status": "ok", "run_id": run_id(spec), "dataset": spec.dataset, "group": spec.group, "model": spec.model, "family": spec.family, "seed": spec.seed, "layers": spec.layers, "max_children": spec.max_children, "state": spec.state, "widths": spec.widths, "centers": spec.centers, "input_dim": data.input_dim, "state_dim": data.input_dim + extra_dims_for_state(spec.state, data.input_dim), "num_classes": data.num_classes, "params": int(count_parameters(model)), "best_epoch": best_epoch, "best_val_acc": round(best_val, 6), "best_val_f1": round(best_f1, 6), "test_acc": round(test_acc, 6), "test_f1": round(test_f1, 6), "train_time_sec": round(time.time() - start, 3), "checkpoint": str(ckpt_path), "selected_children": json.dumps(model.selected_children_by_layer()) if isinstance(model, AATField) else "", "total_children": model.total_children() if isinstance(model, AATField) else ""}


def synthetic_specs() -> List[RunSpec]:
    datasets = ["checkerboard_2d", "checkerboard_3d", "checkerboard_4d", "moons_2d", "circles_2d", "spiral_2d"]
    models = [("AAT-S-x1","aat",4,8,"x1","",0,800,2e-3),("AAT-S-x2","aat",4,8,"x2","",0,800,2e-3),("AAT-S-x4","aat",4,8,"x4","",0,800,2e-3),("AAT-S-x8","aat",4,8,"x8","",0,800,2e-3),("AAT-M-x2","aat",8,12,"x2","",0,800,1.5e-3),("AAT-M-x4","aat",8,12,"x4","",0,800,1.5e-3),("AAT-L-x4","aat",12,16,"x4","",0,800,1e-3),("AAT-L-x8","aat",12,16,"x8","",0,800,1e-3),("MLP-small","mlp",0,0,"","32-32",0,800,1e-3),("MLP-match","mlp",0,0,"","64-64",0,800,1e-3),("MLP-large","mlp",0,0,"","128-128-128",0,800,1e-3),("RBF-small","rbf",0,0,"","",32,800,1e-3),("RBF-match","rbf",0,0,"","",96,800,1e-3)]
    return [RunSpec(d, "synthetic", name, fam, seed, L, K, state, widths, centers, epochs, lr, 256, 4096, 8, "flat") for d in datasets for seed in [0, 1, 2] for name, fam, L, K, state, widths, centers, epochs, lr in models]


def image_specs(dataset: str) -> List[RunSpec]:
    models = [("AAT-Img-S-x1","aat",4,8,"x1","",0,80,1e-3,"flat"),("AAT-Img-S-p256","aat",4,8,"p256","",0,80,1e-3,"flat"),("AAT-Img-S-x2","aat",4,8,"x2","",0,80,1e-3,"flat"),("AAT-Img-M-p256","aat",8,12,"p256","",0,80,8e-4,"flat"),("AAT-Img-M-x2","aat",8,12,"x2","",0,80,8e-4,"flat"),("AAT-Img-L-p512","aat",16,16,"p512","",0,80,6e-4,"flat"),("AAT-Img-XL-x2","aat",16,24,"x2","",0,80,5e-4,"flat"),("MLP-small","mlp",0,0,"","256",0,80,1e-3,"flat"),("MLP-medium","mlp",0,0,"","512-512",0,80,1e-3,"flat"),("MLP-large","mlp",0,0,"","1024-1024",0,80,1e-3,"flat"),("RBF-match","rbf",0,0,"","",512,80,1e-3,"flat"),("CNN-small","cnn",0,0,"","",0,80,1e-3,"image")]
    return [RunSpec(dataset, "image", name, fam, 0, L, K, state, widths, centers, epochs, lr, 256, 8192, 6, mode) for name, fam, L, K, state, widths, centers, epochs, lr, mode in models]


def build_specs(tasks: List[str]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if "synthetic" in tasks: specs += synthetic_specs()
    if "mnist" in tasks: specs += image_specs("mnist")
    if "fashion" in tasks: specs += image_specs("fashion")
    return specs


def get_data(spec: RunSpec, args) -> DataBundle:
    if spec.group == "synthetic": return synthetic_bundle(spec.dataset, spec.seed)
    if spec.dataset in {"mnist", "fashion"}: return image_bundle(spec.dataset, Path(args.data_root), args.image_train_limit, args.image_val_size)
    raise ValueError(f"unknown dataset: {spec.dataset}")


def parse_tasks(raw: str) -> List[str]:
    if raw == "all":
        return ["synthetic", "mnist", "fashion"]
    tasks = [x.strip().lower() for x in raw.split(",") if x.strip()]
    allowed = {"synthetic", "mnist", "fashion"}
    bad = [x for x in tasks if x not in allowed]
    if bad:
        raise ValueError(f"Unknown task(s): {bad}. Allowed: synthetic,mnist,fashion,all")
    return tasks

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./benchmark_runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-train-limit", type=int, default=60000)
    parser.add_argument("--image-val-size", type=int, default=10000)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    device = torch.device(args.device)
    tasks = parse_tasks(args.tasks)
    specs = build_specs(tasks)
    done = load_done(results_csv) if args.resume else set()
    print(f"device={device} runs={len(specs)} resume_done={len(done)} out={out_dir}", flush=True)
    for i, spec in enumerate(specs, 1):
        rid = run_id(spec)
        if rid in done:
            print(f"[{i}/{len(specs)}] skip {rid}", flush=True)
            continue
        print(f"[{i}/{len(specs)}] run {rid}", flush=True)
        try:
            row = fit_one(spec, get_data(spec, args), out_dir, device)
        except Exception as e:
            row = {"status": "failed", "run_id": rid, "dataset": spec.dataset, "group": spec.group, "model": spec.model, "family": spec.family, "seed": spec.seed, "error": repr(e)}
            print(f"FAILED {rid}: {e}", flush=True)
        append_csv(results_csv, row)
        write_summary(results_csv, summary_csv)
    write_summary(results_csv, summary_csv)
    print(f"done: {results_csv}")
    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
