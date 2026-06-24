# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aatfield import AATField, AATFieldConfig  # noqa: E402
try:
    from aatfield.utils import count_parameters
except Exception:  # noqa: BLE001
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Experiment constants
# ============================================================

RUN_NAME = "capacity_scan"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = bool(torch.cuda.is_available())

N_TRAIN = 3072
N_VAL = 2048
N_TEST = 4096
INPUT_DIM = 5
NUM_CLASSES = 2

BATCH_SIZE = 512
EVAL_BATCH_SIZE = 2048
INIT_SAMPLES = 3072
KMEANS_ITERS = 5

LR = 3e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 3.0

QUICK_EPOCHS = 60
LONG_EPOCHS = 500
RANDOM_LABEL_EPOCHS = 200
PATIENCE_QUICK = 8
PATIENCE_LONG = 18

K_VALUES = [2, 4, 8, 16, 32, 64]
EXTRA_RATIOS = [0, 1, 3]       # state_dim = input_dim * (1 + ratio)
LAYER_VALUES = [2, 4, 8]
CORE_SEEDS = [0]

# Keep the sweep broad, then continue only the best configurations.
PROMOTE_TOP_PER_TASK = 7
SAVE_TOP_PROMOTED_MODELS = 12

RANDOM_LABEL_N_VALUES = [512, 1024, 2048, 4096, 8192]
RANDOM_LABEL_K_VALUES = [2, 4, 8, 16, 32, 64]
RANDOM_LABEL_LAYER_VALUES = [4, 8]
RANDOM_LABEL_EXTRA_RATIO = 1

OUT_ROOT = THIS_FILE.parent / "runs"
RUN_DIR = OUT_ROOT / f"{RUN_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
CKPT_DIR = RUN_DIR / "checkpoints"
RESULTS_CSV = RUN_DIR / "results.csv"
RESULTS_JSONL = RUN_DIR / "results.jsonl"
SUMMARY_MD = RUN_DIR / "summary.md"


# ============================================================
# Reproducibility and IO
# ============================================================

def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_s() -> str:
    return datetime.now().strftime("%H:%M:%S")


def ensure_out_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def write_run_config() -> None:
    cfg = {
        "run_name": RUN_NAME,
        "device": str(DEVICE),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo_root": str(REPO_ROOT),
        "run_dir": str(RUN_DIR),
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "input_dim": INPUT_DIM,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "init_samples": INIT_SAMPLES,
        "kmeans_iters": KMEANS_ITERS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "quick_epochs": QUICK_EPOCHS,
        "long_epochs": LONG_EPOCHS,
        "random_label_epochs": RANDOM_LABEL_EPOCHS,
        "k_values": K_VALUES,
        "extra_ratios": EXTRA_RATIOS,
        "layer_values": LAYER_VALUES,
        "core_seeds": CORE_SEEDS,
        "random_label_n_values": RANDOM_LABEL_N_VALUES,
        "random_label_k_values": RANDOM_LABEL_K_VALUES,
        "random_label_layer_values": RANDOM_LABEL_LAYER_VALUES,
    }
    with open(RUN_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


RESULT_FIELDS = [
    "status", "stage", "task", "variant", "run_id", "seed",
    "n_train", "n_val", "n_test", "input_dim", "state_dim", "extra_dims", "extra_ratio",
    "K", "layers", "params", "total_children", "selected_children",
    "epochs", "best_epoch", "runtime_sec",
    "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
    "best_val_loss", "best_val_acc", "best_test_acc_at_end",
    "generalization_gap", "init_sec", "train_sec",
    "mean_move_norm", "mean_sigma", "min_sigma", "max_sigma", "mean_charge_abs",
    "mean_alpha_entropy", "mean_alpha_eff", "mean_top1_entropy", "mean_top1_eff",
    "mean_local_active", "mean_global_positive", "mean_z_norm", "field_scale",
    "checkpoint", "error",
]


def append_result(row: Dict[str, object]) -> None:
    exists = RESULTS_CSV.exists()
    clean = {k: row.get(k, "") for k in RESULT_FIELDS}
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(clean)
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



# ============================================================
# Synthetic task generators
# ============================================================

def _g(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _orthogonal_dirs(m: int, d: int, seed: int) -> torch.Tensor:
    gen = _g(seed)
    u = torch.randn(int(m), int(d), generator=gen)
    u = F.normalize(u, dim=1)
    return u


def _make_split(task_name: str, n_train: int, n_val: int, n_test: int, seed: int, **kw):
    x_train, y_train = make_task(task_name, n_train, seed=seed * 1000 + 11, **kw)
    x_val, y_val = make_task(task_name, n_val, seed=seed * 1000 + 23, **kw)
    x_test, y_test = make_task(task_name, n_test, seed=seed * 1000 + 37, **kw)
    return x_train, y_train, x_val, y_val, x_test, y_test


def make_oblique_parity(n: int, d: int = 5, bits: int = 4, seed: int = 0, noise: float = 0.00):
    gen = _g(seed)
    x = torch.rand(int(n), int(d), generator=gen)
    xc = x * 2.0 - 1.0
    u = _orthogonal_dirs(bits, d, 100_000 + bits)
    b = (xc @ u.t() > 0.0).long()
    y = torch.remainder(b.sum(dim=1), 2).long()
    if noise > 0:
        flip = torch.rand(n, generator=gen) < float(noise)
        y = torch.where(flip, 1 - y, y)
    return x.contiguous(), y.contiguous()


def make_periodic_projection(n: int, d: int = 5, bands: int = 6, axes: int = 2, seed: int = 0, noise: float = 0.00):
    gen = _g(seed)
    x = torch.rand(int(n), int(d), generator=gen)
    xc = x * 2.0 - 1.0
    u = _orthogonal_dirs(axes, d, 200_000 + bands * 31 + axes)
    phase = torch.linspace(0.13, 0.37, int(axes)).view(1, -1)
    v = xc @ u.t() + phase
    band_bits = torch.remainder(torch.floor(v * float(bands)).long(), 2)
    y = torch.remainder(band_bits.sum(dim=1), 2).long()
    if noise > 0:
        flip = torch.rand(n, generator=gen) < float(noise)
        y = torch.where(flip, 1 - y, y)
    return x.contiguous(), y.contiguous()


def make_madelon_lite(n: int, d: int = 5, seed: int = 0, noise_std: float = 0.18):
    gen = _g(seed)
    # 32 hypercube vertices in 5 informative dimensions, random fixed labels.
    verts = torch.tensor(list(__import__("itertools").product([-1.0, 1.0], repeat=int(d))), dtype=torch.float32)
    label_gen = _g(300_000)
    labels = torch.randint(0, 2, (verts.shape[0],), generator=label_gen, dtype=torch.long)
    # Force roughly balanced labels.
    if labels.sum().item() < 8 or labels.sum().item() > 24:
        labels = (torch.arange(verts.shape[0]) % 2).long()
        labels = labels[torch.randperm(verts.shape[0], generator=label_gen)]

    idx = torch.randint(0, verts.shape[0], (int(n),), generator=gen)
    z = 0.55 * verts.index_select(0, idx) + float(noise_std) * torch.randn(int(n), int(d), generator=gen)
    x = ((z + 1.35) / 2.70).clamp(0.0, 1.0)
    y = labels.index_select(0, idx)
    return x.contiguous(), y.contiguous()


def make_random_labels(n: int, d: int = 5, seed: int = 0):
    gen = _g(seed)
    x = torch.rand(int(n), int(d), generator=gen)
    y = torch.randint(0, 2, (int(n),), generator=gen, dtype=torch.long)
    return x.contiguous(), y.contiguous()


def make_hastie_like(n: int, d: int = 5, seed: int = 0, noise: float = 0.0):
    gen = _g(seed)
    x = torch.rand(int(n), int(d), generator=gen)
    xc = x * 2.0 - 1.0
    # Nonlinear but smooth sanity anchor, intentionally not the main probe.
    s = (xc[:, 0] * xc[:, 1] + 0.75 * xc[:, 2].square() - 0.45 * xc[:, 3] + 0.35 * torch.sin(4.0 * xc[:, 4]))
    y = (s > s.median()).long()
    if noise > 0:
        flip = torch.rand(n, generator=gen) < float(noise)
        y = torch.where(flip, 1 - y, y)
    return x.contiguous(), y.contiguous()


def make_task(task_name: str, n: int, seed: int = 0, **kw):
    if task_name == "parity3":
        return make_oblique_parity(n, d=INPUT_DIM, bits=3, seed=seed, noise=kw.get("noise", 0.0))
    if task_name == "parity5":
        return make_oblique_parity(n, d=INPUT_DIM, bits=5, seed=seed, noise=kw.get("noise", 0.0))
    if task_name == "periodic4":
        return make_periodic_projection(n, d=INPUT_DIM, bands=4, axes=2, seed=seed, noise=kw.get("noise", 0.0))
    if task_name == "periodic8":
        return make_periodic_projection(n, d=INPUT_DIM, bands=8, axes=2, seed=seed, noise=kw.get("noise", 0.0))
    if task_name == "madelon_lite":
        return make_madelon_lite(n, d=INPUT_DIM, seed=seed, noise_std=kw.get("noise_std", 0.18))
    if task_name == "hastie_like":
        return make_hastie_like(n, d=INPUT_DIM, seed=seed, noise=kw.get("noise", 0.0))
    if task_name == "random_labels":
        return make_random_labels(n, d=INPUT_DIM, seed=seed)
    raise ValueError(f"unknown task: {task_name}")


# ============================================================
# Metrics and model analysis
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = EVAL_BATCH_SIZE) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_ok = 0
    total_n = 0
    for i in range(0, x.shape[0], int(batch_size)):
        xb = x[i:i + int(batch_size)].to(DEVICE, non_blocking=True)
        yb = y[i:i + int(batch_size)].to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item())
        total_ok += int((logits.argmax(dim=1) == yb).sum().item())
        total_n += int(yb.numel())
    return total_loss / max(total_n, 1), total_ok / max(total_n, 1)


@torch.no_grad()
def analyze_aatfield(model: AATField, x: torch.Tensor, max_samples: int = 1536) -> Dict[str, float]:
    model.eval()
    xb = x[: int(max_samples)].to(DEVICE)
    z = model.lift(xb)

    move_norms: List[float] = []
    sigma_means: List[float] = []
    sigma_mins: List[float] = []
    sigma_maxs: List[float] = []
    charge_abs: List[float] = []
    entropies: List[float] = []
    effs: List[float] = []
    top1_entropies: List[float] = []
    top1_effs: List[float] = []
    local_active: List[float] = []
    global_positive: List[float] = []
    z_norms: List[float] = []
    field_scales: List[float] = []

    for layer in model.layers:
        anchors = layer.all_anchors()
        sigma = layer.sigma()
        sigma2 = sigma.view(1, -1).square().clamp_min(1e-8)
        dist2 = (
            (z * z).sum(dim=-1, keepdim=True)
            + (anchors * anchors).sum(dim=-1).view(1, -1)
            - 2.0 * (z @ anchors.t())
        ).clamp_min(0.0)
        scores = layer.charge.view(1, -1) - dist2 / (2.0 * sigma2)
        alpha = torch.softmax(scores, dim=-1)
        weight = alpha / sigma2
        weighted_anchor = weight @ anchors
        weighted_sum = weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        field_target = weighted_anchor / weighted_sum
        fs = float(getattr(layer, "field_scale", 0.08))
        move = fs * (weighted_anchor - z * weighted_sum)
        z_mid = z + move
        z_next = torch.relu(field_target + torch.relu(z_mid - field_target))

        entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()
        top1 = alpha.argmax(dim=1)
        counts = torch.bincount(top1, minlength=anchors.shape[0]).float()
        p = counts / counts.sum().clamp_min(1.0)
        top_entropy = -(p[p > 0] * p[p > 0].log()).sum()

        move_norms.append(float(move.norm(dim=1).mean().item()))
        sigma_means.append(float(sigma.mean().item()))
        sigma_mins.append(float(sigma.min().item()))
        sigma_maxs.append(float(sigma.max().item()))
        charge_abs.append(float(layer.charge.detach().abs().mean().item()))
        entropies.append(float(entropy.item()))
        effs.append(float(torch.exp(entropy).item()))
        top1_entropies.append(float(top_entropy.item()))
        top1_effs.append(float(torch.exp(top_entropy).item()))
        local_active.append(float((z_mid - field_target > 0).float().mean().item()))
        global_positive.append(float((z_next > 0).float().mean().item()))
        z_norms.append(float(z_next.norm(dim=1).mean().item()))
        field_scales.append(fs)

        z = z_next

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "mean_move_norm": mean(move_norms),
        "mean_sigma": mean(sigma_means),
        "min_sigma": min(sigma_mins) if sigma_mins else 0.0,
        "max_sigma": max(sigma_maxs) if sigma_maxs else 0.0,
        "mean_charge_abs": mean(charge_abs),
        "mean_alpha_entropy": mean(entropies),
        "mean_alpha_eff": mean(effs),
        "mean_top1_entropy": mean(top1_entropies),
        "mean_top1_eff": mean(top1_effs),
        "mean_local_active": mean(local_active),
        "mean_global_positive": mean(global_positive),
        "mean_z_norm": mean(z_norms),
        "field_scale": mean(field_scales),
    }


# ============================================================
# Training
# ============================================================

def train_one_run(
    *,
    stage: str,
    task: str,
    variant: str,
    seed: int,
    K: int,
    layers: int,
    extra_ratio: int,
    epochs: int,
    patience: int,
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    save_checkpoint: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, object]:
    run_t0 = time.time()
    set_seed(seed)
    run_id = run_id or f"{stage}_{task}_{variant}_K{K}_L{layers}_R{extra_ratio}_S{seed}"
    print(f"[{now_s()}] START {run_id}", flush=True)

    x_train, y_train, x_val, y_val, x_test, y_test = _make_split(task, n_train, n_val, n_test, seed)
    extra_dims = int(INPUT_DIM * int(extra_ratio))

    cfg = AATFieldConfig.from_data(
        x_train,
        num_classes=NUM_CLASSES,
        extra_dims=extra_dims,
        layers=int(layers),
        max_children=int(K),
        sigma_init=0.75,
        charge_init=0.08,
        step_cap=1.0,
        gate_bias=True,
        head_bias=True,
        lift_seed=1234 + int(seed),
    )

    model = AATField(cfg).to(DEVICE)

    init_t0 = time.time()
    # Fixed-K capacity mode: max_children=K and min_children=K.
    model.initialize(
        x_train,
        y_train,
        samples=min(int(INIT_SAMPLES), int(n_train)),
        min_children=int(K),
        kmeans_iters=int(KMEANS_ITERS),
    )
    init_sec = time.time() - init_t0

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=bool(torch.cuda.is_available()),
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    train_t0 = time.time()

    last_train_loss = 0.0
    last_train_acc = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0

    for ep in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_ok = 0
        total_n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))
            scaler.step(optimizer)
            scaler.update()

            bs = int(yb.numel())
            total_loss += float(loss.item()) * bs
            total_ok += int((logits.detach().argmax(dim=1) == yb).sum().item())
            total_n += bs

        last_train_loss = total_loss / max(total_n, 1)
        last_train_acc = total_ok / max(total_n, 1)
        last_val_loss, last_val_acc = evaluate(model, x_val, y_val)

        improved = (last_val_acc > best_val_acc + 1e-5) or (
            abs(last_val_acc - best_val_acc) <= 1e-5 and last_val_loss < best_val_loss
        )
        if improved:
            best_val_acc = float(last_val_acc)
            best_val_loss = float(last_val_loss)
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if ep == 1 or ep % 8 == 0 or ep == epochs:
            print(
                f"[{now_s()}] {run_id} ep={ep:03d}/{epochs} "
                f"train={last_train_acc:.4f} val={last_val_acc:.4f} best={best_val_acc:.4f}@{best_epoch}",
                flush=True,
            )

        if stale >= int(patience):
            break

    train_sec = time.time() - train_t0

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    train_loss, train_acc = evaluate(model, x_train, y_train)
    val_loss, val_acc = evaluate(model, x_val, y_val)
    test_loss, test_acc = evaluate(model, x_test, y_test)
    stats = analyze_aatfield(model, x_val)

    ckpt_path = ""
    if save_checkpoint:
        ckpt_path = str(CKPT_DIR / f"{run_id}.pt")
        try:
            model.save_checkpoint(
                ckpt_path,
                metadata={
                    "stage": stage,
                    "task": task,
                    "variant": variant,
                    "seed": int(seed),
                    "K": int(K),
                    "layers": int(layers),
                    "extra_ratio": int(extra_ratio),
                    "best_epoch": int(best_epoch),
                    "best_val_acc": float(best_val_acc),
                    "test_acc": float(test_acc),
                },
            )
        except Exception:
            torch.save({"state_dict": model.state_dict(), "config": model.config_dict()}, ckpt_path)

    row: Dict[str, object] = {
        "status": "ok",
        "stage": stage,
        "task": task,
        "variant": variant,
        "run_id": run_id,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "input_dim": int(INPUT_DIM),
        "state_dim": int(model.state_dim),
        "extra_dims": int(extra_dims),
        "extra_ratio": int(extra_ratio),
        "K": int(K),
        "layers": int(layers),
        "params": int(count_parameters(model)),
        "total_children": int(model.total_children()),
        "selected_children": json.dumps(model.selected_children_by_layer()),
        "epochs": int(ep),
        "best_epoch": int(best_epoch),
        "runtime_sec": float(time.time() - run_t0),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_test_acc_at_end": float(test_acc),
        "generalization_gap": float(train_acc - test_acc),
        "init_sec": float(init_sec),
        "train_sec": float(train_sec),
        "checkpoint": ckpt_path,
        "error": "",
        **stats,
    }
    append_result(row)
    print(
        f"[{now_s()}] DONE  {run_id} train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f} "
        f"gap={train_acc - test_acc:.4f} time={row['runtime_sec']:.1f}s",
        flush=True,
    )
    return row


def run_safe(**kwargs) -> Optional[Dict[str, object]]:
    try:
        return train_one_run(**kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        err = traceback.format_exc()
        print(f"[{now_s()}] ERROR {kwargs.get('run_id', '')}: {e}\n{err}", flush=True)
        row = {
            "status": "error",
            "stage": kwargs.get("stage", ""),
            "task": kwargs.get("task", ""),
            "variant": kwargs.get("variant", ""),
            "run_id": kwargs.get("run_id", ""),
            "seed": kwargs.get("seed", ""),
            "K": kwargs.get("K", ""),
            "layers": kwargs.get("layers", ""),
            "extra_ratio": kwargs.get("extra_ratio", ""),
            "error": str(e),
        }
        append_result(row)
        return None


# ============================================================
# Experiment schedule
# ============================================================

def core_task_names() -> List[str]:
    return [
        "parity3",
        "parity5",
        "periodic4",
        "periodic8",
        "madelon_lite",
        "hastie_like",
    ]


def run_core_sweep() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task in core_task_names():
        for seed in CORE_SEEDS:
            for layers in LAYER_VALUES:
                for extra_ratio in EXTRA_RATIOS:
                    for K in K_VALUES:
                        run_id = f"core_{task}_K{K}_D{1+extra_ratio}x_L{layers}_S{seed}"
                        row = run_safe(
                            stage="core_short",
                            task=task,
                            variant="fixedK_lattice",
                            seed=seed,
                            K=K,
                            layers=layers,
                            extra_ratio=extra_ratio,
                            epochs=QUICK_EPOCHS,
                            patience=PATIENCE_QUICK,
                            run_id=run_id,
                        )
                        if row is not None and row.get("status") == "ok":
                            rows.append(row)
    return rows


def select_promotions(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_task: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        if r.get("stage") != "core_short" or r.get("status") != "ok":
            continue
        by_task.setdefault(str(r["task"]), []).append(r)

    selected: List[Dict[str, object]] = []
    for task, rs in by_task.items():
        # Prefer high validation accuracy, then lower gap, then fewer params.
        rs_sorted = sorted(
            rs,
            key=lambda r: (
                float(r.get("val_acc", 0.0)),
                -abs(float(r.get("generalization_gap", 0.0))),
                -float(r.get("params", 0.0)),
            ),
            reverse=True,
        )
        selected.extend(rs_sorted[:PROMOTE_TOP_PER_TASK])
    return selected


def run_promoted(core_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    promoted = select_promotions(core_rows)
    rows: List[Dict[str, object]] = []
    for i, base in enumerate(promoted):
        save_ckpt = i < SAVE_TOP_PROMOTED_MODELS
        task = str(base["task"])
        K = int(base["K"])
        layers = int(base["layers"])
        extra_ratio = int(base["extra_ratio"])
        seed = int(base["seed"])
        run_id = f"long_{task}_K{K}_D{1+extra_ratio}x_L{layers}_S{seed}"
        row = run_safe(
            stage="promoted_long",
            task=task,
            variant="top_from_core",
            seed=seed,
            K=K,
            layers=layers,
            extra_ratio=extra_ratio,
            epochs=LONG_EPOCHS,
            patience=PATIENCE_LONG,
            save_checkpoint=save_ckpt,
            run_id=run_id,
        )
        if row is not None and row.get("status") == "ok":
            rows.append(row)
    return rows


def run_random_label_strip() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for n_train in RANDOM_LABEL_N_VALUES:
        for layers in RANDOM_LABEL_LAYER_VALUES:
            for K in RANDOM_LABEL_K_VALUES:
                seed = 7000 + int(n_train) + int(K) * 13 + int(layers) * 101
                run_id = f"memorize_random_N{n_train}_K{K}_D2x_L{layers}"
                row = run_safe(
                    stage="random_memorization",
                    task="random_labels",
                    variant="fixedK_train_size_strip",
                    seed=seed,
                    K=K,
                    layers=layers,
                    extra_ratio=RANDOM_LABEL_EXTRA_RATIO,
                    epochs=RANDOM_LABEL_EPOCHS,
                    patience=PATIENCE_LONG,
                    n_train=int(n_train),
                    n_val=min(N_VAL, max(512, int(n_train))),
                    n_test=min(N_TEST, max(1024, int(n_train))),
                    run_id=run_id,
                )
                if row is not None and row.get("status") == "ok":
                    rows.append(row)
    return rows


# ============================================================
# Summary and plots
# ============================================================

def read_ok_rows() -> List[Dict[str, str]]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def write_summary() -> None:
    rows = read_ok_rows()
    if not rows:
        return

    def f(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except Exception:
            return float(default)

    lines: List[str] = []
    lines.append(f"# AATField Capacity Scan Summary\n")
    lines.append(f"Run dir: `{RUN_DIR}`\n")
    lines.append(f"Finished ok runs: **{len(rows)}**\n")

    for stage in ["core_short", "promoted_long", "random_memorization"]:
        stage_rows = [r for r in rows if r.get("stage") == stage]
        if not stage_rows:
            continue
        lines.append(f"\n## {stage}\n")
        for task in sorted(set(r["task"] for r in stage_rows)):
            rs = [r for r in stage_rows if r["task"] == task]
            best = sorted(rs, key=lambda r: f(r, "val_acc"), reverse=True)[:8]
            lines.append(f"\n### {task}\n")
            lines.append("| rank | K | D | L | train | val | test | gap | params | move | alpha_eff | sigma |\n")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for i, r in enumerate(best, start=1):
                lines.append(
                    f"| {i} | {r['K']} | {r['state_dim']} | {r['layers']} | "
                    f"{f(r,'train_acc'):.4f} | {f(r,'val_acc'):.4f} | {f(r,'test_acc'):.4f} | "
                    f"{f(r,'generalization_gap'):.4f} | {int(f(r,'params'))} | "
                    f"{f(r,'mean_move_norm'):.3f} | {f(r,'mean_alpha_eff'):.2f} | {f(r,'mean_sigma'):.3f} |\n"
                )

    SUMMARY_MD.write_text("".join(lines), encoding="utf-8")


def make_plots() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = read_ok_rows()
    if not rows:
        return
    plot_dir = RUN_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    def f(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except Exception:
            return float(default)

    # K scaling by task for core short, grouped by D/L as separate figures.
    core = [r for r in rows if r.get("stage") == "core_short"]
    for task in sorted(set(r["task"] for r in core)):
        task_rows = [r for r in core if r["task"] == task]
        for layers in sorted(set(int(f(r, "layers")) for r in task_rows)):
            fig = plt.figure(figsize=(8, 5))
            for er in sorted(set(int(f(r, "extra_ratio")) for r in task_rows)):
                rs = [r for r in task_rows if int(f(r, "layers")) == layers and int(f(r, "extra_ratio")) == er]
                rs = sorted(rs, key=lambda r: f(r, "K"))
                if not rs:
                    continue
                plt.plot([f(r, "K") for r in rs], [f(r, "val_acc") for r in rs], marker="o", label=f"D={(1+er)}x")
            plt.xscale("log", base=2)
            plt.xlabel("children per class K")
            plt.ylabel("validation accuracy")
            plt.title(f"{task}: K scaling, L={layers}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"core_{task}_L{layers}_K_scaling.png", dpi=150)
            plt.close(fig)

    # Random memorization: train acc vs N by K/L.
    mem = [r for r in rows if r.get("stage") == "random_memorization"]
    if mem:
        for layers in sorted(set(int(f(r, "layers")) for r in mem)):
            fig = plt.figure(figsize=(8, 5))
            for K in sorted(set(int(f(r, "K")) for r in mem)):
                rs = [r for r in mem if int(f(r, "layers")) == layers and int(f(r, "K")) == K]
                rs = sorted(rs, key=lambda r: f(r, "n_train"))
                if rs:
                    plt.plot([f(r, "n_train") for r in rs], [f(r, "train_acc") for r in rs], marker="o", label=f"K={K}")
            plt.xscale("log", base=2)
            plt.xlabel("random-label train set size")
            plt.ylabel("train accuracy")
            plt.title(f"Random label memorization, L={layers}, D=2x")
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"random_memorization_L{layers}.png", dpi=150)
            plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_out_dirs()
    write_run_config()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("=" * 80, flush=True)
    print(f"AATField capacity scan", flush=True)
    print(f"repo_root: {REPO_ROOT}", flush=True)
    print(f"run_dir  : {RUN_DIR}", flush=True)
    print(f"device   : {DEVICE} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print("=" * 80, flush=True)

    all_rows: List[Dict[str, object]] = []

    print(f"[{now_s()}] Stage 1/3: core fixed-K lattice", flush=True)
    core_rows = run_core_sweep()
    all_rows.extend(core_rows)
    write_summary()
    make_plots()

    print(f"[{now_s()}] Stage 2/3: random-label memorization strip", flush=True)
    mem_rows = run_random_label_strip()
    all_rows.extend(mem_rows)
    write_summary()
    make_plots()

    print(f"[{now_s()}] Stage 3/3: promoted long runs", flush=True)
    long_rows = run_promoted(core_rows)
    all_rows.extend(long_rows)
    write_summary()
    make_plots()

    write_summary()
    make_plots()
    print("=" * 80, flush=True)
    print(f"[{now_s()}] Finished. OK rows in memory: {len(all_rows)}", flush=True)
    print(f"Results CSV : {RESULTS_CSV}", flush=True)
    print(f"Results JSON: {RESULTS_JSONL}", flush=True)
    print(f"Summary    : {SUMMARY_MD}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
