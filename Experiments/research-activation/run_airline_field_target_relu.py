# -*- coding: utf-8 -*-
"""
run_airline_field_target_relu.py

Field-target ReLU AAT experiment on Airline Satisfaction.

Goal
----
Test one extremely simple post-transport activation:

    compute the normal potential transport move
    reuse the transport field's implicit target r
    apply z_next = r + ReLU(z_mid - r)

This script does NOT modify the main aatfield package.  It reuses the current
AATField initialization / Auto-K machinery, but overrides the transport logic
inside this experiment file.

Recommended placement:
    C:/Projets/AATField/Experiments/research-activation/run_airline_field_target_relu.py

Default data path:
    ../data/AirlineSatisfaction
"""
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
# Expected layout: <repo>/Experiments/research-activation/this_script.py
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
                vec = [0.0] * (len(vocab) + 1)  # unknown bucket
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
# Conservative / potential-style AAT wrapper
# ============================================================


class ConservativeAATField(AATField):
    """
    Experiment-only model.

    Core transport mode
    -------------------
    potential:
        Treat anchors as a log-sum-exp potential field:

            scores_i = charge_i - ||z-a_i||^2 / (2 sigma_i^2)
            alpha_i  = softmax(scores_i)
            F(z)     = field_scale * sum_i alpha_i * (a_i-z) / sigma_i^2

        This is the negative gradient direction of U(z) = -logsumexp(scores_i)
        with respect to z, up to field_scale. No child gate is used.

    current_nogate:
        The old AAT transport without child gate. This is kept only as a
        diagnostic comparison; it is not guaranteed conservative.
    """

    VALID_CORE_MODES = {"potential", "current_nogate"}
    VALID_ACTIVATIONS = {
        # One method only: reuse the implicit target center already produced
        # by the same potential field that generated the move.
        "field_target_relu",   # z_next = r_target + ReLU(z_mid - r_target)
    }

    def __init__(
        self,
        cfg: AATFieldConfig,
        *,
        core_mode: str = "potential",
        activation_mode: str = "field_target_relu",
        field_scale: float = 1.0,
        boundary_margin: float = 0.30,
        boundary_strength: float = 0.50,
        child_axis_strength: float = 0.50,
        use_step_cap: bool = True,
    ):
        super().__init__(cfg)
        if core_mode not in self.VALID_CORE_MODES:
            raise ValueError(f"Unknown core_mode={core_mode!r}. Valid: {sorted(self.VALID_CORE_MODES)}")
        if activation_mode not in self.VALID_ACTIVATIONS:
            raise ValueError(f"Unknown activation_mode={activation_mode!r}. Valid: {sorted(self.VALID_ACTIVATIONS)}")
        self.core_mode = str(core_mode)
        self.activation_mode = str(activation_mode)
        self.field_scale = float(field_scale)
        self.boundary_margin = float(boundary_margin)
        self.boundary_strength = float(boundary_strength)
        self.child_axis_strength = float(child_axis_strength)
        self.use_step_cap = bool(use_step_cap)

    def _anchors_sigma(self, layer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = layer.num_classes
        child_flat = layer.child_anchors().reshape(layer.child_n, layer.state_dim)
        anchors = torch.cat([layer.parents, child_flat], dim=0)
        sigma = layer.sigma()
        if sigma.shape[0] != anchors.shape[0]:
            raise RuntimeError(
                f"shape mismatch: anchors={anchors.shape[0]} sigma={sigma.shape[0]}. "
                "Initialize before creating optimizer."
            )
        return anchors, sigma, child_flat

    def _potential_scores(self, layer, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors, sigma, child_flat = self._anchors_sigma(layer)
        dist2 = (
            (z * z).sum(dim=-1, keepdim=True)
            + (anchors * anchors).sum(dim=-1).view(1, -1)
            - 2.0 * (z @ anchors.t())
        ).clamp_min(0.0)
        sigma2 = sigma.view(1, -1).square().clamp_min(1e-8)
        # Here charge is treated as an anchor logit / bias in the potential.
        scores = layer.charge.view(1, -1) - dist2 / (2.0 * sigma2)
        return scores, dist2, anchors, sigma

    def _potential_energy(self, layer, z: torch.Tensor) -> torch.Tensor:
        scores, _, _, _ = self._potential_scores(layer, z)
        return -torch.logsumexp(scores, dim=1)

    def _field_step(self, layer, z: torch.Tensor, *, apply_cap: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.core_mode == "potential":
            scores, dist2, anchors, sigma = self._potential_scores(layer, z)
            alpha = torch.softmax(scores, dim=-1)
            sigma2 = sigma.view(1, -1).square().clamp_min(1e-8)
            # Transport vector is the potential gradient.  It can be rewritten as
            #     move = field_scale * W * (r_target - z)
            # where
            #     W = sum_i alpha_i / sigma_i^2
            #     r_target = sum_i (alpha_i / sigma_i^2) * anchor_i / W
            # So r_target is not an extra field-center computation; it is the
            # implicit target point already used by this transport step.
            weight = alpha / sigma2
            weighted_anchor = weight @ anchors
            weighted_sum = weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
            field_target = weighted_anchor / weighted_sum
            move = float(self.field_scale) * (weighted_anchor - z * weighted_sum)
            energy = -torch.logsumexp(scores, dim=1)
        elif self.core_mode == "current_nogate":
            anchors, sigma, _ = self._anchors_sigma(layer)
            dist2 = (
                (z * z).sum(dim=-1, keepdim=True)
                + (anchors * anchors).sum(dim=-1).view(1, -1)
                - 2.0 * (z @ anchors.t())
            ).clamp_min(0.0)
            logits = -dist2 / (2.0 * sigma.view(1, -1).square().clamp_min(1e-8))
            alpha = torch.softmax(logits, dim=-1)
            dist = torch.sqrt(dist2 + 1e-8)
            strength = alpha * layer.charge.view(1, -1)
            beta = strength / dist.clamp_min(1e-6)
            move = beta @ anchors - z * beta.sum(dim=1, keepdim=True)
            scores = logits
            energy = -torch.logsumexp(scores, dim=1)
            # Diagnostic fallback only. The default experiment uses potential.
            pos_weight = beta.abs().clamp_min(0.0)
            denom = pos_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
            field_target = (pos_weight @ anchors) / denom
        else:
            raise RuntimeError(f"bad core_mode={self.core_mode}")

        raw_move = move
        if apply_cap and self.use_step_cap:
            cap = float(layer.cfg.step_cap)
            if cap > 0:
                norm = move.norm(dim=-1, keepdim=True)
                capped = cap * torch.tanh(norm / cap)
                move = move * (capped / norm.clamp_min(1e-6))

        z_mid = z + move
        info = {
            "move": move,
            "raw_move": raw_move,
            "alpha": alpha,
            "scores": scores,
            "dist2": dist2,
            "energy": energy,
            "field_target": field_target,
        }
        return z_mid, info

    def _boundary_push_activation(self, layer, z_mid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parents = layer.parents
        C = int(layer.num_classes)
        if C < 2:
            return z_mid, {"boundary_push_norm": z_mid.new_zeros(z_mid.shape[0]), "boundary_margin_value": z_mid.new_zeros(z_mid.shape[0])}

        dist2 = (
            (z_mid * z_mid).sum(dim=-1, keepdim=True)
            + (parents * parents).sum(dim=-1).view(1, -1)
            - 2.0 * (z_mid @ parents.t())
        ).clamp_min(0.0)
        # nearest parent and nearest competitor parent. Works for 2, 10, 20 classes.
        top2 = torch.topk(dist2, k=2, dim=1, largest=False).indices
        p1 = parents.index_select(0, top2[:, 0])
        p2 = parents.index_select(0, top2[:, 1])
        u = p1 - p2
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mid = 0.5 * (p1 + p2)
        signed = ((z_mid - mid) * u).sum(dim=-1, keepdim=True)
        margin = float(self.boundary_margin)
        # If the point is too close to the local parent-parent boundary, commit it
        # further toward the nearest-parent side. This is a multiclass local boundary rule.
        amount = F.relu(margin - signed)
        push = float(self.boundary_strength) * amount * u
        z_next = z_mid + push
        return z_next, {
            "boundary_push_norm": push.norm(dim=-1),
            "boundary_margin_value": signed.squeeze(1),
        }

    def _child_axis_activation(self, layer, z_mid: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        C = int(layer.num_classes)
        K = int(layer.children_per_class)
        if K < 1:
            return z_mid, {"child_axis_push_norm": z_mid.new_zeros(z_mid.shape[0]), "child_axis_positive_rate": z_mid.new_zeros(z_mid.shape[0])}

        child_alpha = alpha[:, C:]
        idx = child_alpha.argmax(dim=1)  # [B], over C*K children
        child_flat = layer.child_anchors().reshape(C * K, layer.state_dim)
        child = child_flat.index_select(0, idx)
        parent_idx = idx // K
        parent = layer.parents.index_select(0, parent_idx)
        axis = child - parent
        u = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # Post-transport local axis decision. If z_mid has crossed beyond the
        # selected child along the parent->child axis, commit further along that axis.
        s = ((z_mid - child) * u).sum(dim=-1, keepdim=True)
        amount = F.relu(s)
        push = float(self.child_axis_strength) * amount * u
        z_next = z_mid + push
        return z_next, {
            "child_axis_push_norm": push.norm(dim=-1),
            "child_axis_positive_rate": (s.squeeze(1) > 0).float(),
        }

    def _apply_activation(self, layer, z: torch.Tensor, z_mid: torch.Tensor, info: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = self.activation_mode
        if mode != "field_target_relu":
            raise RuntimeError(f"bad activation_mode={mode}")

        # Judge z_mid after transport. The failed value is reset to r, where r is
        # the field target already implied by the same transport step, not a
        # separately recomputed softmax center at z_mid.
        r = info["field_target"]
        delta = z_mid - r

        # Per-dimension ReLU around the transport target:
        #   z_next_d = r_d + ReLU(z_mid_d - r_d)
        #            = max(z_mid_d, r_d)
        z_next = r + F.relu(delta)
        active = (delta > 0).float()
        reset = z_next - z_mid
        return z_next, {
            "field_target_norm": r.norm(dim=-1),
            "field_target_delta_norm": delta.norm(dim=-1),
            "field_target_gate_active_rate": active.float().mean(dim=1),
            "field_target_reset_norm": reset.norm(dim=-1),
        }

    def _layer_forward_with_info(self, layer, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_mid, info = self._field_step(layer, z)
        z_next, act_info = self._apply_activation(layer, z, z_mid, info)
        out = {
            **info,
            **act_info,
            "z_pre": z,
            "z_mid": z_mid,
            "z_next": z_next,
        }
        return z_next, out

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lift(x)
        for layer in self.layers:
            z, _ = self._layer_forward_with_info(layer, z)
        return z

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
        if int(min_children) < 1:
            raise ValueError("min_children must be >= 1")
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
            z, _ = self._layer_forward_with_info(layer, z)
        if was_training:
            self.train()


# ============================================================
# Diagnostics
# ============================================================


@torch.no_grad()
def collect_diagnostics(model: ConservativeAATField, x: torch.Tensor, device: torch.device, batch_size: int = 2048) -> Dict[str, Any]:
    model.eval()
    x = x[: min(int(x.shape[0]), 8192)]
    loader = DataLoader(TensorDataset(x), batch_size=int(batch_size), shuffle=False)

    per_layer: List[Dict[str, List[float]]] = []
    for _ in model.layers:
        per_layer.append({
            "state_norm": [],
            "move_norm": [],
            "move_state_ratio": [],
            "field_target_norm": [],
            "field_target_delta_norm": [],
            "field_target_gate_active_rate": [],
            "field_target_reset_norm": [],
        })

    for (xb,) in loader:
        z = model.lift(xb.to(device))
        for li, layer in enumerate(model.layers):
            z_next, info = model._layer_forward_with_info(layer, z)
            move = info["move"]
            d = per_layer[li]
            d["state_norm"].append(float(z_next.norm(dim=1).mean().item()))
            d["move_norm"].append(float(move.norm(dim=1).mean().item()))
            d["move_state_ratio"].append(float((move.norm(dim=1) / z.norm(dim=1).clamp_min(1e-8)).mean().item()))
            for key in [
                "field_target_norm",
                "field_target_delta_norm",
                "field_target_gate_active_rate",
                "field_target_reset_norm",
            ]:
                if key in info:
                    d[key].append(float(info[key].float().mean().item()))
            z = z_next

    result: Dict[str, Any] = {}
    for li, d in enumerate(per_layer, start=1):
        for key, values in d.items():
            if values:
                result[f"layer{li}_{key}"] = round(sum(values) / len(values), 6)
    for key in ["state_norm", "move_norm", "move_state_ratio", "field_target_norm", "field_target_delta_norm", "field_target_gate_active_rate", "field_target_reset_norm"]:
        vals = [float(result[f"layer{li}_{key}"]) for li in range(1, len(per_layer) + 1) if f"layer{li}_{key}" in result]
        if vals:
            result[f"diag_mean_{key}"] = round(sum(vals) / len(vals), 6)
    return result


# ============================================================
# Experiment loop
# ============================================================


def parse_layers(spec: str) -> List[int]:
    s = str(spec).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def split_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def selected_k_by_layer(model: AATField) -> List[int]:
    out: List[int] = []
    for counts in model.selected_children_by_layer():
        # current Auto-K uses equal K per class; keep average for generality.
        out.append(int(round(sum(counts) / max(len(counts), 1))))
    return out


def row_path_key(core_mode: str, activation_mode: str, layers: int, seed: int) -> str:
    return f"airline__core_{core_mode}__act_{activation_mode}__L{layers}__seed{seed}"


def train_one(
    *,
    data: DataBundle,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    core_mode: str,
    activation_mode: str,
    layers: int,
) -> Dict[str, Any]:
    run_id = row_path_key(core_mode, activation_mode, layers, int(args.seed))
    set_seed(int(args.seed))

    cfg = AATFieldConfig(
        input_dim=data.input_dim,
        extra_dims=int(args.extra_dims),
        num_classes=data.num_classes,
        layers=int(layers),
        max_children=int(args.max_children),
        sigma_init=float(args.sigma_init),
        charge_init=float(args.charge_init),
        step_cap=float(args.step_cap),
        gate_bias=False,  # no contribution-level child gate in this experiment
        head_bias=True,
        lift_seed=int(args.lift_seed),
    )
    model = ConservativeAATField(
        cfg,
        core_mode=core_mode,
        activation_mode=activation_mode,
        field_scale=float(args.field_scale),
        boundary_margin=float(args.boundary_margin),
        boundary_strength=float(args.boundary_strength),
        child_axis_strength=float(args.child_axis_strength),
        use_step_cap=not bool(args.no_step_cap),
    ).to(device)

    row: Dict[str, Any] = {
        "status": "ok",
        "run_id": run_id,
        "seed": int(args.seed),
        "core_mode": core_mode,
        "activation_mode": activation_mode,
        "layers": int(layers),
        "extra_dims": int(args.extra_dims),
        "max_children": int(args.max_children),
        "min_children": int(args.min_children),
        "field_scale": float(args.field_scale),
        "boundary_margin": float(args.boundary_margin),
        "boundary_strength": float(args.boundary_strength),
        "child_axis_strength": float(args.child_axis_strength),
        "params": None,
        "total_children": None,
        "selected_k_by_layer": None,
        "best_epoch": None,
        "best_val_acc": None,
        "best_val_f1": None,
        "test_acc": None,
        "test_f1": None,
        "error": "",
    }

    t0 = time.time()
    try:
        model.initialize(
            data.x_train,
            data.y_train,
            samples=int(args.init_samples),
            min_children=int(args.min_children),
            kmeans_iters=int(args.kmeans_iters),
        )
        init_time = time.time() - t0

        # Important: optimizer after Auto-K materialization.
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        criterion = nn.CrossEntropyLoss()

        best_metric = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0
        best_val_acc = 0.0
        best_val_f1 = 0.0
        train_t0 = time.time()

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                opt.step()

            if epoch == int(args.epochs) or epoch % int(args.eval_every) == 0:
                val_acc, val_f1 = accuracy_and_f1(model, val_loader, device, data.num_classes)
                metric = val_f1 if args.best_metric == "val_f1" else val_acc
                if metric > best_metric:
                    best_metric = float(metric)
                    best_epoch = int(epoch)
                    best_val_acc = float(val_acc)
                    best_val_f1 = float(val_f1)
                    best_state = clone_state_dict_cpu(model)
                print(
                    f"{run_id} | epoch {epoch:03d}/{int(args.epochs)} "
                    f"val_acc={val_acc:.6f} val_f1={val_f1:.6f} best={best_metric:.6f}",
                    flush=True,
                )

        train_time = time.time() - train_t0
        if best_state is not None:
            load_state_dict_to_device(model, best_state, device)
        test_acc, test_f1 = accuracy_and_f1(model, test_loader, device, data.num_classes)
        diag = collect_diagnostics(model, data.x_val, device, batch_size=int(args.eval_batch_size))

        row.update({
            "params": int(count_parameters(model)),
            "total_children": int(model.total_children()),
            "selected_k_by_layer": json.dumps(selected_k_by_layer(model)),
            "best_epoch": int(best_epoch),
            "best_val_acc": round(float(best_val_acc), 6),
            "best_val_f1": round(float(best_val_f1), 6),
            "test_acc": round(float(test_acc), 6),
            "test_f1": round(float(test_f1), 6),
            "init_time_sec": round(float(init_time), 3),
            "train_time_sec": round(float(train_time), 3),
            "total_time_sec": round(float(time.time() - t0), 3),
            **diag,
        })
        if bool(args.save_checkpoints):
            ckpt_dir = Path(args.out_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "ConservativeAATFieldExperimentCheckpoint",
                "config": model.config_dict(),
                "core_mode": core_mode,
                "activation_mode": activation_mode,
                "state_dict": model.state_dict(),
                "row": row,
            }, ckpt_dir / f"{run_id}.pt")
    except Exception as exc:  # keep grid running on one failed run
        row["status"] = "error"
        row["error"] = repr(exc)
        print(f"ERROR in {run_id}: {exc!r}", flush=True)
    return row


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union fieldnames, stable common fields first.
    first = [
        "status", "run_id", "seed", "core_mode", "activation_mode", "layers",
        "best_epoch", "best_val_acc", "best_val_f1", "test_acc", "test_f1",
        "params", "total_children", "selected_k_by_layer", "field_scale",
        "boundary_margin", "boundary_strength", "child_axis_strength", "error",
    ]
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    fieldnames = first + [k for k in keys if k not in first]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: float(r.get("test_f1") or -1), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not ok:
            f.write("no ok rows\n")
            return
        fields = [
            "core_mode", "activation_mode", "layers", "best_val_acc", "best_val_f1",
            "test_acc", "test_f1", "params", "total_children", "selected_k_by_layer",
            "diag_mean_move_norm", "diag_mean_move_state_ratio",
            "diag_mean_field_target_norm", "diag_mean_field_target_delta_norm",
            "diag_mean_field_target_gate_active_rate", "diag_mean_field_target_reset_norm",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ok:
            writer.writerow({k: r.get(k, "") for k in fields})


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Field-target ReLU AAT Airline experiment")
    p.add_argument("--data-dir", type=str, default="../data/AirlineSatisfaction")
    p.add_argument("--out-dir", type=str, default="./airline_field_target_relu")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--core-modes", type=str, default="potential", help="potential,current_nogate")
    p.add_argument(
        "--activation-modes",
        type=str,
        default="field_target_relu",
        help="field_target_relu",
    )
    p.add_argument("--layers", type=str, default="1-8")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--extra-dims", type=int, default=2)
    p.add_argument("--max-children", type=int, default=100)
    p.add_argument("--min-children", type=int, default=2)
    p.add_argument("--init-samples", type=int, default=8192)
    p.add_argument("--kmeans-iters", type=int, default=8)
    p.add_argument("--sigma-init", type=float, default=0.75)
    p.add_argument("--charge-init", type=float, default=0.08)
    p.add_argument("--step-cap", type=float, default=1.0)
    p.add_argument("--lift-seed", type=int, default=1234)

    p.add_argument("--field-scale", type=float, default=0.08, help="Scale for potential gradient transport. 0.05~0.15 is a safe first range.")
    # Kept only for compatibility with shared train_one signature; unused in this script.
    p.add_argument("--boundary-margin", type=float, default=0.30)
    p.add_argument("--boundary-strength", type=float, default=0.50)
    p.add_argument("--child-axis-strength", type=float, default=0.50)
    p.add_argument("--no-step-cap", action="store_true")
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--best-metric", choices=["val_acc", "val_f1"], default="val_f1")
    p.add_argument("--fresh", action="store_true", help="Delete old results.csv/summary.csv before running")
    p.add_argument("--limit", type=int, default=0, help="Run only first N grid items for smoke test")
    p.add_argument("--save-checkpoints", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    if args.fresh:
        for p in [results_csv, summary_csv]:
            if p.exists():
                p.unlink()

    device = torch.device(str(args.device))
    print(f"device={device}", flush=True)
    data = airline_bundle(data_dir / "train.csv", data_dir / "test.csv", val_ratio=float(args.val_ratio), seed=int(args.seed) + 123)
    train_loader, val_loader, test_loader = make_loaders(data, int(args.batch_size))

    grid: List[Tuple[str, str, int]] = []
    for core_mode in split_csv_list(args.core_modes):
        for activation_mode in split_csv_list(args.activation_modes):
            for layers in parse_layers(args.layers):
                grid.append((core_mode, activation_mode, int(layers)))
    if int(args.limit) > 0:
        grid = grid[: int(args.limit)]

    print(f"grid={len(grid)} runs: {grid}", flush=True)
    all_rows: List[Dict[str, Any]] = []
    for i, (core_mode, activation_mode, layers) in enumerate(grid, start=1):
        print(f"\n=== [{i}/{len(grid)}] core={core_mode} activation={activation_mode} layers={layers} ===", flush=True)
        row = train_one(
            data=data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            core_mode=core_mode,
            activation_mode=activation_mode,
            layers=int(layers),
        )
        all_rows.append(row)
        append_csv(results_csv, [row])
        write_summary(summary_csv, all_rows)
        print(f"row: test_acc={row.get('test_acc')} test_f1={row.get('test_f1')} selected={row.get('selected_k_by_layer')}", flush=True)

    print(f"\nSaved: {results_csv}", flush=True)
    print(f"Saved: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
