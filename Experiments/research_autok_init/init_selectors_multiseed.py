# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Fixed experiment config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_SEEDS = [0, 1, 2, 3, 4]
GRID_SIZE = 4
N = 4096

EXTRA_DIMS = 3
MIN_K = 2
MAX_K = 80
KMEANS_ITERS = 8

# The high-performance plateau observed from training sweep.
GOOD_K_MIN = 34
GOOD_K_MAX = 60


# ============================================================
# Basic utilities
# ============================================================

@torch.no_grad()
def pairwise_dist2(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x = x.float()
    centers = centers.float()
    return (
        (x * x).sum(dim=1, keepdim=True)
        + (centers * centers).sum(dim=1).view(1, -1)
        - 2.0 * x @ centers.t()
    ).clamp_min(0.0)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def make_checkerboard_3d(n: int, grid_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    x = torch.rand(int(n), 3, generator=gen)
    cells = torch.floor(x * int(grid_size)).long().clamp(max=int(grid_size) - 1)
    y = (cells.sum(dim=1) % 2).long()

    idx = torch.randperm(int(n), generator=gen)
    return x[idx], y[idx]


@torch.no_grad()
def lift_extra(x: torch.Tensor, extra_dims: int) -> torch.Tensor:
    z = x.float() * 2.0 - 1.0
    if int(extra_dims) > 0:
        z = torch.cat([z, z.new_zeros(z.shape[0], int(extra_dims))], dim=1)
    return z


@torch.no_grad()
def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(1e-12)
    return -(p * p.log()).sum()


# ============================================================
# Original placement logic
# ============================================================

@torch.no_grad()
def boundary_weights(pts: torch.Tensor, class_id: int, parents: torch.Tensor) -> torch.Tensor:
    pts = pts.float()
    parents = parents.float()

    d = torch.sqrt(pairwise_dist2(pts, parents) + 1e-8)
    d_own = d[:, int(class_id)]

    if parents.shape[0] <= 1:
        return torch.ones_like(d_own)

    mask = torch.ones(parents.shape[0], dtype=torch.bool, device=pts.device)
    mask[int(class_id)] = False
    d_other = d[:, mask].min(dim=1).values

    silhouette = (d_other - d_own) / torch.maximum(d_own, d_other).clamp_min(1e-8)
    w = (1.0 - silhouette).clamp_min(1e-3)
    return w / w.mean().clamp_min(1e-8)


@torch.no_grad()
def weighted_kmeans(pts: torch.Tensor, weights: torch.Tensor, k: int, iters: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.float()
    w = weights.float().clamp_min(1e-8)

    n, d = pts.shape
    k = int(max(1, min(int(k), n)))

    centers = torch.empty((k, d), device=pts.device, dtype=pts.dtype)
    centers[0] = (pts * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-8)

    if k > 1:
        nearest = pairwise_dist2(pts, centers[:1]).squeeze(1)
        for j in range(1, k):
            score = nearest * w
            idx = int(score.argmax().item())
            centers[j] = pts[idx]
            nearest = torch.minimum(nearest, pairwise_dist2(pts, centers[j:j + 1]).squeeze(1))

    assign = torch.zeros(n, dtype=torch.long, device=pts.device)
    for _ in range(int(iters)):
        assign = pairwise_dist2(pts, centers).argmin(dim=1)

        weighted_pts = pts * w[:, None]

        sum_w = torch.zeros(k, device=pts.device, dtype=pts.dtype)
        sum_w.scatter_add_(0, assign, w)

        sum_wp = torch.zeros(k, d, device=pts.device, dtype=pts.dtype)
        sum_wp.scatter_add_(0, assign.view(-1, 1).expand(-1, d), weighted_pts)

        updated = sum_wp / sum_w.clamp_min(1e-8).view(-1, 1)
        centers = torch.where((sum_w > 0).view(-1, 1), updated, centers)

    assign = pairwise_dist2(pts, centers).argmin(dim=1)
    return centers, assign


# ============================================================
# Old Fisher score
# ============================================================

@torch.no_grad()
def supervised_fisher_score(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    z = z.float()
    y = y.long().to(z.device)

    c_num = int(num_classes)
    if z.shape[0] == 0 or c_num <= 0:
        return 0.0

    n = max(int(z.shape[0]), 1)
    d = int(z.shape[1])
    global_mu = z.mean(dim=0)

    counts = torch.bincount(y, minlength=c_num)[:c_num].to(device=z.device, dtype=z.dtype)
    valid_mask = counts > 0
    valid = int(valid_mask.sum().item())

    if valid <= 1 or n <= valid:
        return 0.0

    class_sums = torch.zeros(c_num, d, device=z.device, dtype=z.dtype)
    class_sums.scatter_add_(0, y.view(-1, 1).expand(-1, d), z)
    class_means = class_sums / counts.clamp_min(1.0).view(c_num, 1)

    sample_means = class_means.index_select(0, y)
    w = (z - sample_means).square().sum().clamp_min(1e-12)
    b = (counts[valid_mask] * (class_means[valid_mask] - global_mu).square().sum(dim=1)).sum()

    score = (b / max(valid - 1, 1)) / (w / max(n - valid, 1)).clamp_min(1e-12)
    return float(score.item())


@torch.no_grad()
def child_response_features_for_class(
    z: torch.Tensor,
    parent: torch.Tensor,
    centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    if centers.numel() == 0:
        return z.new_zeros((z.shape[0], 1))

    anchors = torch.cat([parent.view(1, -1), centers], dim=0)
    dist2 = pairwise_dist2(z, anchors)
    alpha = torch.softmax(-dist2 / (2.0 * float(sigma) * float(sigma) + 1e-8), dim=-1)[:, 1:]

    axis = centers - parent.view(1, -1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    s = ((z.unsqueeze(1) - centers.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=-1)
    s = s / max(float(sigma), 1e-6)
    return alpha * F.relu(s)


@torch.no_grad()
def old_fisher_score(
    z: torch.Tensor,
    y: torch.Tensor,
    parents: torch.Tensor,
    centers_by_class: List[torch.Tensor],
    sigma: float,
) -> float:
    values = []
    for c in range(parents.shape[0]):
        bin_y = (y == c).long()
        phi = child_response_features_for_class(z, parents[c], centers_by_class[c], sigma)
        sc = supervised_fisher_score(phi, bin_y, num_classes=2)
        values.append(math.log1p(max(sc, 0.0)))
    return float(sum(values) / max(len(values), 1))


# ============================================================
# Global information scores
# ============================================================

@torch.no_grad()
def global_memberships(z: torch.Tensor, centers_by_class: List[torch.Tensor], sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    centers = torch.cat(centers_by_class, dim=0)
    dist2 = pairwise_dist2(z, centers)
    logits = -dist2 / (2.0 * float(sigma) * float(sigma) + 1e-8)

    soft = torch.softmax(logits, dim=1)
    hard_idx = soft.argmax(dim=1)
    hard = F.one_hot(hard_idx, num_classes=centers.shape[0]).float()
    return hard, soft


@torch.no_grad()
def membership_nmi_and_purity(membership: torch.Tensor, y: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    m = membership.float().clamp_min(0.0)
    y = y.long().to(m.device)
    n, k = m.shape
    c_num = int(num_classes)

    m = m / m.sum(dim=1, keepdim=True).clamp_min(1e-8)

    joint = torch.zeros(k, c_num, device=m.device, dtype=m.dtype)
    for c in range(c_num):
        joint[:, c] = m[y == c].sum(dim=0)
    joint = joint / max(n, 1)

    pk = joint.sum(dim=1)
    pc = joint.sum(dim=0)
    denom = pk[:, None] * pc[None, :]

    mask = joint > 1e-12
    mi = (joint[mask] * (joint[mask] / denom[mask].clamp_min(1e-12)).log()).sum()

    nmi = mi / torch.sqrt((entropy(pk) * entropy(pc)).clamp_min(1e-12))
    purity = joint.max(dim=1).values.sum()

    return float(nmi.item()), float(purity.item())


@torch.no_grad()
def weighted_inertia(pts_by_class: List[torch.Tensor], weights_by_class: List[torch.Tensor], centers_by_class: List[torch.Tensor]) -> float:
    values = []
    for pts, w, centers in zip(pts_by_class, weights_by_class, centers_by_class):
        nearest = pairwise_dist2(pts, centers).min(dim=1).values
        values.append(float((nearest * w.float()).sum().item() / w.sum().clamp_min(1e-8).item()))
    return float(sum(values) / max(len(values), 1))


# ============================================================
# Selectors
# ============================================================

def argmax_k(rows: List[Dict[str, float]], key: str) -> int:
    return int(max(rows, key=lambda r: float(r[key]))["K"])


def first_fraction_k(rows: List[Dict[str, float]], key: str, frac: float) -> int:
    best = max(float(r[key]) for r in rows)
    target = best * float(frac)
    for r in rows:
        if float(r[key]) >= target:
            return int(r["K"])
    return int(rows[-1]["K"])


def knee_k(rows: List[Dict[str, float]], key: str, maximize: bool = True) -> int:
    xs = torch.tensor([float(r["K"]) for r in rows])
    ys = torch.tensor([float(r[key]) for r in rows])

    if not maximize:
        ys = -ys

    x = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1e-8)
    y = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1e-8)

    # distance from diagonal line; simple elbow detector
    d = torch.stack([x, y], dim=1).sub(torch.stack([x, x], dim=1)).norm(dim=1)
    return int(rows[int(d.argmax().item())]["K"])


def summarize_selectors(rows: List[Dict[str, float]]) -> Dict[str, int]:
    return {
        "old_fisher_argmax": argmax_k(rows, "old_fisher_log"),
        "hard_nmi_knee": knee_k(rows, "hard_nmi", maximize=True),
        "hard_nmi_95": first_fraction_k(rows, "hard_nmi", 0.95),
        "purity_90": first_fraction_k(rows, "purity", 0.90),
        "purity_92": first_fraction_k(rows, "purity", 0.92),
        "purity_95": first_fraction_k(rows, "purity", 0.95),
        "purity_98": first_fraction_k(rows, "purity", 0.98),
        "inertia_knee": knee_k(rows, "weighted_inertia", maximize=False),
        "hard_nmi_argmax": argmax_k(rows, "hard_nmi"),
        "purity_argmax": argmax_k(rows, "purity"),
    }


# ============================================================
# One seed probe
# ============================================================

@dataclass
class SeedResult:
    seed: int
    rows: List[Dict[str, float]]
    selected: Dict[str, int]


@torch.no_grad()
def run_seed(seed: int, device: torch.device) -> SeedResult:
    set_seed(seed)

    x, y = make_checkerboard_3d(N, GRID_SIZE, seed)
    z = lift_extra(x, EXTRA_DIMS).to(device)
    y = y.to(device)

    c_num = 2
    parents = torch.zeros(c_num, z.shape[1], device=device)
    global_mean = z.mean(dim=0)

    for c in range(c_num):
        pts = z[y == c]
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    pts_by_class = []
    weights_by_class = []
    for c in range(c_num):
        pts = z[y == c]
        if pts.shape[0] == 0:
            pts = z
        pts_by_class.append(pts)
        weights_by_class.append(boundary_weights(pts, c, parents))

    full_centers = []
    for c in range(c_num):
        centers, _ = weighted_kmeans(pts_by_class[c], weights_by_class[c], MAX_K, KMEANS_ITERS)
        full_centers.append(centers)

    full_anchors = torch.cat([parents] + full_centers, dim=0)
    nearest = torch.sqrt(pairwise_dist2(z, full_anchors).min(dim=1).values + 1e-8)
    sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
    sigma = max(0.05, min(3.0, sigma))

    rows: List[Dict[str, float]] = []
    for k in range(MIN_K, MAX_K + 1):
        centers_by_class = []
        for c in range(c_num):
            centers, _ = weighted_kmeans(pts_by_class[c], weights_by_class[c], k, KMEANS_ITERS)
            centers_by_class.append(centers)

        hard, soft = global_memberships(z, centers_by_class, sigma)
        hard_nmi, purity = membership_nmi_and_purity(hard, y, c_num)
        soft_nmi, soft_purity = membership_nmi_and_purity(soft, y, c_num)

        row = {
            "K": int(k),
            "old_fisher_log": old_fisher_score(z, y, parents, centers_by_class, sigma),
            "hard_nmi": hard_nmi,
            "soft_nmi": soft_nmi,
            "purity": purity,
            "soft_purity": soft_purity,
            "weighted_inertia": weighted_inertia(pts_by_class, weights_by_class, centers_by_class),
            "sigma": sigma,
        }
        rows.append(row)

    return SeedResult(seed=seed, rows=rows, selected=summarize_selectors(rows))


# ============================================================
# Print + plot
# ============================================================

def mean_std(values: List[int]) -> Tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.mean().item()), float(t.std(unbiased=True).item()) if len(values) > 1 else 0.0


def print_results(results: List[SeedResult]) -> None:
    selector_names = list(results[0].selected.keys())

    print("\n=== selected K per seed ===")
    header = "seed " + " ".join([f"{name:>17s}" for name in selector_names])
    print(header)
    print("-" * len(header))

    for res in results:
        line = f"{res.seed:4d} " + " ".join([f"{res.selected[name]:17d}" for name in selector_names])
        print(line)

    print("\n=== selector stability ===")
    print("selector              mean     std    min    max   in_34_60")
    print("-" * 62)

    for name in selector_names:
        ks = [res.selected[name] for res in results]
        m, s = mean_std(ks)
        hit = sum(GOOD_K_MIN <= k <= GOOD_K_MAX for k in ks)
        print(f"{name:20s}{m:7.2f}{s:8.2f}{min(ks):7d}{max(ks):7d}{hit:9d}/{len(ks)}")

    print("\n=== top rows by hard_nmi for each seed ===")
    for res in results:
        top = sorted(res.rows, key=lambda r: r["hard_nmi"], reverse=True)[:5]
        print(f"\n[seed {res.seed}]")
        for r in top:
            print(
                f"K={int(r['K']):02d} "
                f"hard_nmi={r['hard_nmi']:.4f} "
                f"soft_nmi={r['soft_nmi']:.4f} "
                f"purity={r['purity']:.4f} "
                f"old={r['old_fisher_log']:.4f} "
                f"inertia={r['weighted_inertia']:.6f}"
            )


def plot_curves(results: List[SeedResult]) -> None:
    for key, title in [
        ("hard_nmi", "Global hard NMI across data seeds"),
        ("purity", "Global purity across data seeds"),
        ("old_fisher_log", "Old Fisher score across data seeds"),
        ("weighted_inertia", "Weighted inertia across data seeds"),
    ]:
        plt.figure(figsize=(13, 5))
        for res in results:
            xs = [r["K"] for r in res.rows]
            ys = [r[key] for r in res.rows]
            plt.plot(xs, ys, marker="o", markersize=2, label=f"seed {res.seed}")

        plt.xlabel("K")
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    selector_names = list(results[0].selected.keys())
    plt.figure(figsize=(14, 6))
    for i, name in enumerate(selector_names):
        ys = [res.selected[name] for res in results]
        xs = [i + 1] * len(ys)
        plt.scatter(xs, ys, s=70)
        m, _ = mean_std(ys)
        plt.scatter([i + 1], [m], s=180, marker="x")

    plt.xticks(range(1, len(selector_names) + 1), selector_names, rotation=35, ha="right")
    plt.ylabel("selected K")
    plt.title("Selected K by selector across data seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


def main() -> None:
    device = torch.device(DEVICE)

    print(
        f"dataset=checkerboard3d_g{GRID_SIZE} N={N} state_dim={3 + EXTRA_DIMS} "
        f"K={MIN_K}..{MAX_K} seeds={DATA_SEEDS} device={device}"
    )
    print("placement=boundary_weights + weighted_kmeans")

    results = []
    for seed in DATA_SEEDS:
        print(f"\nrun seed {seed}")
        results.append(run_seed(seed, device))

    print_results(results)
    plot_curves(results)


if __name__ == "__main__":
    main()
