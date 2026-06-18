# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
GRID_SIZE = 4
N = 4096
EXTRA_DIMS = 3
MIN_K = 2
MAX_K = 80
KMEANS_ITERS = 8

METHODS = [
    "weighted_kmeans",
    "unweighted_kmeans",
    "farthest_boundary",
    "kmeanspp",
    "pca_split",
]


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


@torch.no_grad()
def farthest_boundary(pts: torch.Tensor, weights: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.float()
    w = weights.float().clamp_min(1e-8)
    n, d = pts.shape
    k = int(max(1, min(int(k), n)))

    centers = torch.empty((k, d), device=pts.device, dtype=pts.dtype)
    centers[0] = (pts * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-8)
    nearest = pairwise_dist2(pts, centers[:1]).squeeze(1)

    for j in range(1, k):
        idx = int((nearest * w).argmax().item())
        centers[j] = pts[idx]
        nearest = torch.minimum(nearest, pairwise_dist2(pts, centers[j:j + 1]).squeeze(1))

    assign = pairwise_dist2(pts, centers).argmin(dim=1)
    return centers, assign


@torch.no_grad()
def random_kmeanspp(
    pts: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    weighted_update: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.float()
    w = weights.float().clamp_min(1e-8)
    n, d = pts.shape
    k = int(max(1, min(int(k), n)))

    gen = torch.Generator(device=pts.device)
    gen.manual_seed(int(seed))

    probs = w / w.sum().clamp_min(1e-8)
    first = int(torch.multinomial(probs, 1, generator=gen).item())

    centers = torch.empty((k, d), device=pts.device, dtype=pts.dtype)
    centers[0] = pts[first]
    nearest = pairwise_dist2(pts, centers[:1]).squeeze(1)

    for j in range(1, k):
        score = (nearest * w).clamp_min(1e-12)
        probs = score / score.sum().clamp_min(1e-8)
        idx = int(torch.multinomial(probs, 1, generator=gen).item())
        centers[j] = pts[idx]
        nearest = torch.minimum(nearest, pairwise_dist2(pts, centers[j:j + 1]).squeeze(1))

    ww = w if weighted_update else torch.ones_like(w)
    assign = torch.zeros(n, dtype=torch.long, device=pts.device)
    for _ in range(int(iters)):
        assign = pairwise_dist2(pts, centers).argmin(dim=1)

        weighted_pts = pts * ww[:, None]
        sum_w = torch.zeros(k, device=pts.device, dtype=pts.dtype)
        sum_w.scatter_add_(0, assign, ww)

        sum_wp = torch.zeros(k, d, device=pts.device, dtype=pts.dtype)
        sum_wp.scatter_add_(0, assign.view(-1, 1).expand(-1, d), weighted_pts)

        updated = sum_wp / sum_w.clamp_min(1e-8).view(-1, 1)
        centers = torch.where((sum_w > 0).view(-1, 1), updated, centers)

    assign = pairwise_dist2(pts, centers).argmin(dim=1)
    return centers, assign


@torch.no_grad()
def pca_split(pts: torch.Tensor, weights: torch.Tensor, k: int, iters: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.float()
    w = weights.float().clamp_min(1e-8)
    n, d = pts.shape
    k = int(max(1, min(int(k), n)))

    clusters: List[torch.Tensor] = [torch.arange(n, device=pts.device)]
    while len(clusters) < k:
        sizes = torch.tensor([idx.numel() for idx in clusters], device=pts.device)
        ci = int(sizes.argmax().item())
        idx = clusters.pop(ci)

        if idx.numel() <= 1:
            clusters.append(idx)
            break

        sub = pts[idx]
        sub_w = w[idx]
        mu = (sub * sub_w[:, None]).sum(dim=0) / sub_w.sum().clamp_min(1e-8)
        xc = sub - mu
        cov = (xc * sub_w[:, None]).t() @ xc / sub_w.sum().clamp_min(1e-8)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        axis = eigvecs[:, eigvals.argmax()]
        proj = sub @ axis
        cut = torch.median(proj)

        left = idx[proj <= cut]
        right = idx[proj > cut]
        if left.numel() == 0 or right.numel() == 0:
            perm = idx[torch.argsort(proj)]
            half = max(1, perm.numel() // 2)
            left, right = perm[:half], perm[half:]

        clusters.append(left)
        if right.numel() > 0:
            clusters.append(right)

    if len(clusters) > k:
        clusters = clusters[:k]

    centers = torch.zeros(k, d, device=pts.device, dtype=pts.dtype)
    for i, idx in enumerate(clusters):
        if idx.numel() == 0:
            centers[i] = pts[int(torch.argmax(w).item())]
        else:
            ww = w[idx]
            centers[i] = (pts[idx] * ww[:, None]).sum(dim=0) / ww.sum().clamp_min(1e-8)

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


@torch.no_grad()
def cluster_points(method: str, pts: torch.Tensor, weights: torch.Tensor, k: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "weighted_kmeans":
        return weighted_kmeans(pts, weights, k, KMEANS_ITERS)
    if method == "unweighted_kmeans":
        return weighted_kmeans(pts, torch.ones_like(weights), k, KMEANS_ITERS)
    if method == "farthest_boundary":
        return farthest_boundary(pts, weights, k)
    if method == "kmeanspp":
        return random_kmeanspp(pts, weights, k, KMEANS_ITERS, seed, weighted_update=True)
    if method == "pca_split":
        return pca_split(pts, weights, k, iters=max(2, KMEANS_ITERS // 2))
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp_min(1e-12)
    return -(p * p.log()).sum()


@torch.no_grad()
def weighted_binary_nmi(membership: torch.Tensor, y_bin: torch.Tensor) -> Tuple[float, float, float]:
    m = membership.float().clamp_min(0.0)
    y_bin = y_bin.long().to(m.device)
    n, k = m.shape
    m = m / m.sum(dim=1, keepdim=True).clamp_min(1e-8)

    joint = torch.zeros(k, 2, device=m.device, dtype=m.dtype)
    joint[:, 0] = m[y_bin == 0].sum(dim=0)
    joint[:, 1] = m[y_bin == 1].sum(dim=0)
    joint = joint / max(n, 1)

    pk = joint.sum(dim=1)
    py = joint.sum(dim=0)
    denom = pk[:, None] * py[None, :]

    mask = joint > 1e-12
    mi = (joint[mask] * (joint[mask] / denom[mask].clamp_min(1e-12)).log()).sum()
    nmi = mi / torch.sqrt((entropy(pk) * entropy(py)).clamp_min(1e-12))
    purity = joint.max(dim=1).values.sum()
    return float(mi.item()), float(nmi.item()), float(purity.item())


@torch.no_grad()
def memberships_from_centers(z: torch.Tensor, centers: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    dist2 = pairwise_dist2(z, centers)
    logits = -dist2 / (2.0 * float(sigma) * float(sigma) + 1e-8)
    soft = torch.softmax(logits, dim=1)
    hard_idx = soft.argmax(dim=1)
    hard = F.one_hot(hard_idx, num_classes=centers.shape[0]).float()
    return hard, soft


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
def child_response_features_for_class(z: torch.Tensor, parent: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
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
def response_membership(z: torch.Tensor, parent: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    phi = child_response_features_for_class(z, parent, centers, sigma)
    row_sum = phi.sum(dim=1, keepdim=True)
    _, soft = memberships_from_centers(z, centers, sigma)
    return torch.where(row_sum > 1e-8, phi / row_sum.clamp_min(1e-8), soft)


@torch.no_grad()
def inertia_for_class(pts: torch.Tensor, centers: torch.Tensor, weights: torch.Tensor) -> float:
    nearest = pairwise_dist2(pts, centers).min(dim=1).values
    return float((nearest * weights.float()).sum().item() / weights.sum().clamp_min(1e-8).item())


@torch.no_grad()
def score_k(
    z: torch.Tensor,
    y: torch.Tensor,
    parents: torch.Tensor,
    centers_by_class: List[torch.Tensor],
    sigma: float,
) -> Dict[str, float]:
    old_vals, hard_nmi_vals, soft_nmi_vals, resp_nmi_vals, purity_vals = [], [], [], [], []

    for c in range(parents.shape[0]):
        centers = centers_by_class[c]
        bin_y = (y == c).long()

        phi = child_response_features_for_class(z, parents[c], centers, sigma)
        old_vals.append(math.log1p(max(supervised_fisher_score(phi, bin_y, 2), 0.0)))

        hard, soft = memberships_from_centers(z, centers, sigma)
        _, hard_nmi, purity = weighted_binary_nmi(hard, bin_y)
        _, soft_nmi, _ = weighted_binary_nmi(soft, bin_y)
        _, resp_nmi, _ = weighted_binary_nmi(response_membership(z, parents[c], centers, sigma), bin_y)

        hard_nmi_vals.append(hard_nmi)
        soft_nmi_vals.append(soft_nmi)
        resp_nmi_vals.append(resp_nmi)
        purity_vals.append(purity)

    def avg(xs: List[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "old_fisher_log": avg(old_vals),
        "hard_nmi": avg(hard_nmi_vals),
        "soft_nmi": avg(soft_nmi_vals),
        "resp_nmi": avg(resp_nmi_vals),
        "purity": avg(purity_vals),
    }


def argmax_k(rows: List[Dict[str, float]], key: str) -> int:
    return int(max(rows, key=lambda r: float(r[key]))["K"])


def first95_k(rows: List[Dict[str, float]], key: str) -> int:
    target = max(float(r[key]) for r in rows) * 0.95
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
    d = torch.stack([x, y], dim=1).sub(torch.stack([x, x], dim=1)).norm(dim=1)
    return int(rows[int(d.argmax().item())]["K"])


@dataclass
class ProbeResult:
    method: str
    rows: List[Dict[str, float]]


@torch.no_grad()
def run_probe_for_method(
    method: str,
    z: torch.Tensor,
    y: torch.Tensor,
    parents: torch.Tensor,
    weights_by_class: List[torch.Tensor],
) -> ProbeResult:
    c_num = parents.shape[0]
    pts_by_class = [z[y == c] if (y == c).any() else z for c in range(c_num)]

    full_centers = []
    for c in range(c_num):
        centers, _ = cluster_points(method, pts_by_class[c], weights_by_class[c], MAX_K, SEED + 1000 * c + MAX_K)
        full_centers.append(centers)

    full_anchors = torch.cat([parents] + full_centers, dim=0)
    nearest = torch.sqrt(pairwise_dist2(z, full_anchors).min(dim=1).values + 1e-8)
    sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
    sigma = max(0.05, min(3.0, sigma))

    rows: List[Dict[str, float]] = []
    for k in range(MIN_K, MAX_K + 1):
        centers_by_class = []
        inertia_vals = []

        for c in range(c_num):
            centers, _ = cluster_points(method, pts_by_class[c], weights_by_class[c], k, SEED + 1000 * c + k)
            centers_by_class.append(centers)
            inertia_vals.append(inertia_for_class(pts_by_class[c], centers, weights_by_class[c]))

        row = {
            "K": int(k),
            "sigma": float(sigma),
            "weighted_inertia": float(sum(inertia_vals) / max(len(inertia_vals), 1)),
            "neglog_inertia": float(-math.log(max(sum(inertia_vals) / max(len(inertia_vals), 1), 1e-12))),
        }
        row.update(score_k(z, y, parents, centers_by_class, sigma))
        rows.append(row)

    return ProbeResult(method=method, rows=rows)


def print_summary(results: List[ProbeResult]) -> None:
    print("\n=== selected K by method ===")
    print("method                old_arg  hard95  hard_knee  hard_arg  soft95  resp95  purity95  inertia_knee")
    print("-" * 105)

    for res in results:
        print(
            f"{res.method:20s}"
            f"{argmax_k(res.rows, 'old_fisher_log'):8d}"
            f"{first95_k(res.rows, 'hard_nmi'):8d}"
            f"{knee_k(res.rows, 'hard_nmi', True):11d}"
            f"{argmax_k(res.rows, 'hard_nmi'):10d}"
            f"{first95_k(res.rows, 'soft_nmi'):8d}"
            f"{first95_k(res.rows, 'resp_nmi'):8d}"
            f"{first95_k(res.rows, 'purity'):10d}"
            f"{knee_k(res.rows, 'weighted_inertia', False):14d}"
        )

    print("\n=== top hard_nmi rows ===")
    for res in results:
        top = sorted(res.rows, key=lambda r: r["hard_nmi"], reverse=True)[:5]
        print(f"\n[{res.method}]")
        for r in top:
            print(
                f"K={int(r['K']):02d} "
                f"hard_nmi={r['hard_nmi']:.4f} "
                f"soft_nmi={r['soft_nmi']:.4f} "
                f"resp_nmi={r['resp_nmi']:.4f} "
                f"purity={r['purity']:.4f} "
                f"old={r['old_fisher_log']:.4f} "
                f"inertia={r['weighted_inertia']:.6f}"
            )


def plot_results(results: List[ProbeResult]) -> None:
    plots = [
        ("hard_nmi", "Hard NMI"),
        ("soft_nmi", "Soft NMI"),
        ("resp_nmi", "Response NMI"),
        ("purity", "Purity"),
        ("old_fisher_log", "Old Fisher Score"),
        ("neglog_inertia", "-log Weighted Inertia"),
    ]

    for key, title in plots:
        plt.figure(figsize=(13, 5))
        for res in results:
            xs = [r["K"] for r in res.rows]
            ys = [r[key] for r in res.rows]
            plt.plot(xs, ys, marker="o", markersize=2, label=res.method)

        plt.xlabel("K")
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    plt.show()


def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)

    x, y = make_checkerboard_3d(N, GRID_SIZE, SEED)
    z = lift_extra(x, EXTRA_DIMS).to(device)
    y = y.to(device)

    parents = torch.zeros(2, z.shape[1], device=device)
    global_mean = z.mean(dim=0)
    for c in range(2):
        pts = z[y == c]
        parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

    weights_by_class = []
    for c in range(2):
        pts = z[y == c]
        weights_by_class.append(boundary_weights(pts, c, parents))

    print(
        f"dataset=checkerboard3d_g{GRID_SIZE} n={N} "
        f"state_dim={z.shape[1]} K={MIN_K}..{MAX_K} device={device}"
    )
    print(f"methods={METHODS}")
    print(f"parents_norm={[float(v) for v in parents.norm(dim=1).detach().cpu()]}")

    results = []
    for method in METHODS:
        print(f"\nrun method: {method}")
        results.append(run_probe_for_method(method, z, y, parents, weights_by_class))

    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main()
