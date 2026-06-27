# -*- coding: utf-8 -*-
r"""
experiment_mnist_aat_field_oriented_activation.py

MNIST AAT Gaussian-field experiment for field-oriented activation.

Default scan:
  K                = 16, 32, 64
  depth            = 3, 6, 10
  activation modes = old_response, field_dir

All runs keep the same transport mechanism:
  - label-free k-means anchor initialization
  - trainable anchor positions
  - shared sigma
  - Gaussian signed conservative terrain transport
  - linear head

The only intended mechanism change is the activation between layers.

Gaussian field:
  G_j(z) = exp(-||z-a_j||^2 / (2 sigma^2))
  F(z)   = eta / sqrt(K) * sum_j c_j G_j(z) (a_j-z) / sigma^2
  z'     = z + F(z)

This isolates the question:
  Can a field-generated local activation direction replace the expensive learned rotation matrix,
  while keeping the conservative transport field unchanged?

Activation modes:
  old_response:
    old response-center axis-aligned ReLU: r + ReLU(z_mid - r)

  field_dir:
    global-zero directional ReLU. A separate scalar activation terrain Psi gives
    u(z)=grad Psi(z)/||grad Psi(z)||. Then activation keeps the perpendicular part and
    applies ReLU only along u:
      z_out = z_mid + (ReLU(<z_mid,u>) - <z_mid,u>) u

Run:
  python experiment_mnist_aat_field_oriented_activation.py
  python experiment_mnist_aat_field_oriented_activation.py --k-list 16,32,64 --depth-list 3,6,10 --activation-modes old_response,field_dir
"""

import argparse
import csv
import gzip
import json
import math
import random
import struct
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Defaults
# ============================================================

SEED = 42
MNIST_RAW_DIR = r"C:\Projets\AATField\Experiments\data\MNIST\raw"

N_TRAIN = 10000
N_VAL = 10000
BATCH_SIZE = 256
LR = 2e-3
SIGMA_LR = 5e-4
USE_AMP = True
BASE_DIM = 28 * 28
C_INIT_STD = 0.04
EPS = 1e-12

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


# ============================================================
# Variants
# ============================================================

@dataclass
class Variant:
    name: str
    init_method: str = "kmeans"       # keep this experiment focused on k-means data anchors
    per_anchor_sigma: bool = False    # shared sigma by default; same as the current best Gaussian result
    eta: float = 1.0
    zero_mean_c: bool = True
    use_activation: bool = True
    sigma_scale: float = 1.0
    sigma_knn: int = 8
    trainable_anchors: bool = False
    depth_override: int = 0           # 0 means use --depth; otherwise use this depth
    activation_mode: str = "old_response"  # old_response | field_dir


# Variants are created dynamically from --k-list, --depth-list, and --activation-modes.
# Every scan entry uses trainable transport anchors; K/depth/activation mode are the scanned variables.
ALL_VARIANTS: List[Variant] = []


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_quantile(x: torch.Tensor, q: float) -> float:
    x = x.detach().float().flatten().cpu()
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, torch.tensor(float(q))).item())


def gini_nonnegative(x: torch.Tensor) -> float:
    x = x.detach().float().flatten().cpu().clamp_min(0.0)
    n = x.numel()
    if n == 0:
        return float("nan")
    s = x.sum()
    if float(s.item()) <= 0.0:
        return 0.0
    xs, _ = torch.sort(x)
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    g = (2.0 * (idx * xs).sum() / (n * s)) - (n + 1.0) / n
    return float(g.item())


def effective_count_from_mass(x: torch.Tensor) -> float:
    x = x.detach().float().flatten().cpu().clamp_min(0.0)
    s = x.sum()
    if x.numel() == 0 or float(s.item()) <= 0.0:
        return 0.0
    return float((s * s / (x.square().sum() + EPS)).item())


def top_mass_frac(x: torch.Tensor, topk: int) -> float:
    x = x.detach().float().flatten().cpu().clamp_min(0.0)
    s = x.sum()
    if x.numel() == 0 or float(s.item()) <= 0.0:
        return 0.0
    vals = torch.topk(x, k=min(topk, x.numel())).values
    return float((vals.sum() / s).item())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# MNIST loader
# ============================================================

def _open_idx(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _find_file(root: Path, names: List[str]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing MNIST file in {root}. Tried: {names}")


def read_idx_images(path: Path) -> torch.Tensor:
    with _open_idx(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image idx file {path}: magic={magic}")
        data = f.read(rows * cols * n)
    x = torch.frombuffer(bytearray(data), dtype=torch.uint8).float().view(n, rows * cols) / 255.0
    return x


def read_idx_labels(path: Path) -> torch.Tensor:
    with _open_idx(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label idx file {path}: magic={magic}")
        data = f.read(n)
    y = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return y


def _has_mnist_idx_files(root: Path) -> bool:
    if not root.exists():
        return False
    expected = [
        ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"],
        ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"],
        ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"],
        ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"],
    ]
    return all(any((root / name).exists() for name in group) for group in expected)


def download_mnist_raw(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES:
        target = root / filename
        if target.exists():
            continue
        last_error = None
        for base in MNIST_MIRRORS:
            url = base + filename
            try:
                print(f"Downloading {url} -> {target}")
                urllib.request.urlretrieve(url, target)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                try:
                    target.unlink()
                except OSError:
                    pass
        if last_error is not None:
            raise RuntimeError(f"Failed to download {filename}. Last error: {last_error}")


def resolve_mnist_raw_dir(root: str) -> Path:
    requested = Path(root)
    if _has_mnist_idx_files(requested):
        return requested
    candidates = [
        Path.cwd() / "data" / "MNIST" / "raw",
        Path(__file__).resolve().parent / "data" / "MNIST" / "raw",
        Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw",
    ]
    for candidate in candidates:
        if _has_mnist_idx_files(candidate):
            return candidate
    download_dir = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"
    print(f"MNIST raw files not found. Downloading to: {download_dir}")
    download_mnist_raw(download_dir)
    return download_dir


def load_mnist_raw(root: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_path = resolve_mnist_raw_dir(root)
    train_images = _find_file(root_path, ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"])
    train_labels = _find_file(root_path, ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"])
    test_images = _find_file(root_path, ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"])
    test_labels = _find_file(root_path, ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"])
    print(f"Using MNIST raw dir: {root_path}")
    return read_idx_images(train_images), read_idx_labels(train_labels), read_idx_images(test_images), read_idx_labels(test_labels)


def normalize_to_margin_field(x: torch.Tensor) -> torch.Tensor:
    return x - 0.5


def make_loaders(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int):
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


# ============================================================
# Anchor initialization
# ============================================================

def make_sobol_anchors(dim: int, k_points: int, seed: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return engine.draw(k_points).float() * (high - low) + low


def sample_data_anchors(x: torch.Tensor, k_points: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n = x.size(0)
    if k_points <= n:
        idx = torch.randperm(n, generator=g)[:k_points]
    else:
        idx = torch.randint(0, n, (k_points,), generator=g)
    return x[idx].float().clone()


@torch.no_grad()
def assign_to_centers(x: torch.Tensor, centers: torch.Tensor, batch_size: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return hard assignments and nearest squared distances on CPU tensors."""
    device = centers.device
    idx_all = []
    d2_all = []
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size].to(device)
        d2 = torch.cdist(xb, centers).square()
        vals, idx = d2.min(dim=1)
        idx_all.append(idx.cpu())
        d2_all.append(vals.cpu())
    return torch.cat(idx_all, dim=0), torch.cat(d2_all, dim=0)


@torch.no_grad()
def hard_kmeans_anchors(
    x: torch.Tensor,
    k_points: int,
    *,
    seed: int,
    device: torch.device,
    iters: int = 25,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Label-free hard k-means over the training data."""
    centers = sample_data_anchors(x, k_points, seed).to(device)
    x_cpu = x.float().cpu()
    for it in range(iters):
        sums = torch.zeros(k_points, x.size(1), device=device)
        counts = torch.zeros(k_points, device=device)
        total_inertia = 0.0
        for start in range(0, x_cpu.size(0), batch_size):
            xb = x_cpu[start:start + batch_size].to(device)
            d2 = torch.cdist(xb, centers).square()
            vals, idx = d2.min(dim=1)
            total_inertia += float(vals.sum().item())
            sums.index_add_(0, idx, xb)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        if (~nonempty).any():
            refill = sample_data_anchors(x_cpu, int((~nonempty).sum().item()), seed + 10000 + it).to(device)
            centers[~nonempty] = refill
        if it in {0, iters - 1}:
            print(f"  kmeans iter {it+1:02d}/{iters}: inertia={total_inertia / x_cpu.size(0):.4f}, empty={(~nonempty).sum().item()}")
    return centers.detach().cpu()


@torch.no_grad()
def estimate_temperature(x: torch.Tensor, centers: torch.Tensor, device: torch.device, sample_n: int = 2048) -> Tuple[float, float]:
    n = x.size(0)
    idx = torch.randperm(n)[:min(sample_n, n)]
    xb = x[idx].to(device)
    c = centers.to(device)
    d2 = torch.cdist(xb, c).square()
    nearest = d2.min(dim=1).values
    nearest_med = float(torch.median(nearest).item())
    all_med = float(torch.median(d2.flatten()).item())
    # Start broad but not so broad that all centers collapse to the global mean.
    t_start = max(nearest_med * 4.0, all_med * 0.15, 1e-4)
    t_end = max(nearest_med * 0.20, 1e-5)
    return t_start, t_end


@torch.no_grad()
def annealed_soft_kmeans_anchors(
    x: torch.Tensor,
    k_points: int,
    *,
    seed: int,
    device: torch.device,
    stages: int = 8,
    iters_per_stage: int = 2,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Annealed soft k-means.

    This is only an initializer. It does not use labels, c_j, sigma_j, or the field loss.
    """
    x_cpu = x.float().cpu()
    centers = sample_data_anchors(x_cpu, k_points, seed).to(device)
    t_start, t_end = estimate_temperature(x_cpu, centers, device=device)
    if stages <= 1:
        temps = [t_end]
    else:
        temps = torch.logspace(math.log10(t_start), math.log10(t_end), steps=stages).tolist()
    print(f"  anneal T_start={t_start:.4g}, T_end={t_end:.4g}, stages={stages}, iters/stage={iters_per_stage}")

    for si, temp in enumerate(temps):
        T = float(temp)
        for _ in range(iters_per_stage):
            sums = torch.zeros(k_points, x_cpu.size(1), device=device)
            weights = torch.zeros(k_points, device=device)
            entropy_acc = 0.0
            n_seen = 0
            for start in range(0, x_cpu.size(0), batch_size):
                xb = x_cpu[start:start + batch_size].to(device)
                d2 = torch.cdist(xb, centers).square()
                w = torch.softmax(-d2 / max(T, 1e-8), dim=1)
                sums += w.t() @ xb
                weights += w.sum(dim=0)
                entropy_acc += float((-(w * (w.clamp_min(1e-12)).log()).sum(dim=1)).sum().item())
                n_seen += xb.size(0)
            nonempty = weights > 1e-6
            centers[nonempty] = sums[nonempty] / weights[nonempty].unsqueeze(1)
            if (~nonempty).any():
                centers[~nonempty] = sample_data_anchors(x_cpu, int((~nonempty).sum().item()), seed + 20000 + si).to(device)
        if si in {0, len(temps) // 2, len(temps) - 1}:
            eff = math.exp(entropy_acc / max(n_seen, 1))
            print(f"  anneal stage {si+1:02d}/{len(temps)}: T={T:.4g}, soft_eff≈{eff:.1f}, empty={(~nonempty).sum().item()}")
    return centers.detach().cpu()


def anchor_knn_sigma(anchors: torch.Tensor, knn: int = 8, scale: float = 1.0) -> Tuple[torch.Tensor, float]:
    """Return per-anchor sigma and shared sigma from anchor spacing."""
    with torch.no_grad():
        k = anchors.size(0)
        kk = max(1, min(int(knn), k - 1))
        d = torch.cdist(anchors.float(), anchors.float())
        d.fill_diagonal_(float("inf"))
        kth = torch.topk(d, k=kk, largest=False, dim=1).values[:, -1]
        per = (kth * float(scale)).clamp_min(1e-3)
        shared = float(torch.median(per).item())
    return per, shared


def initialize_anchors(
    method: str,
    x_train: torch.Tensor,
    dim: int,
    k_points: int,
    seed: int,
    device: torch.device,
    *,
    kmeans_iters: int,
    anneal_stages: int,
    anneal_iters_per_stage: int,
) -> torch.Tensor:
    method = method.lower()
    if method == "sobol":
        return make_sobol_anchors(dim, k_points, seed, low=-1.0, high=1.0)
    if method == "data_sample":
        return sample_data_anchors(x_train, k_points, seed)
    if method == "kmeans":
        return hard_kmeans_anchors(x_train, k_points, seed=seed, device=device, iters=kmeans_iters)
    if method == "anneal":
        return annealed_soft_kmeans_anchors(
            x_train, k_points, seed=seed, device=device,
            stages=anneal_stages, iters_per_stage=anneal_iters_per_stage,
        )
    raise ValueError(f"Unknown init method: {method}")


# ============================================================
# Gaussian AAT terrain model
# ============================================================

class GaussianFieldLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        k_points: int,
        anchors: torch.Tensor,
        sigma_init: torch.Tensor,
        variant: Variant,
    ):
        super().__init__()
        self.dim = int(dim)
        self.k_points = int(k_points)
        self.variant = variant
        self.force_denom = math.sqrt(float(k_points))
        a_init = anchors.float().contiguous()
        if variant.trainable_anchors:
            self.a = nn.Parameter(a_init.clone())
        else:
            self.register_buffer("a", a_init)

        if variant.per_anchor_sigma:
            if sigma_init.numel() == 1:
                sigma_init = sigma_init.expand(k_points)
            self.log_sigma = nn.Parameter(sigma_init.float().clamp_min(1e-3).log().clone())
        else:
            shared = float(torch.median(sigma_init.float()).item()) if sigma_init.numel() > 1 else float(sigma_init.item())
            self.log_sigma = nn.Parameter(torch.tensor([math.log(max(shared, 1e-3))], dtype=torch.float32))

        self.c = nn.Parameter(torch.randn(k_points, dtype=torch.float32) * C_INIT_STD)

        # Extra scalar terrain used only by field-oriented activation.
        # It shares the transport anchors/sigma to keep the activation mechanism lightweight:
        # only K extra coefficients per layer, not a D x D rotation matrix.
        if variant.activation_mode == "field_dir":
            self.act_c = nn.Parameter(torch.randn(k_points, dtype=torch.float32) * C_INIT_STD)
        else:
            self.register_parameter("act_c", None)

    def sigma(self) -> torch.Tensor:
        sigma = self.log_sigma.exp()
        if sigma.numel() == 1:
            return sigma.expand(self.k_points)
        return sigma

    def effective_c(self) -> torch.Tensor:
        c = self.c
        if self.variant.zero_mean_c:
            c = c - c.mean()
        return c

    def _dist2(self, z: torch.Tensor) -> torch.Tensor:
        a = self.a.to(dtype=z.dtype, device=z.device)
        return (
            z.square().sum(dim=1, keepdim=True)
            + a.square().sum(dim=1).view(1, -1)
            - 2.0 * z @ a.t()
        ).clamp_min(0.0)

    def kernel_terms(self, z: torch.Tensor):
        dist2 = self._dist2(z)
        sigma = self.sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        kernel = torch.exp(-dist2 / (2.0 * sigma2))
        c_vec = self.effective_c().to(dtype=z.dtype, device=z.device).view(1, -1)
        weighted = kernel * c_vec / sigma2
        potential = (kernel * c_vec).sum(dim=1) / self.force_denom
        return dist2, sigma, sigma2, kernel, c_vec, weighted, potential

    def force_from_weighted(self, z: torch.Tensor, weighted: torch.Tensor) -> torch.Tensor:
        a = self.a.to(dtype=z.dtype, device=z.device)
        wa = weighted @ a
        wsum = weighted.sum(dim=1, keepdim=True)
        return float(self.variant.eta) * (wa - wsum * z) / self.force_denom

    def force(self, z: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, weighted, _ = self.kernel_terms(z)
        return self.force_from_weighted(z, weighted)

    def effective_act_c(self) -> torch.Tensor:
        if self.act_c is None:
            return self.effective_c()
        c = self.act_c
        if self.variant.zero_mean_c:
            c = c - c.mean()
        return c

    def activation_direction(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unit direction u from grad Psi, plus potential and grad norm.

        Psi(z) = sum_j act_c_j exp(-||z-a_j||^2 / (2 sigma_j^2))
        grad Psi = sum_j act_c_j G_j(z) (a_j - z) / sigma_j^2

        This is intentionally the same cheap O(BKD) geometry as the conservative field,
        not a D x D learned rotation.
        """
        a = self.a.to(dtype=z.dtype, device=z.device)
        dist2 = self._dist2(z)
        sigma = self.sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        kernel = torch.exp(-dist2 / (2.0 * sigma2))
        c_vec = self.effective_act_c().to(dtype=z.dtype, device=z.device).view(1, -1)
        weighted = kernel * c_vec / sigma2
        grad = weighted @ a - weighted.sum(dim=1, keepdim=True) * z
        norm = grad.norm(dim=1, keepdim=True)
        u = grad / (norm + 1e-6)
        potential = (kernel * c_vec).sum(dim=1, keepdim=True) / self.force_denom
        return u, potential, norm

    def response_center(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Same old local activation reference: response is computed by softmax over signed terrain scores.
        a = self.a.to(dtype=z.dtype, device=z.device)
        dist2 = self._dist2(z)
        sigma = self.sigma().to(dtype=z.dtype, device=z.device).view(1, -1)
        sigma2 = sigma.square() + EPS
        c_vec = self.effective_c().to(dtype=z.dtype, device=z.device).view(1, -1)
        score = c_vec - dist2 / (2.0 * sigma2)
        alpha = torch.softmax(score, dim=1)
        return alpha @ a, alpha


class GaussianAATModel(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        k_points: int,
        anchors_by_layer: List[torch.Tensor],
        sigmas_by_layer: List[torch.Tensor],
        variant: Variant,
    ):
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.k_points = int(k_points)
        self.variant = variant

        layers = []
        for anchors, sigmas in zip(anchors_by_layer, sigmas_by_layer):
            layers.append(GaussianFieldLayer(
                dim=dim,
                k_points=k_points,
                anchors=anchors,
                sigma_init=sigmas,
                variant=variant,
            ))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor):
        z = x
        metrics = []
        for i, layer in enumerate(self.layers):
            z_before = z
            f = layer.force(z)
            z_mid = z + f
            if self.variant.use_activation and i < len(self.layers) - 1:
                if self.variant.activation_mode == "old_response":
                    r, _ = layer.response_center(z)
                    z_after = r + F.relu(z_mid - r)
                elif self.variant.activation_mode == "field_dir":
                    # Global-zero directional ReLU. The activation axis is local and generated
                    # by the activation terrain gradient at z_mid.
                    u, _, _ = layer.activation_direction(z_mid)
                    s = (z_mid * u).sum(dim=1, keepdim=True)
                    z_after = z_mid + (F.relu(s) - s) * u
                else:
                    raise ValueError(f"Unknown activation_mode: {self.variant.activation_mode}")
                act_delta = z_after - z_mid
            else:
                z_after = z_mid
                act_delta = torch.zeros_like(z_after)
            metrics.append((
                float(f.detach().abs().mean().item()),
                float(f.detach().norm(dim=1).mean().item()),
                float(act_delta.detach().abs().mean().item()),
                float((z_after - z_before).detach().norm(dim=1).mean().item()),
            ))
            z = z_after
        return self.head(z), metrics


# ============================================================
# Training / evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def split_params(model: nn.Module):
    sigma_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "log_sigma" in name:
            sigma_params.append(p)
        else:
            base_params.append(p)
    return base_params, sigma_params


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    print_every: int,
) -> Dict[str, float]:
    base_params, sigma_params = split_params(model)
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": LR, "weight_decay": 0.0},
            {"params": sigma_params, "lr": SIGMA_LR, "weight_decay": 0.0},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = -1

    print("metric format per layer: force_abs / force_norm / act_abs / delta_norm")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        metric_accum = None
        metric_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits, metrics = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += bs
            if metric_accum is None:
                metric_accum = [[0.0, 0.0, 0.0, 0.0] for _ in metrics]
            for li, m in enumerate(metrics):
                for mi in range(4):
                    metric_accum[li][mi] += m[mi]
            metric_count += 1

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch

        if epoch % print_every == 0 or epoch == epochs - 1:
            parts = []
            if metric_accum is not None and metric_count > 0:
                for li, vals in enumerate(metric_accum):
                    avg = [v / metric_count for v in vals]
                    parts.append(f"L{li+1}:{avg[0]:.4g}/{avg[1]:.4g}/{avg[2]:.4g}/{avg[3]:.4g}")
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train loss {total_loss/max(total,1):.4f} acc {total_correct/max(total,1):.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"best {best_acc:.4f}@{best_epoch} | metrics " + "; ".join(parts)
            )

    return {"best_acc": float(best_acc), "best_epoch": int(best_epoch), "best_loss": float(best_loss)}


@torch.no_grad()
def quick_terrain_summary(
    model: GaussianAATModel,
    x: torch.Tensor,
    *,
    device: torch.device,
    max_n: int,
    batch_size: int,
) -> List[Dict[str, float]]:
    model.eval()
    x = x[:max_n].to(device)
    z = x
    rows = []

    for li, layer in enumerate(model.layers):
        c = layer.effective_c().detach().float().cpu()
        sigma = layer.sigma().detach().float().cpu()
        abs_usage = torch.zeros(layer.k_points, dtype=torch.float64)
        signed_usage = torch.zeros(layer.k_points, dtype=torch.float64)
        force_norms = []
        act_norms = []
        kernel_eff_samples = []
        kernel_max_samples = []

        z_next_chunks = []
        for start in range(0, z.size(0), batch_size):
            zb = z[start:start + batch_size]
            dist2, sigma_b, sigma2_b, kernel, c_vec, weighted, _ = layer.kernel_terms(zb)
            f = layer.force_from_weighted(zb, weighted)
            z_mid = zb + f
            if layer.variant.use_activation and li < len(model.layers) - 1:
                if layer.variant.activation_mode == "old_response":
                    r, _ = layer.response_center(zb)
                    z_after = r + F.relu(z_mid - r)
                elif layer.variant.activation_mode == "field_dir":
                    u, _, _ = layer.activation_direction(z_mid)
                    s = (z_mid * u).sum(dim=1, keepdim=True)
                    z_after = z_mid + (F.relu(s) - s) * u
                else:
                    raise ValueError(f"Unknown activation_mode: {layer.variant.activation_mode}")
                act_delta = z_after - z_mid
            else:
                z_after = z_mid
                act_delta = torch.zeros_like(z_after)

            usage = (kernel.detach().float().cpu() * c.abs().view(1, -1)).sum(dim=0).double()
            signed = (kernel.detach().float().cpu() * c.view(1, -1)).sum(dim=0).double()
            abs_usage += usage
            signed_usage += signed
            force_norms.append(f.detach().norm(dim=1).float().cpu())
            act_norms.append(act_delta.detach().norm(dim=1).float().cpu())
            k_mass = kernel.detach().float().cpu().clamp_min(0.0)
            k_sum = k_mass.sum(dim=1)
            k_eff = (k_sum.square() / (k_mass.square().sum(dim=1) + EPS)).cpu()
            kernel_eff_samples.append(k_eff)
            kernel_max_samples.append((k_mass.max(dim=1).values / (k_sum + EPS)).cpu())
            z_next_chunks.append(z_after.detach())

        force_norm = torch.cat(force_norms)
        act_norm = torch.cat(act_norms)
        kernel_eff = torch.cat(kernel_eff_samples)
        kernel_max = torch.cat(kernel_max_samples)

        rows.append({
            "layer": li + 1,
            "c_mean": float(c.mean().item()),
            "c_std": float(c.std().item()),
            "c_pos_frac": float((c > 0).float().mean().item()),
            "c_neg_frac": float((c < 0).float().mean().item()),
            "abs_c_q50": tensor_quantile(c.abs(), 0.50),
            "abs_c_q95": tensor_quantile(c.abs(), 0.95),
            "sigma_q05": tensor_quantile(sigma, 0.05),
            "sigma_q50": tensor_quantile(sigma, 0.50),
            "sigma_q95": tensor_quantile(sigma, 0.95),
            "usage_eff_abs": effective_count_from_mass(abs_usage.float()),
            "usage_gini_abs": gini_nonnegative(abs_usage.float()),
            "usage_top10_abs": top_mass_frac(abs_usage.float(), 10),
            "signed_usage_abs_sum": float(signed_usage.abs().sum().item()),
            "kernel_eff_sample_mean": float(kernel_eff.mean().item()),
            "kernel_max_sample_mean": float(kernel_max.mean().item()),
            "force_norm_mean": float(force_norm.mean().item()),
            "act_norm_mean": float(act_norm.mean().item()),
        })
        z = torch.cat(z_next_chunks, dim=0)
    return rows


# ============================================================
# Variant helpers and main
# ============================================================


def parse_int_list(text: str) -> List[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(list(range(int(a), int(b) + 1)))
        else:
            vals.append(int(part))
    # preserve order but remove duplicates
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def parse_str_list(text: str) -> List[str]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(part)
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def make_scan_variants(k_list: List[int], depth_list: List[int], activation_modes: List[str]) -> List[Tuple[int, int, Variant]]:
    configs = []
    for k_points in k_list:
        for depth in depth_list:
            for mode in activation_modes:
                if mode not in {"old_response", "field_dir"}:
                    raise ValueError(f"Unknown activation mode: {mode}")
                configs.append((
                    int(k_points),
                    int(depth),
                    Variant(
                        name=f"K{k_points}_L{depth}_{mode}",
                        init_method="kmeans",
                        per_anchor_sigma=False,
                        eta=1.0,
                        zero_mean_c=True,
                        use_activation=True,
                        sigma_scale=1.0,
                        sigma_knn=8,
                        trainable_anchors=True,
                        depth_override=int(depth),
                        activation_mode=mode,
                    ),
                ))
    return configs

def build_anchors_and_sigmas(
    variant: Variant,
    x_train: torch.Tensor,
    *,
    dim: int,
    depth: int,
    k_points: int,
    seed: int,
    device: torch.device,
    kmeans_iters: int,
    anneal_stages: int,
    anneal_iters_per_stage: int,
    anchor_cache: Optional[Dict[Tuple[int, int, int], torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    anchors_by_layer: List[torch.Tensor] = []
    sigmas_by_layer: List[torch.Tensor] = []
    for li in range(depth):
        layer_seed = seed + 1000 * (li + 1)
        print(f"  init layer {li+1}/{depth}: method={variant.init_method}, seed={layer_seed}")
        cache_key = (int(k_points), int(li), int(layer_seed))
        if anchor_cache is not None and cache_key in anchor_cache:
            anchors = anchor_cache[cache_key].clone()
            print("    using cached anchors")
        else:
            anchors = initialize_anchors(
                variant.init_method, x_train, dim, k_points, layer_seed, device,
                kmeans_iters=kmeans_iters,
                anneal_stages=anneal_stages,
                anneal_iters_per_stage=anneal_iters_per_stage,
            )
            if anchor_cache is not None:
                anchor_cache[cache_key] = anchors.clone()
        per_sigma, shared_sigma = anchor_knn_sigma(
            anchors, knn=variant.sigma_knn, scale=variant.sigma_scale
        )
        sigma_init = per_sigma if variant.per_anchor_sigma else torch.tensor([shared_sigma], dtype=torch.float32)
        print(
            f"    anchors mean/std={anchors.mean().item():.4f}/{anchors.std().item():.4f}, "
            f"sigma q50={tensor_quantile(per_sigma,0.50):.4f}, q05/q95={tensor_quantile(per_sigma,0.05):.4f}/{tensor_quantile(per_sigma,0.95):.4f}"
        )
        anchors_by_layer.append(anchors)
        sigmas_by_layer.append(sigma_init)
    return anchors_by_layer, sigmas_by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-root", type=str, default=MNIST_RAW_DIR)
    parser.add_argument("--k-list", type=str, default="16,32,64", help="Comma-separated K values, e.g. 16,32,64.")
    parser.add_argument("--depth-list", type=str, default="3,6,10", help="Comma-separated depths or ranges, e.g. 3,6,10 or 1-10.")
    parser.add_argument("--activation-modes", type=str, default="old_response,field_dir", help="Comma-separated activation modes: old_response,field_dir.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--analyze-n", type=int, default=2048)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--anneal-stages", type=int, default=8)
    parser.add_argument("--anneal-iters-per-stage", type=int, default=2)
    parser.add_argument("--no-zero-mean", action="store_true", help="Disable zero-mean c for all runs.")
    parser.add_argument("--no-activation", action="store_true", help="Disable response-center ReLU activation for all runs.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N configs after expanding the scan. 0 means all.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N configs after expanding the scan.")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_list = parse_int_list(args.k_list)
    depth_list = parse_int_list(args.depth_list)
    activation_modes = parse_str_list(args.activation_modes)
    configs = make_scan_variants(k_list, depth_list, activation_modes)
    if args.no_zero_mean:
        configs = [(k, d, Variant(**{**asdict(v), "zero_mean_c": False})) for k, d, v in configs]
    if args.no_activation:
        configs = [(k, d, Variant(**{**asdict(v), "use_activation": False})) for k, d, v in configs]
    if args.start_index > 0:
        configs = configs[args.start_index:]
    if args.limit and args.limit > 0:
        configs = configs[:args.limit]

    print(f"Device: {device}")
    print(f"Settings: K={k_list}, depth={depth_list}, activation_modes={activation_modes}, epochs={args.epochs}, analyze_n={args.analyze_n}")
    print("Mechanism: conservative Gaussian transport unchanged; compare old response activation vs field-directed global-zero activation")
    print(f"Total configs: {len(configs)}")
    print("Configs:", ", ".join(v.name for _, _, v in configs))

    x_train_raw, y_train_all, x_test_raw, y_test_all = load_mnist_raw(args.mnist_root)
    x_train = normalize_to_margin_field(x_train_raw[:N_TRAIN]).contiguous()
    y_train = y_train_all[:N_TRAIN].contiguous()
    x_val = normalize_to_margin_field(x_test_raw[:N_VAL]).contiguous()
    y_val = y_test_all[:N_VAL].contiguous()
    print(f"Loaded train {x_train.shape}, val {x_val.shape}")
    print(f"x_train range [{x_train.min().item():.3f}, {x_train.max().item():.3f}], mean/std {x_train.mean().item():.4f}/{x_train.std().item():.4f}")

    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, args.batch_size)

    out_dir = Path("gaussian_field_oriented_activation_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    k_tag = "-".join(str(k) for k in k_list)
    d_tag = "-".join(str(d) for d in depth_list)
    m_tag = "-".join(activation_modes)
    json_path = out_dir / f"summary_K{k_tag}_D{d_tag}_{m_tag}.json"
    csv_path = out_dir / f"terrain_K{k_tag}_D{d_tag}_{m_tag}.csv"
    summary_csv_path = out_dir / f"summary_K{k_tag}_D{d_tag}_{m_tag}.csv"

    results = []
    terrain_csv_rows = []
    summary_rows = []
    anchor_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    for ci, (k_points, effective_depth, variant) in enumerate(configs):
        print("\n" + "=" * 96)
        print(f"[{ci+1}/{len(configs)}] {variant.name}")
        print("=" * 96)
        print(json.dumps({"K": k_points, "depth": effective_depth, **asdict(variant)}, ensure_ascii=False))
        set_seed(SEED + ci * 1000)

        # Use a layer-based seed independent of config index so K_L3 reuses the same first 3 layer seeds as K_L10.
        anchors_by_layer, sigmas_by_layer = build_anchors_and_sigmas(
            variant,
            x_train,
            dim=BASE_DIM,
            depth=effective_depth,
            k_points=k_points,
            seed=SEED,
            device=device,
            kmeans_iters=args.kmeans_iters,
            anneal_stages=args.anneal_stages,
            anneal_iters_per_stage=args.anneal_iters_per_stage,
            anchor_cache=anchor_cache,
        )

        model = GaussianAATModel(
            dim=BASE_DIM,
            depth=effective_depth,
            k_points=k_points,
            anchors_by_layer=anchors_by_layer,
            sigmas_by_layer=sigmas_by_layer,
            variant=variant,
        ).to(device)

        params = count_params(model)
        print(
            f"params={params}, dim={BASE_DIM}, K={k_points}, depth={effective_depth}, init={variant.init_method}, "
            f"trainable_anchors={variant.trainable_anchors}, per_anchor_sigma={variant.per_anchor_sigma}, "
            f"zero_mean_c={variant.zero_mean_c}, activation={variant.use_activation}, activation_mode={variant.activation_mode}"
        )

        train_info = train_model(
            model, train_loader, val_loader,
            device=device, epochs=args.epochs, print_every=args.print_every,
        )
        terrain = quick_terrain_summary(
            model, x_val, device=device, max_n=args.analyze_n, batch_size=args.batch_size,
        )

        print("Terrain quick summary:")
        for row in terrain:
            print(
                f"  L{row['layer']}: c_std {row['c_std']:.4g}, pos/neg {row['c_pos_frac']:.2f}/{row['c_neg_frac']:.2f}, "
                f"sigma q05/q50/q95 {row['sigma_q05']:.3g}/{row['sigma_q50']:.3g}/{row['sigma_q95']:.3g}, "
                f"usage_eff {row['usage_eff_abs']:.1f}, top10 {row['usage_top10_abs']:.3f}, "
                f"kernel_eff {row['kernel_eff_sample_mean']:.1f}, kernel_max {row['kernel_max_sample_mean']:.3f}, "
                f"force {row['force_norm_mean']:.3g}, act {row['act_norm_mean']:.3g}"
            )

        result = {
            "name": variant.name,
            "K": k_points,
            "params": params,
            "effective_depth": effective_depth,
            "variant": asdict(variant),
            "train_info": train_info,
            "terrain": terrain,
        }
        results.append(result)
        summary_rows.append({
            "name": variant.name,
            "K": k_points,
            "depth": effective_depth,
            "params": params,
            "best_acc": train_info["best_acc"],
            "best_epoch": train_info["best_epoch"],
            "best_loss": train_info["best_loss"],
            "trainable_anchors": variant.trainable_anchors,
            "activation": variant.use_activation,
            "activation_mode": variant.activation_mode,
        })
        for row in terrain:
            csv_row = {"variant": variant.name, "K": k_points, "effective_depth": effective_depth, **row, **train_info}
            terrain_csv_rows.append(csv_row)

        # Save after every config so a long scan is not lost if interrupted.
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if summary_rows:
            with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        if terrain_csv_rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(terrain_csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(terrain_csv_rows)

    print("\n" + "=" * 96)
    print("FINAL SUMMARY")
    print("=" * 96)
    for r in results:
        ti = r["train_info"]
        print(
            f"{r['name']:<24} | K {r['K']:3d} | depth {r['effective_depth']:2d} | params {r['params']:7d} | "
            f"best_val_acc {ti['best_acc']:.4f}@{ti['best_epoch']} | val_loss {ti['best_loss']:.4f}"
        )

    print(f"Saved JSON summary: {json_path}")
    print(f"Saved summary CSV:  {summary_csv_path}")
    print(f"Saved terrain CSV:  {csv_path}")


if __name__ == "__main__":
    main()
