from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_dist2(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x = x.float()
    centers = centers.float()
    return ((x * x).sum(1, keepdim=True) + (centers * centers).sum(1).view(1, -1) - 2.0 * x @ centers.t()).clamp_min(0.0)


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y.float().clamp_min(1e-8)).clamp_min(1e-8))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lift(x: torch.Tensor, extra_dims: int) -> torch.Tensor:
    z = x.float().view(x.shape[0], -1) * 2.0 - 1.0
    if extra_dims > 0:
        z = torch.cat([z, torch.zeros(z.shape[0], extra_dims, dtype=z.dtype, device=z.device)], dim=1)
    return z.contiguous()


def load_mnist(data_root: str, n_train: int, n_val: int, download: bool, smoke_test: bool, seed: int):
    if smoke_test:
        g = torch.Generator().manual_seed(seed)
        return (
            torch.rand(n_train, 784, generator=g),
            torch.randint(0, 10, (n_train,), generator=g),
            torch.rand(n_val, 784, generator=g),
            torch.randint(0, 10, (n_val,), generator=g),
        )
    from torchvision import datasets
    train_ds = datasets.MNIST(root=data_root, train=True, download=download)
    val_ds = datasets.MNIST(root=data_root, train=False, download=download)
    x_train = train_ds.data.float().view(-1, 784) / 255.0
    y_train = train_ds.targets.long()
    x_val = val_ds.data.float().view(-1, 784) / 255.0
    y_val = val_ds.targets.long()
    return x_train[:n_train], y_train[:n_train], x_val[:n_val], y_val[:n_val]


def boundary_weights(pts: torch.Tensor, class_id: int, parents: torch.Tensor) -> torch.Tensor:
    d = torch.sqrt(pairwise_dist2(pts, parents) + 1e-8)
    own = d[:, class_id]
    mask = torch.ones(parents.shape[0], dtype=torch.bool, device=pts.device)
    mask[class_id] = False
    other = d[:, mask].min(1).values
    sil = (other - own) / torch.maximum(own, other).clamp_min(1e-8)
    w = (1.0 - sil).clamp_min(1e-3)
    return w / w.mean().clamp_min(1e-8)


@torch.no_grad()
def weighted_kmeans(pts: torch.Tensor, weights: torch.Tensor, k: int, iters: int = 8):
    pts = pts.float()
    w = weights.float().clamp_min(1e-8)
    n, d = pts.shape
    k = max(1, min(k, n))
    centers = torch.empty(k, d, device=pts.device, dtype=pts.dtype)
    centers[0] = (pts * w[:, None]).sum(0) / w.sum().clamp_min(1e-8)
    if k > 1:
        nearest = pairwise_dist2(pts, centers[:1]).squeeze(1)
        for j in range(1, k):
            idx = int((nearest * w).argmax().item())
            centers[j] = pts[idx]
            nearest = torch.minimum(nearest, pairwise_dist2(pts, centers[j:j + 1]).squeeze(1))
    assign = torch.zeros(n, dtype=torch.long, device=pts.device)
    for _ in range(iters):
        assign = pairwise_dist2(pts, centers).argmin(1)
        sum_w = torch.zeros(k, device=pts.device)
        sum_w.scatter_add_(0, assign, w)
        sum_wp = torch.zeros(k, d, device=pts.device)
        sum_wp.scatter_add_(0, assign[:, None].expand(-1, d), pts * w[:, None])
        new_centers = torch.where((sum_w > 0)[:, None], sum_wp / sum_w.clamp_min(1e-8)[:, None], centers)
        if (new_centers - centers).norm().item() < 1e-5:
            centers = new_centers
            break
        centers = new_centers
    return centers


def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum()


@torch.no_grad()
def choose_anchors(z: torch.Tensor, y: torch.Tensor, kmeans_iters: int = 8):
    z = z.float()
    y = y.long().to(z.device)
    C, D = 10, z.shape[1]
    parents = torch.zeros(C, D, device=z.device)
    class_points = []
    for c in range(C):
        pts = z[y == c]
        class_points.append(pts)
        parents[c] = pts.mean(0)
    common_max_k = min(100, min(int(p.shape[0]) for p in class_points))
    common_min_k = min(2, common_max_k)
    k_values = list(range(common_min_k, common_max_k + 1))
    needed_ks = list(range(1, common_max_k + 1))
    centers_by_class: List[Dict[int, torch.Tensor]] = []
    for c in range(C):
        pts = class_points[c]
        w = boundary_weights(pts, c, parents)
        mp = {}
        for k in needed_ks:
            mp[k] = weighted_kmeans(pts, w, k, kmeans_iters).detach().clone()
        centers_by_class.append(mp)

    def hard_nmi(k: int) -> float:
        anchors = torch.cat([centers_by_class[c][k] for c in range(C)], 0)
        assign = pairwise_dist2(z, anchors).argmin(1)
        flat = assign * C + y
        joint_flat = torch.zeros(anchors.shape[0] * C, device=z.device)
        joint_flat.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        joint = joint_flat.view(anchors.shape[0], C) / max(z.shape[0], 1)
        pa = joint.sum(1)
        py = joint.sum(0)
        denom = pa[:, None] * py[None, :]
        m = joint > 1e-12
        mi = (joint[m] * (joint[m] / denom[m].clamp_min(1e-12)).log()).sum()
        return float((mi / torch.sqrt((entropy(pa) * entropy(py)).clamp_min(1e-12))).item())

    def macro_f1(k: int) -> float:
        anchors = torch.cat([centers_by_class[c][k] for c in range(C)], 0)
        labels = torch.arange(C, device=z.device).repeat_interleave(k)
        pred = labels[pairwise_dist2(z, anchors).argmin(1)]
        scores = []
        for c in range(C):
            tp = ((pred == c) & (y == c)).sum().float()
            fp = ((pred == c) & (y != c)).sum().float()
            fn = ((pred != c) & (y == c)).sum().float()
            scores.append(float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item()))
        return sum(scores) / C

    nmi = [hard_nmi(k) for k in k_values]
    f1 = [macro_f1(k) for k in k_values]
    xs = torch.tensor(k_values, device=z.device, dtype=z.dtype)
    ys = torch.tensor(nmi, device=z.device, dtype=z.dtype)
    xn = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1e-8)
    yn = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1e-8)
    k_info = int(k_values[int((torch.stack([xn, yn], 1) - torch.stack([xn, xn], 1)).norm(dim=1).argmax().item())])
    target = 0.99 * max(f1)
    k_geo = k_values[-1]
    for k, s in zip(k_values, f1):
        if s >= target:
            k_geo = k
            break
    best_k = max(common_min_k, min(int(round((k_info + k_geo) / 2.0)) + 5, common_max_k))
    anchors = torch.cat([centers_by_class[c][best_k] for c in range(C)], 0).contiguous()
    nearest = torch.sqrt(pairwise_dist2(z, torch.cat([parents, anchors], 0)).min(1).values + 1e-8)
    sigma = max(0.05, min(3.0, float(torch.quantile(nearest, 0.20).item()) * 0.75))
    return anchors, torch.full((anchors.shape[0],), sigma, device=z.device), {"k": best_k, "k_info": k_info, "k_geo": k_geo}


class PotentialField(nn.Module):
    def __init__(self, anchors: torch.Tensor, sigma: torch.Tensor, charge: torch.Tensor | None = None):
        super().__init__()
        self.anchors = nn.Parameter(anchors.detach().float().clone())
        self.raw_sigma = nn.Parameter(inv_softplus(sigma.detach().float().clone()))
        if charge is None:
            charge = torch.randn(anchors.shape[0], device=anchors.device) * 1e-2
        self.charge = nn.Parameter(charge.detach().float().clone())

    def sigma(self):
        return F.softplus(self.raw_sigma) + 1e-4

    def kernel(self, z: torch.Tensor):
        sigma = self.sigma()
        dist2 = pairwise_dist2(z, self.anchors)
        sigma2 = sigma[None, :].square().clamp_min(1e-8)
        k = torch.exp(-dist2 / (2.0 * sigma2))
        return k, sigma2

    def center(self, z: torch.Tensor):
        k, _ = self.kernel(z)
        w = self.charge.float().abs()[None, :] * k
        return (w @ self.anchors.float()) / w.sum(1, keepdim=True).clamp_min(1e-8)

    def hvp(self, z: torch.Tensor, v: torch.Tensor):
        k, sigma2 = self.kernel(z)
        c = self.charge.float()[None, :]
        av = self.anchors.float() @ v.float().t()
        av = av.t()
        zv = (z.float() * v.float()).sum(1, keepdim=True)
        rdotv = av - zv
        sigma4 = sigma2.square().clamp_min(1e-12)
        w1 = c * k * rdotv / sigma4
        term1 = w1 @ self.anchors.float() - z.float() * w1.sum(1, keepdim=True)
        term2 = v.float() * (c * k / sigma2).sum(1, keepdim=True)
        return term1 - term2

    @torch.no_grad()
    def stats(self, prefix: str):
        s = self.sigma().detach()
        c = self.charge.detach()
        return {f"{prefix}_sigma": float(s.mean().item()), f"{prefix}_c_abs": float(c.abs().mean().item())}


class DualCurvatureLayer(nn.Module):
    def __init__(self, anchors: torch.Tensor, sigma: torch.Tensor, charge: torch.Tensor):
        super().__init__()
        self.size_field = PotentialField(anchors, sigma, charge)
        self.dir_field = PotentialField(anchors, sigma, charge)

    @staticmethod
    def unit(x: torch.Tensor):
        return x / x.norm(2, dim=1, keepdim=True).clamp_min(1e-8)

    def curvature_vector(self, field: PotentialField, z: torch.Tensor):
        center = field.center(z)
        v = self.unit(z.float() - center.float())
        return field.hvp(z, v)

    def forward(self, z: torch.Tensor):
        zf = z.float()
        h_size = self.curvature_vector(self.size_field, zf)
        h_dir = self.curvature_vector(self.dir_field, zf)
        mag = h_size.norm(2, dim=1, keepdim=True)
        direction = self.unit(h_dir)
        move = mag * direction
        return zf + move, {"move": move.detach().norm(2, dim=1), "mag": mag.detach().squeeze(1)}


class DualCurvatureStack(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.head = nn.Linear(dim, 10)

    def add_layer(self, layer: DualCurvatureLayer):
        self.layers.append(layer)

    def features(self, z: torch.Tensor):
        infos = []
        out = z.float()
        for layer in self.layers:
            out, info = layer(out)
            infos.append(info)
        return out, infos

    def forward(self, z: torch.Tensor):
        out, infos = self.features(z)
        total_move = torch.zeros(z.shape[0], device=z.device)
        last_move = torch.zeros(z.shape[0], device=z.device)
        if infos:
            last_move = infos[-1]["move"]
            total_move = torch.stack([i["move"] for i in infos], 0).sum(0)
        return self.head(out), {"last_move": last_move, "total_move": total_move}


@torch.no_grad()
def collect_features(model: DualCurvatureStack, loader, device):
    model.eval()
    xs, ys = [], []
    for z, y in loader:
        z = z.to(device)
        feat, _ = model.features(z)
        xs.append(feat.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(xs, 0), torch.cat(ys, 0)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for z, y in loader:
        z, y = z.to(device), y.to(device)
        logits, _ = model(z)
        loss = F.cross_entropy(logits, y)
        total += y.numel()
        correct += int((logits.argmax(1) == y).sum().item())
        loss_sum += float(loss.item()) * y.numel()
    return correct / max(total, 1), loss_sum / max(total, 1)


def train_stage(model: DualCurvatureStack, stage: int, train_loader, val_loader, device, epochs: int, lr: float, amp: bool, out_dir: Path):
    model.to(device)
    if stage == 1:
        params = [{"params": model.parameters(), "lr": lr}]
    else:
        old_params = []
        for layer in list(model.layers)[:-1]:
            old_params += list(layer.parameters())
        new_params = list(model.layers[-1].parameters()) + list(model.head.parameters())
        params = [{"params": old_params, "lr": lr * 0.1}, {"params": new_params, "lr": lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))
    best, best_epoch = 0.0, 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = move_sum = last_sum = 0.0
        for z, y in train_loader:
            z, y = z.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
                logits, info = model(z)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = y.numel()
            total += bs
            correct += int((logits.detach().argmax(1) == y).sum().item())
            loss_sum += float(loss.detach().item()) * bs
            move_sum += float(info["total_move"].mean().item()) * bs
            last_sum += float(info["last_move"].mean().item()) * bs
        val_acc, val_loss = evaluate(model, val_loader, device)
        if val_acc > best:
            best, best_epoch = val_acc, epoch
        row = {
            "epoch": epoch,
            "train_acc": correct / max(total, 1),
            "train_loss": loss_sum / max(total, 1),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "best_acc": best,
            "total_move": move_sum / max(total, 1),
            "last_move": last_sum / max(total, 1),
        }
        rows.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"dual_L{stage} epoch {epoch:04d}/{epochs} | train_acc={row['train_acc']:.4f} loss={row['train_loss']:.4f} | "
                f"val_acc={val_acc:.4f} loss={val_loss:.4f} best={best:.4f}@{best_epoch} | "
                f"last_move={row['last_move']:.4f} total_move={row['total_move']:.4f}"
            )
    with (out_dir / f"dual_L{stage}_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"stage": stage, "best_acc": best, "best_epoch": best_epoch, "params": count_parameters(model), "supports": sum(l.size_field.anchors.shape[0] for l in model.layers)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="mnist_progressive_dual_curvature")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--init-samples", type=int, default=8192)
    p.add_argument("--extra-dims", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = load_mnist(args.data_root, args.n_train, args.n_val, args.download, args.smoke_test, args.seed)
    z_train = lift(x_train, args.extra_dims)
    z_val = lift(x_val, args.extra_dims)
    dim = z_train.shape[1]

    train_loader = DataLoader(TensorDataset(z_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    init_loader = DataLoader(TensorDataset(z_train, y_train), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(TensorDataset(z_val, y_val), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    print("=" * 110)
    print("Progressive dual-curvature field | flattened MNIST")
    print("=" * 110)
    print(f"device={device} seed={args.seed} train={z_train.shape[0]} val={z_val.shape[0]} dim={dim} depth={args.max_depth}")
    print("lift: flattened image -> [2*x-1, hidden zeros], no permutation")
    print("layer: size-field curvature magnitude * direction-field curvature direction")

    model = DualCurvatureStack(dim).to(device)
    summaries = []

    for stage in range(1, args.max_depth + 1):
        if stage == 1:
            init_z, init_y = z_train, y_train
        else:
            init_z, init_y = collect_features(model, init_loader, device)
        if args.init_samples > 0 and args.init_samples < init_z.shape[0]:
            idx = torch.randperm(init_z.shape[0])[:args.init_samples]
            init_z_s, init_y_s = init_z[idx], init_y[idx]
        else:
            init_z_s, init_y_s = init_z, init_y
        t0 = time.time()
        anchors, sigma, meta = choose_anchors(init_z_s.to(device), init_y_s.to(device))
        base_charge = torch.randn(anchors.shape[0], device=device) * 1e-2
        model.add_layer(DualCurvatureLayer(anchors, sigma, base_charge).to(device))
        print(
            f"L{stage} init done in {time.time() - t0:.2f}s | selected_per_class={meta['k']} "
            f"total_supports={anchors.shape[0]} k_info={meta['k_info']} k_geo={meta['k_geo']} sigma={float(sigma.mean().item()):.4f}"
        )
        print(f"AAT dual depth={stage} params={count_parameters(model):,} supports={sum(l.size_field.anchors.shape[0] for l in model.layers)}")
        summaries.append(train_stage(model, stage, train_loader, val_loader, device, args.epochs, args.lr, args.amp, out_dir))

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "best_acc", "best_epoch", "params", "supports"])
        w.writeheader()
        w.writerows(summaries)
    print("=" * 110)
    for s in summaries:
        print(f"L{s['stage']} | best_acc={s['best_acc']:.4f}@{s['best_epoch']} | params={s['params']:,} | supports={s['supports']}")
    print(f"summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
