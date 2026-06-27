# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def add_project_root(project_root: str | None) -> None:
    if project_root:
        root = Path(project_root).resolve()
    else:
        here = Path(__file__).resolve()
        candidates = [here.parent, *here.parents]
        root = None
        for p in candidates:
            if (p / "aatfield").is_dir():
                root = p
                break
        if root is None:
            root = here.parents[1] if len(here.parents) > 1 else here.parent
    sys.path.insert(0, str(root))


def load_mnist(data_root: str, n_train: int, n_val: int, download: bool):
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
    except Exception as exc:
        raise RuntimeError("This script needs torchvision to load MNIST.") from exc

    train_set = MNIST(root=data_root, train=True, download=download, transform=ToTensor())
    val_set = MNIST(root=data_root, train=False, download=download, transform=ToTensor())

    n_train = min(int(n_train), len(train_set))
    n_val = min(int(n_val), len(val_set))

    x_train = torch.stack([train_set[i][0] for i in range(n_train)], dim=0)
    y_train = torch.tensor([int(train_set[i][1]) for i in range(n_train)], dtype=torch.long)
    x_val = torch.stack([val_set[i][0] for i in range(n_val)], dim=0)
    y_val = torch.tensor([int(val_set[i][1]) for i in range(n_val)], dtype=torch.long)
    return x_train, y_train, x_val, y_val


def quantiles(x: torch.Tensor, qs: Iterable[float]) -> list[float]:
    x = x.detach().float().flatten()
    return [float(torch.quantile(x, float(q)).item()) for q in qs]


def stats_str(name: str, x: torch.Tensor) -> str:
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return f"{name}: empty"
    q05, q50, q95 = quantiles(x, [0.05, 0.50, 0.95])
    return (
        f"{name}: mean={x.mean().item():.6f} std={x.std(unbiased=False).item():.6f} "
        f"min={x.min().item():.6f} q05={q05:.6f} q50={q50:.6f} q95={q95:.6f} max={x.max().item():.6f}"
    )


@torch.no_grad()
def layer_diagnostics(model, x: torch.Tensor, y: torch.Tensor, batch_size: int):
    from aatfield.utils import pairwise_dist2

    model.eval()
    device = next(model.parameters()).device
    x = x[:batch_size].to(device)
    y = y[:batch_size].to(device)
    z = model.lift(x)

    print("\n" + "=" * 110)
    print("INITIALIZED MODEL STRUCTURE")
    print("=" * 110)
    print(f"selected_children_by_layer={model.selected_children_by_layer()}")
    print(f"total_children={model.total_children()}")
    print(f"state_dim={model.state_dim} input_dim={model.input_dim} extra_dims={model.extra_dims}")
    print(f"diagnostic_batch={z.shape[0]}")
    print(stats_str("z0_norm", z.norm(dim=1)))

    rows = []
    for li, layer in enumerate(model.layers, start=1):
        anchors = layer.all_anchors()
        sigma = layer.sigma()
        charge = layer.charge.detach()
        dist2 = pairwise_dist2(z, anchors)
        dist = torch.sqrt(dist2 + 1e-8)
        nearest = dist.min(dim=1).values
        sigma2 = sigma.view(1, -1).square().clamp_min(1e-8)
        scores = charge.view(1, -1) - dist2 / (2.0 * sigma2)
        alpha = torch.softmax(scores, dim=-1)
        entropy = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(dim=1)
        effective = entropy.exp()
        top_values = torch.topk(alpha, k=min(5, alpha.shape[1]), dim=1).values

        move, field_target = layer._potential_response(z)
        z_mid = z + move
        z_out = layer(z)
        fold_delta = z_out - z_mid

        print("\n" + "-" * 110)
        print(f"L{li}: anchors={anchors.shape[0]} children_per_class={layer.children_per_class} field_scale={getattr(layer, 'field_scale', None)}")
        print(stats_str(f"L{li}_sigma", sigma))
        print(stats_str(f"L{li}_charge", charge))
        print(stats_str(f"L{li}_charge_abs", charge.abs()))
        print(stats_str(f"L{li}_anchor_norm", anchors.norm(dim=1)))
        print(stats_str(f"L{li}_nearest_dist", nearest))
        print(stats_str(f"L{li}_score_max", scores.max(dim=1).values))
        print(stats_str(f"L{li}_alpha_max", alpha.max(dim=1).values))
        print(stats_str(f"L{li}_alpha_effective_count", effective))
        print(stats_str(f"L{li}_alpha_top5_mass", top_values.sum(dim=1)))
        print(stats_str(f"L{li}_move_norm", move.norm(dim=1)))
        print(stats_str(f"L{li}_fold_delta_norm", fold_delta.norm(dim=1)))
        print(stats_str(f"L{li}_z_out_norm", z_out.norm(dim=1)))

        rows.append({
            "layer": li,
            "anchors": int(anchors.shape[0]),
            "children_per_class": int(layer.children_per_class),
            "sigma_mean": float(sigma.mean().item()),
            "sigma_min": float(sigma.min().item()),
            "sigma_max": float(sigma.max().item()),
            "nearest_mean": float(nearest.mean().item()),
            "nearest_p20": float(torch.quantile(nearest, 0.20).item()),
            "alpha_max_mean": float(alpha.max(dim=1).values.mean().item()),
            "alpha_effective_mean": float(effective.mean().item()),
            "move_mean": float(move.norm(dim=1).mean().item()),
            "fold_delta_mean": float(fold_delta.norm(dim=1).mean().item()),
            "z_out_norm_mean": float(z_out.norm(dim=1).mean().item()),
        })
        z = z_out

    logits = model.head(z)
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    print("\n" + "=" * 110)
    print(f"random/untrained head acc on diagnostic batch after init = {acc:.4f}")
    print("=" * 110)
    return rows


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.shape[0]
        correct += int((model(xb).argmax(dim=1) == yb).sum().item())
        total += int(xb.shape[0])
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * xb.shape[0]
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(xb.shape[0])
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--extra-dims", type=int, default=1)
    p.add_argument("--max-children", type=int, default=100)
    p.add_argument("--min-children", type=int, default=2)
    p.add_argument("--init-samples", type=int, default=8192)
    p.add_argument("--kmeans-iters", type=int, default=8)
    p.add_argument("--sigma-init", type=float, default=0.75)
    p.add_argument("--charge-init", type=float, default=0.08)
    p.add_argument("--lift-seed", type=int, default=1234)
    p.add_argument("--diag-batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=str, default="aatfield_existing_mnist_diagnostics.csv")
    args = p.parse_args()

    add_project_root(args.project_root)
    from aatfield import AATField, AATFieldConfig
    from aatfield.utils import count_parameters

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    x_train, y_train, x_val, y_val = load_mnist(args.data_root, args.n_train, args.n_val, args.download)

    cfg = AATFieldConfig.from_data(
        x_train,
        num_classes=10,
        extra_dims=int(args.extra_dims),
        layers=int(args.layers),
        max_children=int(args.max_children),
        sigma_init=float(args.sigma_init),
        charge_init=float(args.charge_init),
        lift_seed=int(args.lift_seed),
    )
    model = AATField(cfg).to(device)

    print("=" * 110)
    print("Existing aatfield MNIST initialization diagnostic")
    print("=" * 110)
    print(f"device={device} seed={args.seed} n_train={x_train.shape[0]} n_val={x_val.shape[0]}")
    print(f"cfg={cfg}")
    print(f"init_samples={args.init_samples} min_children={args.min_children} kmeans_iters={args.kmeans_iters}")

    model.initialize(
        x_train,
        y_train,
        samples=int(args.init_samples),
        min_children=int(args.min_children),
        kmeans_iters=int(args.kmeans_iters),
    )
    print(f"params_after_init={count_parameters(model):,}")

    rows = layer_diagnostics(model, x_train, y_train, batch_size=int(args.diag_batch))
    out_path = Path(args.out)
    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"diagnostics_csv={out_path}")

    if int(args.epochs) > 0:
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(args.batch_size), shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(args.batch_size), shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
        best = 0.0
        best_epoch = 0
        for epoch in range(1, int(args.epochs) + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            va_loss, va_acc = evaluate(model, val_loader, device)
            if va_acc > best:
                best = va_acc
                best_epoch = epoch
            if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
                print(
                    f"epoch {epoch:04d}/{args.epochs} | "
                    f"train_acc={tr_acc:.4f} loss={tr_loss:.4f} | "
                    f"val_acc={va_acc:.4f} loss={va_loss:.4f} best={best:.4f}@{best_epoch}"
                )


if __name__ == "__main__":
    main()
