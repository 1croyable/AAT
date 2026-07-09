from __future__ import annotations

from pathlib import Path

import math
import torch
import torch.nn.functional as F

from aatfield import AAT, AATConfig


def make_triple_helix(
    n: int,
    *,
    turns: float = 3.0,
    radius: float = 1.0,
    height: float = 2.0,
    noise: float = 0.06,
    seed: int = 0,
):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    classes = 3
    n_each = math.ceil(int(n) / classes)
    xs = []
    ys = []

    for cls in range(classes):
        t = torch.rand((n_each,), generator=gen) * (2.0 * math.pi * float(turns))
        phase = 2.0 * math.pi * float(cls) / float(classes)

        x = float(radius) * torch.cos(t + phase)
        y = float(radius) * torch.sin(t + phase)
        z = float(height) * (t / (2.0 * math.pi * float(turns)) - 0.5)

        pts = torch.stack((x, y, z), dim=1)
        pts = pts + torch.randn(pts.shape, generator=gen) * float(noise)

        xs.append(pts)
        ys.append(torch.full((n_each,), int(cls), dtype=torch.long))

    x = torch.cat(xs, dim=0)[: int(n)]
    y = torch.cat(ys, dim=0)[: int(n)]

    perm = torch.randperm(x.shape[0], generator=gen)
    return x.index_select(0, perm), y.index_select(0, perm)


@torch.no_grad()
def accuracy(model: AAT, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


def clone_state_dict(model: AAT) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_epoch(model: AAT, x: torch.Tensor, y: torch.Tensor, optimizer, batch_size: int) -> float:
    model.train()
    perm = torch.randperm(x.shape[0], device=x.device)
    total_loss = 0.0
    total_n = 0

    for start in range(0, x.shape[0], int(batch_size)):
        idx = perm[start:start + int(batch_size)]
        xb = x.index_select(0, idx)
        yb = y.index_select(0, idx)

        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(xb.shape[0])
        total_n += int(xb.shape[0])

    return total_loss / max(total_n, 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triple_helix_3d.pt"

    seed = 0
    n_train = 6000
    n_val = 3000
    epochs = 600
    batch_size = 512
    lr = 3e-3
    weight_decay = 1e-4

    torch.manual_seed(seed)

    x_train, y_train = make_triple_helix(n_train, seed=seed)
    x_val, y_val = make_triple_helix(n_val, seed=seed + 1)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    cfg = AATConfig(
        input_dim=3,
        num_classes=3,
        layers=8,
        rays=32,
        kappa=6.0,
        ray_dropout=0.15,
    )
    model = AAT(cfg).to(device)
    model.fit_state(x_train)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = 0
    best_train_acc = 0.0
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, x_train, y_train, optimizer, batch_size)
        val_acc = accuracy(model, x_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_acc = accuracy(model, x_train, y_train)
            best_state = clone_state_dict(model)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            train_acc = accuracy(model, x_train, y_train)
            print(
                f"epoch={epoch:04d} loss={loss:.4f} train={train_acc:.4f} "
                f"val={val_acc:.4f} best={best_val_acc:.4f}@{best_epoch}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("No best state was recorded.")

    model.load_state_dict(best_state)
    model.save_checkpoint(out_path)

    print(f"saved: {out_path}")
    print(f"best_epoch={best_epoch}")
    print(f"best_train_acc={best_train_acc:.4f}")
    print(f"best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
