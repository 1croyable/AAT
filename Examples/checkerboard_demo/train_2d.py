from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F

from aatfield import AATField, AATFieldConfig


def make_checkerboard(n: int, grid_size: int = 4):
    x = torch.rand(n, 2)
    cells = torch.floor(x * grid_size).long().clamp(max=grid_size - 1)
    y = ((cells[:, 0] + cells[:, 1]) % 2).long()
    return x, y


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return (pred == y).float().mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "checkerboard_2d.pt"

    grid_size = 4
    n_train = 4096
    n_val = 2048
    epochs = 800
    lr = 2e-3

    # prepare datas
    x_train, y_train = make_checkerboard(n_train, grid_size)
    x_val, y_val = make_checkerboard(n_val, grid_size)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # configuration
    cfg = AATFieldConfig(
        input_dim=2,
        extra_dims=0,
        num_classes=2,
        layers=8,
        max_children=12,
        sigma_init=0.75,
        charge_init=0.08,
        step_cap=1.0,
    )

    # use aat model
    model = AATField(cfg).to(device)
    model.initialize(x_train, y_train, samples=n_train, min_children=8, kmeans_iters=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_train_acc = -1.0
    best_epoch = 0
    best_state = None

    # do training
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        val_acc = accuracy(model, x_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = accuracy(model, x_train, y_train)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            train_acc = accuracy(model, x_train, y_train)
            print(
                f"epoch={epoch} "
                f"loss={loss.item():.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} "
                f"best_val={best_val_acc:.4f}@{best_epoch}"
            )

    checkpoint = {
        "config": asdict(cfg),
        "state_dict": best_state,
        "grid_size": grid_size,
        "best_epoch": best_epoch,
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "selected_children": model.selected_children_by_layer(),
        "total_children": model.total_children(),
    }

    torch.save(checkpoint, out_path)

    print(f"saved best checkpoint: {out_path}")
    print(f"best_epoch: {best_epoch}")
    print(f"best_train_acc: {best_train_acc:.4f}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"selected_children: {model.selected_children_by_layer()}")
    print(f"total_children: {model.total_children()}")


if __name__ == "__main__":
    main()