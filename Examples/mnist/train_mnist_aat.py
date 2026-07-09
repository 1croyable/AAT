from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from aatfield import AAT, AATConfig


@torch.no_grad()
def accuracy(model: AAT, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def clone_state_dict(model: AAT) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_epoch(model: AAT, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(xb.shape[0])
        total_n += int(xb.shape[0])

    return total_loss / max(total_n, 1)


def load_mnist(data_root: Path, device: torch.device):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    x_all = full_train.data.float().div(255.0).unsqueeze(1)
    y_all = full_train.targets.long()

    x_train = x_all[:50000].to(device)
    y_train = y_all[:50000].to(device)
    x_val = x_all[50000:].to(device)
    y_val = y_all[50000:].to(device)

    x_test = test_set.data.float().div(255.0).unsqueeze(1).to(device)
    y_test = test_set.targets.long().to(device)

    return x_train, y_train, x_val, y_val, x_test, y_test


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = out_dir / "data"
    out_path = out_dir / "mnist_aat.pt"

    seed = 0
    epochs = 60
    batch_size = 128
    eval_batch_size = 512
    lr = 3e-3
    weight_decay = 1e-4

    torch.manual_seed(seed)
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist(data_root, device)

    cfg = AATConfig(input_dim=28 * 28, num_classes=10, layers=4, rays=32, kappa=6.0, ray_dropout=0.2)
    model = AAT(cfg).to(device)
    model.fit_state(x_train)

    train_loader = DataLoader(
        TensorDataset(x_train.cpu(), y_train.cpu()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_epoch = 0
    best_train_acc = 0.0
    best_val_acc = -1.0
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = accuracy(model, x_val, y_val, batch_size=eval_batch_size)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_acc = accuracy(model, x_train, y_train, batch_size=eval_batch_size)
            best_test_acc = accuracy(model, x_test, y_test, batch_size=eval_batch_size)
            best_state = clone_state_dict(model)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            train_acc = accuracy(model, x_train, y_train, batch_size=eval_batch_size)
            test_acc = accuracy(model, x_test, y_test, batch_size=eval_batch_size)
            print(
                f"epoch={epoch:04d} loss={loss:.4f} train={train_acc:.4f} "
                f"val={val_acc:.4f} test={test_acc:.4f} best={best_val_acc:.4f}@{best_epoch}",
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
    print(f"best_test_acc={best_test_acc:.4f}")


if __name__ == "__main__":
    main()
