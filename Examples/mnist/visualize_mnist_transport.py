from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from aatfield import AAT


def load_one_per_digit(data_root: Path, device: torch.device):
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    xs = []
    ys = []
    found = set()

    for image, label in test_set:
        label = int(label)
        if label in found:
            continue
        xs.append(image)
        ys.append(label)
        found.add(label)
        if len(found) == 10:
            break

    if len(xs) != 10:
        raise RuntimeError("Could not find one sample for every digit.")

    order = sorted(range(10), key=lambda i: ys[i])
    x = torch.stack([xs[i] for i in order], dim=0).to(device)
    y = torch.tensor([ys[i] for i in order], dtype=torch.long, device=device)
    return x, y


def polar_to_image(model: AAT, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    center = model.center.to(device=rho.device, dtype=rho.dtype)
    r_min = model.r_min.to(device=rho.device, dtype=rho.dtype)
    r_max = model.r_max.to(device=rho.device, dtype=rho.dtype)

    r = 0.5 * (rho + 1.0) * (r_max - r_min) + r_min
    z = center + r * u
    return z.view(-1, 28, 28)


@torch.no_grad()
def collect_transport_images(model: AAT, x: torch.Tensor):
    model.eval()

    images = [x.squeeze(1).detach().cpu()]
    rho, u = model.to_polar(x)

    for layer in model.layers:
        rho, u = layer(rho, u)
        img = polar_to_image(model, rho, u)
        images.append(img.detach().cpu())

    return images


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent
    data_root = out_dir / "data"
    checkpoint_path = out_dir / "mnist_aat.pt"
    out_path = out_dir / "mnist_transport_layers.png"

    model = AAT.from_checkpoint(checkpoint_path, map_location=device).to(device)
    x, y = load_one_per_digit(data_root, device)

    images = collect_transport_images(model, x)
    titles = ["input"] + [f"layer {i}" for i in range(1, len(images))]

    rows = 10
    cols = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.7 * rows))

    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            img = images[col][row].clamp(0.0, 1.0)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    str(int(y[row].item())),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    pred = model(x).argmax(dim=1)
    print(f"checkpoint: {checkpoint_path}")
    print(f"saved: {out_path}")
    print("digits:", [int(v) for v in y.detach().cpu().tolist()])
    print("preds: ", [int(v) for v in pred.detach().cpu().tolist()])


if __name__ == "__main__":
    main()
