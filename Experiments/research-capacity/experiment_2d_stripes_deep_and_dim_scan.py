import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
BASE_DIM = 2
N_TRAIN = 4096
N_VAL = 4096
BATCH_SIZE = 256
EPOCHS = 500
LR = 3e-3
N_INTEGRAL_POINTS = 1024
X_RANGE = 2.4
STRIPE_FREQ = 3.0
C_INIT_STD = 0.06
SIGMA_INIT = 0.22
PRINT_EVERY = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def make_data(n):
    x = (torch.rand(n, BASE_DIM) * 2.0 - 1.0) * X_RANGE
    y = ((torch.sin(STRIPE_FREQ * x[:, 0]) * torch.sin(STRIPE_FREQ * x[:, 1])) > 0).long()
    return x, y


def make_permutation(dim):
    g = torch.Generator().manual_seed(SEED)
    return torch.randperm(dim, generator=g)


def expand_data(x, dim, perm):
    if dim == BASE_DIM:
        return x
    zeros = torch.zeros(x.shape[0], dim - BASE_DIM, dtype=x.dtype)
    x_pad = torch.cat([x, zeros], dim=1)
    return x_pad[:, perm]


class IntegralTerrainLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("a", (torch.rand(N_INTEGRAL_POINTS, dim) * 2.0 - 1.0) * X_RANGE)
        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(SIGMA_INIT), dtype=torch.float32))

    def sigma(self):
        return torch.exp(self.log_sigma)

    def force(self, z):
        sigma = self.sigma()
        diff = self.a[None, :, :] - z[:, None, :]
        dist2 = (diff * diff).sum(dim=-1)
        k = torch.exp(-dist2 / (2.0 * sigma * sigma))
        return (k[:, :, None] * self.c[None, :, None] * diff).mean(dim=1) / (sigma * sigma)

    def response_center(self, z):
        sigma = self.sigma()
        diff = self.a[None, :, :] - z[:, None, :]
        dist2 = (diff * diff).sum(dim=-1)
        score = self.c[None, :] - dist2 / (2.0 * sigma * sigma)
        alpha = torch.softmax(score, dim=1)
        return alpha @ self.a


class AATModel(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([IntegralTerrainLayer(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, 2)

    def local_activation(self, z, z_mid, layer):
        r = layer.response_center(z)
        return r + torch.relu(z_mid - r)

    def forward(self, x):
        z = x
        forces = []
        for i, layer in enumerate(self.layers):
            f = layer.force(z)
            z_mid = z + f
            if i < len(self.layers) - 1:
                z = self.local_activation(z, z_mid, layer)
            else:
                z = z_mid
            forces.append(f.detach().abs().mean().item())
        return self.head(z), forces

    def sigmas(self):
        return [layer.sigma().detach().item() for layer in self.layers]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    force_sum = None

    for start in range(0, len(x), BATCH_SIZE):
        xb = x[start:start + BATCH_SIZE].to(DEVICE)
        yb = y[start:start + BATCH_SIZE].to(DEVICE)
        logits, forces = model(xb)
        total_loss += F.cross_entropy(logits, yb).item() * len(xb)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(xb)
        if force_sum is None:
            force_sum = [0.0 for _ in forces]
        for i, value in enumerate(forces):
            force_sum[i] += value * len(xb)

    return total_loss / total, total_correct / total, [v / total for v in force_sum]


def train_one(name, dim, depth, x_train, y_train, x_val, y_val, seed_offset):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    torch.manual_seed(SEED + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + seed_offset)

    model = AATModel(dim=dim, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(
        f"Params: {count_params(model)} | dim: {dim} | layers: {depth} | "
        f"hidden local activation, last plain | sigma_init: {SIGMA_INIT:.3f} | "
        f"integral points/layer: {N_INTEGRAL_POINTS}"
    )

    best_acc = 0.0
    best_epoch = 0
    best_loss = 1e9
    best_sigmas = None
    n = len(x_train)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = x_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            train_loss, train_acc, _ = evaluate(model, x_train, y_train)
            val_loss, val_acc, val_force = evaluate(model, x_val, y_val)
            sigmas = model.sigmas()
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_loss = val_loss
                best_sigmas = sigmas

            force_text = ", ".join(f"L{i + 1}:{v:.4f}" for i, v in enumerate(val_force))
            sigma_text = ", ".join(f"L{i + 1}:{v:.3f}" for i, v in enumerate(sigmas))
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"force {force_text} | sigma {sigma_text} | "
                f"best {best_acc:.4f}@{best_epoch}"
            )

    return {
        "name": name,
        "params": count_params(model),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_sigmas": best_sigmas,
    }


def main():
    print(f"Device: {DEVICE}")
    print("Task: 2D sinusoidal stripe XOR")
    print(f"Train: {N_TRAIN}, Val: {N_VAL}")
    print("AAT: trainable per-layer sigma, hidden local activation, last layer plain")
    print(f"Integral points/layer: {N_INTEGRAL_POINTS}")

    x_train_base, y_train = make_data(N_TRAIN)
    x_val_base, y_val = make_data(N_VAL)

    specs = []
    for depth in [18, 20, 24]:
        specs.append((f"AAT_L{depth}_2D_hidden_local_activation_last_plain", BASE_DIM, depth))
    for dim in [4, 5, 6, 7, 8]:
        specs.append((f"AAT_L2_{dim}D_padded_swapped_hidden_local_activation_last_plain", dim, 2))

    results = []
    for i, (name, dim, depth) in enumerate(specs):
        perm = make_permutation(dim)
        x_train = expand_data(x_train_base, dim, perm)
        x_val = expand_data(x_val_base, dim, perm)
        if dim > BASE_DIM:
            print(f"\n{name} permutation: {perm.tolist()}")
        results.append(train_one(name, dim, depth, x_train, y_train, x_val, y_val, seed_offset=1000 + i))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        sigma_text = ", ".join(f"{v:.3f}" for v in result["best_sigmas"])
        print(
            f"{result['name']:<62} | params {result['params']:6d} | "
            f"best_val_acc {result['best_acc']:.4f}@{result['best_epoch']} | "
            f"val_loss {result['best_loss']:.4f} | best_sigmas [{sigma_text}]"
        )


if __name__ == "__main__":
    main()
