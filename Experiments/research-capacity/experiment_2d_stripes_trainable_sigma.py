import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
BASE_DIM = 2
EXPANDED_DIM = 3
N_TRAIN = 4096
N_VAL = 4096
BATCH_SIZE = 256
EPOCHS = 500
LR = 3e-3
N_INTEGRAL_POINTS = 1024
X_RANGE = 2.4
STRIPE_FREQ = 3.0
C_INIT_STD = 0.06
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

expand_perm = torch.randperm(EXPANDED_DIM)


def make_data(n):
    x = (torch.rand(n, BASE_DIM) * 2.0 - 1.0) * X_RANGE
    y = ((torch.sin(STRIPE_FREQ * x[:, 0]) * torch.sin(STRIPE_FREQ * x[:, 1])) > 0).long()
    return x, y


def expand_by_swap(x):
    z = torch.zeros(x.size(0), EXPANDED_DIM, dtype=x.dtype)
    z[:, :BASE_DIM] = x
    return z[:, expand_perm]


class IntegralTerrainLayer(nn.Module):
    def __init__(self, dim, sigma_init):
        super().__init__()
        self.register_buffer("a", (torch.rand(N_INTEGRAL_POINTS, dim) * 2.0 - 1.0) * X_RANGE)
        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init), dtype=torch.float32))

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
    def __init__(self, dim, layers, sigma_init):
        super().__init__()
        self.layers = nn.ModuleList([IntegralTerrainLayer(dim, sigma_init) for _ in range(layers)])
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        z = x
        force_values = []
        for i, layer in enumerate(self.layers):
            f = layer.force(z)
            z_mid = z + f
            if i < len(self.layers) - 1:
                r = layer.response_center(z)
                z = torch.relu(r + torch.relu(z_mid - r))
            else:
                z = z_mid
            force_values.append(f.detach().abs().mean().item())
        return self.head(z), force_values

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


def train_one(name, layers, dim, sigma_init, x_train, y_train, x_val, y_val, seed_offset):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    torch.manual_seed(SEED + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + seed_offset)

    model = AATModel(dim, layers, sigma_init).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(
        f"Params: {count_params(model)} | dim: {dim} | layers: {layers} | "
        f"integral points/layer: {N_INTEGRAL_POINTS} | sigma_init: {sigma_init:.3f}"
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

        if epoch % 25 == 0 or epoch == EPOCHS - 1:
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
    print("AAT: trainable per-layer sigma, hidden AAT activation, last layer no activation")
    print(f"Integral points/layer: {N_INTEGRAL_POINTS}")
    print(f"3D expansion permutation: {expand_perm.tolist()}")

    x_train, y_train = make_data(N_TRAIN)
    x_val, y_val = make_data(N_VAL)
    x_train_3 = expand_by_swap(x_train)
    x_val_3 = expand_by_swap(x_val)

    configs = [
        ("AAT_L3_2D_trainable_sigma_init_0p60", 3, BASE_DIM, 0.60, x_train, x_val, 10),
        ("AAT_L3_2D_trainable_sigma_init_0p35", 3, BASE_DIM, 0.35, x_train, x_val, 20),
        ("AAT_L3_2D_trainable_sigma_init_0p22", 3, BASE_DIM, 0.22, x_train, x_val, 30),
        ("AAT_L2_3D_padded_swapped_trainable_sigma_init_0p60", 2, EXPANDED_DIM, 0.60, x_train_3, x_val_3, 40),
        ("AAT_L2_3D_padded_swapped_trainable_sigma_init_0p35", 2, EXPANDED_DIM, 0.35, x_train_3, x_val_3, 50),
        ("AAT_L2_3D_padded_swapped_trainable_sigma_init_0p22", 2, EXPANDED_DIM, 0.22, x_train_3, x_val_3, 60),
    ]

    results = []
    for name, layers, dim, sigma_init, x_tr, x_va, seed_offset in configs:
        results.append(train_one(name, layers, dim, sigma_init, x_tr, y_train, x_va, y_val, seed_offset))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        sigma_text = ", ".join(f"{v:.3f}" for v in r["best_sigmas"])
        print(
            f"{r['name']:<55} | params {r['params']:6d} | "
            f"best_val_acc {r['best_acc']:.4f}@{r['best_epoch']} | "
            f"val_loss {r['best_loss']:.4f} | best_sigmas [{sigma_text}]"
        )


if __name__ == "__main__":
    main()
