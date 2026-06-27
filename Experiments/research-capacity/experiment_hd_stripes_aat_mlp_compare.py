import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
BASE_DIM = 8
EXPANDED_DIM = 10
N_TRAIN = 8192
N_VAL = 4096
BATCH_SIZE = 256
EPOCHS_AAT = 450
EPOCHS_MLP = 450
LR_AAT = 3e-3
LR_MLP = 1e-3
WEIGHT_DECAY_MLP = 1e-4
N_INTEGRAL_POINTS = 1024
SIGMA_START = 2.4
SIGMA_END = 0.85
STRIPE_FREQ = 4.0
C_INIT_STD = 0.08
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def unit(v):
    return v / (v.norm() + 1e-12)


u1 = unit(torch.randn(BASE_DIM))
u2 = torch.randn(BASE_DIM)
u2 = unit(u2 - (u2 @ u1) * u1)
expand_perm = torch.randperm(EXPANDED_DIM)


def make_data(n):
    x = torch.randn(n, BASE_DIM)
    s1 = x @ u1
    s2 = x @ u2
    y = ((torch.sin(STRIPE_FREQ * s1) * torch.sin(STRIPE_FREQ * s2)) > 0).long()
    return x, y


def expand_by_swap(x):
    z = torch.zeros(x.size(0), EXPANDED_DIM, dtype=x.dtype)
    z[:, :BASE_DIM] = x
    return z[:, expand_perm]


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        layers = []
        last = dim
        for width in hidden:
            layers.append(nn.Linear(last, width))
            layers.append(nn.GELU())
            last = width
        layers.append(nn.Linear(last, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IntegralTerrainLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("a", torch.randn(N_INTEGRAL_POINTS, dim))
        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)

    def force(self, z, sigma):
        diff = self.a[None, :, :] - z[:, None, :]
        dist2 = (diff * diff).sum(dim=-1)
        k = torch.exp(-dist2 / (2.0 * sigma * sigma))
        return (k[:, :, None] * self.c[None, :, None] * diff).mean(dim=1) / (sigma * sigma)

    def response_center(self, z, sigma):
        diff = self.a[None, :, :] - z[:, None, :]
        dist2 = (diff * diff).sum(dim=-1)
        score = self.c[None, :] - dist2 / (2.0 * sigma * sigma)
        alpha = torch.softmax(score, dim=1)
        return alpha @ self.a


class AATModel(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.layers = nn.ModuleList([IntegralTerrainLayer(dim) for _ in range(layers)])
        self.head = nn.Linear(dim, 2)

    def forward(self, x, sigma):
        z = x
        force_values = []
        for i, layer in enumerate(self.layers):
            f = layer.force(z, sigma)
            z_mid = z + f
            if i < len(self.layers) - 1:
                r = layer.response_center(z, sigma)
                z = torch.relu(r + torch.relu(z_mid - r))
            else:
                z = z_mid
            force_values.append(f.detach().abs().mean().item())
        return self.head(z), force_values


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sigma_at(epoch):
    t = epoch / max(1, EPOCHS_AAT - 1)
    t = 0.5 - 0.5 * math.cos(math.pi * t)
    return SIGMA_START + (SIGMA_END - SIGMA_START) * t


@torch.no_grad()
def evaluate_mlp(model, x, y):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, len(x), BATCH_SIZE):
        xb = x[start:start + BATCH_SIZE].to(DEVICE)
        yb = y[start:start + BATCH_SIZE].to(DEVICE)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb).item() * len(xb)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate_aat(model, x, y, sigma):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    force_sum = None
    for start in range(0, len(x), BATCH_SIZE):
        xb = x[start:start + BATCH_SIZE].to(DEVICE)
        yb = y[start:start + BATCH_SIZE].to(DEVICE)
        logits, forces = model(xb, sigma)
        total_loss += F.cross_entropy(logits, yb).item() * len(xb)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(xb)
        if force_sum is None:
            force_sum = [0.0 for _ in forces]
        for i, value in enumerate(forces):
            force_sum[i] += value * len(xb)
    return total_loss / total, total_correct / total, [v / total for v in force_sum]


def train_mlp(name, hidden, x_train, y_train, x_val, y_val, seed_offset):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    torch.manual_seed(SEED + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + seed_offset)
    model = MLP(BASE_DIM, hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MLP, weight_decay=WEIGHT_DECAY_MLP)
    print(f"Params: {count_params(model)}")
    best_acc = 0.0
    best_epoch = 0
    best_loss = 1e9
    n = len(x_train)
    for epoch in range(EPOCHS_MLP):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = x_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 25 == 0 or epoch == EPOCHS_MLP - 1:
            train_loss, train_acc = evaluate_mlp(model, x_train, y_train)
            val_loss, val_acc = evaluate_mlp(model, x_val, y_val)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_loss = val_loss
            print(
                f"Epoch {epoch:4d}/{EPOCHS_MLP} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"best {best_acc:.4f}@{best_epoch}"
            )
    return {"name": name, "params": count_params(model), "best_acc": best_acc, "best_epoch": best_epoch, "best_loss": best_loss}


def train_aat(name, layers, dim, x_train, y_train, x_val, y_val, seed_offset):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    torch.manual_seed(SEED + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + seed_offset)
    model = AATModel(dim, layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_AAT)
    print(f"Params: {count_params(model)} | dim: {dim} | layers: {layers} | integral points/layer: {N_INTEGRAL_POINTS}")
    best_acc = 0.0
    best_epoch = 0
    best_sigma = SIGMA_START
    best_loss = 1e9
    n = len(x_train)
    for epoch in range(EPOCHS_AAT):
        model.train()
        sigma = sigma_at(epoch)
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = x_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            logits, _ = model(xb, sigma)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 25 == 0 or epoch == EPOCHS_AAT - 1:
            train_loss, train_acc, _ = evaluate_aat(model, x_train, y_train, sigma)
            val_loss, val_acc, val_force = evaluate_aat(model, x_val, y_val, sigma)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_sigma = sigma
                best_loss = val_loss
            force_text = ", ".join([f"L{i + 1}:{v:.4f}" for i, v in enumerate(val_force)])
            print(
                f"Epoch {epoch:4d}/{EPOCHS_AAT} | sigma {sigma:.4f} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"force {force_text} | best {best_acc:.4f}@{best_epoch}"
            )
    return {"name": name, "params": count_params(model), "best_acc": best_acc, "best_epoch": best_epoch, "best_sigma": best_sigma, "best_loss": best_loss}


def main():
    print(f"Device: {DEVICE}")
    print("Task: 8D two-projection sinusoidal stripe XOR")
    print(f"Train: {N_TRAIN}, Val: {N_VAL}")
    print(f"AAT: hidden AAT activation, last layer no activation, integral points/layer: {N_INTEGRAL_POINTS}")
    x_train, y_train = make_data(N_TRAIN)
    x_val, y_val = make_data(N_VAL)
    x_train_10 = expand_by_swap(x_train)
    x_val_10 = expand_by_swap(x_val)
    print(f"10D expansion permutation: {expand_perm.tolist()}")

    results = []
    results.append(train_mlp("MLP_small_2x64", [64, 64], x_train, y_train, x_val, y_val, 10))
    results.append(train_mlp("MLP_big_4x512", [512, 512, 512, 512], x_train, y_train, x_val, y_val, 20))
    results.append(train_aat("AAT_L6_8D_AAT_hidden_activation", 6, BASE_DIM, x_train, y_train, x_val, y_val, 30))
    results.append(train_aat("AAT_L8_8D_AAT_hidden_activation", 8, BASE_DIM, x_train, y_train, x_val, y_val, 40))
    results.append(train_aat("AAT_L6_10D_padded_swapped_activation", 6, EXPANDED_DIM, x_train_10, y_train, x_val_10, y_val, 50))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        sigma_text = f" | sigma {r['best_sigma']:.4f}" if "best_sigma" in r else ""
        print(
            f"{r['name']:38s} | params {r['params']:8d} | "
            f"best_val_acc {r['best_acc']:.4f}@{r['best_epoch']} | "
            f"val_loss {r['best_loss']:.4f}{sigma_text}"
        )


if __name__ == "__main__":
    main()
