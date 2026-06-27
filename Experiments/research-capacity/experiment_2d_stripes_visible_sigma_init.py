import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
DIM = 2
N_TRAIN = 4096
N_VAL = 4096
BATCH_SIZE = 256
EPOCHS = 500
LR = 3e-3
N_INTEGRAL_POINTS = 1024
X_RANGE = 2.4
STRIPE_FREQ = 3.0
C_INIT_STD = 0.06
PRINT_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sigma initializer settings.
# Pure version: no self subtraction, no random-label baseline, no explicit penalty.
INIT_SAMPLE_SIZE = 4096
SIGMA_GRID_SIZE = 180
SIGMA_MIN = 0.05
SIGMA_MAX = 2.00
PROFILE_TOP_FRAC = 0.72


random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def make_data(n):
    x = (torch.rand(n, DIM) * 2.0 - 1.0) * X_RANGE
    y = ((torch.sin(STRIPE_FREQ * x[:, 0]) * torch.sin(STRIPE_FREQ * x[:, 1])) > 0).long()
    return x, y


@torch.no_grad()
def visible_structure_scores(x, y, sigma_grid):
    """
    Compute the pure Gaussian-eye visibility score

        J(sigma) = integral || sum_i exp(-||u-x_i||^2/(2 sigma^2)) s_i ||^2 du

    up to a positive constant. For finite samples this is proportional to

        sigma^d * sum_ij <s_i, s_j> exp(-||x_i-x_j||^2/(4 sigma^2)).

    Here s_i is the centered one-hot label vector e_y - mean(e_y).
    No self correction, no random-label baseline, no extra penalty.
    """
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    n = x.size(0)
    num_classes = int(y.max().item()) + 1

    onehot = F.one_hot(y, num_classes=num_classes).float()
    s = onehot - onehot.mean(dim=0, keepdim=True)

    # Pairwise squared distance. For 4096 samples this is about 67 MB in float32.
    dist2 = torch.cdist(x, x, p=2).pow(2)

    scores = []
    for sigma in sigma_grid.to(DEVICE):
        k = torch.exp(-dist2 / (4.0 * sigma * sigma))
        ks = k @ s
        energy = (ks * s).sum() / (n * n)
        score = (sigma ** x.size(1)) * energy
        scores.append(score.detach().float().cpu())

    return torch.stack(scores)


@torch.no_grad()
def select_sigma_profile_from_visibility(x_train, y_train, depth):
    n = min(INIT_SAMPLE_SIZE, x_train.size(0))
    x = x_train[:n].contiguous()
    y = y_train[:n].contiguous()

    sigma_grid = torch.exp(torch.linspace(math.log(SIGMA_MIN), math.log(SIGMA_MAX), SIGMA_GRID_SIZE))
    scores = visible_structure_scores(x, y, sigma_grid)

    # Numerical safety: this score should be non-negative, but clamp tiny negatives from float error.
    scores = torch.clamp(scores, min=0.0)
    max_score = float(scores.max().item())
    peak_idx = int(scores.argmax().item())

    if max_score <= 0.0 or not math.isfinite(max_score):
        profile = [0.22 for _ in range(depth)]
        return profile, sigma_grid, scores

    # First try true local peaks.
    peak_indices = []
    for i in range(1, SIGMA_GRID_SIZE - 1):
        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            peak_indices.append(i)
    peak_indices = sorted(peak_indices, key=lambda i: float(scores[i].item()), reverse=True)

    if len(peak_indices) >= depth:
        chosen = sorted(peak_indices[:depth])
        profile = [float(sigma_grid[i].item()) for i in chosen]
        return profile, sigma_grid, scores

    # If the curve is mostly single-peaked, use the high-visibility band and take log-quantiles.
    threshold = PROFILE_TOP_FRAC * max_score
    high = torch.nonzero(scores >= threshold, as_tuple=False).flatten()

    if high.numel() >= depth:
        log_high = torch.log(sigma_grid[high])
        q = torch.linspace(0.0, 1.0, depth)
        positions = q * (log_high.numel() - 1)
        idx0 = torch.floor(positions).long()
        idx1 = torch.clamp(idx0 + 1, max=log_high.numel() - 1)
        frac = positions - idx0.float()
        log_profile = log_high[idx0] * (1.0 - frac) + log_high[idx1] * frac
        profile = [float(v.exp().item()) for v in log_profile]
        return profile, sigma_grid, scores

    # Last fallback: spread around the best sigma in log space.
    peak_log = float(torch.log(sigma_grid[peak_idx]).item())
    spread = 0.50
    if depth == 1:
        logs = [peak_log]
    else:
        logs = torch.linspace(peak_log - spread, peak_log + spread, depth).tolist()
    profile = [float(max(SIGMA_MIN, min(SIGMA_MAX, math.exp(v)))) for v in logs]
    profile.sort()
    return profile, sigma_grid, scores


class IntegralTerrainLayer(nn.Module):
    def __init__(self, dim, sigma_init):
        super().__init__()
        self.register_buffer("a", (torch.rand(N_INTEGRAL_POINTS, dim) * 2.0 - 1.0) * X_RANGE)
        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(float(sigma_init)), dtype=torch.float32))

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
    def __init__(self, depth, sigma_profile):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            IntegralTerrainLayer(DIM, sigma_profile[i])
            for i in range(depth)
        ])
        self.head = nn.Linear(DIM, 2)

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


def train_one(name, depth, sigma_profile, x_train, y_train, x_val, y_val):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    # Same seed convention as previous sigma-policy scans.
    torch.manual_seed(SEED + depth * 100)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + depth * 100)

    model = AATModel(depth=depth, sigma_profile=sigma_profile).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    init_text = ", ".join(f"{v:.4f}" for v in sigma_profile)
    print(
        f"Params: {count_params(model)} | dim: {DIM} | layers: {depth} | "
        f"sigma_policy: trainable | init: visible-structure | init_sigmas [{init_text}] | "
        f"hidden local activation, last plain | integral points/layer: {N_INTEGRAL_POINTS}"
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
        "init_sigmas": sigma_profile,
    }


def main():
    print(f"Device: {DEVICE}")
    print("Task: 2D sinusoidal stripe XOR")
    print(f"Train: {N_TRAIN}, Val: {N_VAL}")
    print("AAT: visible-structure sigma initializer, hidden local activation, last layer plain")
    print(f"Integral points/layer: {N_INTEGRAL_POINTS}")

    x_train, y_train = make_data(N_TRAIN)
    x_val, y_val = make_data(N_VAL)

    print("\n" + "=" * 80)
    print("VISIBLE-STRUCTURE SIGMA INITIALIZER")
    print("=" * 80)
    profiles = {}
    for depth in [1, 2, 3, 4]:
        profile, sigma_grid, scores = select_sigma_profile_from_visibility(x_train, y_train, depth)
        profiles[depth] = profile
        peak_idx = int(scores.argmax().item())
        peak_sigma = float(sigma_grid[peak_idx].item())
        peak_score = float(scores[peak_idx].item())
        profile_text = ", ".join(f"{v:.4f}" for v in profile)
        print(
            f"L{depth}: peak_sigma {peak_sigma:.4f} | peak_score {peak_score:.6e} | "
            f"selected [{profile_text}]"
        )

    specs = []
    for depth in [1, 2, 3, 4]:
        specs.append((
            f"AAT_L{depth}_2D_visible_sigma_init_trainable",
            depth,
            profiles[depth],
        ))

    results = []
    for name, depth, sigma_profile in specs:
        results.append(train_one(name, depth, sigma_profile, x_train, y_train, x_val, y_val))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        init_text = ", ".join(f"{v:.3f}" for v in result["init_sigmas"])
        sigma_text = ", ".join(f"{v:.3f}" for v in result["best_sigmas"])
        print(
            f"{result['name']:<48} | params {result['params']:6d} | "
            f"best_val_acc {result['best_acc']:.4f}@{result['best_epoch']} | "
            f"val_loss {result['best_loss']:.4f} | init_sigmas [{init_text}] | best_sigmas [{sigma_text}]"
        )


if __name__ == "__main__":
    main()
