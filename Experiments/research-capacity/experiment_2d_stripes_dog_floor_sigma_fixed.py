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
# DoG version: use a center-surround / Difference-of-Gaussians visibility score.
# Fixed-sigma experiment: sigmas are selected once and then kept constant during training.
INIT_SAMPLE_SIZE = 4096
SIGMA_GRID_SIZE = 220
SIGMA_MIN_BASE = 0.04
SIGMA_MAX = 2.00
PROFILE_TOP_FRAC = 0.70
DOG_KAPPA = 1.60

# Resolution floor: the DoG eye must not look below the sampling resolution.
# It is computed from the median nearest-neighbor distance of the initialization sample.
NN_FLOOR_MULT = 5.0
MIN_PROFILE_LOG_SPAN = 0.55


random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def make_data(n):
    x = (torch.rand(n, DIM) * 2.0 - 1.0) * X_RANGE
    y = ((torch.sin(STRIPE_FREQ * x[:, 0]) * torch.sin(STRIPE_FREQ * x[:, 1])) > 0).long()
    return x, y


@torch.no_grad()
def dog_visible_structure_scores(x, y, sigma_grid):
    """
    Compute a center-surround / Difference-of-Gaussians visibility score.

    The previous raw Gaussian visibility was a low-pass score:
        R_sigma(u) = sum_i G_sigma(u - x_i) s_i
        J(sigma) = integral ||R_sigma(u)||^2 du

    That can prefer overly coarse sigmas. Here we use a band-pass eye:
        D_sigma(x) = phi_sigma(x) - phi_{kappa sigma}(x)
        B_sigma(u) = sum_i D_sigma(u - x_i) s_i
        J_DoG(sigma) = integral ||B_sigma(u)||^2 du

    phi is the normalized Gaussian density. The integral can be written as a
    pairwise kernel without explicitly sliding a grid:

        integral phi_a(u-x_i) phi_b(u-x_j) du
        = phi_{sqrt(a^2+b^2)}(x_i-x_j).

    Therefore the DoG pair kernel is:
        K = phi_{sqrt(2) sigma}
            - 2 phi_{sqrt(1+kappa^2) sigma}
            + phi_{sqrt(2) kappa sigma}.

    Labels are centered so that global class imbalance is not treated as structure.
    No self/random correction is used here; the band-pass filter itself is the
    structure detector.
    """
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    n = x.size(0)
    d = x.size(1)
    num_classes = int(y.max().item()) + 1

    onehot = F.one_hot(y, num_classes=num_classes).float()
    s = onehot - onehot.mean(dim=0, keepdim=True)

    dist2 = torch.cdist(x, x, p=2).pow(2)
    scores = []

    two_pi = 2.0 * math.pi
    kappa = DOG_KAPPA

    for sigma in sigma_grid.to(DEVICE):
        # Variance of product-integral kernels.
        v_cc = 2.0 * sigma * sigma
        v_cs = (1.0 + kappa * kappa) * sigma * sigma
        v_ss = 2.0 * kappa * kappa * sigma * sigma

        # Normalized Gaussian densities in d dimensions.
        norm_cc = (two_pi * v_cc).pow(torch.tensor(-0.5 * d, device=DEVICE))
        norm_cs = (two_pi * v_cs).pow(torch.tensor(-0.5 * d, device=DEVICE))
        norm_ss = (two_pi * v_ss).pow(torch.tensor(-0.5 * d, device=DEVICE))

        k_cc = norm_cc * torch.exp(-dist2 / (2.0 * v_cc))
        k_cs = norm_cs * torch.exp(-dist2 / (2.0 * v_cs))
        k_ss = norm_ss * torch.exp(-dist2 / (2.0 * v_ss))

        kernel = k_cc - 2.0 * k_cs + k_ss
        score = (kernel @ s * s).sum() / (n * n)
        scores.append(score.detach().float().cpu())

    return torch.stack(scores)


@torch.no_grad()
def nearest_neighbor_resolution_floor(x):
    """
    Estimate the smallest meaningful visual scale from sample spacing.

    DoG visibility without a resolution floor tends to select the smallest sigma,
    because it sees point-level sampling granularity as high-frequency structure.
    This floor plays the role of eye/sensor resolution: the eye is not allowed to
    inspect below a few nearest-neighbor distances.
    """
    x = x.to(DEVICE)
    dist = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float("inf"))
    nn = dist.min(dim=1).values
    median_nn = float(nn.median().item())
    sigma_floor = max(SIGMA_MIN_BASE, NN_FLOOR_MULT * median_nn)
    sigma_floor = min(sigma_floor, SIGMA_MAX * 0.5)
    return sigma_floor, median_nn


@torch.no_grad()
def select_sigma_profile_from_visibility(x_train, y_train, depth):
    n = min(INIT_SAMPLE_SIZE, x_train.size(0))
    x = x_train[:n].contiguous()
    y = y_train[:n].contiguous()

    sigma_floor, median_nn = nearest_neighbor_resolution_floor(x)
    sigma_grid = torch.exp(torch.linspace(math.log(sigma_floor), math.log(SIGMA_MAX), SIGMA_GRID_SIZE))
    scores = dog_visible_structure_scores(x, y, sigma_grid)

    # DoG scores should be non-negative in theory, but numerical noise can be tiny negative.
    scores = torch.clamp(scores, min=0.0)
    max_score = float(scores.max().item())
    peak_idx = int(scores.argmax().item())

    if max_score <= 0.0 or not math.isfinite(max_score):
        profile = [0.22 for _ in range(depth)]
        meta = {
            "sigma_floor": sigma_floor,
            "median_nn": median_nn,
            "peak_sigma": 0.22,
            "peak_score": 0.0,
        }
        return profile, sigma_grid, scores, meta

    peak_sigma = float(sigma_grid[peak_idx].item())
    log_floor = math.log(sigma_floor)
    log_max = math.log(SIGMA_MAX)

    # For depth=1, use the best visible scale directly.
    if depth == 1:
        profile = [peak_sigma]
        meta = {
            "sigma_floor": sigma_floor,
            "median_nn": median_nn,
            "peak_sigma": peak_sigma,
            "peak_score": max_score,
        }
        return profile, sigma_grid, scores, meta

    # First try the high-score band. If the band is broad enough, take quantiles.
    threshold = PROFILE_TOP_FRAC * max_score
    high = torch.nonzero(scores >= threshold, as_tuple=False).flatten()

    use_band = False
    if high.numel() >= depth:
        log_high = torch.log(sigma_grid[high])
        if float((log_high[-1] - log_high[0]).item()) >= MIN_PROFILE_LOG_SPAN:
            use_band = True

    if use_band:
        q = torch.linspace(0.0, 1.0, depth)
        positions = q * (log_high.numel() - 1)
        idx0 = torch.floor(positions).long()
        idx1 = torch.clamp(idx0 + 1, max=log_high.numel() - 1)
        frac = positions - idx0.float()
        log_profile = log_high[idx0] * (1.0 - frac) + log_high[idx1] * frac
        profile = [float(v.exp().item()) for v in log_profile]
    else:
        # If DoG still peaks at the resolution floor, do not let all layers collapse
        # to the same tiny scale. Give the network a small structural scale ladder.
        # This is not a model parameter; it only spreads the initializer over the
        # visible band above the sensor resolution.
        peak_log = math.log(peak_sigma)
        if peak_log - log_floor < 0.35 * MIN_PROFILE_LOG_SPAN:
            lo = log_floor
            hi = min(log_max, lo + MIN_PROFILE_LOG_SPAN)
        else:
            lo = max(log_floor, peak_log - 0.5 * MIN_PROFILE_LOG_SPAN)
            hi = min(log_max, peak_log + 0.5 * MIN_PROFILE_LOG_SPAN)
            if hi - lo < MIN_PROFILE_LOG_SPAN:
                if lo <= log_floor + 1e-12:
                    hi = min(log_max, lo + MIN_PROFILE_LOG_SPAN)
                else:
                    lo = max(log_floor, hi - MIN_PROFILE_LOG_SPAN)
        log_profile = torch.linspace(lo, hi, depth)
        profile = [float(v.exp().item()) for v in log_profile]

    profile.sort()
    meta = {
        "sigma_floor": sigma_floor,
        "median_nn": median_nn,
        "peak_sigma": peak_sigma,
        "peak_score": max_score,
    }
    return profile, sigma_grid, scores, meta


class IntegralTerrainLayer(nn.Module):
    def __init__(self, dim, sigma_init):
        super().__init__()
        self.register_buffer("a", (torch.rand(N_INTEGRAL_POINTS, dim) * 2.0 - 1.0) * X_RANGE)
        self.c = nn.Parameter(torch.randn(N_INTEGRAL_POINTS) * C_INIT_STD)
        # Fixed sigma: initialized by the dog-visible scan, not trained.
        self.register_buffer("sigma_value", torch.tensor(float(sigma_init), dtype=torch.float32))

    def sigma(self):
        return self.sigma_value

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
        f"sigma_policy: fixed | init: dog-visible | init_sigmas [{init_text}] | "
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
    print("AAT: dog-visible fixed sigma initializer with resolution floor, hidden local activation, last layer plain")
    print(f"Integral points/layer: {N_INTEGRAL_POINTS}")

    x_train, y_train = make_data(N_TRAIN)
    x_val, y_val = make_data(N_VAL)

    print("\n" + "=" * 80)
    print("DOG VISIBLE-STRUCTURE FIXED SIGMA INITIALIZER + RESOLUTION FLOOR")
    print("=" * 80)
    profiles = {}
    for depth in [1, 2, 3, 4]:
        profile, sigma_grid, scores, meta = select_sigma_profile_from_visibility(x_train, y_train, depth)
        profiles[depth] = profile
        peak_sigma = float(meta["peak_sigma"])
        peak_score = float(meta["peak_score"])
        sigma_floor = float(meta["sigma_floor"])
        median_nn = float(meta["median_nn"])
        profile_text = ", ".join(f"{v:.4f}" for v in profile)
        print(
            f"L{depth}: median_nn {median_nn:.4f} | sigma_floor {sigma_floor:.4f} | "
            f"peak_sigma {peak_sigma:.4f} | peak_score {peak_score:.6e} | "
            f"selected [{profile_text}]"
        )

    specs = []
    for depth in [1, 2, 3, 4]:
        specs.append((
            f"AAT_L{depth}_2D_dog_floor_sigma_init_fixed",
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
