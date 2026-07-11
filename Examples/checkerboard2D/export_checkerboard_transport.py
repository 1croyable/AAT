from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path

import torch

from aatfield import AAT


POINTS = 3000
DATA_SEED = 1
GRID_SIZE = 220
BOUNDARY_SAMPLES = 2048
EPS = 1e-8


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("checkerboard_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_dataset_generator(module, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = ["make_checkerboard"]
    candidates.extend(
        name
        for name, value in vars(module).items()
        if callable(value)
        and getattr(value, "__module__", None) == module.__name__
        and "checker" in name.lower()
        and name not in candidates
    )

    for name in candidates:
        fn = getattr(module, name, None)
        if fn is None:
            continue

        signature = inspect.signature(fn)
        kwargs = {}
        valid = True

        for parameter in signature.parameters.values():
            key = parameter.name.lower()
            if key in {"n", "n_samples", "num_samples", "samples"}:
                kwargs[parameter.name] = int(n)
            elif key in {"seed", "random_seed"}:
                kwargs[parameter.name] = int(seed)
            elif parameter.default is inspect.Parameter.empty:
                valid = False
                break

        if not valid:
            continue

        result = fn(**kwargs)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            x = torch.as_tensor(result[0]).float()
            y = torch.as_tensor(result[1]).long()
            if x.ndim == 2 and x.shape[0] == y.shape[0] and x.shape[1] == 2:
                return x[:n], y[:n]

    raise RuntimeError("Could not find a compatible checkerboard generator in train_2d.py.")


def find_checkpoint(directory: Path) -> Path:
    preferred = [
        directory / "checkerboard_2d.pt",
        directory / "checkerboard2d.pt",
        directory / "checkerboard.pt",
    ]
    for path in preferred:
        if path.exists():
            return path

    matches = sorted(directory.glob("*checker*.pt"))
    if matches:
        return matches[0]

    raise FileNotFoundError("No checkerboard checkpoint was found in this directory.")


def inverse_radius(model: AAT, rho: torch.Tensor) -> torch.Tensor:
    r_min = model.r_min.to(device=rho.device, dtype=rho.dtype)
    r_max = model.r_max.to(device=rho.device, dtype=rho.dtype)
    return 0.5 * (rho + 1.0) * (r_max - r_min) + r_min


def state_to_cartesian(model: AAT, rho: torch.Tensor, u: torch.Tensor, *, clip_radius: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    center = model.center.to(device=rho.device, dtype=rho.dtype)
    radius = inverse_radius(model, rho)
    vis_radius = radius.clamp_min(0.0) if clip_radius else radius
    positions = center + vis_radius * u
    return positions, radius


def cartesian_to_state(model: AAT, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return model.to_polar(x)


def theta_from_u(u: torch.Tensor) -> torch.Tensor:
    return torch.atan2(u[:, 1], u[:, 0])


def rounded_list(x: torch.Tensor, decimals: int = 6):
    x = x.detach().cpu().float()
    factor = float(10 ** decimals)
    return (x * factor).round().div(factor).tolist()


@torch.no_grad()
def collect_frames(model: AAT, x: torch.Tensor, y: torch.Tensor) -> list[dict]:
    model.eval()
    rho, u = model.to_polar(x)
    frames = []

    def append_frame(name: str, layer_index: int) -> None:
        state = torch.cat((rho, u), dim=1)
        logits = model.head(state)
        probabilities = logits.softmax(dim=1)
        predictions = logits.argmax(dim=1)

        cart_positions, raw_radius = state_to_cartesian(model, rho, u, clip_radius=True)
        polar_positions = torch.stack((theta_from_u(u), rho.squeeze(1)), dim=1)

        frame_separability = float((predictions == y).float().mean().item())
        negative_fraction = float((raw_radius < 0).float().mean().item())

        frames.append({
            "name": name,
            "layer": int(layer_index),
            "head_separability": round(frame_separability, 6),
            "negative_radius_fraction": round(negative_fraction, 6),
            "positions": rounded_list(cart_positions),
            "polar_positions": rounded_list(polar_positions),
            "predictions": predictions.detach().cpu().tolist(),
            "confidence": rounded_list(probabilities.max(dim=1).values),
        })

    append_frame("input", 0)
    for layer_index, layer in enumerate(model.layers, start=1):
        rho, u = layer(rho, u)
        append_frame(f"layer_{layer_index}", layer_index)

    return frames


def positions_bounds(frames: list[dict], key: str, *, theta_fixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        [point for frame in frames for point in frame[key]],
        dtype=torch.float32,
    )
    lower = positions.min(dim=0).values
    upper = positions.max(dim=0).values
    if theta_fixed:
        lower[0] = -math.pi
        upper[0] = math.pi
    span = (upper - lower).clamp_min(1e-3)
    padding = span * 0.08
    if theta_fixed:
        padding[0] = 0.0
    return lower - padding, upper + padding


@torch.no_grad()
def evaluate_head_on_xy_grid(
    model: AAT,
    lower: torch.Tensor,
    upper: torch.Tensor,
    device: torch.device,
    grid_size: int = GRID_SIZE,
) -> dict:
    xs = torch.linspace(float(lower[0].item()), float(upper[0].item()), grid_size, device=device)
    ys = torch.linspace(float(lower[1].item()), float(upper[1].item()), grid_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

    rho, u = cartesian_to_state(model, points)
    state = torch.cat((rho, u), dim=1)
    probs = model.head(state).softmax(dim=1)[:, 1].reshape(grid_size, grid_size)

    return {
        "bounds": {"min": rounded_list(lower), "max": rounded_list(upper)},
        "x_values": rounded_list(xs),
        "y_values": rounded_list(ys),
        "prob_class1": rounded_list(probs),
    }


@torch.no_grad()
def evaluate_full_model_on_input_grid(
    model: AAT,
    lower: torch.Tensor,
    upper: torch.Tensor,
    device: torch.device,
    grid_size: int = GRID_SIZE,
) -> dict:
    xs = torch.linspace(float(lower[0].item()), float(upper[0].item()), grid_size, device=device)
    ys = torch.linspace(float(lower[1].item()), float(upper[1].item()), grid_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

    probs = model(points).softmax(dim=1)[:, 1].reshape(grid_size, grid_size)

    return {
        "bounds": {"min": rounded_list(lower), "max": rounded_list(upper)},
        "x_values": rounded_list(xs),
        "y_values": rounded_list(ys),
        "prob_class1": rounded_list(probs),
    }


@torch.no_grad()
def evaluate_head_on_polar_grid(
    model: AAT,
    rho_min: float,
    rho_max: float,
    device: torch.device,
    grid_size: int = GRID_SIZE,
) -> dict:
    theta = torch.linspace(-math.pi, math.pi, grid_size, device=device)
    rho = torch.linspace(rho_min, rho_max, grid_size, device=device)
    rr, tt = torch.meshgrid(rho, theta, indexing="ij")

    u = torch.stack((tt.cos(), tt.sin()), dim=-1).reshape(-1, 2)
    state = torch.cat((rr.reshape(-1, 1), u), dim=1)
    probs = model.head(state).softmax(dim=1)[:, 1].reshape(grid_size, grid_size)

    return {
        "bounds": {"min": [-math.pi, float(rho_min)], "max": [math.pi, float(rho_max)]},
        "theta_values": rounded_list(theta),
        "rho_values": rounded_list(rho),
        "prob_class1": rounded_list(probs),
    }


@torch.no_grad()
def exact_binary_head_boundary_polar(model: AAT, device: torch.device, rho_min: float, rho_max: float) -> dict:
    if model.num_classes != 2:
        raise RuntimeError("The exact 2D boundary exporter requires a binary classifier.")

    weight = model.head.weight.detach().to(device)
    bias = model.head.bias.detach().to(device)

    delta_w = weight[1] - weight[0]
    delta_b = bias[1] - bias[0]

    a_rho = delta_w[0]
    a_u0 = delta_w[1]
    a_u1 = delta_w[2]

    theta = torch.linspace(0.0, 2.0 * math.pi, BOUNDARY_SAMPLES + 1, device=device)
    rho = -(a_u0 * theta.cos() + a_u1 * theta.sin() + delta_b) / (a_rho + EPS)
    curve = torch.stack((theta, rho), dim=1)

    return {
        "space": "theta_rho",
        "decision_difference": {
            "weights": rounded_list(delta_w),
            "bias": round(float(delta_b.item()), 6),
        },
        "suggested_rho_range": [float(rho_min), float(rho_max)],
        "boundary_curve": rounded_list(curve),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = Path(__file__).resolve().parent
    training_script = directory / "train_2d.py"
    output_path = directory / "checkerboard_transport.json"

    module = load_training_module(training_script)
    x, y = call_dataset_generator(module, POINTS, DATA_SEED)
    x = x.to(device)
    y = y.to(device)

    checkpoint_path = find_checkpoint(directory)
    model = AAT.from_checkpoint(checkpoint_path, map_location=device).to(device)
    model.eval()

    frames = collect_frames(model, x, y)

    cart_lower, cart_upper = positions_bounds(frames, "positions")
    polar_lower, polar_upper = positions_bounds(frames, "polar_positions", theta_fixed=True)

    state_cartesian_grid = evaluate_head_on_xy_grid(model, cart_lower, cart_upper, device)

    input_lower = torch.tensor([x[:, 0].min().item(), x[:, 1].min().item()], device=device) - 0.03
    input_upper = torch.tensor([x[:, 0].max().item(), x[:, 1].max().item()], device=device) + 0.03
    input_model_grid = evaluate_full_model_on_input_grid(model, input_lower, input_upper, device)

    polar_grid = evaluate_head_on_polar_grid(
        model,
        rho_min=float(polar_lower[1].item()),
        rho_max=float(polar_upper[1].item()),
        device=device,
    )
    polar_boundary = exact_binary_head_boundary_polar(
        model,
        device=device,
        rho_min=float(polar_lower[1].item()),
        rho_max=float(polar_upper[1].item()),
    )

    payload = {
        "schema_version": 3,
        "demo": "checkerboard_2d",
        "dimension": 2,
        "checkpoint": checkpoint_path.name,
        "config": model.config_dict(),
        "point_count": int(x.shape[0]),
        "labels": y.detach().cpu().tolist(),
        "frames": frames,
        "views": {
            "transport_cartesian": state_cartesian_grid,
            "transport_polar": {
                **polar_grid,
                "boundary_curve": polar_boundary["boundary_curve"],
            },
            "input_model": input_model_grid,
        },
        "notes": {
            "transport_cartesian": "Head decision region evaluated on a Cartesian grid and converted to polar internally. More stable than reconstructing a boundary curve from possibly negative inverse radii.",
            "input_model": "True model decision region in the original checkerboard input plane.",
            "transport_polar": "Head decision region and exact linear boundary in (theta, rho) coordinates.",
        },
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"checkpoint: {checkpoint_path}")
    print(f"saved: {output_path}")
    final_acc = float((model(x).argmax(dim=1) == y).float().mean().item())
    print(f"final_model_accuracy_on_export_points={final_acc:.4f}")
    for frame in frames:
        print(
            f"{frame['name']:>10s} "
            f"head_sep={frame['head_separability']:.4f} "
            f"negative_radius={frame['negative_radius_fraction']:.4f}"
        )


if __name__ == "__main__":
    main()