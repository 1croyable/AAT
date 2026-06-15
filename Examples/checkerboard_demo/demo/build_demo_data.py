from pathlib import Path
import json
import torch

from aatfield import AATField, AATFieldConfig


def make_checkerboard(n: int, grid_size: int = 4):
    x = torch.rand(n, 2)
    cells = torch.floor(x * grid_size).long().clamp(max=grid_size - 1)
    y = ((cells[:, 0] + cells[:, 1]) % 2).long()
    return x, y


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = AATFieldConfig(**ckpt["config"])
    model = AATField(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_states(model: AATField, x: torch.Tensor):
    z = model.lift(x)
    states = [z.detach().cpu()]
    for layer in model.layers:
        z = layer(z)
        states.append(z.detach().cpu())
    return states


@torch.no_grad()
def predict_by_state(model: AATField, states):
    preds = []
    probs = []
    device = next(model.parameters()).device
    for z in states:
        logits = model.head(z.to(device))
        p = torch.softmax(logits, dim=1).detach().cpu()
        preds.append(p.argmax(dim=1))
        probs.append(p[:, 1])
    return preds, probs


def accuracy(pred: torch.Tensor, y: torch.Tensor):
    return float((pred == y.cpu()).float().mean().item())


def binary_head_boundary(model: AATField):
    w = model.head.weight.detach().cpu()
    b = model.head.bias.detach().cpu() if model.head.bias is not None else torch.zeros(2)
    normal = w[1] - w[0]
    bias = b[1] - b[0]
    return normal, float(bias.item())


def bounds_for_state(z: torch.Tensor, pad: float = 0.15):
    mins = z.min(dim=0).values
    maxs = z.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    return mins - span * pad, maxs + span * pad


def boundary_line_2d(normal: torch.Tensor, bias: float, z: torch.Tensor):
    mins, maxs = bounds_for_state(z)
    a = float(normal[0].item())
    b = float(normal[1].item())
    c = float(bias)

    if abs(b) >= abs(a) and abs(b) > 1e-8:
        x1 = float(mins[0].item())
        x2 = float(maxs[0].item())
        y1 = -(a * x1 + c) / b
        y2 = -(a * x2 + c) / b
        return [[x1, y1], [x2, y2]]

    if abs(a) > 1e-8:
        y1 = float(mins[1].item())
        y2 = float(maxs[1].item())
        x1 = -(b * y1 + c) / a
        x2 = -(b * y2 + c) / a
        return [[x1, y1], [x2, y2]]

    return [[0.0, 0.0], [0.0, 0.0]]


def boundary_plane_3d(normal: torch.Tensor, bias: float, z: torch.Tensor, steps: int = 14):
    mins, maxs = bounds_for_state(z)
    n = [float(v) for v in normal[:3].tolist()]
    c = float(bias)

    solve_axis = max([0, 1, 2], key=lambda i: abs(n[i]))
    grid_axes = [i for i in [0, 1, 2] if i != solve_axis]
    ranges = [torch.linspace(float(mins[ax].item()), float(maxs[ax].item()), steps) for ax in grid_axes]

    points = []
    denom = n[solve_axis]
    if abs(denom) < 1e-8:
        return {"solve_axis": solve_axis, "points": points}

    for u in ranges[0]:
        row = []
        for v in ranges[1]:
            p = [0.0, 0.0, 0.0]
            p[grid_axes[0]] = float(u.item())
            p[grid_axes[1]] = float(v.item())
            p[solve_axis] = -(n[grid_axes[0]] * p[grid_axes[0]] + n[grid_axes[1]] * p[grid_axes[1]] + c) / denom
            row.append(p)
        points.append(row)

    return {"solve_axis": solve_axis, "points": points}


def parent_positions(model: AATField):
    return [
        [[float(v) for v in row.tolist()] for row in layer.parents.detach().cpu()]
        for layer in model.layers
    ]


def pack_model_data(name: str, model: AATField, ckpt: dict, x: torch.Tensor, y: torch.Tensor):
    states = collect_states(model, x)
    preds, probs = predict_by_state(model, states)
    normal, bias = binary_head_boundary(model)
    accuracy_by_layer = [accuracy(pred, y) for pred in preds]

    if model.state_dim == 2:
        boundaries = [{"layer": i, "type": "line", "points": boundary_line_2d(normal, bias, states[i])} for i in range(len(states))]
    elif model.state_dim == 3:
        boundaries = [{"layer": i, "type": "plane", **boundary_plane_3d(normal, bias, states[i])} for i in range(len(states))]
    else:
        boundaries = []

    return {
        "name": name,
        "state_dim": int(model.state_dim),
        "layers": len(model.layers),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_train_acc": float(ckpt.get("best_train_acc", -1.0)),
        "best_val_acc": float(ckpt.get("best_val_acc", -1.0)),
        "selected_children": ckpt.get("selected_children", model.selected_children_by_layer()),
        "total_children": int(ckpt.get("total_children", model.total_children())),
        "accuracy_by_layer": accuracy_by_layer,
        "parents_by_layer": parent_positions(model),
        "boundary": {
            "normal": [float(v) for v in normal.tolist()],
            "bias": float(bias),
            "by_layer": boundaries,
        },
        "states": [[[float(v) for v in row.tolist()] for row in state] for state in states],
        "preds": [[int(v) for v in pred.tolist()] for pred in preds],
        "prob_class_1": [[float(v) for v in prob.tolist()] for prob in probs],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_dir = Path(__file__).resolve().parent
    root_dir = demo_dir.parent

    ckpt_2d_path = root_dir / "checkerboard_2d.pt"
    ckpt_3d_path = root_dir / "checkerboard_3d.pt"
    out_path = demo_dir / "demo_data.js"

    grid_size = 4
    n_points = 2000
    trace_count = 20

    x, y = make_checkerboard(n_points, grid_size)
    trace_indices = torch.randperm(n_points)[:trace_count].tolist()

    x_device = x.to(device)
    y_device = y.to(device)

    model_2d, ckpt_2d = load_model(ckpt_2d_path, device)
    model_3d, ckpt_3d = load_model(ckpt_3d_path, device)

    data_2d = pack_model_data("2d", model_2d, ckpt_2d, x_device, y_device)
    data_3d = pack_model_data("3d", model_3d, ckpt_3d, x_device, y_device)

    trace_set = set(trace_indices)
    points = []
    for i in range(n_points):
        points.append({
            "id": int(i),
            "input": [float(v) for v in x[i].tolist()],
            "label": int(y[i].item()),
            "trace": int(i) in trace_set,
        })

    payload = {
        "task": "checkerboard",
        "grid_size": grid_size,
        "n_points": n_points,
        "trace_indices": [int(i) for i in trace_indices],
        "points": points,
        "models": {
            "2d": data_2d,
            "3d": data_3d,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("const DEMO_DATA = ")
        json.dump(payload, f, ensure_ascii=False)
        f.write(";\n")

    print(f"saved: {out_path}")
    print(f"2d accuracy by layer: {[round(v, 4) for v in data_2d['accuracy_by_layer']]}")
    print(f"3d accuracy by layer: {[round(v, 4) for v in data_3d['accuracy_by_layer']]}")
    print(f"trace indices: {trace_indices}")


if __name__ == "__main__":
    main()