# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AATFieldConfig, AATFieldLayerConfig
from .utils import inv_softplus, make_permutation, pairwise_dist2
from .initialize import boundary_weights, weighted_kmeans, supervised_fisher_score, child_response_features_for_class


class AATFieldLayer(nn.Module):
    def __init__(self, cfg: AATFieldLayerConfig):
        super().__init__()
        self.cfg = cfg
        C = int(cfg.num_classes)
        M = int(cfg.max_children)
        D = int(cfg.state_dim)

        self.parents = nn.Parameter(torch.randn(C, D) * 0.20)
        self.child_offsets = nn.Parameter(torch.randn(C, M, D) * 0.06)
        self.charge = nn.Parameter(torch.randn(cfg.candidate_anchors_n) * float(cfg.charge_init))
        self.raw_sigma = nn.Parameter(torch.full((cfg.candidate_anchors_n,), inv_softplus(float(cfg.sigma_init))))
        if cfg.gate_bias:
            self.child_gate_bias = nn.Parameter(torch.zeros(cfg.candidate_child_n))
        else:
            self.register_parameter("child_gate_bias", None)

        self.selected_counts: List[int] = [M for _ in range(C)]

    @property
    def state_dim(self) -> int:
        return int(self.cfg.state_dim)

    @property
    def num_classes(self) -> int:
        return int(self.cfg.num_classes)

    @property
    def children_per_class(self) -> int:
        return int(self.child_offsets.shape[1])

    @property
    def child_n(self) -> int:
        return int(self.num_classes * self.children_per_class)

    @property
    def anchors_n(self) -> int:
        return int(self.num_classes + self.child_n)

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 1e-4

    def child_anchors(self) -> torch.Tensor:
        return self.parents.unsqueeze(1) + self.child_offsets

    def all_anchors(self) -> torch.Tensor:
        return torch.cat([self.parents, self.child_anchors().reshape(self.child_n, self.state_dim)], dim=0)

    def _child_gate(self, z: torch.Tensor, child_flat: torch.Tensor, sigma_child: torch.Tensor) -> torch.Tensor:
        C = self.num_classes
        K = self.children_per_class
        axis = self.child_offsets.reshape(C * K, self.state_dim)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        s = ((z.unsqueeze(1) - child_flat.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=-1)
        s = s / sigma_child.view(1, -1).clamp_min(1e-6)
        if self.child_gate_bias is not None:
            s = s + self.child_gate_bias.view(1, -1)
        return F.relu(s)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        C = self.num_classes
        child_flat = self.child_anchors().reshape(self.child_n, self.state_dim)
        anchors = torch.cat([self.parents, child_flat], dim=0)
        sigma = self.sigma()

        if sigma.shape[0] != anchors.shape[0]:
            raise RuntimeError(
                f"AATFieldLayer parameter shape mismatch: anchors={anchors.shape[0]}, sigma={sigma.shape[0]}. "
                "This usually means the layer was re-materialized after the optimizer was created. "
                "Call model.initialize(...) before creating the optimizer."
            )

        dist2 = (
            (z * z).sum(dim=-1, keepdim=True)
            + (anchors * anchors).sum(dim=-1).view(1, -1)
            - 2.0 * (z @ anchors.t())
        ).clamp_min(0.0)
        logits = -dist2 / (2.0 * sigma.view(1, -1).square() + 1e-8)
        alpha = torch.softmax(logits, dim=-1)

        dist = torch.sqrt(dist2 + 1e-8)
        strength = alpha * self.charge.view(1, -1) * sigma.view(1, -1)
        gate = self._child_gate(z, child_flat, sigma[C:])
        strength = torch.cat([strength[:, :C], strength[:, C:] * gate], dim=1)

        diff = anchors.unsqueeze(0) - z.unsqueeze(1)
        move = ((strength / dist.clamp_min(1e-6)).unsqueeze(-1) * diff).sum(dim=1)

        cap = float(self.cfg.step_cap)
        if cap > 0:
            norm = move.norm(dim=-1, keepdim=True)
            capped = cap * torch.tanh(norm / cap)
            move = move * (capped / norm.clamp_min(1e-6))
        return z + move

    @torch.no_grad()
    def auto_k_init(self, z: torch.Tensor, y: torch.Tensor, *, min_children: int, kmeans_iters: int) -> None:
        z = z.detach().float()
        y = y.detach().long().to(z.device)
        C = self.num_classes
        M = int(self.cfg.max_children)
        D = self.state_dim
        min_children = max(1, min(int(min_children), M))

        parents = torch.zeros(C, D, device=z.device, dtype=z.dtype)
        global_mean = z.mean(dim=0)
        for c in range(C):
            pts = z[y == c]
            parents[c] = pts.mean(dim=0) if pts.shape[0] > 0 else global_mean

        centers_by_class: List[Dict[int, torch.Tensor]] = []
        max_k_by_class: List[int] = []
        for c in range(C):
            pts = z[y == c]
            if pts.shape[0] == 0:
                pts = z
            w = boundary_weights(pts, c, parents)
            max_k = min(M, int(pts.shape[0]))
            max_k_by_class.append(max_k)

            centers_map: Dict[int, torch.Tensor] = {}
            for k in range(1, max_k + 1):
                centers, _ = weighted_kmeans(pts, w, k=k, iters=int(kmeans_iters))
                centers_map[int(k)] = centers.detach().clone()
            centers_by_class.append(centers_map)

        common_max_k = min(max_k_by_class) if max_k_by_class else M
        common_min_k = min(max(min_children, 1), common_max_k)

        all_centers_full = [centers_by_class[c][max_k_by_class[c]] for c in range(C)]
        full_anchors = torch.cat([parents] + all_centers_full, dim=0)
        nearest = torch.sqrt(pairwise_dist2(z, full_anchors).min(dim=1).values + 1e-8)
        sigma = float(torch.quantile(nearest, 0.20).item()) * 0.75
        sigma = max(0.05, min(3.0, sigma))

        best_k = int(common_min_k)
        best_score = -float("inf")
        for k in range(int(common_min_k), int(common_max_k) + 1):
            info_values: List[float] = []
            for c in range(C):
                bin_y = (y == c).long()
                phi = child_response_features_for_class(
                    z=z,
                    parent=parents[c],
                    centers=centers_by_class[c][int(k)],
                    sigma_value=sigma,
                )
                sc = supervised_fisher_score(phi, bin_y, num_classes=2)
                info_values.append(float(math.log1p(max(sc, 0.0))))
            layer_score = float(sum(info_values) / max(len(info_values), 1))
            if layer_score > best_score:
                best_score = layer_score
                best_k = int(k)

        child_centers = torch.zeros(C, best_k, D, device=z.device, dtype=z.dtype)
        for c in range(C):
            child_centers[c] = centers_by_class[c][best_k][:best_k]
        self._materialize(parents, child_centers, sigma)

    @torch.no_grad()
    def _materialize(self, parents: torch.Tensor, child_centers: torch.Tensor, sigma: float) -> None:
        C, K, _ = child_centers.shape
        device = self.parents.device
        dtype = self.parents.dtype
        parents = parents.to(device=device, dtype=dtype)
        child_centers = child_centers.to(device=device, dtype=dtype)

        self.parents.copy_(parents)
        self.child_offsets = nn.Parameter((child_centers - parents[:, None, :]).contiguous())

        anchors_n = int(C + C * K)
        self.raw_sigma = nn.Parameter(torch.full((anchors_n,), inv_softplus(float(sigma)), device=device, dtype=dtype))
        self.charge = nn.Parameter(torch.full((anchors_n,), float(self.cfg.charge_init), device=device, dtype=dtype))
        if self.cfg.gate_bias:
            self.child_gate_bias = nn.Parameter(torch.zeros(C * K, device=device, dtype=dtype))
        else:
            self.register_parameter("child_gate_bias", None)

        self.selected_counts = [int(K) for _ in range(C)]

    @torch.no_grad()
    def materialize_child_count(self, k: int) -> None:
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1.")

        C = self.num_classes
        D = self.state_dim
        device = self.parents.device
        dtype = self.parents.dtype

        self.child_offsets = nn.Parameter(torch.zeros(C, k, D, device=device, dtype=dtype))

        anchors_n = int(C + C * k)
        self.raw_sigma = nn.Parameter(torch.full((anchors_n,), inv_softplus(float(self.cfg.sigma_init)), device=device, dtype=dtype))
        self.charge = nn.Parameter(torch.full((anchors_n,), float(self.cfg.charge_init), device=device, dtype=dtype))
        if self.cfg.gate_bias:
            self.child_gate_bias = nn.Parameter(torch.zeros(C * k, device=device, dtype=dtype))
        else:
            self.register_parameter("child_gate_bias", None)

        self.selected_counts = [int(k) for _ in range(C)]


class AATField(nn.Module):
    def __init__(self, cfg: AATFieldConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = int(cfg.input_dim)
        self.extra_dims = int(cfg.extra_dims)
        self.state_dim = int(cfg.state_dim)
        self.num_classes = int(cfg.num_classes)

        layer_cfg = AATFieldLayerConfig(
            state_dim=self.state_dim,
            max_children=int(cfg.max_children),
            num_classes=int(cfg.num_classes),
            sigma_init=float(cfg.sigma_init),
            charge_init=float(cfg.charge_init),
            step_cap=float(cfg.step_cap),
            gate_bias=bool(cfg.gate_bias),
        )
        self.layers = nn.ModuleList([AATFieldLayer(layer_cfg) for _ in range(int(cfg.layers))])
        self.head = nn.Linear(self.state_dim, self.num_classes, bias=bool(cfg.head_bias))

        if self.extra_dims > 0:
            lift_perm = make_permutation(self.state_dim, int(cfg.lift_seed))
        else:
            lift_perm = torch.arange(self.state_dim, dtype=torch.long)
        self.register_buffer("lift_perm", lift_perm.long(), persistent=True)

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        x = x.float()
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")
        z = x * 2.0 - 1.0
        if self.extra_dims > 0:
            z = torch.cat([z, z.new_zeros((z.shape[0], self.extra_dims))], dim=-1)
        return z.index_select(dim=-1, index=self.lift_perm.to(z.device))

    def transport(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lift(x)
        for layer in self.layers:
            z = layer(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.transport(x))


    @torch.no_grad()
    def materialize_from_state_dict(self, state_dict) -> None:
        for li, layer in enumerate(self.layers):
            key = f"layers.{li}.child_offsets"
            if key not in state_dict:
                continue

            value = state_dict[key]
            if value.dim() != 3:
                raise RuntimeError(f"Invalid checkpoint tensor shape for {key}: {tuple(value.shape)}")

            C, K, D = map(int, value.shape)
            if C != layer.num_classes:
                raise RuntimeError(f"Checkpoint class count mismatch in layer {li}: checkpoint={C}, model={layer.num_classes}")
            if D != layer.state_dim:
                raise RuntimeError(f"Checkpoint state_dim mismatch in layer {li}: checkpoint={D}, model={layer.state_dim}")
            if layer.children_per_class != K:
                layer.materialize_child_count(K)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.materialize_from_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, y: torch.Tensor, *, samples: int = 8192, min_children: int = 2, kmeans_iters: int = 8, seed: int = 123) -> None:
        if int(min_children) < 2:
            raise ValueError("min_children must be >= 2 for AATField initialization.")

        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device=device, dtype=torch.long)

        if int(samples) > 0 and x.shape[0] > int(samples):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            idx = torch.randperm(x.shape[0], generator=gen, device=device)[: int(samples)]
            x = x[idx]
            y = y[idx]

        z = self.lift(x)
        for layer in self.layers:
            layer.auto_k_init(
                z,
                y,
                min_children=int(min_children),
                kmeans_iters=int(kmeans_iters),
            )
            z = layer(z)

        if was_training:
            self.train()

    @torch.no_grad()
    def selected_children_by_layer(self) -> List[List[int]]:
        return [list(map(int, layer.selected_counts)) for layer in self.layers]

    @torch.no_grad()
    def total_children(self) -> int:
        return int(sum(sum(counts) for counts in self.selected_children_by_layer()))


__all__ = [
    "AATFieldLayer",
    "AATField",
]
