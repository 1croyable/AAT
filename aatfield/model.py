# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AATFieldConfig, AATFieldLayerConfig
from .utils import inv_softplus, make_permutation
from .initialize import initialize_layer_auto_k


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
        self.selected_counts: List[int] = [M for _ in range(C)]
        self.field_scale = float(getattr(cfg, "field_scale", 0.08))

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

    def _potential_response(self, z: torch.Tensor):
        anchors = self.all_anchors()
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
        sigma2 = sigma.view(1, -1).square().clamp_min(1e-8)
        scores = self.charge.view(1, -1) - dist2 / (2.0 * sigma2)
        alpha = torch.softmax(scores, dim=-1)
        weight = alpha / sigma2
        weighted_anchor = weight @ anchors
        weighted_sum = weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        field_target = weighted_anchor / weighted_sum
        move = self.field_scale * (weighted_anchor - z * weighted_sum)
        return move, field_target

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        move, field_target = self._potential_response(z)
        z_mid = z + move
        return torch.relu(field_target + torch.relu(z_mid - field_target))

    @torch.no_grad()
    def auto_k_init(self, z: torch.Tensor, y: torch.Tensor, *, min_children: int, kmeans_iters: int) -> None:
        initialize_layer_auto_k(self, z, y, min_children=int(min_children), kmeans_iters=int(kmeans_iters))

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
    def initialize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        samples: int = 8192,
        min_children: int = 2,
        kmeans_iters: int = 8,
    ) -> None:
        if int(min_children) < 2:
            raise ValueError("min_children must be >= 2 for AATField initialization.")

        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device=device, dtype=torch.long)

        if int(samples) > 0 and x.shape[0] > int(samples):
            idx = torch.randperm(x.shape[0], device=device)[: int(samples)]
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

    def config_dict(self) -> Dict[str, object]:
        cfg = self.cfg
        return {
            "input_dim": int(cfg.input_dim),
            "extra_dims": int(cfg.extra_dims),
            "num_classes": int(cfg.num_classes),
            "layers": int(cfg.layers),
            "max_children": int(cfg.max_children),
            "sigma_init": float(cfg.sigma_init),
            "charge_init": float(cfg.charge_init),
            "step_cap": float(cfg.step_cap),
            "gate_bias": bool(cfg.gate_bias),
            "head_bias": bool(cfg.head_bias),
            "lift_seed": int(cfg.lift_seed),
        }

    @torch.no_grad()
    def checkpoint_dict(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, object]:
        ckpt: Dict[str, object] = {
            "format": "AATFieldCheckpoint",
            "version": 1,
            "config": self.config_dict(),
            "state_dict": self.state_dict(),
            "selected_children": self.selected_children_by_layer(),
            "total_children": self.total_children(),
        }
        if metadata:
            ckpt["metadata"] = dict(metadata)
            for key, value in metadata.items():
                if key not in ckpt:
                    ckpt[key] = value
        return ckpt

    @torch.no_grad()
    def save_checkpoint(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        torch.save(self.checkpoint_dict(metadata=metadata), path)

    @staticmethod
    def _torch_load(path: str | Path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

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
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location=None) -> "AATField":
        ckpt = cls._torch_load(path, map_location=map_location)
        if not isinstance(ckpt, dict) or "config" not in ckpt:
            raise RuntimeError(
                "This checkpoint does not contain an AATField config. "
                "It may be an old raw state_dict checkpoint. Recreate the model with "
                "the original AATFieldConfig and call load_state_dict(...), or resave it "
                "with model.save_checkpoint(...)."
            )

        state = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        if state is None:
            raise RuntimeError("Checkpoint contains config but no state_dict/model_state_dict.")

        config = dict(ckpt["config"])
        model = cls(AATFieldConfig(**config))
        model.load_state_dict(state)
        return model

    @torch.no_grad()
    def load_checkpoint(self, path: str | Path, map_location=None, strict: bool = True) -> None:
        ckpt = self._torch_load(path, map_location=map_location)
        if not isinstance(ckpt, dict):
            raise RuntimeError("Invalid checkpoint object.")

        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        self.load_state_dict(state, strict=strict)


__all__ = [
    "AATFieldLayer",
    "AATField",
]
