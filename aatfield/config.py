# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AATFieldConfig:
    input_dim: int
    extra_dims: int = 0
    num_classes: int = 2
    layers: int = 4

    # Maximum candidate children per class used during Auto-K initialization.
    # After initialization, each layer is materialized with its selected child count.
    max_children: int = 10

    sigma_init: float = 0.75
    charge_init: float = 0.08
    step_cap: float = 1.0
    gate_bias: bool = True
    head_bias: bool = True
    lift_seed: int = 1234

    @property
    def state_dim(self) -> int:
        return int(self.input_dim) + int(self.extra_dims)

    @classmethod
    def from_data(
        cls,
        x: Any,
        *,
        num_classes: int,
        extra_dims: int = 0,
        layers: int = 4,
        max_children: int = 10,
        sigma_init: float = 0.75,
        charge_init: float = 0.08,
        step_cap: float = 1.0,
        gate_bias: bool = True,
        head_bias: bool = True,
        lift_seed: int = 1234,
    ) -> "AATFieldConfig":
        shape = getattr(x, "shape", None)
        if shape is None:
            raise ValueError("x must have a shape attribute.")

        shape = tuple(shape)
        if len(shape) < 2:
            raise ValueError("x must have shape [N, input_dim] or [N, ...].")

        input_dim = 1
        for s in shape[1:]:
            input_dim *= int(s)

        extra_dims_int = int(extra_dims)
        if extra_dims_int < 0:
            raise ValueError("extra_dims must be non-negative.")

        return cls(
            input_dim=int(input_dim),
            extra_dims=int(extra_dims_int),
            num_classes=int(num_classes),
            layers=int(layers),
            max_children=int(max_children),
            sigma_init=float(sigma_init),
            charge_init=float(charge_init),
            step_cap=float(step_cap),
            gate_bias=bool(gate_bias),
            head_bias=bool(head_bias),
            lift_seed=int(lift_seed),
        )

@dataclass
class AATFieldLayerConfig:
    state_dim: int
    max_children: int
    num_classes: int
    sigma_init: float
    charge_init: float
    step_cap: float
    gate_bias: bool = True

    @property
    def candidate_child_n(self) -> int:
        return int(self.num_classes) * int(self.max_children)

    @property
    def candidate_anchors_n(self) -> int:
        return int(self.num_classes) + self.candidate_child_n

__all__ = [
    "AATFieldConfig",
    "AATFieldLayerConfig",
]