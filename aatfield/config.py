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
        extra_dims: int | str = 0,
        layers: int = 4,
        max_children: int = 10,
        sigma_init: float = 0.75,
        charge_init: float = 0.08,
        step_cap: float = 1.0,
        gate_bias: bool = True,
        head_bias: bool = True,
    ) -> "AATFieldConfig":
        """
        extra_dims:
            0       -> no extra dimensions
            32      -> add 32 zero dimensions
            "x2"    -> state_dim = 2 * input_dim
            "x3"    -> state_dim = 3 * input_dim
            "p256"  -> add 256 zero dimensions
        """
        shape = tuple(x.shape)
        if len(shape) < 2:
            raise ValueError("x must have shape [N, input_dim] or [N, ...].")

        if len(shape) > 2:
            input_dim = 1
            for s in shape[1:]:
                input_dim *= int(s)
        else:
            input_dim = int(shape[-1])

        if isinstance(extra_dims, str):
            key = extra_dims.lower().strip()

            if key in {"x1", "none", "0"}:
                extra_dims_int = 0
            elif key.startswith("x"):
                factor = float(key[1:])
                if factor < 1:
                    raise ValueError("extra_dims multiplier must be >= 1, e.g. 'x2'.")
                extra_dims_int = int(round(input_dim * (factor - 1)))
            elif key.startswith("p"):
                extra_dims_int = int(key[1:])
                if extra_dims_int < 0:
                    raise ValueError("p-style extra_dims must be non-negative, e.g. 'p256'.")
            else:
                raise ValueError(f"Unknown extra_dims format: {extra_dims!r}")
        else:
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