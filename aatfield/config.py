
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

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