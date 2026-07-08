from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class AATConfig:
    input_dim: int
    num_classes: int
    layers: int
    rays: int | Sequence[int]
    kappa: float = 6.0
    ray_dropout: float = 0.15

    @property
    def state_dim(self) -> int:
        return int(self.input_dim)

    @property
    def ray_counts(self) -> tuple[int, ...]:
        if isinstance(self.rays, int):
            return tuple(int(self.rays) for _ in range(int(self.layers)))
        counts = tuple(int(v) for v in self.rays)
        if len(counts) == 0:
            raise ValueError("rays must not be empty.")
        return counts

    def validate(self) -> None:
        if int(self.input_dim) <= 0:
            raise ValueError("input_dim must be positive.")
        if int(self.num_classes) <= 1:
            raise ValueError("num_classes must be greater than 1.")
        
        counts = self.ray_counts
        if any(v <= 0 for v in counts):
            raise ValueError("all ray counts must be positive.")
        
        if isinstance(self.rays, int):
            if int(self.layers) <= 0:
                raise ValueError("layers must be positive.")
        else:
            self.layers = len(counts)

        if float(self.kappa) <= 0.0:
            raise ValueError("kappa must be positive.")
        if not (0.0 <= float(self.ray_dropout) < 1.0):
            raise ValueError("ray_dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "input_dim": int(self.input_dim),
            "num_classes": int(self.num_classes),
            "layers": int(self.layers),
            "rays": list(self.ray_counts),
            "kappa": float(self.kappa),
            "ray_dropout": float(self.ray_dropout),
        }

    @classmethod
    def from_data(
        cls,
        x: Any,
        *,
        num_classes: int,
        layers: int,
        rays: int | Sequence[int],
        kappa: float = 6.0,
        ray_dropout: float = 0.15,
    ) -> "AATConfig":
        shape = getattr(x, "shape", None)
        if shape is None:
            raise ValueError("x must have a shape attribute.")
        shape = tuple(shape)
        if len(shape) < 2:
            raise ValueError("x must have shape [N, input_dim] or [N, ...].")
        input_dim = 1
        for s in shape[1:]:
            input_dim *= int(s)
        cfg = cls(input_dim=int(input_dim), num_classes=int(num_classes), layers=int(layers), rays=rays, kappa=float(kappa), ray_dropout=float(ray_dropout))
        cfg.validate()
        return cfg


__all__ = ["AATConfig"]
