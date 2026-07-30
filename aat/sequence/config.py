from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AATConfig:
    d_model: int
    encoder_layers: int
    decoder_layers: int
    heads: int
    rays: int
    chunk_size: int
    position_scale: float
    transport_steps: int = 3
    dropout: float = 0.10
    kappa: float = 6.0
    score_clip: float = 30.0
    gradient_checkpointing: bool = True
    # Kept only so older keyword-based configs can still be loaded. Sequence
    # AAT no longer contains a feed-forward branch.
    ffn_dim: int | None = None

    @property
    def head_dim(self) -> int:
        return int(self.d_model) // int(self.heads)

    def validate(self) -> None:
        if int(self.d_model) <= 0 or int(self.heads) <= 0:
            raise ValueError("d_model and heads must be positive.")
        if int(self.d_model) % int(self.heads) != 0:
            raise ValueError("d_model must be divisible by heads.")
        if int(self.encoder_layers) < 0 or int(self.decoder_layers) < 0:
            raise ValueError("encoder_layers and decoder_layers must be non-negative.")
        if int(self.encoder_layers) + int(self.decoder_layers) == 0:
            raise ValueError("at least one encoder or decoder layer is required.")
        if min(int(self.rays), int(self.chunk_size)) <= 0:
            raise ValueError("rays and chunk_size must be positive.")
        if int(self.transport_steps) < 0:
            raise ValueError("transport_steps must be non-negative.")
        if self.ffn_dim is not None and int(self.ffn_dim) <= 0:
            raise ValueError("ffn_dim must be positive when provided.")
        if float(self.position_scale) <= 0.0:
            raise ValueError("position_scale must be positive.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if float(self.kappa) <= 0.0 or float(self.score_clip) <= 0.0:
            raise ValueError("kappa and score_clip must be positive.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_sequence_length(
        cls,
        sequence_length: int,
        **values: Any,
    ) -> "AATConfig":
        if int(sequence_length) <= 1:
            raise ValueError("sequence_length must be greater than one.")
        cfg = cls(
            **values,
            position_scale=float(int(sequence_length) - 1),
        )
        cfg.validate()
        return cfg


__all__ = ["AATConfig"]
