from __future__ import annotations

import math

import torch
import torch.nn as nn


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, width = x.shape
    if width % int(heads) != 0:
        raise ValueError("the last dimension must be divisible by heads.")
    return x.view(batch, tokens, int(heads), width // int(heads)).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, _, tokens, _ = x.shape
    return x.transpose(1, 2).contiguous().view(batch, tokens, -1)


def make_padding_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    shape = (x.shape[0], x.shape[1])
    if mask is None:
        return torch.zeros(shape, device=x.device, dtype=torch.bool)
    if tuple(mask.shape) != shape:
        raise ValueError(f"padding_mask must have shape {shape}.")
    return mask.to(device=x.device, dtype=torch.bool)


def init_identity(projection: nn.Linear) -> None:
    if projection.in_features != projection.out_features:
        raise ValueError("identity initialization requires a square projection.")
    with torch.no_grad():
        projection.weight.copy_(
            torch.eye(
                projection.in_features,
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
        )
        if projection.bias is not None:
            projection.bias.zero_()


def pad_tokens(x: torch.Tensor, tokens: int) -> torch.Tensor:
    if int(tokens) < x.shape[2]:
        raise ValueError("tokens cannot be smaller than the current token count.")
    if int(tokens) == x.shape[2]:
        return x
    shape = list(x.shape)
    shape[2] = int(tokens) - x.shape[2]
    return torch.cat((x, x.new_zeros(shape)), dim=2)


def prefix_memory(
    values: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    score_clip: float,
) -> torch.Tensor:
    if values.ndim != 4 or scores.ndim != 4 or values.shape[:3] != scores.shape[:3]:
        raise ValueError("values and scores must have matching [B,H,T] axes.")
    if tuple(mask.shape) != (values.shape[0], values.shape[2]):
        raise ValueError("padding_mask must have shape [B,T].")

    dtype = values.dtype
    with torch.autocast(device_type=values.device.type, enabled=False):
        values = values.float()
        weights = scores.float().clamp(-float(score_clip), float(score_clip)).exp()
        weights = weights * (~mask)[:, None, :, None].float()
        numerator = (weights.unsqueeze(-1) * values.unsqueeze(-2)).cumsum(dim=2)
        denominator = weights.cumsum(dim=2).clamp_min(1e-12)
        memory = (numerator / denominator.unsqueeze(-1)).masked_fill(
            mask[:, None, :, None, None],
            0.0,
        )
    return memory.to(dtype)


def ordered_state(
    address: torch.Tensor,
    payload: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if address.ndim != 4 or payload.ndim != 4 or address.shape[:3] != payload.shape[:3]:
        raise ValueError("address and payload must have matching [B,H,T] axes.")
    if tuple(mask.shape) != (address.shape[0], address.shape[2]):
        raise ValueError("padding_mask must have shape [B,T].")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")

    dtype = payload.dtype
    batch, heads, tokens, rays = address.shape
    payload_dim = payload.shape[-1]
    chunks = math.ceil(tokens / int(chunk_size))
    padded_tokens = chunks * int(chunk_size)

    with torch.autocast(device_type=payload.device.type, enabled=False):
        valid = (~mask)[:, None, :, None].float()
        address = pad_tokens(address.float() * valid, padded_tokens)
        payload = pad_tokens(payload.float() * valid, padded_tokens)
        address = address.view(batch, heads, chunks, int(chunk_size), rays)
        payload = payload.view(batch, heads, chunks, int(chunk_size), payload_dim)
        memory = payload.new_zeros(batch, heads, rays, payload_dim)
        innovations: list[torch.Tensor] = []
        boundaries: list[torch.Tensor] = []

        for index in range(chunks):
            chunk_address = address[:, :, index]
            chunk_payload = payload[:, :, index]
            boundaries.append(memory)
            residual = chunk_payload - torch.einsum(
                "bhcr,bhrd->bhcd",
                chunk_address,
                memory,
            )
            gram = torch.matmul(chunk_address, chunk_address.transpose(-1, -2))
            innovation = torch.linalg.solve_triangular(
                torch.tril(gram, diagonal=-1),
                residual,
                upper=False,
                unitriangular=True,
            )
            innovations.append(innovation)
            memory = memory + torch.einsum(
                "bhcr,bhcd->bhrd",
                chunk_address,
                innovation,
            )

    return (
        address.to(dtype),
        torch.stack(innovations, dim=2).to(dtype),
        torch.stack(boundaries, dim=2).to(dtype),
    )


def ordered_read(
    query: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    address, innovations, boundaries = state
    if query.ndim != 4 or address.ndim != 5 or innovations.ndim != 5 or boundaries.ndim != 5:
        raise ValueError("invalid ordered memory state.")
    if address.shape[:3] != innovations.shape[:3] or address.shape[:3] != boundaries.shape[:3]:
        raise ValueError("ordered memory chunk axes do not match.")

    batch, heads, tokens, rays = query.shape
    if tuple(mask.shape) != (batch, tokens):
        raise ValueError("padding_mask must have shape [B,T].")
    if address.shape[:2] != (batch, heads) or address.shape[-1] != rays:
        raise ValueError("query and ordered memory axes do not match.")

    chunks, chunk_size = address.shape[2:4]
    padded_tokens = chunks * chunk_size
    query = pad_tokens(
        query * (~mask)[:, None, :, None].to(query.dtype),
        padded_tokens,
    ).view(batch, heads, chunks, chunk_size, rays)
    read = torch.einsum("bhncr,bhnrd->bhncd", query, boundaries)
    coefficients = torch.einsum("bhnir,bhnjr->bhnij", query, address)
    causal = torch.ones(
        chunk_size,
        chunk_size,
        device=query.device,
        dtype=torch.bool,
    ).tril()
    coefficients = coefficients.masked_fill(
        ~causal.view(1, 1, 1, chunk_size, chunk_size),
        0.0,
    )
    read = read + torch.einsum("bhnij,bhnjd->bhnid", coefficients, innovations)
    return read.reshape(batch, heads, padded_tokens, innovations.shape[-1])[
        :, :, :tokens
    ].masked_fill(mask[:, None, :, None], 0.0)


__all__ = [
    "init_identity",
    "make_padding_mask",
    "merge_heads",
    "ordered_read",
    "ordered_state",
    "prefix_memory",
    "split_heads",
]
