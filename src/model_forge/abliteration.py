"""Deterministic abliteration: direction measurement and weight projection.

Implements the published refusal-direction removal math:
  W' = W - r @ (r.T @ W)   for row-major residual-writing matrices

where r is the unit-norm harmful-minus-harmless direction at the target layer.
All projection math is performed in float32 for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from model_forge.selectors import is_vision_tensor as is_vision_tensor
from model_forge.selectors import matches_selector as matches_selector


@dataclass(frozen=True)
class DirectionResult:
    direction: torch.Tensor  # unit-norm float32 [hidden_dim]
    harmful_mean: torch.Tensor
    harmless_mean: torch.Tensor
    raw_diff_norm: float
    layer: int
    seed: int


def mask_massive_activations(
    activations: torch.Tensor,
    *,
    absolute_threshold: float = 100.0,
    median_multiplier: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask input-agnostic massive-activation dimensions before mean-difference."""
    values = activations.float()
    medians = values.abs().median(dim=1, keepdim=True).values.clamp_min(1e-12)
    hits = (values.abs() > absolute_threshold) & (
        values.abs() > median_multiplier * medians
    )
    masked_dimensions = hits.any(dim=0)
    result = values.clone()
    result[:, masked_dimensions] = 0
    return result, masked_dimensions


def compute_refusal_direction(
    harmful_activations: torch.Tensor,
    harmless_activations: torch.Tensor,
    *,
    orthogonalize_harmless: bool = False,
    seed: int = 42,
    layer: int = 38,
) -> DirectionResult:
    """Compute normalized harmful-minus-harmless direction from last-token activations.

    Args:
        harmful_activations: [n_harmful, hidden_dim] float32
        harmless_activations: [n_harmless, hidden_dim] float32
        orthogonalize_harmless: if True, subtract harmless projection from direction
        seed: deterministic seed (unused in math but recorded for provenance)
        layer: layer index (recorded for provenance)
    """
    assert harmful_activations.ndim == 2
    assert harmless_activations.ndim == 2
    assert harmful_activations.shape[1] == harmless_activations.shape[1]

    harmful_activations = harmful_activations.float()
    harmless_activations = harmless_activations.float()

    harmful_mean = harmful_activations.mean(dim=0)
    harmless_mean = harmless_activations.mean(dim=0)

    direction = harmful_mean - harmless_mean

    if orthogonalize_harmless:
        harmless_norm = harmless_mean.norm()
        if harmless_norm == 0:
            raise ValueError("Cannot orthogonalize against a zero harmless mean")
        harmless_unit = harmless_mean / harmless_norm
        projection = (direction @ harmless_unit) * harmless_unit
        direction = direction - projection

    raw_norm = direction.norm().item()
    if raw_norm == 0:
        raise ValueError("Harmful and harmless means produced a zero refusal direction")
    direction = direction / raw_norm

    return DirectionResult(
        direction=direction,
        harmful_mean=harmful_mean,
        harmless_mean=harmless_mean,
        raw_diff_norm=raw_norm,
        layer=layer,
        seed=seed,
    )


def project_weight(
    weight: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Apply W' = W - r @ (r^T @ W) in float32.

    For a residual-writing matrix with shape [out_features, in_features],
    the direction is projected along the output dimension (rows).
    For embed_tokens with shape [vocab, hidden], direction projects along columns.
    """
    orig_dtype = weight.dtype
    w = weight.float()
    r = direction.float()

    if w.ndim == 1:
        # 1-D parameter (e.g. norm_f gain): w' = w - (w . r) r
        proj = (w @ r) * r
        w = w - proj
    elif w.shape[0] == r.shape[0]:
        # [out, in] with direction along output dim: W' = W - r @ (r^T @ W)
        proj = torch.outer(r, r @ w)
        w = w - proj
    elif w.shape[1] == r.shape[0]:
        # [vocab, hidden] embed_tokens: W' = W - (W @ r) @ r^T
        proj = torch.outer(w @ r, r)
        w = w - proj
    else:
        raise ValueError(
            f"Direction dim {r.shape[0]} incompatible with weight shape {w.shape}"
        )

    return w.to(orig_dtype)


def compute_leakage(
    weight: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    """Compute residual refusal-direction leakage: |r^T @ W| / |W|."""
    w = weight.float()
    r = direction.float()
    if w.shape[0] == r.shape[0]:
        proj_norm = (r @ w).norm().item()
    elif w.shape[1] == r.shape[0]:
        proj_norm = (w @ r).norm().item()
    else:
        return 0.0
    w_norm: float = w.norm().item()
    if w_norm == 0:
        return 0.0
    return float(proj_norm / w_norm)


def biproject_direction(
    direction: torch.Tensor,
    harmless_mean: torch.Tensor,
) -> torch.Tensor:
    """Orthogonalize a refusal direction against the harmless mean activation.

    jim-plus/llm-abliteration "projected" measurement (mlabonne biprojected
    abliteration): the ablated axis carries only the harmful-conditional
    component, leaving shared harmless content intact. Returns a unit vector.
    """
    d = direction.float()
    n = harmless_mean.float()
    n_norm_sq = n.norm().pow(2)
    if n_norm_sq.item() < 1e-12:
        return d / d.norm()
    d = d - (d @ n) * n / n_norm_sq
    return d / d.norm()


def norm_preserving_project(
    weight: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Ablate then restore each edited row to its pre-edit L2 norm.

    grimjim norm-preserving biprojected abliteration: prevents the global
    norm shrink that degrades non-refusal behavior when many rows lose the
    same direction component. Applies :func:`project_weight` then rescales.
    """
    edited = project_weight(weight, direction)
    if edited.ndim != 2 or weight.ndim != 2:
        return edited
    orig_norms = weight.float().norm(dim=1, keepdim=True)
    new_norms = edited.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (orig_norms / new_norms).to(edited.dtype)
    return (edited.float() * scale).to(edited.dtype)
