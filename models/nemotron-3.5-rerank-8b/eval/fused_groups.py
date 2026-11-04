"""Fused-group scale check for a Nemotron-3.5-Rerank-8B ModelOpt export (issue #32).

vLLM fuses q/k/v into one GEMM and gate/up into another, and applies one activation scale (and, for NVFP4, one weight
global scale) to the fused weight. So within each group:
  - NVFP4 shards must share input_scale and weight_scale_2;
  - FP8 shards must share input_scale (vLLM requantizes FP8 weights to the group maximum at load, so their per-shard
    weight scales may differ);
  - every shard must use the same format.
Pure function over a scalar reader, so it is unit-tested without tensors (tests/test_nemotron_rerank_fused_groups.py)
and called by validate_export.py on the real safetensors."""
from __future__ import annotations

from typing import Callable

GROUPS = {"self_attn": ("q_proj", "k_proj", "v_proj"), "mlp": ("gate_proj", "up_proj")}
SHARED = {"nvfp4": ("input_scale", "weight_scale_2"), "fp8": ("input_scale",), "bf16": ()}


def fused_group_mismatches(read: Callable[[str], float], plan: dict[str, str], layers: int) -> dict:
    """read(tensor_name) returns the scalar value of a per-tensor scale; plan maps projection -> nvfp4 | fp8 | bf16.
    Returns the number of shared-scale groups checked, the mismatch count and the first mismatches."""
    checked, mismatches = 0, []
    for layer in range(layers):
        for sub, projs in GROUPS.items():
            fmts = {plan[p] for p in projs}
            if len(fmts) != 1:
                mismatches.append({"layer": layer, "group": sub, "scale": None,
                                   "reason": f"mixed formats {sorted(fmts)}"})
                continue
            for scale in SHARED[fmts.pop()]:
                vals = [read(f"model.language_model.layers.{layer}.{sub}.{p}.{scale}") for p in projs]
                checked += 1
                if max(vals) != min(vals):
                    mismatches.append({"layer": layer, "group": sub, "scale": scale, "values": vals})
    return {"groups_checked": checked, "mismatched": len(mismatches), "mismatches": mismatches[:8]}
