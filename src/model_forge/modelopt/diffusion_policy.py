"""Diffusion/DiT quantization policy for image families.

Architecture-aware pin machinery for diffusion transformers, parallel to
``policy.py`` (text models — do not touch). Pins are derived from the ACTUAL
2.1 module tree (verified against diffusers ``transformer_qwenimage21.py`` and
the upstream ``Qwen/Qwen-Image-2.1`` safetensors index at revision
``b3179ad355be050328e483a9dfdd9e60cd62adfa``); they are NOT substitutions of
the 1.0 recipe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 2.1 DiT anatomy (source-verified, diffusers main transformer_qwenimage21.py)
# ---------------------------------------------------------------------------
#
# Per transformer block (32 blocks, single-stream, NO biases anywhere):
#   img_norm1            LayerNorm(elementwise_affine=False)  -> no weights to quantize
#   attn.to_q/to_k/to_v  nn.Linear(bias=False)                -> quantizable
#   attn.norm_q/norm_k   RMSNorm                              -> pinned BF16
#   attn.to_out.0        nn.Linear(bias=False)                -> quantizable
#   img_norm2            LayerNorm(elementwise_affine=False)  -> no weights
#   img_mlp.gate_layer   nn.Linear(bias=False) SwiGLU gate    -> quantizable
#   img_mlp.proj         nn.Linear(bias=False) SwiGLU up      -> quantizable
#   img_mlp.out          nn.Linear(bias=False) SwiGLU down    -> quantizable
#
# Top level:
#   modulation.N         shared modulation MLP (sliced per block) -> pinned BF16
#   time_text_embed.timestep_embedder.linear_{1,2}                -> pinned BF16
#   txt_in.{in_layer,out_layer}  text projection                  -> pinned BF16
#   txt_in.text_norm                                             -> norm, pinned
#   img_in                                                       -> pinned BF16
#   norm_out.linear                                              -> pinned BF16
#   proj_out                                                     -> pinned BF16

# Quantizable linear-layer glob patterns inside transformer blocks.
QUANTIZABLE_DIT_PATTERNS: tuple[str, ...] = (
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.img_mlp.gate_layer",
    "transformer_blocks.*.img_mlp.proj",
    "transformer_blocks.*.img_mlp.out",
)

# Layers that must stay BF16 for output fidelity (norms, embeddings, modulation,
# input/output projections of the residual stream).
PROTECTED_DIT_PATTERNS: tuple[str, ...] = (
    "img_in",
    "txt_in",
    "modulation",
    "time_text_embed",
    "norm_out",
    "proj_out",
    "transformer_blocks.*.img_norm1",
    "transformer_blocks.*.img_norm2",
    "transformer_blocks.*.attn.norm_q",
    "transformer_blocks.*.attn.norm_k",
)

# Diffusion candidates declare this architecture name in the ledger.
DIFFUSION_ARCHITECTURE = "diffusion"

# Precision vocabulary for diffusion candidate ids / precision_class values.
DIFFUSION_PRECISION_CLASSES: tuple[str, ...] = ("NVFP4", "FP8", "NVFP4-Mixed-FP8")


@dataclass(frozen=True)
class DiffusionPolicy:
    """A diffusion quantization policy: which DiT modules quantize, which pin."""

    name: str
    quant_format: str  # "nvfp4" | "fp8"
    block_range_exclude_first: int = 2
    block_range_exclude_last: int = 2
    quantize_mha: bool = False  # attention bmm quantization (NVFP4 activations in qkv GEMM)
    quantized_patterns: tuple[str, ...] = QUANTIZABLE_DIT_PATTERNS
    protected_patterns: tuple[str, ...] = PROTECTED_DIT_PATTERNS
    extras: dict[str, Any] = field(default_factory=dict)

    def module_quant_decision(self, module_name: str) -> str:
        """Return 'quantize', 'protected', or 'outside' for a fully-qualified module name."""
        for pattern in self.protected_patterns:
            if _glob_match(pattern, module_name):
                return "protected"
        for pattern in self.quantized_patterns:
            if _glob_match(pattern, module_name):
                return "quantize"
        return "outside"

    def coverage_report(self, module_names: list[str]) -> dict[str, list[str]]:
        """Classify every module name; exposed so validators can assert pin coverage."""
        buckets: dict[str, list[str]] = {"quantize": [], "protected": [], "outside": []}
        for name in module_names:
            buckets[self.module_quant_decision(name)].append(name)
        return buckets


def _glob_match(pattern: str, name: str) -> bool:
    """Match a dotted module glob ('transformer_blocks.*.attn.to_q') against a name.

    ``*`` matches a single path segment (block indices), unlike fnmatch.
    """
    regex = "^" + re.escape(pattern).replace(r"\.\*\.", r"\.[^.]+\.") + "$"
    return re.match(regex, name) is not None


# The 2.1 primary NVFP4 recipe: quantize interior blocks' linear layers,
# protect the first/last 2 blocks entirely (empirically re-derived for 2.1 in
# validation; the 1.0 exclusion is the starting hypothesis, not the answer).
NVFP4_PRIMARY = DiffusionPolicy(
    name="qwen-image-2.1-base-nvfp4",
    quant_format="nvfp4",
    quantize_mha=False,
)

# FP8 CI-fast recipe: whole-DiT FP8, no MHA quantization, same pins.
FP8_FAST = DiffusionPolicy(
    name="qwen-image-2.1-base-fp8",
    quant_format="fp8",
    block_range_exclude_first=0,
    block_range_exclude_last=0,
)


def assert_policy_covers_module_tree(
    policy: DiffusionPolicy, module_names: list[str]
) -> list[str]:
    """Fail closed if the policy leaves unexpected unclassified linear modules.

    Every ``nn.Linear``-shaped module inside ``transformer_blocks`` must be
    either quantized or explicitly protected. Returns a list of violations
    (empty = ok).
    """
    violations: list[str] = []
    block_linears = [
        n
        for n in module_names
        if n.startswith("transformer_blocks.")
        and not any(
            tag in n for tag in ("norm", "img_norm")
        )
    ]
    for name in block_linears:
        decision = policy.module_quant_decision(name)
        if decision == "outside":
            violations.append(f"{name}: in-block linear module not classified by policy")
    return violations
