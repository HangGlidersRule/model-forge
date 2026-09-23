"""Tests for the diffusion quantization policy and the 2.1 bridge-fork overlay."""

from model_forge.modelopt.diffusion_policy import (
    FP8_FAST,
    NVFP4_PRIMARY,
    DiffusionPolicy,
    assert_policy_covers_module_tree,
)

# Actual module names from the verified 2.1 tree (diffusers main
# transformer_qwenimage21.py + upstream safetensors index @ b3179ad).
BLOCK_LINEARS = [
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.img_mlp.gate_layer",
    "transformer_blocks.0.img_mlp.proj",
    "transformer_blocks.0.img_mlp.out",
    "transformer_blocks.17.attn.to_q",
    "transformer_blocks.31.img_mlp.out",
]
NON_LINEAR_MODULES = [
    "img_in",
    "txt_in.in_layer",
    "modulation.0",
    "time_text_embed.timestep_embedder.linear_1",
    "norm_out.linear",
    "proj_out",
    "transformer_blocks.5.img_norm1",
    "transformer_blocks.5.img_norm2",
    "transformer_blocks.5.attn.norm_q",
    "transformer_blocks.5.attn.norm_k",
]


def test_block_linears_quantize() -> None:
    for name in BLOCK_LINEARS:
        assert NVFP4_PRIMARY.module_quant_decision(name) == "quantize", name


def test_norms_embeddings_modulation_protected() -> None:
    for name in NON_LINEAR_MODULES:
        decision = NVFP4_PRIMARY.module_quant_decision(name)
        assert decision in ("protected", "outside"), name


def test_policy_covers_real_tree_without_violations() -> None:
    assert assert_policy_covers_module_tree(NVFP4_PRIMARY, BLOCK_LINEARS + NON_LINEAR_MODULES) == []


def test_unclassified_block_linear_fails_closed() -> None:
    # a hypothetical 2.1 module the policy does not know about
    tree = BLOCK_LINEARS + ["transformer_blocks.5.some_new_linear"]
    violations = assert_policy_covers_module_tree(NVFP4_PRIMARY, tree)
    assert any("some_new_linear" in v for v in violations)


def test_fp8_fast_recipe_differs_only_in_format_and_range() -> None:
    assert FP8_FAST.quant_format == "fp8"
    assert FP8_FAST.block_range_exclude_first == 0
    assert FP8_FAST.quantized_patterns == NVFP4_PRIMARY.quantized_patterns


def test_precision_vocabulary() -> None:
    assert set(DiffusionPolicy.__dataclass_params__.converters if False else __import__(
        "model_forge.modelopt.diffusion_policy", fromlist=["DIFFUSION_PRECISION_CLASSES"]
    ).DIFFUSION_PRECISION_CLASSES) == {"NVFP4", "FP8", "NVFP4-Mixed-FP8"}
