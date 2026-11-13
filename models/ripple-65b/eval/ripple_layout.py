"""Ripple-65B Preview (#39): the campaign's candidate layouts, pins and the rules every script shares.

Pure Python (stdlib only), so the build, the validator, the gate, the window's checks and the tests read one source.
Candidates (posted pre-registration, ModelOpt 0.48.0):
  A  MLP gate/up/down W4A4 NVFP4, attention q/k/v/o/g FP8 W8A8 (static per-tensor), lm_head W4A16 NVFP4
  C  MLP W4A4 NVFP4, attention BF16, lm_head W4A16 NVFP4                      (only if A fails a bar)
  B  MLP W4A16 NVFP4, attention FP8 W8A8, lm_head W4A16 NVFP4                 (only if C also fails)
Embeddings, norms and value_res_lambda stay BF16, byte-identical to upstream; KV cache BF16.
"""

from __future__ import annotations

import fnmatch
import re

SOURCE_REPO = "ricursiveAI/Ripple-65B-Preview"
SOURCE_REV = "240d5966cbe00f0627c699e59ef58560b60303cf"
VENDOR_REPO = "ricursiveAI/Ripple-65B-Preview-NVFP4"
VENDOR_REV = "99eea4477a001a41b86914e41feb4b2ed86105e5"
DRAFTER_REPO = "ricursiveAI/Ripple-65B-Preview-DFlash2"
DRAFTER_REV = "35609b3be3dd08a822b3ba1257c1ea5dd6f87632"
NUM_LAYERS = 80

RECIPE_DIR = "configs/modelopt/recipes/ripple-65b"
RECIPES = {
    "A": ("a_nvfp4_w4a4_mlp-fp8_attn-kv_bf16_local_hessian_layerwise.yaml",
          "624185c8df017e0130979905a36885bd761131baae6d6535472fa9d3515f9e4e"),
    "B": ("b_w4a16_nvfp4_mlp-fp8_attn-kv_bf16_local_hessian_layerwise.yaml",
          "ceafd52fdc19eeab65d4fcb8649406767107c0ddcc34201b549566baa02d302a"),
    "C": ("c_nvfp4_w4a4_mlp-bf16_attn-kv_bf16_local_hessian_layerwise.yaml",
          "ae77800f164a123d5fbea4697e378a134de4fa9f7b366135c86131d38e296f32"),
}
LM_HEAD_RECIPE = ("lm_head_w4a16_nvfp4_mse.yaml", "123e4e99af9634a16d5b6c467e8341ad6c0e72b93038f94fc2181b79f61e74f6")

# Remote code, chat template and vocabulary files pinned on the issue (byte-identical in the vendor NVFP4 repo), in
# sha256sum format.
SOURCE_FILE_SUMS = """\
f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3  modeling_ricursive.py
002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6  configuration_ricursive.py
877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3  chat_template.jinja
73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca  tokenizer.json
"""
SOURCE_FILE_PINS = {line.split()[1]: line.split()[0] for line in SOURCE_FILE_SUMS.splitlines()}
# Files the export must carry byte-identical from the source (the exporter writes config.json and the quant config).
COPIED_FILES = ("modeling_ricursive.py", "configuration_ricursive.py", "chat_template.jinja", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "LICENSE")

CALIB_SHA256 = "403730e417b94a5e412144ee72b47975c3996c61abf8a32ac7580cf0c73ca508"
CALIB_ROWS = 512
CALIB_TOKENS = 927_103
CALIB_ROW_LIST_DIGEST = "4c8c67af0c16d2810c945f86ff3c373396d1d6f3352131b138cc4ed376b806d5"
HELDOUT_SHA256 = "c258051a0a8fa7bb281fc54c55452b68f8e5e0b8eb904505bb59cc0c307b3adc"
SOURCES_IN_FILE_ORDER = ("cnn_dailymail", "nemotron_if_chat_reasoning_on", "swe_smith_tool", "hermes_function_calling")

KNOWN_ANSWERS = (("The capital of France is", " Paris"), ("The chemical symbol for gold is", " Au"))

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
# Architecture fields of config.json that the export must carry unchanged (posted artifact check).
ARCH_FIELDS = ("architectures", "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
               "num_attention_heads", "num_key_value_heads", "head_dim", "layer_types", "hybrid_layer_pattern",
               "sliding_window", "embed_scale", "use_valres", "ffn_clamp_threshold", "use_gated_attention",
               "gate_full_precision", "is_hybrid_swa", "rms_norm_eps", "rope_theta", "vocab_size",
               "tie_word_embeddings", "max_position_embeddings", "auto_map", "bos_token_id", "eos_token_id",
               "pad_token_id")

# Per candidate: format of the attention linears, the MLP linears and lm_head (None = BF16, not quantized).
PLANS = {
    "A": {"attn": "FP8", "mlp": "NVFP4", "lm_head": "W4A16_NVFP4"},
    "B": {"attn": "FP8", "mlp": "W4A16_NVFP4", "lm_head": "W4A16_NVFP4"},
    "C": {"attn": None, "mlp": "NVFP4", "lm_head": "W4A16_NVFP4"},
}
PRODUCT_IDS = {
    "A": "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8",
    "B": "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8",
    "C": "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4",
}
# Enabled quantizers after stage 1 + stage 2 (posted dry-run counts).
QUANTIZER_TOTALS = {"A": 1281, "B": 1041, "C": 481}
# Fused runtime groups (vLLM ricursive.py): qkg_proj = q/k/g, gate_up_proj = gate/up. v_proj, o_proj, down_proj alone.
FUSED_GROUPS = (("self_attn", ("q_proj", "k_proj", "g_proj")), ("mlp", ("gate_proj", "up_proj")))


def check_letter(letter: str) -> str:
    if letter not in PLANS:
        raise ValueError(f"candidate letter must be one of {sorted(PLANS)}, got {letter!r}")
    return letter


def quantized_layers(letter: str, num_layers: int = NUM_LAYERS) -> dict[str, dict]:
    """The quantized_layers map the export must declare (config.json and hf_quant_config.json)."""
    plan = PLANS[check_letter(letter)]
    out: dict[str, dict] = {}

    def entry(fmt: str) -> dict:
        return {"quant_algo": fmt} if fmt == "FP8" else {"quant_algo": fmt, "group_size": 16}

    for i in range(num_layers):
        if plan["attn"]:
            for p in ATTN:
                out[f"model.layers.{i}.self_attn.{p}"] = entry(plan["attn"])
        for p in MLP:
            out[f"model.layers.{i}.mlp.{p}"] = entry(plan["mlp"])
    out["lm_head"] = entry(plan["lm_head"])
    return out


def expected_algo_counts(letter: str, num_layers: int = NUM_LAYERS) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in quantized_layers(letter, num_layers).values():
        counts[v["quant_algo"]] = counts.get(v["quant_algo"], 0) + 1
    return counts


def expected_quantizer_total(letter: str, num_layers: int = NUM_LAYERS) -> int:
    """Weight quantizers for every quantized linear, plus input quantizers for W8A8 and W4A4 (not W4A16)."""
    total = 0
    for v in quantized_layers(letter, num_layers).values():
        total += 1 if v["quant_algo"] == "W4A16_NVFP4" else 2
    return total


def linear_of(tensor_name: str) -> str | None:
    """'model.layers.3.mlp.up_proj.weight_scale' -> 'model.layers.3.mlp.up_proj'; lm_head.* -> 'lm_head'."""
    m = re.match(r"^(model\.layers\.\d+\.(?:self_attn|mlp)\.[a-z_]+_proj|lm_head)\.", tensor_name)
    return m.group(1) if m else None


def calib_take(rows: list[dict], n: int) -> list[dict]:
    """Calibration rows for an n-row build: the full file (n = 512) in file order, or for a smoke the first
    ceil(n / 4) rows of each source in file order (8 rows: 2 per source), kept in file order."""
    if n >= len(rows):
        return list(rows)
    if n <= 0:
        raise ValueError("rows must be positive")
    per = -(-n // len(SOURCES_IN_FILE_ORDER))
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        k = r["source"]
        if seen.get(k, 0) < per:
            out.append(r)
            seen[k] = seen.get(k, 0) + 1
    return out[:n] if len(out) > n else out


# ---- vLLM nightly (f5d93296) modelopt_mixed resolution, re-implemented for offline checks --------------------------
VLLM_PACKED = {"qkg_proj": ("q_proj", "k_proj", "g_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
VLLM_PREFIXES = tuple(f"model.layers.{{i}}.{s}" for s in ("self_attn.qkg_proj", "self_attn.v_proj",
                                                            "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"))


def vllm_resolve(quantized: dict[str, dict], excluded: list[str], prefix: str) -> str | None:
    """Algorithm vLLM's modelopt_mixed config gives a runtime layer: exact entry, else the packed shards' common
    entry (mixed shards raise in vLLM), else None (unquantized). Exclusions: exact, substring or fnmatch."""
    for pat in excluded:
        if prefix == pat or pat in prefix or fnmatch.fnmatch(prefix, pat):
            return None
    if prefix in quantized:
        return quantized[prefix]["quant_algo"]
    head, _, last = prefix.rpartition(".")
    if last in VLLM_PACKED:
        algos = {quantized.get(f"{head}.{s}", {}).get("quant_algo") for s in VLLM_PACKED[last]}
        if len(algos) > 1:
            raise ValueError(f"{prefix}: mixed algorithms inside a fused layer: {sorted(map(str, algos))}")
        return algos.pop()
    return None


def runtime_plan(letter: str, num_layers: int = NUM_LAYERS) -> dict[str, str | None]:
    plan = PLANS[check_letter(letter)]
    out: dict[str, str | None] = {}
    for i in range(num_layers):
        for s in ("self_attn.qkg_proj", "self_attn.v_proj", "self_attn.o_proj"):
            out[f"model.layers.{i}.{s}"] = plan["attn"]
        for s in ("mlp.gate_up_proj", "mlp.down_proj"):
            out[f"model.layers.{i}.{s}"] = plan["mlp"]
    out["lm_head"] = plan["lm_head"]
    out["model.embed_tokens"] = None
    return out
