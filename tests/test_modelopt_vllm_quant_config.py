"""Regression: partial NVFP4 exports must get modelopt_mixed quantization_config.

vLLM maps hf_quant_config quant_algo=NVFP4 -> modelopt_fp4, which produces NaN
logits for partially-quantized nemotron_h artifacts (experts-only NVFP4, rest
BF16). The proven-working layout is a config.json quantization_config block
with quant_method=modelopt + MIXED_PRECISION + quantized_layers, which vLLM
maps to modelopt_mixed (the darkstar-Qwen prod path).
"""
import json

from model_forge.modelopt.write_vllm_quant_config import (
    collect_quantized_layers,
    write_quantization_config,
)


def _make_fake_export(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    keys = {
        "backbone.layers.1.mixer.experts.0.up_proj.weight": "m0001.safetensors",
        "backbone.layers.1.mixer.experts.0.up_proj.weight_scale": "m0001.safetensors",
        "backbone.layers.1.mixer.experts.0.down_proj.weight": "m0001.safetensors",
        "backbone.layers.1.mixer.shared_experts.up_proj.weight": "m0001.safetensors",
        "backbone.layers.1.mixer.shared_experts.down_proj.weight": "m0001.safetensors",
        "backbone.layers.1.mixer.q_proj.weight": "m0001.safetensors",  # BF16: NOT quantized
        "model.embed_tokens.weight": "m0001.safetensors",  # BF16: NOT quantized
        "lm_head.weight": "m0001.safetensors",  # BF16 protected
    }
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": keys})
    )
    # source BF16 config.json
    (tmp_path / "bf16").mkdir()
    (tmp_path / "bf16" / "config.json").write_text(
        json.dumps({"hidden_size": 2688, "num_hidden_layers": 52})
    )
    return export, tmp_path / "bf16"


def test_collect_quantized_layers_only_experts(tmp_path):
    export, _ = _make_fake_export(tmp_path)
    ql = collect_quantized_layers(export)
    assert set(ql) == {
        "backbone.layers.1.mixer.experts.0.up_proj",
        "backbone.layers.1.mixer.experts.0.down_proj",
        "backbone.layers.1.mixer.shared_experts.up_proj",
        "backbone.layers.1.mixer.shared_experts.down_proj",
    }, f"got {sorted(ql)}"
    # BF16-protected modules must NOT be in the quantized map
    assert not any("q_proj" in k for k in ql)
    assert not any("embed_tokens" in k for k in ql)
    assert not any("lm_head" in k for k in ql)


def test_write_quantization_config_maps_to_modelopt_mixed(tmp_path):
    export, bf16 = _make_fake_export(tmp_path)
    write_quantization_config(export, bf16)
    cfg = json.loads((export / "config.json").read_text())
    qc = cfg["quantization_config"]
    # The convertor only enters the quant_algo remap branch when there is no
    # explicit quant_method; modelopt_mixed comes from the config block shape.
    assert qc["quant_method"] == "modelopt"
    assert qc["quant_algo"] == "MIXED_PRECISION"
    assert len(qc["quantized_layers"]) == 4
    assert qc["quantized_layers"]["backbone.layers.1.mixer.shared_experts.down_proj"] == {
        "quant_algo": "W4A16_NVFP4"
    }
    # preserve structural fields from the BF16 source
    assert cfg["hidden_size"] == 2688
