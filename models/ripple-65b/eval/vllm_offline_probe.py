#!/usr/bin/env python3
"""Ripple-65B Preview (#39): offline (CPU) check that the pinned vLLM nightly resolves an export as planned.

Runs inside the vLLM image, no GPU. Loads the export's config the way the serve does (vLLM's get_config, no remote
code: it must be vLLM's own RicursiveConfig). Feeds the export's config.json quantization_config to vLLM's ModelOpt
detection exactly as the loader does (override_quantization_method over the ModelOpt config classes), builds the
selected config with from_config, applies the hf_to_vllm_mapper rename and packed_modules_mapping of RicursiveForCausalLM,
and asks vLLM for every runtime prefix of every layer (qkg_proj, v_proj, o_proj, gate_up_proj, down_proj), lm_head
and the embeddings: excluded or not, and the resolved algorithm. Every answer must equal the candidate's plan
(eval/ripple_layout.py runtime_plan) and the method must be modelopt_mixed.
Usage: vllm_offline_probe.py EXPORT_DIR LETTER     Prints OFFLINE-PROBE-OK or OFFLINE-PROBE-FAILED (exit 1).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ripple_layout as lay  # noqa: E402


def main() -> None:
    export, letter = Path(sys.argv[1]), lay.check_letter(sys.argv[2])
    from vllm.model_executor.layers.quantization import modelopt as mo
    from vllm.model_executor.models.ricursive import RicursiveForCausalLM

    # The config vLLM itself builds for this checkpoint, without remote code (the serve passes no --trust-remote-code):
    # model_type ricursive must resolve to vLLM's registered RicursiveConfig and keep the architecture fields.
    from vllm.transformers_utils.config import get_config

    vcfg = get_config(str(export), trust_remote_code=False)
    if type(vcfg).__name__ != "RicursiveConfig" or vcfg.num_hidden_layers != json.loads(
            (export / "config.json").read_text())["num_hidden_layers"]:
        print(f"OFFLINE-PROBE-FAILED: vLLM's config is {type(vcfg).__name__} with {vcfg.num_hidden_layers} layers")
        sys.exit(1)
    for f in ("sliding_window", "embed_scale", "use_valres", "ffn_clamp_threshold", "layer_types"):
        if getattr(vcfg, f, None) is None:
            print(f"OFFLINE-PROBE-FAILED: vLLM's config lost {f}")
            sys.exit(1)
    cfg_json = json.loads((export / "config.json").read_text())
    qc = cfg_json.get("quantization_config")
    if not qc:
        print("OFFLINE-PROBE-FAILED: config.json has no quantization_config")
        sys.exit(1)
    picks = {c.__name__: c.override_quantization_method(qc, None) for c in
             (mo.ModelOptMixedPrecisionConfig, mo.ModelOptNvFp4Config, mo.ModelOptFp8Config)}
    if picks.get("ModelOptMixedPrecisionConfig") != "modelopt_mixed":
        print(f"OFFLINE-PROBE-FAILED: vLLM picks {picks}, not modelopt_mixed")
        sys.exit(1)
    cfg = mo.ModelOptMixedPrecisionConfig.from_config(qc)
    cfg.apply_vllm_mapper(RicursiveForCausalLM.hf_to_vllm_mapper.get_rename_mapper())
    cfg.packed_modules_mapping = dict(RicursiveForCausalLM.packed_modules_mapping)
    n_layers = int(cfg_json["num_hidden_layers"])
    bad, seen = [], {}
    for prefix, want in lay.runtime_plan(letter, n_layers).items():
        try:
            excluded = cfg.is_layer_excluded(prefix)
            got = None if excluded else cfg._resolve_quant_algo(prefix)
        except Exception as e:  # noqa: BLE001 - vLLM raises on a mixed fused group; report it as a mismatch
            bad.append(f"{prefix}: {type(e).__name__}: {str(e)[:120]}")
            continue
        seen[str(got)] = seen.get(str(got), 0) + 1
        if got != want:
            bad.append(f"{prefix}: vLLM {got}, plan {want}")
    if bad:
        print(f"OFFLINE-PROBE-FAILED {letter}: {len(bad)} prefixes differ: {bad[:5]}")
        sys.exit(1)
    print(f"OFFLINE-PROBE-OK {letter} modelopt_mixed | {type(vcfg).__name__} | runtime prefixes by algorithm "
          f"{json.dumps(seen, sort_keys=True)}")


if __name__ == "__main__":
    main()
