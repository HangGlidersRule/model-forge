# Artifact lineage and build record

## Source checkpoint

- Model: `Qwen/Qwen3.8-27B`
- Revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Precision: BF16
- Architecture: Qwen3.8 multimodal / Qwen3.5-family text architecture
- Native MTP inventory: 15 tensors under `mtp.*`
- Vision inventory observed in the source index: 333 tensors

This official BF16 checkpoint is an external upstream reference. It is not branded Darkstar because
HangGlidersRule did not produce it.

## Upstream Base BF16 reference (clean)

The Base BF16 cell evaluates the unchanged upstream checkpoint (no refusal-direction edit, no
quantization). It is not a Darkstar-owned artifact or publication target. Its evaluation profile is
complete: pinned recipe
`recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-bf16.yaml`, frozen full-denominator GPQA `157/198 =
79.29%` (0 timeout/parse/error), and an independent single-stream throughput winner of `130.158 tok/s`
at MTP8. It is the reference against which the quantization and edit deltas are computed.

## R3 refusal-direction measurement

- Measurement layer: `38`
- Hidden dimension: `5120`
- Seed: `42`
- Harmful corpus: `mlabonne/harmful_behaviors@01cead01398926d81f7c52bdb790ee8cf77ebba7`
- Harmless corpus: `mlabonne/harmless_alpaca@02c6a92cfcf11bb0c387334f8146d149d65b587f`
- Samples: 128 harmful + 128 harmless after deterministic normalization/deduplication
- Prompt formatting: model chat template, generation prompt enabled, thinking disabled
- Activation: final-token residual at language-model layer 38
- Direction: normalized float32 difference between harmful and harmless means

The massive-activation masking diagnostic used the documented criteria `abs(x) > 100` and `abs(x) > 1000 * median(abs(row))`. It found `0/5120` dimensions for this measurement, so the masked and unmasked directions were identical.

## Projection targets

Exactly 131 residual-writing tensors were edited:

- 16 self-attention output projections
- 48 linear-attention/GDN output projections
- 64 MLP down projections
- 1 token embedding matrix
- 1 MTP self-attention output projection
- 1 MTP MLP down projection

The vision tower was not targeted. Edits were applied in float32 and cast back to source dtype. Structural validation retained all 333 vision tensors and all 15 MTP tensors. Maximum normalized residual leakage was approximately `0.000195`.

## Darkstar Abliterated BF16 artifact (R3)

The Darkstar Abliterated BF16 checkpoint is the direct projected derivative of the pinned official
source. R3 remains its internal edit-lineage id. It is the editable source artifact for future
quantization experiments.

Required publication validation:

- exact file manifest and SHA-256 checksums;
- tokenizer, processor, chat template, and generation config preserved;
- 131 changed tensors accounted for;
- non-target tensors byte-identical where applicable;
- 333 vision tensors present;
- 15 BF16 MTP tensors present;
- model boots in compiled vLLM and passes text, vision, tools, JSON, and long-context gates.

## Darkstar Abliterated NVFP4 artifact (R3)

### Rejected historical path (do not republish)

The first R3 NVFP4 build used llm-compressor compressed-tensors NVFP4 W4A4
(`neuralmagic/calibration`, 32×8192, basic pipeline) with a post-export MTP graft.
Those compressed-tensors artifacts are **rejected/historical**. Keep them for lineage
only; never overwrite their directories or publish them as the current release.

### Active rebuild path (NVIDIA ModelOpt)

Publication NVFP4 is rebuilt with pinned ModelOpt `0.46.0rc2`
(`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`) using the **selected** clean-base recipe:

- Selected recipe: `configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml`
  (mixed precision; runtime/export KV stays BF16).
- W4A16 NVFP4 group 16 on language MLP gate/up/down **and `lm_head`**; FP8 (e4m3) on
  self-attention q/k/v/o and the GatedDeltaNet projections.
- BF16: vision, MTP, `conv1d`, norms, embeddings, and runtime KV.
- Calibration: deterministic `cnn_dailymail` + `nemotron-post-training-dataset-v2`,
  512+512 samples, batch 1, sequence 2048, seed 1234, layerwise false.
- Export: ModelOpt `examples/hf_ptq/hf_ptq.py` → unified HF `hf_quant_config.json`.
- Preserve/reattach all 15 BF16 MTP tensors via ModelOpt's MTP path.
- Darkstar Abliterated NVFP4 (internal edit lineage R3) is the **selected/promoted, locally complete**
  mixed build at
  `${PUBLIC_ARTIFACT_PATH}`.
- `_SUCCESS.json` SHA-256:
  `3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79`.
- `manifest.sha256` SHA-256:
  `642dbbe89b085a2daf5119c37c0496576a475ed64c36653fc993c04abaf2ca9f`.
- Recipe SHA-256:
  `90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642`.
- Runtime snapshots: inspect
  `f1e94a763d5bb71de0a9991b2c1211db6ae6c3294dbf8a03b38f53af95235222`, logs
  `4ffe8666015e4c4f4f45cd31c84e4a2e056a0575dfe3aa45e32e0011012d2582`, operator snapshot Compose
  `85ba68155418dad7387219f62889def88c62a0e2ca35d15e3f83d62879077088`.
- Tracked serve Compose (this repository's deterministic rendering of the same frozen profile,
  [`containers/serve/darkstar-qwen38-abliterated-nvfp4.yml`](../../containers/serve/darkstar-qwen38-abliterated-nvfp4.yml))
  SHA-256: `5434c2a99bdadce512bd87b65c30f830c21fc2eae647182ffa89e77b174833cc`. It is a different file
  from the operator snapshot above and its digest is never presented as that snapshot's.

Required publication validation:

- ModelOpt provenance (commit, wheel hash, recipe hash, calibration contract);
- no FP8 KV metadata; no NaN/Inf/zero/empty scales;
- no quantized vision; all 15 MTP tensors BF16;
- no mixed precision in fused q/k/v, gate/up, linear-attn qkv+z, a+b groups;
- tokenizer/processor/chat template unchanged except quantization metadata;
- full SHA manifests; GPQA 198/198 full-denominator.

## Current serving result

The selected Base ModelOpt mixed W4A16-NVFP4+FP8 product's frozen serve profile:

- vLLM `0.27.1`
- compiled mode
- Flash Attention
- BF16 KV cache
- native MTP depth 4 (single-stream winner: 203.636 tok/s)
- context length 126,144
- `max_num_seqs=16`
- `max_num_batched_tokens=32768`
- prefix caching with xxhash
- chunked prefill
- multimodal limits: 4 images, 2 videos per prompt

Each BF16 product has its own frozen single-stream profile (Base BF16: MTP8, 64K budget, 130.158 tok/s;
Abliterated BF16: MTP11, 16K budget, 144.502 tok/s).

The selected Abliterated ModelOpt mixed W4A16-NVFP4+FP8 product freezes vLLM `0.27.1`, compiled mode,
FlashAttention, BF16 KV, context 126144, MTP10, scheduler budget 32768, `max_num_seqs=16`, prefix
caching, and chunked prefill. Its MTP1–12 sweep had a headline peak at MTP8 (`251.316 tok/s`), but the
nonmonotonic confirmation `10->8->8->10` selected MTP10 by mean throughput
(`251.889` vs MTP8 `250.862`). Runtime alias:
`darkstar-qwen38-abliterated-nvfp4`; container:
`vllm-darkstar-qwen38-abliterated-modelopt`.
