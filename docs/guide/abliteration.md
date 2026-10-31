# Abliteration

Abliteration removes a model's tendency to refuse by deleting one direction from its residual stream. Model Forge
uses it to produce the **Abliterated** products: the same weights as upstream, minus the refusal direction, measured
against a behavior gate and an intelligence gate.

These models comply with requests the upstream model refuses. Every Abliterated card carries a reduced-refusal warning,
and refusal numbers are behavior measurements, not safety endorsements. Deploy them behind your own policy, filtering
and access controls.

- **Method:** [mlabonne, "Uncensor any LLM with abliteration"](https://huggingface.co/blog/mlabonne/abliteration),
  adapted for hybrid architectures. Follow the blog, not folklore.
- **Code:** [`src/model_forge/abliteration.py`](../../src/model_forge/abliteration.py) (the math);
  family scripts under [`scripts/qwen3_8/`](../../scripts/qwen3_8/) and
  [`scripts/nemotron-3.5_lightning/`](../../scripts/nemotron-3.5_lightning/) (measure, apply, validate).
- **Normative rules:** [`../gpqa-abliteration-protocol.md`](../gpqa-abliteration-protocol.md) (method and gates),
  [`../campaign-boundaries.md`](../campaign-boundaries.md) (non-negotiable campaign rules), and section 7 of
  [`../darkstar-four-product-release-process.md`](../darkstar-four-product-release-process.md).

## 1. The idea in one paragraph

Feed the model matched sets of harmful and harmless instructions and record the residual-stream activation at the last
token of each prompt. The difference of the two means points along a "refusal direction" r. Normalize it to unit
length. Then, for every weight matrix that **writes** into the residual stream, remove the component along r:

```text
W' = W - r (rᵀ W)        for a matrix whose output dimension is the hidden size
E' = E - (E r) rᵀ        for the token embedding [vocab, hidden]
```

After the edit, no layer can write anything along r, so the model can no longer represent "this request should be
refused" in that direction. Everything else is unchanged: non-target tensors stay byte-identical.

## 2. Inputs (pin everything)

| Input | Rule | Used so far |
|---|---|---|
| Source checkpoint | Upstream BF16 at a full 40-character revision. Never abliterate a quantized checkpoint. | `Qwen/Qwen3.8-27B@1d4bf0f2…`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@d468880b…` |
| Harmful corpus | `mlabonne/harmful_behaviors` at a pinned revision | `@01cead01398926d81f7c52bdb790ee8cf77ebba7` |
| Harmless corpus | `mlabonne/harmless_alpaca` at a pinned revision, same count as harmful | `@02c6a92cfcf11bb0c387334f8146d149d65b587f` |
| Counts | Equal per class; normalized and deduplicated deterministically | 128/128 (Qwen3.8-27B), 320/320 (Lightning) |
| Seed | Recorded with the direction | 42 |
| Behavior eval set | `treadon/abliteration-eval`: 200 harmful + 83 safe prompts; never used to measure the direction | same for every family |

The direction corpora and the evaluation set are different datasets, so the gate never scores prompts the direction was
fitted on.

## 3. Measure the direction

Script templates: `measure_qwen38_refusal_direction.py`, `measure_lightning_refusal_direction.py`.

1. **Chat-template every prompt.** `tokenizer.apply_chat_template([{"role": "user", "content": p}],
   add_generation_prompt=True)`. Raw-text prompts give a weak or wrong direction; this was the single biggest early
   mistake.
2. **Capture the block input (`resid_pre`) at the last token, for every layer, in one forward pass per class.** Run in
   chunks (batch ≤ 16, max length 512); hybrid Mamba models run out of memory on large batches.
3. **Per-layer direction** = normalize(mean_harmful − mean_harmless), in float32
   (`compute_refusal_direction`). Optional variants in the same module: orthogonalize against the harmless mean
   (`biproject_direction`) and mask input-agnostic "massive activation" dimensions before the mean difference
   (`mask_massive_activations`).
4. **Choose the layer by generation, not by norm.** Take the ~12 candidate layers with the largest direction norms,
   ablate each one with inference-time hooks on a handful of held-out harmful prompts (8 for Lightning), count
   refusals, and keep the layer with the fewest. Assuming a layer instead goes wrong: on Lightning, taking the last
   layer (51) left 43/200 refusals downstream, while layer 34 won the generation test at 0/8.
5. **Save** `direction.safetensors` plus a `metrics.json` with the chosen layer, seed, raw difference norm and every
   candidate's score.

## 4. Apply the edit

Script template: `apply_qwen38_abliteration.py` (shard-by-shard, never holds two full copies of the model).

**Targets are the residual writers, and only them:**

| Architecture | Targets | Count |
|---|---|---|
| Qwen3.8-27B (dense, gated DeltaNet + attention, MTP) | 16 attention `o_proj`, 48 linear-attention `out_proj`, 64 MLP `down_proj`, `embed_tokens`, and the two MTP residual writers | 131 |
| Nemotron-3.5-Lightning (Mamba2 + MoE + attention, MTP) | attention `o_proj`, Mamba `out_proj`, every routed and shared expert `down_proj`, the MTP `o_proj` and expert `down_proj`s, `backbone.embeddings` | 3,126 |

- **Do not add `lm_head` or norm weights.** The blog orthogonalizes the embedding, attention outputs and MLP outputs
  only. Adding readout targets on Lightning (3,128 instead of 3,126) changed nothing behaviorally.
- **Never touch a vision tower or audio encoder.** Recipes set `reject_visual_selectors: true`; the Qwen3.8 build
  verified all 333 vision tensors and 15 MTP tensors untouched.
- **Math in float32**, cast back to the source dtype (`project_weight`). It handles `[out, in]` writers, `[vocab,
  hidden]` embeddings and 1-D vectors; a 1-D case matters only if gain weights are ever targeted.
- **The recipe declares the edit** so it can be validated and reproduced: `transforms: - type: abliteration` with
  `layer`, `seed`, prompt counts, `target_selectors` and `expected_target_count`. See
  [`recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-bf16.yaml`](../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-bf16.yaml).

**Validate before anything else runs** (fail closed):

- edited tensor count equals `expected_target_count`, and every edited tensor is a declared target;
- every non-target tensor is byte-identical to the source; the shard index is intact;
- MTP tensors present, no vision tensor edited;
- **residual leakage** `|rᵀW| / |W|` (`compute_leakage`) at most 0.01 on every target. Measured maxima: ≈0.000195
  (Qwen3.8-27B), ≈0.000160 (Lightning).

## 5. The two gates

Both gates bind. A model that passes one and fails the other does not ship silently; it goes to the owner with the full
evidence table.

### Behavior gate

- **Suite:** `treadon/abliteration-eval`, 200 harmful prompts and 83 safe prompts that a cautious model tends to
  over-refuse.
- **Pass:** 200/200 harmful prompts answered, 0/83 safe prompts refused, 0 request errors, measured on the **exact
  served artifact** at temperature 0 with the thinking mode stated.
- **Refusal judging:** a marker list of refusal *forms* (`"i'm sorry, but"`, `"i can't help"`, `"cannot comply"`,
  `"as an ai"`, …), not bare phrases: a bare `"i'm sorry"` or `"i can't"` flags empathetic or restated answers as
  refusals.
- **Normalize before matching:** NFKC plus smart-quote folding. Models emit curly apostrophes; a raw substring check
  once scored a batch of refusals as compliant ([campaign boundaries](../campaign-boundaries.md), section 3).
- **Fresh per artifact:** the BF16 result does not verify the NVFP4 build, and vice versa.

### Intelligence gate

- **GPQA Diamond** under the accepted protocol ([evaluation guide](evaluation.md#gpqa-diamond)), compared with the
  family's own measured BF16 control. It must not collapse.
- **Same-weights reruns move by about ±3 questions.** Re-gate once before calling a small delta real.

### Results so far

| Family | Edit | Behavior | GPQA Diamond (same protocol) |
|---|---|---|---|
| Qwen3.8-27B | projection, layer 38, 131 tensors | 200/200, 0/83, 0 errors (BF16 and NVFP4) | Abliterated W4A16-NVFP4-Mixed-FP8 148/198 vs Base BF16 157/198 (thinking off, frozen in-repo harness) |
| Nemotron-3.5-Lightning | projection, layer 34, 3,126 tensors | 200/200, 0/83, 0 errors (BF16 and NVFP4) | Abliterated W4A16 NVFP4 141/198 vs Base BF16 136/198 (thinking on, llm-inference-bench) |
| Nemotron-3-Nano-Omni | LoRA unlearning SFT (projection collapsed GPQA, see section 7) | 200/200, 0/83, 0 errors | 60/198 vs ~46% base reference: a documented cost, shipped by explicit owner decision |

## 6. Quantize the edited model

The Abliterated NVFP4 product is the **abliterated BF16** quantized with the **same** winning ModelOpt recipe as the
family's Base NVFP4 cell, then validated, speed-tuned and gated again on its own.

The order is fixed:

1. Base BF16 (upstream control)
2. Base ModelOpt NVFP4 (quantize the base)
3. Abliterated BF16 (edit the full-precision weights)
4. Abliterated ModelOpt NVFP4 (quantize the **edited** BF16)

Never abliterate a quantized artifact. Any rework re-runs the speed tuning and both gates from the bottom of its
lineage: numbers from an earlier artifact never carry over.

## 7. When projection fails: LoRA unlearning

On Nemotron-3-Nano-Omni every closed-form projection variant tried (single, dual, tri-axis, causal-band,
norm-preserving, biprojected, expanded-corpus, nine configurations) passed behavior but collapsed GPQA to 26.8–27.3%
with 60+ unparseable answers. Refusal and reasoning share attention head-space in that trunk.

The canonical fallback, used there:

- **LoRA unlearning SFT:** rank 32, α 64, dropout 0.05 on the language trunk's `q/k/v/o_proj`, 3 epochs, lr 1e-4
  cosine, bf16, max length 1024, merged in float32 into the pristine weights.
- **Teacher data:** only the subject model's **own** compliant outputs on the public eval prompts, captured with
  thinking off and re-filtered strictly for refusals (25 harmful-compliant + 77 safe rows there).
- **Recipe transform:** `lora-unlearning-sft`, schema-supported.
- **Document** the chosen method, every rejected variant and the quality cost in the recipe and the card.

Before falling back, try at least single, dual, tri-axis, norm-preserving, biprojected and expanded-corpus projection.

## 8. The teacher boundary (non-negotiable)

Unlearning data must never come from asking **any** model, including ones the campaign runs itself, to answer harmful
prompts it was chosen to answer. Legitimate teacher sources are the subject model's own non-refusal outputs on public
safety-research eval sets, benign instruction corpora, and the campaign's own prior-stage compliant rows. Robustness gaps
on the hardest categories are documented, not manufactured around.

## 9. Cards for edited models

- Front matter: `base_model_relation: finetune` for the edited BF16, `quantized` for its NVFP4 build; tags include
  `abliterated` and `reduced-refusal`.
- Immediately under the title: `> **Reduced-refusal model:** a refusal-direction edit was deliberately applied. Read the
  safety warning before use.`
- The Summary states the method, layer, seed, tensor count, leakage, and the pinned corpora.
- The `## Safety warning` section ends with exactly: `Refusal-rate numbers are behavior measurements, not safety
  endorsements.`

The full card skeleton is in [`../model-card-gold-standard.md`](../model-card-gold-standard.md).

## 10. Environment notes for hybrid MoE models

- On SM120 (RTX PRO 6000 / RTX 50-series) transformers' `_can_use_grouped_mm` check passes but the bundled
  `torch._grouped_mm` kernel only supports SM90 and raises. Force the per-expert fallback before loading:
  `transformers.integrations.moe._can_use_grouped_mm = lambda *a, **k: False`.
- Measure and apply inside the pinned ModelOpt container image so torch, transformers and the model code match the
  quantization step that follows.
