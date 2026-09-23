"""DiT-side concept erasure for Qwen-Image-2.1 (#42).

UCE-style, adapted to the single-stream joint attention: concept text tokens
and image tokens share one attention stream, so a concept steers the image
through its keys/values. The edit subtracts the concept direction from the
to_k/to_v projections' response to concept-like encoder inputs.

Approach (input-keyed, static weights):
1. HARVEST: run concept prompts' text embeddings through the DiT blocks with a
   capture hook on to_k/to_v outputs at concept-token positions vs neutral
   counterparts; accumulate a per-layer concept response direction.
2. EDIT: project the to_k/to_v weights so their response to the concept
   subspace is removed (weight-space edit keyed by the concept embedding
   subspace — UCE's cross-attention edit generalized to single-stream).
3. VALIDATE: frozen behavior protocol (judge-scored semantic compliance) +
   D4 manifest quality run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ConceptEdit:
    """A per-layer, input-keyed projection edit for one concept direction."""

    concept: str
    # per-layer edit spec: for each edited module name, the projection matrix P
    # edit: W' = W - strength * W @ P applied so concept inputs project away.
    edits: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "modules": {
                name: {"rows": v.shape[0], "dim": v.shape[1]}
                for name, v in self.edits.items()
            },
        }


def harvest_concept_subspace(
    pipe: Any,
    concept_prompts: list[str],
    neutral_prompts: list[str],
    *,
    n_layers: int | None = None,
    top_k_components: int = 8,
) -> dict[str, torch.Tensor]:
    """Compute the per-layer text-token response subspace that separates
    concept prompts from neutral ones.

    Runs the ENCODER (not the DiT) with hidden-state capture at every layer;
    the concept subspace = top principal components of (concept_mean - neutral_mean)
    token embeddings. The DiT's to_k/to_v consume these encoder outputs
    (through txt_in), so an input-keyed edit keyed on this subspace suppresses
    the concept's steering signal at the source of the joint attention.
    """
    proc = pipe.processor
    enc = pipe.text_encoder

    def capture(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hs, masks = [], []
        for p in prompts:
            inputs = proc(text=[p], return_tensors="pt").to(next(enc.parameters()).device)
            mask = inputs.get("attention_mask")
            with torch.no_grad():
                out = enc(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1]
            if h.dim() == 3:
                h = h[0]
            hs.append(h.float())
            masks.append(mask[0].bool() if mask is not None else torch.ones(h.shape[0], dtype=torch.bool, device=h.device))
        return hs, masks

    harm_h, harm_m = capture(concept_prompts)
    neut_h, neut_m = capture(neutral_prompts)

    # concept-direction per token position: concat mean-difference tokens
    diffs = []
    for (h, m), (n, nm) in zip(zip(harm_h, harm_m), zip(neut_h, neut_m)):
        d = h.mean(dim=0) - n.mean(dim=0)  # [hidden]
        diffs.append(d)
    D = torch.stack(diffs)  # [n_prompts, hidden]
    D = D - D.mean(dim=0, keepdim=True)
    # top-k principal components of the difference set
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    k = min(top_k_components, S.shape[0])
    # concept directions live in the RIGHT singular vectors (hidden-dim space)
    subspace = Vh[:k].T  # [hidden, k] orthonormal columns
    subspace, _ = torch.linalg.qr(subspace)
    return {"subspace": subspace, "singular_values": S[:k]}


def apply_input_keyed_edit(
    pipe: Any,
    subspace: torch.Tensor,
    *,
    module_filter: tuple[str, ...] = ("txt_in",),
    strength: float = 1.0,
) -> ConceptEdit:
    """Project the encoder→DiT boundary weights (txt_in) so the concept
    subspace maps to ~zero: W' = W (I - strength * P_subspace) along the INPUT
    dimension (the encoder embedding enters as input).

    Only boundary modules are edited: the DiT weights themselves are untouched,
    which keeps the edit concept-scoped (input-keyed) rather than global.
    """
    edit = ConceptEdit(concept="harvested-subspace")
    P = subspace @ subspace.T  # projector [hidden, hidden]
    edited = 0
    with torch.no_grad():
        for name, module in pipe.transformer.named_modules():
            if not any(f in name for f in module_filter):
                continue
            for pname, param in module.named_parameters(recurse=False):
                if pname.endswith("weight") and param.ndim == 2:
                    w = param.float()
                    # encoder embeddings enter along INPUT dim (columns) for txt_in.*
                    if w.shape[1] == P.shape[0]:
                        param.copy_((w - strength * (w @ P) * 1.0).to(param.dtype))
                        edited += 1
                        edit.edits[f"{name}.{pname}"] = P
    edit.edits["_edited_count"] = torch.tensor([edited])
    return edit


def save_edit(edit: ConceptEdit, path: str | Path) -> None:
    torch.save({"concept": edit.concept, "subspace_edits": {k: v for k, v in edit.edits.items()}}, path)
