"""Diffusion/DiT candidate precision validation (contract v2 extension)."""

from model_forge.release import (
    candidate_precision_errors,
    candidate_requires_mixed_fp8,
)


def _dit_candidate(**overrides):
    candidate = {
        "candidate_id": "Darkstar-QwenImage21-Base-NVFP4",
        "precision_class": "NVFP4",
        "architecture": "diffusion",
        "precision_map": {
            "dit_blocks": "NVFP4",
            "attention": "NVFP4",
            "text_encoder": "BF16",
            "vae": "BF16",
            "protected_blocks": "BF16",
        },
    }
    candidate.update(overrides)
    return candidate


def test_diffusion_nvfp4_candidate_validates_clean() -> None:
    assert candidate_precision_errors(_dit_candidate()) == []


def test_diffusion_candidate_requires_explicit_architecture() -> None:
    candidate = _dit_candidate()
    del candidate["architecture"]
    errors = candidate_precision_errors(candidate)
    assert any("architecture is not declared" in e for e in errors)


def test_diffusion_mixed_fp8_requires_fp8_and_nvfp4_weights() -> None:
    candidate = _dit_candidate(
        candidate_id="Darkstar-QwenImage21-Base-NVFP4-Mixed-FP8",
        precision_class="NVFP4-Mixed-FP8",
        precision_map={
            "dit_blocks": "NVFP4",
            "attention": "FP8",
            "text_encoder": "BF16",
            "vae": "BF16",
            "protected_blocks": "BF16",
        },
    )
    assert candidate_precision_errors(candidate) == []
    assert candidate_requires_mixed_fp8(candidate) is True


def test_diffusion_candidate_complete_map_required() -> None:
    candidate = _dit_candidate()
    del candidate["precision_map"]["vae"]
    errors = candidate_precision_errors(candidate)
    assert any("omits components ['vae']" in e for e in errors)


def test_text_candidates_unchanged_by_diffusion_extension() -> None:
    text = {
        "candidate_id": "Darkstar-Qwen38-Base-W4A16-NVFP4",
        "precision_class": "W4A16-NVFP4",
        "precision_map": {
            "language_mlp": "NVFP4",
            "lm_head": "BF16",
            "self_attention": "NVFP4",
            "gdn_projections": "BF16",
            "kv_cache": "BF16",
            "protected": "BF16",
        },
    }
    assert candidate_precision_errors(text) == []
    # a text candidate must still fail when a text component is missing
    broken = dict(text)
    broken["precision_map"] = {k: v for k, v in text["precision_map"].items() if k != "lm_head"}
    errors = candidate_precision_errors(broken)
    assert any("omits components ['lm_head']" in e for e in errors)
