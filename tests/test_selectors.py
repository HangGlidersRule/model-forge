"""Tests for tensor name selectors (no torch dependency)."""
from __future__ import annotations

from model_forge.selectors import is_vision_tensor, matches_selector


class TestMatchesSelector:
    def test_exact_match(self) -> None:
        assert matches_selector("model.embed_tokens.weight", ["model.embed_tokens.weight"])

    def test_regex_match(self) -> None:
        sel = [r"re:model\.layers\.\d+\.self_attn\.o_proj\.weight"]
        assert matches_selector("model.layers.5.self_attn.o_proj.weight", sel)
        assert not matches_selector("model.layers.5.self_attn.q_proj.weight", sel)

    def test_no_match(self) -> None:
        assert not matches_selector("unrelated.tensor", ["model.embed_tokens.weight"])

    def test_selector_count_for_qwen38_27b(self) -> None:
        """Verify that selectors produce exactly 131 matches for Qwen3.8-27B architecture."""
        selectors = [
            r"re:model\.language_model\.layers\.\d+\.self_attn\.o_proj\.weight",
            r"re:model\.language_model\.layers\.\d+\.linear_attn\.out_proj\.weight",
            r"re:model\.language_model\.layers\.\d+\.mlp\.down_proj\.weight",
            "model.language_model.embed_tokens.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        ]
        # Qwen3.8-27B: 64 layers (16 with self_attn, 48 with linear_attn, all with mlp)
        # 1 MTP block with 1 layer (has self_attn + mlp) + shared embed_tokens
        names: list[str] = []
        for i in range(64):
            if i % 4 == 3:
                names.append(f"model.language_model.layers.{i}.self_attn.o_proj.weight")
            else:
                names.append(f"model.language_model.layers.{i}.linear_attn.out_proj.weight")
            names.append(f"model.language_model.layers.{i}.mlp.down_proj.weight")
        names.append("model.language_model.embed_tokens.weight")
        names.append("mtp.layers.0.self_attn.o_proj.weight")
        names.append("mtp.layers.0.mlp.down_proj.weight")
        matched = [n for n in names if matches_selector(n, selectors)]
        assert len(matched) == 131

    def test_selector_rejects_vision(self) -> None:
        selectors = [r"re:model\.layers\.\d+\.mlp\.down_proj\.weight"]
        assert not matches_selector("model.visual.layers.0.mlp.down_proj.weight", selectors)


class TestIsVisionTensor:
    def test_visual_detected(self) -> None:
        assert is_vision_tensor("model.visual.encoder.layers.0.weight")

    def test_non_visual(self) -> None:
        assert not is_vision_tensor("model.layers.0.mlp.down_proj.weight")

    def test_case_insensitive(self) -> None:
        assert is_vision_tensor("model.Visual.block.weight")
