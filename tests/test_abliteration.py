"""Tests for abliteration math with tiny synthetic tensors (requires torch)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_forge.abliteration import (  # noqa: E402
    compute_leakage,
    compute_refusal_direction,
    mask_massive_activations,
    project_weight,
)


@pytest.fixture
def direction_4d() -> "torch.Tensor":
    """A known unit-norm direction in 4D."""
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    return r / r.norm()


class TestComputeRefusalDirection:
    def test_direction_is_unit_norm(self) -> None:
        harmful = torch.randn(10, 64)
        harmless = torch.randn(10, 64)
        result = compute_refusal_direction(harmful, harmless)
        assert abs(result.direction.norm().item() - 1.0) < 1e-6

    def test_direction_sign_orientation(self) -> None:
        """Direction should point from harmless mean toward harmful mean."""
        harmful = torch.ones(5, 8) * 2.0
        harmless = torch.zeros(5, 8)
        result = compute_refusal_direction(harmful, harmless)
        assert (result.direction > 0).all()

    def test_deterministic(self) -> None:
        torch.manual_seed(99)
        harmful = torch.randn(20, 32)
        harmless = torch.randn(20, 32)
        r1 = compute_refusal_direction(harmful, harmless)
        r2 = compute_refusal_direction(harmful, harmless)
        assert torch.allclose(r1.direction, r2.direction)

    def test_orthogonalize_harmless(self) -> None:
        harmful = torch.tensor([[3.0, 1.0]])
        harmless = torch.tensor([[1.0, 0.0]])
        result = compute_refusal_direction(
            harmful, harmless, orthogonalize_harmless=True
        )
        harmless_unit = harmless.mean(0) / harmless.mean(0).norm()
        proj = (result.direction @ harmless_unit).abs().item()
        assert proj < 1e-6

    def test_records_metadata(self) -> None:
        harmful = torch.randn(5, 16)
        harmless = torch.randn(5, 16)
        result = compute_refusal_direction(harmful, harmless, layer=38, seed=7)
        assert result.layer == 38
        assert result.seed == 7
        assert result.raw_diff_norm > 0


def test_mask_massive_activations_masks_dimension_globally() -> None:
    acts = torch.tensor([[1.0, 2.0, 10000.0], [2.0, 3.0, 4.0]])
    masked, dimensions = mask_massive_activations(acts)
    assert dimensions.tolist() == [False, False, True]
    assert masked[:, 2].tolist() == [0.0, 0.0]
    assert torch.equal(masked[:, :2], acts[:, :2])


class TestProjectWeight:
    def test_row_projection_removes_direction(self, direction_4d: "torch.Tensor") -> None:
        """After projection, weight rows should have zero component along direction."""
        w = torch.randn(4, 8)
        w_proj = project_weight(w, direction_4d)
        leakage = (direction_4d @ w_proj.float()).norm().item()
        assert leakage < 1e-5

    def test_embed_projection_removes_direction(self, direction_4d: "torch.Tensor") -> None:
        """embed_tokens: [vocab, hidden]. After projection, W@r should be zero."""
        w = torch.randn(10, 4)
        w_proj = project_weight(w, direction_4d)
        leakage = (w_proj.float() @ direction_4d).norm().item()
        assert leakage < 1e-5

    def test_preserves_dtype(self) -> None:
        w = torch.randn(8, 16).to(torch.bfloat16)
        r = torch.randn(8)
        r = r / r.norm()
        result = project_weight(w, r)
        assert result.dtype == torch.bfloat16

    def test_unchanged_orthogonal_weight(self, direction_4d: "torch.Tensor") -> None:
        """Weight orthogonal to direction should be unchanged."""
        w = torch.zeros(4, 3)
        w[1, 0] = 1.0
        w[2, 1] = 2.0
        w_proj = project_weight(w, direction_4d)
        assert torch.allclose(w, w_proj, atol=1e-6)

    def test_incompatible_shapes_raises(self) -> None:
        w = torch.randn(5, 7)
        r = torch.randn(3)
        r = r / r.norm()
        with pytest.raises(ValueError, match="incompatible"):
            project_weight(w, r)


class TestComputeLeakage:
    def test_zero_leakage_after_projection(self, direction_4d: "torch.Tensor") -> None:
        w = torch.randn(4, 6)
        w_proj = project_weight(w, direction_4d)
        leakage = compute_leakage(w_proj, direction_4d)
        assert leakage < 1e-5

    def test_nonzero_leakage_before_projection(self) -> None:
        r = torch.tensor([1.0, 0.0, 0.0, 0.0])
        w = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        leakage = compute_leakage(w, r)
        assert leakage > 0.1
