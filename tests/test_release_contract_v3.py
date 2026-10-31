"""Red tests for Darkstar release contract v3.

These lock the v3 rules before the implementation exists: KV-cache effect status, several products
per role, the task-scoped quality gate, version dispatch, fail-closed ``validate_ledger``, and
alias extraction that sees every ``--served-model-name`` value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from model_forge.release import extract_serve_model_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT / "contracts" / "darkstar-release"

# sha256 of the v1 and v2 contract files at base commit 6ee5cd4. v3 must not rewrite them.
_V1_V2_SHA256 = {
    "contracts/darkstar-release/v1/contract.json": (
        "d057d9da55fdc3c6f783ec2798c030819586876a7334bac07148f36e565269fd"
    ),
    "contracts/darkstar-release/v1/ledger.schema.json": (
        "85f209257fee6d622f931105a585a81df5c3a0f6eacfa6c3626835932dadfd67"
    ),
    "contracts/darkstar-release/v2/contract.json": (
        "2cbe7a89aaef96c4d7b2535a09bdae1fb50a3ea777aeaf08abbc0431b7109405"
    ),
    "contracts/darkstar-release/v2/ledger.schema.json": (
        "b6e3f9341bb8db9cc4a93c9e91d27efa8e237f26526f2ddcf28e8ecf6873bcc8"
    ),
}


def _nvfp4_product(*, kv_cache: str, gate_status: str) -> dict:
    return {
        "product_id": "Darkstar-Example-Base-ModelOpt-NVFP4",
        "role": "base-modelopt-nvfp4",
        "candidates": [
            {
                "candidate_id": "Darkstar-Example-Base-ModelOpt-W4A4-NVFP4",
                "selection": "selected",
                "precision_map": {"kv_cache": kv_cache},
            }
        ],
        "gates": [
            {
                "id": "kv_cache_effect_measured",
                "status": gate_status,
                "evidence": "measured",
                "evidence_path": None,
                "next_missing_proof": "none",
            }
        ],
    }


def _ledger_product(
    *,
    product_id: str,
    target_repository: str | None,
    target_repository_status: str,
    served_model_alias: str,
    runtime_container_name: str,
) -> dict:
    return {
        "product_id": product_id,
        "role": "base-modelopt-nvfp4",
        "target_repository": target_repository,
        "target_repository_status": target_repository_status,
        "served_model_alias": served_model_alias,
        "runtime_container_name": runtime_container_name,
    }


def _gate(gate_id: str, **extra: str) -> dict:
    gate = {
        "id": gate_id,
        "status": "verified",
        "evidence": "measured on this artifact",
        "evidence_path": "results/evidence.json",
        "next_missing_proof": "none",
    }
    gate.update(extra)
    return gate


def test_kv_cache_gate_follows_the_selected_candidate_precision() -> None:
    from model_forge.release import kv_cache_gate_errors

    fp8 = "FP8 E4M3 (calibrated)"
    assert kv_cache_gate_errors(_nvfp4_product(kv_cache=fp8, gate_status="not_applicable"))
    assert kv_cache_gate_errors(_nvfp4_product(kv_cache=fp8, gate_status="verified")) == []
    assert kv_cache_gate_errors(_nvfp4_product(kv_cache="BF16", gate_status="not_applicable")) == []
    assert kv_cache_gate_errors(_nvfp4_product(kv_cache="BF16", gate_status="verified"))


def test_kv_cache_gate_on_a_product_without_a_selected_candidate() -> None:
    """A placeholder role with no built candidate (a reranker's abliterated cell) serves nothing, so its KV gate is
    not_applicable; any other status there is an error, and a missing gate is still an error."""
    from model_forge.release import kv_cache_gate_errors

    placeholder = _nvfp4_product(kv_cache="BF16", gate_status="not_applicable")
    for candidate in placeholder["candidates"]:
        candidate["selection"] = "not_built"
    assert kv_cache_gate_errors(placeholder) == []
    claimed = _nvfp4_product(kv_cache="BF16", gate_status="verified")
    for candidate in claimed["candidates"]:
        candidate["selection"] = "not_built"
    assert kv_cache_gate_errors(claimed)
    no_gate = _nvfp4_product(kv_cache="BF16", gate_status="not_applicable")
    no_gate["gates"] = [g for g in no_gate["gates"] if g["id"] != "kv_cache_effect_measured"]
    for candidate in no_gate["candidates"]:
        candidate["selection"] = "not_built"
    assert kv_cache_gate_errors(no_gate)


def test_ledger_allows_several_products_per_role_when_identities_are_unique() -> None:
    from model_forge.release import ledger_uniqueness_errors

    first = _ledger_product(
        product_id="Darkstar-Example-Base-ModelOpt-NVFP4-A",
        target_repository="HangGlidersRule/Darkstar-Example-A",
        target_repository_status="resolved",
        served_model_alias="darkstar-examplea-base-nvfp4",
        runtime_container_name="vllm-darkstar-examplea-base-modelopt",
    )
    second = _ledger_product(
        product_id="Darkstar-Example-Base-ModelOpt-NVFP4-B",
        target_repository="HangGlidersRule/Darkstar-Example-B",
        target_repository_status="resolved",
        served_model_alias="darkstar-exampleb-base-nvfp4",
        runtime_container_name="vllm-darkstar-exampleb-base-modelopt",
    )
    assert ledger_uniqueness_errors({"products": [first, second]}) == []

    same_id = dict(second)
    same_id["product_id"] = first["product_id"]
    assert ledger_uniqueness_errors({"products": [first, same_id]})

    same_alias = dict(second)
    same_alias["served_model_alias"] = first["served_model_alias"]
    assert ledger_uniqueness_errors({"products": [first, same_alias]})

    same_target = dict(second)
    same_target["target_repository"] = first["target_repository"]
    assert ledger_uniqueness_errors({"products": [first, same_target]})

    same_container = dict(second)
    same_container["runtime_container_name"] = first["runtime_container_name"]
    assert ledger_uniqueness_errors({"products": [first, same_container]})

    unresolved_a = _ledger_product(
        product_id="Darkstar-Example-Unresolved-A",
        target_repository=None,
        target_repository_status="unresolved_pending_precision_winner",
        served_model_alias="darkstar-unresolveda-base-nvfp4",
        runtime_container_name="vllm-darkstar-unresolveda-base-modelopt",
    )
    unresolved_b = _ledger_product(
        product_id="Darkstar-Example-Unresolved-B",
        target_repository=None,
        target_repository_status="unresolved_pending_precision_winner",
        served_model_alias="darkstar-unresolvedb-base-nvfp4",
        runtime_container_name="vllm-darkstar-unresolvedb-base-modelopt",
    )
    assert ledger_uniqueness_errors({"products": [unresolved_a, unresolved_b]}) == []


def test_quality_gate_follows_task() -> None:
    from model_forge.release import quality_gate_errors

    quality = _gate(
        "quality_matched_full_denominator",
        metric="nDCG@10",
        protocol="mteb-nano-v1",
    )
    gpqa = _gate("gpqa_matched_full_denominator")

    embedding_without_quality = {"product_id": "embed", "task": "embedding", "gates": []}
    assert quality_gate_errors(embedding_without_quality)

    embedding_ok = {"product_id": "embed", "task": "embedding", "gates": [quality]}
    assert quality_gate_errors(embedding_ok) == []

    embedding_with_gpqa = {
        "product_id": "embed",
        "task": "embedding",
        "gates": [quality, gpqa],
    }
    assert quality_gate_errors(embedding_with_gpqa)

    chat_without_gpqa = {"product_id": "chat", "task": "chat", "gates": []}
    assert quality_gate_errors(chat_without_gpqa)


def test_validate_ledger_reports_legacy_freeform_without_raising(tmp_path: Path) -> None:
    from model_forge.release import validate_ledger

    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps({"family": "x", "products": [{"product": "Base-BF16"}]}),
        encoding="utf-8",
    )
    errors = validate_ledger(path)
    assert errors
    assert all(isinstance(error, str) for error in errors)


def test_contract_paths_for_dispatches_on_major_version() -> None:
    from model_forge.release import contract_paths_for

    expected = {
        "1.0.0": "v1",
        "2.0.0": "v2",
        "3.0.0": "v3",
    }
    for version, directory in expected.items():
        contract_path, schema_path = contract_paths_for({"contract_version": version})
        assert contract_path == CONTRACT_ROOT / directory / "contract.json"
        assert schema_path == CONTRACT_ROOT / directory / "ledger.schema.json"

    with pytest.raises(ValueError):
        contract_paths_for({"contract_version": "4.0.0"})


def test_extract_serve_model_ids_returns_aliases_outside_the_registered_pattern() -> None:
    card = """\
# Nano Omni

```bash
vllm serve x --served-model-name nano-omni-base-nvfp4
```
"""
    assert extract_serve_model_ids(card, source_format=".md") == ["nano-omni-base-nvfp4"]


def test_v1_and_v2_contract_files_match_base_commit_bytes() -> None:
    for relative, digest in _V1_V2_SHA256.items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, relative
