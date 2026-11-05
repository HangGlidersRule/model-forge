"""Omitted contract gates must not count as satisfied.

These tests are the spec for the publication-readiness check. They fail while
``validate_ledger`` treats a gate the ledger simply leaves out as satisfied, and
they pass once every applicable gate has to be present and verified.

Applicability is the contract for that ledger version: ``applies_to`` contains
the product role, and ``applies_to_tasks`` contains the product task when the
gate defines that field. ``kv_cache_effect_measured`` on a ModelOpt role whose
selected candidate stores BF16 KV stays ``not_applicable``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from model_forge.release import (
    contract_paths_for,
    expected_lifecycle,
    kv_cache_gate_errors,
    ledger_lifecycle_errors,
    load_json,
    product_build_complete,
    product_lifecycle_errors,
    product_publication_ready,
    quality_gate_errors,
    validate_ledger,
)

_UPSTREAMS = (
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen-Image-2.1",
    "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen3-Reranker-4B",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    "nvidia/Nemotron-3.5-Rerank-8B-BF16",
    "nvidia/Nemotron-3.5-Embed-8B-BF16",
)

_BF16_MAP = {
    "language_mlp": "W4A16-NVFP4-g16",
    "lm_head": "BF16",
    "self_attention": "W4A16-NVFP4-g16",
    "gdn_projections": "BF16",
    "kv_cache": "BF16",
    "protected": "embeddings and norms (BF16)",
}


def _gate(gate_id: str, status: str = "verified", **extra: str) -> dict[str, Any]:
    gate = {
        "id": gate_id,
        "status": status,
        "evidence": extra.pop("evidence", f"{gate_id} {status}"),
        "evidence_path": None,
        "next_missing_proof": "none",
    }
    gate.update(extra)
    return gate


def _applicable_ids(contract: dict[str, Any], role: str, task: str | None) -> list[str]:
    ids: list[str] = []
    for spec in contract["gates"]:
        if role not in spec["applies_to"]:
            continue
        tasks = spec.get("applies_to_tasks")
        if tasks is not None and task not in tasks:
            continue
        ids.append(str(spec["id"]))
    return ids


def _product(**fields: Any) -> dict[str, Any]:
    product: dict[str, Any] = {
        "model_card": "models/probe/card.md",
        "recipe": None,
        "target_repository_status": "resolved",
        "publication_claim": False,
        "lifecycle": "in_progress",
    }
    product.update(fields)
    return product


def _shell(version: str, family: str, products: list[dict[str, Any]]) -> dict[str, Any]:
    major = version.split(".", 1)[0]
    return {
        "schema_version": "1.0.0",
        "contract": f"contracts/darkstar-release/v{major}/contract.json",
        "contract_version": version,
        "family": family,
        "source_revision": "probe",
        "any_publication_claim": any(bool(product.get("publication_claim")) for product in products),
        "products": products,
        "rejected_historical": [],
        "related_issues": [],
    }


def _modelopt_candidate(stem: str) -> dict[str, Any]:
    candidate_id = f"Darkstar-{stem}-ModelOpt-W4A16-NVFP4"
    return {
        "candidate_id": candidate_id,
        "precision_class": "W4A16-NVFP4",
        "target_repository": f"HangGlidersRule/{candidate_id}",
        "selection": "selected",
        "recipe": None,
        "precision_map": dict(_BF16_MAP),
    }


def _v3_filler(index: int, target: str) -> dict[str, Any]:
    token = f"fill{index}"
    return _product(
        product_id=f"Darkstar-Filler-{index}",
        role="base-bf16",
        task="chat",
        target_repository=target,
        served_model_alias=f"darkstar-{token}-base-bf16",
        runtime_container_name=f"vllm-darkstar-{token}-base-bf16",
        gates=[_gate("gpqa_matched_full_denominator", "in_progress", evidence=f"open {index}")],
    )


def _with_v3_fillers(victim: dict[str, Any], *, version: str = "3.0.0") -> dict[str, Any]:
    used = {victim.get("target_repository")}
    fillers: list[dict[str, Any]] = []
    for target in _UPSTREAMS:
        if target in used:
            continue
        fillers.append(_v3_filler(len(fillers), target))
        if len(fillers) == 3:
            break
    if len(fillers) != 3:
        raise AssertionError("not enough distinct upstream targets for fillers")
    return _shell(version, "probe", [victim, *fillers])


def _schema_errors(ledger: dict[str, Any]) -> list[str]:
    _contract_path, schema_path = contract_paths_for(ledger)
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    return [error.message for error in validator.iter_errors(ledger)]


def _write(tmp_path: Path, ledger: dict[str, Any]) -> Path:
    path = tmp_path / "publication-readiness-ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path


def _victim(ledger: dict[str, Any], product_id: str) -> dict[str, Any]:
    return next(product for product in ledger["products"] if product["product_id"] == product_id)


def _omitted(ledger: dict[str, Any], product: dict[str, Any]) -> list[str]:
    contract = load_json(contract_paths_for(ledger)[0])
    present = {str(gate["id"]) for gate in product["gates"]}
    task = product.get("task")
    task_text = task if isinstance(task, str) else None
    return [
        gate_id
        for gate_id in _applicable_ids(contract, str(product["role"]), task_text)
        if gate_id not in present
    ]


def _assert_omitted_gates_block(
    tmp_path: Path, ledger: dict[str, Any], product_id: str
) -> None:
    """Schema-valid ledger; every applicable gate that was left out is an error."""
    assert _schema_errors(ledger) == []
    product = _victim(ledger, product_id)
    omitted = _omitted(ledger, product)
    assert omitted
    if str(ledger["contract_version"]).startswith("3."):
        assert quality_gate_errors(product) == []
        assert kv_cache_gate_errors(product) == []
    errors = validate_ledger(_write(tmp_path, ledger))
    for gate_id in omitted:
        assert any(
            product_id in error and gate_id in error and ("missing" in error or "requires" in error)
            for error in errors
        ), (gate_id, errors)
    assert product_publication_ready(product) is False
    assert product_build_complete(product) is False
    assert expected_lifecycle(product) == "in_progress"
    assert product_lifecycle_errors(product)
    assert any(product_id in error for error in ledger_lifecycle_errors(ledger))


def _complete_gates(contract: dict[str, Any], role: str, task: str, owner: str) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for gate_id in _applicable_ids(contract, role, task):
        if gate_id == "kv_cache_effect_measured":
            gates.append(_gate(gate_id, "not_applicable", evidence=f"{owner} bf16 kv"))
            continue
        extra: dict[str, str] = {"evidence": f"{owner} {gate_id}"}
        if gate_id == "quality_matched_full_denominator":
            extra["metric"] = "nDCG@10"
            extra["protocol"] = "frozen full denominator"
        gates.append(_gate(gate_id, **extra))
    return gates


def test_image_base_bf16_single_gate_is_not_published(tmp_path: Path) -> None:
    victim = _product(
        product_id="Darkstar-Probe-Image-Base-BF16",
        role="image-base-bf16",
        task="image",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen-Image-2.1",
        served_model_alias="darkstar-probeimg-base-bf16",
        runtime_container_name="vllm-darkstar-probeimg-base-bf16",
        gates=[_gate("placeholder_gate", evidence="unrelated verified row")],
    )
    _assert_omitted_gates_block(tmp_path, _with_v3_fillers(victim), victim["product_id"])


def test_chat_product_with_only_gpqa_is_not_published(tmp_path: Path) -> None:
    victim = _product(
        product_id="Darkstar-Probe-Chat-Base-BF16",
        role="base-bf16",
        task="chat",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen3.8-27B",
        served_model_alias="darkstar-probechat-base-bf16",
        runtime_container_name="vllm-darkstar-probechat-base-bf16",
        gates=[_gate("gpqa_matched_full_denominator", evidence="198/198 on this chat product")],
    )
    _assert_omitted_gates_block(tmp_path, _with_v3_fillers(victim), victim["product_id"])


@pytest.mark.parametrize("task", ["embedding", "rerank", "decision"])
def test_quality_task_with_only_the_quality_gate_is_not_published(tmp_path: Path, task: str) -> None:
    targets = {
        "embedding": "Qwen/Qwen3-Embedding-8B",
        "rerank": "Qwen/Qwen3-Reranker-4B",
        "decision": "HangGlidersRule/Darkstar-ProbeDecision-Base-BF16",
    }
    victim = _product(
        product_id=f"Darkstar-Probe-{task}-Base-BF16",
        role="base-bf16",
        task=task,
        lifecycle="published",
        publication_claim=True,
        target_repository=targets[task],
        served_model_alias=f"darkstar-probe{task}-base-bf16",
        runtime_container_name=f"vllm-darkstar-probe{task}-base-bf16",
        gates=[
            _gate(
                "quality_matched_full_denominator",
                evidence=f"{task} full denominator",
                metric="nDCG@10",
                protocol="frozen full denominator",
            )
        ],
    )
    _assert_omitted_gates_block(tmp_path, _with_v3_fillers(victim), victim["product_id"])


@pytest.mark.parametrize(
    ("role", "behavior"),
    [("base-modelopt-nvfp4", "base"), ("abliterated-modelopt-nvfp4", "abliterated")],
)
def test_modelopt_bf16_kv_with_only_the_task_gate_is_not_published(
    tmp_path: Path, role: str, behavior: str
) -> None:
    candidate = _modelopt_candidate(f"Probe{behavior}")
    victim = _product(
        product_id=f"Darkstar-Probe-{behavior}-ModelOpt",
        role=role,
        task="chat",
        lifecycle="published",
        publication_claim=True,
        target_repository=candidate["target_repository"],
        served_model_alias=f"darkstar-probe{behavior}-base-nvfp4"
        if behavior == "base"
        else f"darkstar-probe{behavior}-abliterated-nvfp4",
        runtime_container_name=f"vllm-darkstar-probe{behavior}-{behavior}-modelopt"
        if behavior == "abliterated"
        else "vllm-darkstar-probebase-base-modelopt",
        candidates=[candidate],
        gates=[
            _gate("gpqa_matched_full_denominator", evidence=f"{behavior} gpqa 198/198"),
            _gate("kv_cache_effect_measured", "not_applicable", evidence="kv_cache is BF16"),
        ],
    )
    _assert_omitted_gates_block(tmp_path, _with_v3_fillers(victim), victim["product_id"])


def test_partial_build_gates_are_not_locally_complete(tmp_path: Path) -> None:
    victim = _product(
        product_id="Darkstar-Probe-Image-Build",
        role="image-base-bf16",
        task="image",
        lifecycle="locally_complete_unpublished",
        publication_claim=False,
        target_repository="Qwen/Qwen-Image-2.1",
        served_model_alias="darkstar-probebuild-base-bf16",
        runtime_container_name="vllm-darkstar-probebuild-base-bf16",
        gates=[
            _gate("artifact_manifest", evidence="checksums for this image"),
            _gate("release_tag", "in_progress", evidence="tag not cut"),
        ],
    )
    _assert_omitted_gates_block(tmp_path, _with_v3_fillers(victim, version="3.1.0"), victim["product_id"])


def test_v1_products_cannot_publish_on_one_gate(tmp_path: Path) -> None:
    products = [
        _product(
            product_id="Darkstar-V1-base-bf16",
            role="base-bf16",
            lifecycle="published",
            publication_claim=True,
            target_repository="Qwen/Qwen3.8-27B",
            served_model_alias="darkstar-v1base-base-bf16",
            runtime_container_name="vllm-darkstar-v1base-base-bf16",
            gates=[_gate("placeholder_gate", evidence="v1 base")],
        ),
        _product(
            product_id="Darkstar-V1-abliterated-bf16",
            role="abliterated-bf16",
            lifecycle="published",
            publication_claim=True,
            target_repository="HangGlidersRule/Darkstar-V1Probe-Abliterated-BF16",
            served_model_alias="darkstar-v1abl-abliterated-bf16",
            runtime_container_name="vllm-darkstar-v1abl-abliterated-bf16",
            gates=[_gate("placeholder_gate", evidence="v1 abliterated")],
        ),
    ]
    for role, behavior, stem in (
        ("base-modelopt-nvfp4", "base", "V1Base"),
        ("abliterated-modelopt-nvfp4", "abliterated", "V1Abliterated"),
    ):
        candidate = _modelopt_candidate(stem)
        products.append(
            _product(
                product_id=f"Darkstar-V1-{role}",
                role=role,
                lifecycle="published",
                publication_claim=True,
                target_repository=candidate["target_repository"],
                served_model_alias=f"darkstar-v1{behavior}-base-nvfp4"
                if behavior == "base"
                else "darkstar-v1abliterated-abliterated-nvfp4",
                runtime_container_name=f"vllm-darkstar-v1{behavior}-base-modelopt"
                if behavior == "base"
                else "vllm-darkstar-v1abliterated-abliterated-modelopt",
                candidates=[candidate],
                gates=[_gate("placeholder_gate", evidence=f"v1 {role}")],
            )
        )
    for product in products:
        product.pop("task", None)
    ledger = _shell("1.0.0", "probe-v1", products)
    for product in products:
        _assert_omitted_gates_block(tmp_path, ledger, str(product["product_id"]))


def test_v2_image_product_cannot_publish_on_one_gate(tmp_path: Path) -> None:
    victim = _product(
        product_id="Darkstar-V2-Image",
        role="image-base-bf16",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen-Image-2.1",
        served_model_alias="darkstar-v2img-base-bf16",
        runtime_container_name="vllm-darkstar-v2img-base-bf16",
        gates=[_gate("placeholder_gate", evidence="v2 image")],
    )
    fillers = [
        _product(
            product_id="Darkstar-V2-Base",
            role="base-bf16",
            lifecycle="in_progress",
            target_repository="Qwen/Qwen3.8-27B",
            served_model_alias="darkstar-v2base-base-bf16",
            runtime_container_name="vllm-darkstar-v2base-base-bf16",
            gates=[_gate("provenance_ownership", "in_progress", evidence="v2 base open")],
        ),
        _product(
            product_id="Darkstar-V2-Abliterated",
            role="abliterated-bf16",
            lifecycle="in_progress",
            target_repository="HangGlidersRule/Darkstar-V2Probe-Abliterated-BF16",
            served_model_alias="darkstar-v2abl-abliterated-bf16",
            runtime_container_name="vllm-darkstar-v2abl-abliterated-bf16",
            gates=[_gate("provenance_ownership", "in_progress", evidence="v2 abl open")],
        ),
        _product(
            product_id="Darkstar-V2-Image-FP8",
            role="image-base-fp8",
            lifecycle="in_progress",
            target_repository="HangGlidersRule/Darkstar-V2Probe-Base-FP8",
            served_model_alias="darkstar-v2fp8-base-fp8",
            runtime_container_name="vllm-darkstar-v2fp8-base-fp8",
            gates=[_gate("provenance_ownership", "in_progress", evidence="v2 fp8 open")],
        ),
    ]
    for product in [victim, *fillers]:
        product.pop("task", None)
    _assert_omitted_gates_block(tmp_path, _shell("2.0.0", "probe-v2", [victim, *fillers]), victim["product_id"])


def test_applicable_gate_marked_not_applicable_does_not_publish(tmp_path: Path) -> None:
    contract = load_json(contract_paths_for({"contract_version": "3.0.0"})[0])
    gates = _complete_gates(contract, "image-base-bf16", "image", "na-image")
    for gate in gates:
        if gate["id"] == "provenance_ownership":
            gate["status"] = "not_applicable"
            gate["evidence"] = "marked not applicable"
    victim = _product(
        product_id="Darkstar-Probe-Image-NotApplicable",
        role="image-base-bf16",
        task="image",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen-Image-2.1",
        served_model_alias="darkstar-probena-base-bf16",
        runtime_container_name="vllm-darkstar-probena-base-bf16",
        gates=gates,
    )
    ledger = _with_v3_fillers(victim)
    assert _schema_errors(ledger) == []
    assert quality_gate_errors(victim) == []
    assert kv_cache_gate_errors(victim) == []
    errors = validate_ledger(_write(tmp_path, ledger))
    assert any(
        victim["product_id"] in error
        and "provenance_ownership" in error
        and "not_applicable" in error
        for error in errors
    ), errors
    assert product_publication_ready(victim) is False
    assert product_build_complete(victim) is False
    assert expected_lifecycle(victim) == "in_progress"


def test_task_gate_marked_not_applicable_does_not_publish(tmp_path: Path) -> None:
    contract = load_json(contract_paths_for({"contract_version": "3.0.0"})[0])
    gates = _complete_gates(contract, "base-bf16", "chat", "na-chat")
    for gate in gates:
        if gate["id"] == "gpqa_matched_full_denominator":
            gate["status"] = "not_applicable"
            gate["evidence"] = "marked not applicable"
    victim = _product(
        product_id="Darkstar-Probe-Chat-NotApplicable",
        role="base-bf16",
        task="chat",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen3.8-27B",
        served_model_alias="darkstar-probenachat-base-bf16",
        runtime_container_name="vllm-darkstar-probenachat-base-bf16",
        gates=gates,
    )
    ledger = _with_v3_fillers(victim)
    assert _schema_errors(ledger) == []
    assert quality_gate_errors(victim) == []
    errors = validate_ledger(_write(tmp_path, ledger))
    assert any(
        victim["product_id"] in error
        and "gpqa_matched_full_denominator" in error
        and "not_applicable" in error
        for error in errors
    ), errors
    assert product_publication_ready(victim) is False
    assert expected_lifecycle(victim) == "in_progress"


def test_complete_applicable_gates_still_publish(tmp_path: Path) -> None:
    """A product that actually lists every applicable gate still validates.

    The ModelOpt product keeps ``kv_cache_effect_measured`` at ``not_applicable``
    because the selected candidate's KV cache is BF16.
    """
    contract = load_json(contract_paths_for({"contract_version": "3.1.0"})[0])
    image = _product(
        product_id="Darkstar-Pos-Image",
        role="image-base-bf16",
        task="image",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen-Image-2.1",
        served_model_alias="darkstar-posimg-base-bf16",
        runtime_container_name="vllm-darkstar-posimg-base-bf16",
        gates=_complete_gates(contract, "image-base-bf16", "image", "pos-image"),
    )
    chat = _product(
        product_id="Darkstar-Pos-Chat",
        role="base-bf16",
        task="chat",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen3.8-27B",
        served_model_alias="darkstar-poschat-base-bf16",
        runtime_container_name="vllm-darkstar-poschat-base-bf16",
        gates=_complete_gates(contract, "base-bf16", "chat", "pos-chat"),
    )
    rerank = _product(
        product_id="Darkstar-Pos-Rerank",
        role="base-bf16",
        task="rerank",
        lifecycle="published",
        publication_claim=True,
        target_repository="Qwen/Qwen3-Reranker-4B",
        served_model_alias="darkstar-posrerank-base-bf16",
        runtime_container_name="vllm-darkstar-posrerank-base-bf16",
        gates=_complete_gates(contract, "base-bf16", "rerank", "pos-rerank"),
    )
    candidate = _modelopt_candidate("PosMopt")
    modelopt = _product(
        product_id="Darkstar-PosMopt-Base-ModelOpt",
        role="base-modelopt-nvfp4",
        task="chat",
        lifecycle="published",
        publication_claim=True,
        target_repository=candidate["target_repository"],
        served_model_alias="darkstar-posmopt-base-nvfp4",
        runtime_container_name="vllm-darkstar-posmopt-base-modelopt",
        candidates=[candidate],
        gates=_complete_gates(contract, "base-modelopt-nvfp4", "chat", "pos-mopt"),
    )
    ledger = _shell("3.1.0", "probe-complete", [image, chat, rerank, modelopt])
    assert _schema_errors(ledger) == []
    assert validate_ledger(_write(tmp_path, ledger)) == []
    for product in ledger["products"]:
        assert product_publication_ready(product) is True
        assert product_build_complete(product) is True
        assert expected_lifecycle(product) == "published"
        assert product_lifecycle_errors(product) == []
    assert ledger_lifecycle_errors(ledger) == []
