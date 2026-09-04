"""Tests for the Darkstar four-product release contract and the Qwen3.8 publication-readiness ledger.

The ledger JSON is the machine-readable source of truth. These tests enforce schema validity, the
four canonical products, gate presence, valid statuses and applicability, the publication rule (no
publication claim unless every applicable gate is verified), synchronization between the ledger and the
rendered LEDGER-SYNC blocks in the model cards and benchmark matrix, valid links/paths, and a
no-weakening guard on the frozen gate set.

They also enforce the actual-identifier rules fail-closed: every target repository, candidate
repository, model-card identity heading, model source, and recipe publication target must encode
ModelOpt plus W4A16 or W4A4 (plus Mixed-FP8 where applicable). The separate runtime API identity is a
registered concise lowercase alias. Bare NVFP4 publication ids and W4A16/W4A4/Mixed-FP8 conflation are
rejected, and a ModelOpt product with no built precision winner must leave its target explicitly
unresolved instead of reserving a bare slot id.

The Mixed-FP8 requirement itself is derived from each candidate's precision map rather than from the
names it declares, so shortening a mixed candidate's precision class and ids to W4A16-NVFP4 while FP8
stays in the map is a failure and not a silent pass.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import jsonschema
import pytest
import yaml

from model_forge.release import (
    CANONICAL_ROLES,
    DARKSTAR_HF_NAMESPACE,
    FROZEN_REQUIRED_GATES,
    LIFECYCLE_STATES,
    MODELOPT_NVFP4_ROLES,
    PUBLICATION_ONLY_GATES,
    REQUIRED_PRECISION_MAP_COMPONENTS,
    RUNTIME_PRECISION_MAP_COMPONENTS,
    VALID_STATUSES,
    WEIGHT_PRECISION_MAP_COMPONENTS,
    PublicationSourceParseError,
    candidate_precision_errors,
    candidate_promoted,
    candidate_promotion_blocked,
    candidate_requires_mixed_fp8,
    candidate_selection,
    contract_gate_ids,
    expected_card_identity_ids,
    expected_lifecycle,
    extract_model_sources,
    extract_repository_ids,
    extract_serve_model_ids,
    gate_applies_to,
    is_bare_nvfp4_id,
    ledger_candidate_precision_errors,
    ledger_lifecycle_errors,
    ledger_mixed_fp8_requirements,
    ledger_target_repository_errors,
    load_json,
    model_card_identity_ids,
    model_source_precision_errors,
    precision_encoded_id_errors,
    precision_map_requires_mixed_fp8,
    product_build_complete,
    product_candidates,
    product_gate_statuses,
    product_publication_ready,
    target_repository_errors,
    validate_ledger_schema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "contracts" / "darkstar-release" / "v1" / "contract.json"
SCHEMA_PATH = REPO_ROOT / "contracts" / "darkstar-release" / "v1" / "ledger.schema.json"
LEDGER_PATH = REPO_ROOT / "models" / "qwen3.8-27b-r3" / "results" / "publication-readiness-ledger.json"

_MARKER = re.compile(r"<!--\s*LEDGER-SYNC product=(Darkstar-\S+?)\s*-->")
_CANDIDATE_MARKER = re.compile(r"<!--\s*CANDIDATE-SYNC candidate=(Darkstar-\S+?)\s*-->")


def _contract() -> dict:
    return load_json(CONTRACT_PATH)


def _schema() -> dict:
    return load_json(SCHEMA_PATH)


def _ledger() -> dict:
    return load_json(LEDGER_PATH)


def _markdown_files() -> list[Path]:
    ignore = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}
    return sorted(
        p
        for p in REPO_ROOT.rglob("*.md")
        if not any(part in ignore for part in p.relative_to(REPO_ROOT).parts)
    )


def _parse_ledger_sync_blocks(text: str) -> dict[str, dict[str, str]]:
    lines = text.splitlines()
    blocks: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(lines):
        match = _MARKER.search(lines[i])
        if not match:
            i += 1
            continue
        product = match.group(1)
        statuses: dict[str, str] = {}
        started = False
        j = i + 1
        while j < len(lines):
            line = lines[j].strip()
            if line.startswith("|"):
                started = True
                cells = [c.strip() for c in line.strip("|").split("|")]
                gate = cells[0]
                status = cells[1] if len(cells) > 1 else ""
                if gate.lower() != "gate" and set(gate) - set("-: "):
                    statuses[gate] = status
            elif started:
                break
            j += 1
        blocks[product] = statuses
        i = j
    return blocks


def _parse_candidate_sync_blocks(text: str) -> dict[str, dict[str, str]]:
    lines = text.splitlines()
    blocks: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(lines):
        match = _CANDIDATE_MARKER.search(lines[i])
        if not match:
            i += 1
            continue
        candidate = match.group(1)
        rows: dict[str, str] = {}
        started = False
        j = i + 1
        while j < len(lines):
            line = lines[j].strip()
            if line.startswith("|"):
                started = True
                cells = [c.strip() for c in line.strip("|").split("|")]
                key = cells[0]
                val = cells[1] if len(cells) > 1 else ""
                if key.lower() != "component" and set(key) - set("-: "):
                    rows[key] = val
            elif started:
                break
            j += 1
        blocks[candidate] = rows
        i = j
    return blocks


def _all_candidates() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for product in _ledger()["products"]:
        for cand in product_candidates(product):
            out[cand["candidate_id"]] = cand
    return out


def test_contract_loads_with_required_keys() -> None:
    contract = _contract()
    for key in ("contract_version", "canonical_roles", "statuses", "publication_rule", "gates"):
        assert key in contract
    assert list(contract["canonical_roles"]) == list(CANONICAL_ROLES)
    assert list(contract["statuses"]) == list(VALID_STATUSES)


def test_ledger_matches_schema() -> None:
    validate_ledger_schema(_ledger(), _schema())


def test_four_canonical_products() -> None:
    ledger = _ledger()
    products = ledger["products"]
    assert len(products) == 4
    roles = [p["role"] for p in products]
    assert set(roles) == set(CANONICAL_ROLES)
    assert len(set(roles)) == 4
    family = "Qwen3.8-27B"
    expected_ids = {
        "base-bf16": f"Darkstar-{family}-Base-BF16",
        "base-modelopt-nvfp4": f"Darkstar-{family}-Base-ModelOpt-NVFP4",
        "abliterated-bf16": f"Darkstar-{family}-Abliterated-BF16",
        "abliterated-modelopt-nvfp4": f"Darkstar-{family}-Abliterated-ModelOpt-NVFP4",
    }
    for product in products:
        assert product["product_id"] == expected_ids[product["role"]]


def test_gate_presence_matches_contract() -> None:
    contract = _contract()
    contract_ids = set(contract_gate_ids(contract))
    for product in _ledger()["products"]:
        assert set(product_gate_statuses(product)) == contract_ids


def test_statuses_valid_and_applicability() -> None:
    contract = _contract()
    for product in _ledger()["products"]:
        role = product["role"]
        for gate in product["gates"]:
            gid = gate["id"]
            status = gate["status"]
            assert status in VALID_STATUSES
            if gate_applies_to(contract, gid, role):
                assert status != "not_applicable", f"{product['product_id']}/{gid} wrongly N/A"
            else:
                assert status == "not_applicable", f"{product['product_id']}/{gid} must be N/A"


def test_publication_rule_no_unearned_claim() -> None:
    ledger = _ledger()
    any_claim = False
    for product in ledger["products"]:
        claim = bool(product["publication_claim"])
        any_claim = any_claim or claim
        if claim:
            assert product_publication_ready(product), (
                f"{product['product_id']} claims publication without all applicable gates verified"
            )
    assert ledger["any_publication_claim"] == any_claim
    # Nothing is published yet: no product is publication-ready (publication-only gates remain open).
    assert all(product_publication_ready(p) for p in ledger["products"])


def test_lifecycle_matches_gate_statuses() -> None:
    ledger = _ledger()
    assert ledger_lifecycle_errors(ledger) == []
    by_role = {p["role"]: p for p in ledger["products"]}

    for role in CANONICAL_ROLES:
        product = by_role[role]
        assert product["lifecycle"] == "published"
        assert product["lifecycle"] in LIFECYCLE_STATES
        # Build-complete but not publication-ready == locally complete, unpublished.
        assert product_build_complete(product)
        assert product_publication_ready(product)
        assert expected_lifecycle(product) == "published"

def test_publication_only_gates_are_the_only_open_gates_for_local_products() -> None:
    ledger = _ledger()
    for role in CANONICAL_ROLES:
        product = next(p for p in ledger["products"] if p["role"] == role)
        open_gates = {
            g["id"]
            for g in product["gates"]
            if g["status"] not in ("verified", "not_applicable")
        }
        assert open_gates <= PUBLICATION_ONLY_GATES, (
            f"{role} has non-publication gates still open: {open_gates - PUBLICATION_ONLY_GATES}"
        )


def test_no_gate_weakening() -> None:
    contract = _contract()
    contract_ids = set(contract_gate_ids(contract))
    assert FROZEN_REQUIRED_GATES <= contract_ids
    assert set(contract["statuses"]) == set(VALID_STATUSES)
    assert "verified" in contract["publication_rule"]
    for gate in contract["gates"]:
        assert gate["applies_to"], f"gate {gate['id']} applies to no role"
        assert set(gate["applies_to"]) <= set(CANONICAL_ROLES)


def test_ledger_and_rendered_blocks_are_in_sync() -> None:
    ledger = _ledger()
    expected = {p["product_id"]: product_gate_statuses(p) for p in ledger["products"]}
    found_in_matrix: set[str] = set()
    card_by_product = {p["product_id"]: (REPO_ROOT / p["model_card"]) for p in ledger["products"]}
    found_in_card: set[str] = set()
    matrix_path = REPO_ROOT / "models" / "qwen3.8-27b-r3" / "benchmark-matrix.md"

    for md_path in _markdown_files():
        blocks = _parse_ledger_sync_blocks(md_path.read_text(encoding="utf-8"))
        for product_id, rendered in blocks.items():
            assert product_id in expected, f"{md_path} renders unknown product {product_id}"
            assert rendered == expected[product_id], f"{md_path} out of sync for {product_id}"
            if md_path == matrix_path:
                found_in_matrix.add(product_id)
            if md_path == card_by_product.get(product_id):
                found_in_card.add(product_id)

    assert found_in_matrix == set(expected), "benchmark-matrix.md must render all four products"
    assert found_in_card == set(expected), "each model card must render its own product block"


@pytest.mark.private_source_only
def test_valid_links_and_paths() -> None:
    ledger = _ledger()
    contract = _contract()
    assert (REPO_ROOT / ledger["contract"]).exists()
    assert (REPO_ROOT / contract["ledger_schema"]).exists()
    assert (REPO_ROOT / contract["process_document"]).exists()
    for product in ledger["products"]:
        assert (REPO_ROOT / product["model_card"]).exists()
        if product["recipe"] is not None:
            assert (REPO_ROOT / product["recipe"]).exists()
        for gate in product["gates"]:
            path = gate["evidence_path"]
            if path is not None:
                assert (REPO_ROOT / path).exists(), f"missing evidence_path {path}"
    for entry in ledger["rejected_historical"]:
        path = entry["evidence_path"]
        if path is not None:
            assert (REPO_ROOT / path).exists(), f"missing rejected_historical path {path}"


def test_all_four_products_have_no_stale_build_blockers() -> None:
    """All products are locally complete: every applicable *build* gate is verified.

    This is the regression guard against the old conservative snapshot, which wrongly recorded these
    products' build gates as missing/in_progress/blocked. Only publication-only gates may remain open.
    """
    ledger = _ledger()
    by_role = {p["role"]: product_gate_statuses(p) for p in ledger["products"]}

    for role in CANONICAL_ROLES:
        statuses = by_role[role]
        for gid, status in statuses.items():
            if gid in PUBLICATION_ONLY_GATES:
                continue
            assert status in ("verified", "not_applicable"), (
                f"{role}/{gid} is {status!r}; a locally complete product must have no open build gate"
            )
        # The full-denominator GPQA and independent performance/serving profiles are all verified.
        assert statuses["gpqa_matched_full_denominator"] == "verified"
        assert statuses["performance_profile"] == "verified"
        assert statuses["serving_capacity_profile"] == "verified"
        if role.startswith("abliterated-"):
            assert statuses["behavior_refusal_eval"] == "verified"
        else:
            assert statuses["behavior_refusal_eval"] == "not_applicable"

    # Provenance/ownership is resolved for every product (no lingering "undecided ownership").
    for role in CANONICAL_ROLES:
        assert by_role[role]["provenance_ownership"] == "verified"


def test_product_4_is_locally_complete_and_never_claims_publication() -> None:
    ledger = _ledger()
    p4 = next(p for p in ledger["products"] if p["role"] == "abliterated-modelopt-nvfp4")
    statuses = product_gate_statuses(p4)

    assert p4["lifecycle"] == "published"
    assert product_build_complete(p4)
    assert product_publication_ready(p4)
    for gid, status in statuses.items():
        if gid in PUBLICATION_ONLY_GATES:
            continue
        assert status == "verified", f"{gid} should be verified, got {status!r}"
    # No gate anywhere in the ledger uses a "blocked" status.
    for product in ledger["products"]:
        for gate in product["gates"]:
            assert gate["status"] != "blocked"


def test_all_applicable_release_gates_verified_after_tag() -> None:
    by_role = {p["role"]: product_gate_statuses(p) for p in _ledger()["products"]}
    for role in CANONICAL_ROLES:
        expected = "not_applicable" if role == "base-bf16" else "verified"
        assert by_role[role]["publication_targets_hf_ghcr"] == expected
        assert by_role[role]["clean_download_boot_smoke"] == expected
        assert by_role[role]["model_card_final"] == "verified"
        assert by_role[role]["release_tag"] == "verified"


def test_superseded_results_recorded_as_rejected_historical() -> None:
    ledger = _ledger()
    # Old compressed-tensors and mixed-protocol GPQA are preserved as rejected_historical.
    rejected = " ".join(e["item"] + " " + e["reason"] for e in ledger["rejected_historical"])
    assert "compressed-tensors" in rejected
    assert "164/198" in rejected


@pytest.mark.parametrize("issue", [f"#{n}" for n in range(4, 15)])
def test_related_issues_referenced(issue: str) -> None:
    assert issue in _ledger()["related_issues"]


# --- Candidate precision-encoding: prevent precision-map omission or conflation ---


def test_modelopt_products_declare_precision_encoded_candidates() -> None:
    for product in _ledger()["products"]:
        if product["role"] in MODELOPT_NVFP4_ROLES:
            candidates = product_candidates(product)
            assert candidates, f"{product['product_id']} must declare NVFP4 candidates"
            for cand in candidates:
                # A candidate id must never hide its precision behind the bare ModelOpt-NVFP4 slot.
                assert "ModelOpt" in cand["candidate_id"]
                assert ("W4A16" in cand["candidate_id"]) or ("W4A4" in cand["candidate_id"])
        else:
            assert not product_candidates(product), (
                f"{product['product_id']} is not a ModelOpt NVFP4 role and must not list candidates"
            )


def test_candidate_precision_maps_are_complete_and_unconflated() -> None:
    candidates = _all_candidates()
    assert candidates, "expected NVFP4 candidates in the ledger"
    for cand in candidates.values():
        assert set(cand["precision_map"]) == set(REQUIRED_PRECISION_MAP_COMPONENTS)
        errors = candidate_precision_errors(cand)
        assert not errors, "precision encoding violations:\n" + "\n".join(errors)


def test_candidate_ids_are_unique() -> None:
    ids = [c["candidate_id"] for p in _ledger()["products"] for c in product_candidates(p)]
    assert len(ids) == len(set(ids))


def test_mixed_and_w4a4_candidates_are_distinct_not_conflated() -> None:
    candidates = _all_candidates()
    mixed = candidates["Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8"]
    w4a4 = candidates["Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"]

    # The mixed candidate is genuinely W4A16 NVFP4 on MLP + lm_head, FP8 on attention + GDN.
    assert mixed["precision_map"]["language_mlp"].startswith("W4A16")
    assert mixed["precision_map"]["lm_head"].startswith("W4A16")
    assert mixed["precision_map"]["self_attention"].startswith("FP8")
    assert mixed["precision_map"]["gdn_projections"].startswith("FP8")
    assert mixed["precision_map"]["kv_cache"] == "BF16"

    # The uniform W4A4 candidate must not smuggle FP8 or W4A16 into its map.
    joined = " ".join(w4a4["precision_map"].values())
    assert "FP8" not in joined and "W4A16" not in joined


def test_candidate_sync_blocks_render_precision_maps_in_docs() -> None:
    expected = {cid: cand["precision_map"] for cid, cand in _all_candidates().items()}
    rendered_for: set[str] = set()
    for md_path in _markdown_files():
        blocks = _parse_candidate_sync_blocks(md_path.read_text(encoding="utf-8"))
        for candidate_id, rendered in blocks.items():
            assert candidate_id in expected, f"{md_path} renders unknown candidate {candidate_id}"
            assert rendered == expected[candidate_id], f"{md_path} precision map out of sync for {candidate_id}"
            rendered_for.add(candidate_id)
    # The two base ModelOpt candidates must be rendered somewhere (no omission from the docs).
    assert "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8" in rendered_for
    assert "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4" in rendered_for


# --- Actual identifiers: target repositories, model-card identity, serve ids ---

MIXED_BASE_CANDIDATE = "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8"
W4A4_BASE_CANDIDATE = "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"

# Directories that publish identifiers. tests/ is deliberately excluded: it declares the rules and
# carries bare/conflated ids as negative fixtures.
_PUBLICATION_SURFACE_ROOTS = (
    "README.md",
    "configs",
    "containers",
    "contracts",
    "docs",
    "models",
    "recipes",
    "scripts",
    "src",
)
_PUBLICATION_SURFACE_SUFFIXES = {".md", ".yaml", ".yml", ".json", ".py", ".sh", ".toml"}


def _publication_surface_files() -> list[Path]:
    ignore = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", ".hermes"}
    files: list[Path] = []
    for name in _PUBLICATION_SURFACE_ROOTS:
        root = REPO_ROOT / name
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _PUBLICATION_SURFACE_SUFFIXES:
                continue
            if any(part in ignore for part in path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def _product_by_role(role: str) -> dict:
    return next(p for p in _ledger()["products"] if p["role"] == role)


def test_ledger_target_repositories_are_precision_encoded() -> None:
    errors = ledger_target_repository_errors(_ledger())
    assert not errors, "target repository violations:\n" + "\n".join(errors)


def test_base_bf16_is_exact_upstream_reference_not_owned_target() -> None:
    product = _product_by_role("base-bf16")
    assert product["target_repository"] == "Qwen/Qwen3.8-27B"
    assert product["target_repository_status"] == "resolved"
    assert "upstream reference only" in product["target_repository_note"].lower()
    assert "HangGlidersRule" not in product["target_repository"]
    assert not product_candidates(product)


def test_public_hf_clean_smoke_and_final_card_evidence_is_pinned() -> None:
    by_role = {p["role"]: p for p in _ledger()["products"]}
    expected = {
        "base-modelopt-nvfp4": "c3f03c5bf5a28a636d72cd979323ff2f80668fb0",
        "abliterated-bf16": "0181d5d178a15c694b1d6708d3ee3d08d2d9db5e",
        "abliterated-modelopt-nvfp4": "2e25bd97fd1b6e6c7989e74c261d93a8702496e8",
    }
    for role, revision in expected.items():
        product = by_role[role]
        statuses = product_gate_statuses(product)
        gates = {g["id"]: g for g in product["gates"]}
        assert revision in gates["publication_targets_hf_ghcr"]["evidence"]
        assert "GHCR is explicitly not required" in gates["publication_targets_hf_ghcr"]["evidence"]
        assert revision in gates["clean_download_boot_smoke"]["evidence"]
        assert "strict JSON" in gates["clean_download_boot_smoke"]["evidence"]
        assert "darkstar-qwen3.8-27b-v1.0.0" in gates["model_card_final"]["evidence"]
        assert statuses["model_card_final"] == "verified"
        assert statuses["publication_targets_hf_ghcr"] == "verified"
        assert statuses["clean_download_boot_smoke"] == "verified"
        assert statuses["release_tag"] == "verified"
        assert product["publication_claim"]


def test_runtime_aliases_are_concise_unique_and_separate_from_repository_ids() -> None:
    aliases = [p["served_model_alias"] for p in _ledger()["products"]]
    assert len(aliases) == len(set(aliases)) == 4
    for product, alias in zip(_ledger()["products"], aliases, strict=True):
        assert re.fullmatch(
            r"darkstar-[a-z0-9]+-(base|abliterated)-(bf16|nvfp4)", alias
        )
        assert alias != product["target_repository"]
        assert product["runtime_container_name"].startswith("vllm-darkstar-")
        card = (REPO_ROOT / product["model_card"]).read_text(encoding="utf-8")
        assert f"--served-model-name {alias}" in card

    p4 = _product_by_role("abliterated-modelopt-nvfp4")
    assert p4["served_model_alias"] == "darkstar-qwen38-abliterated-nvfp4"
    assert p4["runtime_container_name"] == "vllm-darkstar-qwen38-abliterated-modelopt"
    assert p4["serve_compose"] == "containers/serve/darkstar-qwen38-abliterated-nvfp4.yml"


def test_base_modelopt_target_resolves_to_the_tested_mixed_candidate() -> None:
    product = _product_by_role("base-modelopt-nvfp4")
    assert product["target_repository_status"] == "resolved"
    assert product["target_repository"] == f"{DARKSTAR_HF_NAMESPACE}/{MIXED_BASE_CANDIDATE}"

    # The unbuilt uniform W4A4 candidate reserves its own id and never absorbs the mixed target.
    targets = {c["candidate_id"]: c["target_repository"] for c in product_candidates(product)}
    assert targets[W4A4_BASE_CANDIDATE] == f"{DARKSTAR_HF_NAMESPACE}/{W4A4_BASE_CANDIDATE}"
    assert targets[MIXED_BASE_CANDIDATE] != targets[W4A4_BASE_CANDIDATE]


def test_abliterated_modelopt_target_resolves_to_the_selected_local_product() -> None:
    product = _product_by_role("abliterated-modelopt-nvfp4")
    mixed = "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
    w4a4 = "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A4-NVFP4"

    assert product["lifecycle"] == "published"
    assert product["target_repository_status"] == "resolved"
    assert product["target_repository"] == f"{DARKSTAR_HF_NAMESPACE}/{mixed}"

    by_id = {c["candidate_id"]: c for c in product_candidates(product)}
    assert set(by_id) == {mixed, w4a4}
    assert by_id[mixed]["selection"] == "selected"
    assert by_id[w4a4]["selection"] == "not_built"
    assert candidate_promoted(by_id[mixed])
    assert not candidate_promoted(by_id[w4a4])
    for candidate in by_id.values():
        assert not is_bare_nvfp4_id(candidate["target_repository"])
    # The resolved target names the built candidate; no target-repository violations.
    assert not ledger_target_repository_errors(_ledger())


def test_base_modelopt_mixed_candidate_selected_and_w4a4_rejected() -> None:
    product = _product_by_role("base-modelopt-nvfp4")
    by_id = {c["candidate_id"]: c for c in product_candidates(product)}
    mixed = by_id[MIXED_BASE_CANDIDATE]
    w4a4 = by_id[W4A4_BASE_CANDIDATE]

    assert candidate_selection(mixed) == "selected"
    assert candidate_promoted(mixed)
    assert candidate_selection(w4a4) == "rejected"
    assert not candidate_promoted(w4a4)


def test_resolved_targets_never_name_an_unbuilt_candidate() -> None:
    for product in _ledger()["products"]:
        if product["target_repository_status"] != "resolved":
            continue
        for candidate in product_candidates(product):
            if candidate["selection"] == "not_built":
                assert product["target_repository"] != candidate["target_repository"]


def test_no_bare_nvfp4_repository_id_on_any_publication_surface() -> None:
    required = ledger_mixed_fp8_requirements(_ledger())
    violations: list[str] = []
    for path in _publication_surface_files():
        text = path.read_text(encoding="utf-8")
        for repo_id in extract_repository_ids(text):
            errors = precision_encoded_id_errors(repo_id, required.get(repo_id))
            violations.extend(f"{path.relative_to(REPO_ROOT)}: {error}" for error in errors)
    assert not violations, "bare or conflated NVFP4 repository ids:\n" + "\n".join(violations)


def test_model_card_identity_matches_the_ledger_target() -> None:
    required = ledger_mixed_fp8_requirements(_ledger())
    for product in _ledger()["products"]:
        card = REPO_ROOT / product["model_card"]
        text = card.read_text(encoding="utf-8")
        if product["role"] == "base-bf16":
            assert "# Qwen/Qwen3.8-27B upstream BF16 control" in text
            assert "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16" not in text
            continue
        identity = model_card_identity_ids(text)
        expected = expected_card_identity_ids(product)
        assert identity == expected, (
            f"{product['model_card']} identity heading declares {identity}, expected {expected}"
        )
        for name in identity:
            assert not precision_encoded_id_errors(name, required.get(name)), (
                f"{card}: {name} hides its precision class"
            )


def test_serve_examples_use_registered_runtime_aliases() -> None:
    known_aliases = {
        p["served_model_alias"] for p in _ledger()["products"] if p.get("served_model_alias")
    }
    violations: list[str] = []
    for path in _publication_surface_files():
        for served in extract_serve_model_ids(
            path.read_text(encoding="utf-8"), source_format=path.suffix
        ):
            rel = path.relative_to(REPO_ROOT)
            if served not in known_aliases:
                violations.append(f"{rel}: served alias {served} is not a ledger runtime alias")
    assert not violations, "serve example violations:\n" + "\n".join(violations)


def test_serve_model_id_parser_reads_canonical_product4_compose() -> None:
    compose = REPO_ROOT / "containers/serve/darkstar-qwen38-abliterated-nvfp4.yml"
    assert extract_serve_model_ids(
        compose.read_text(encoding="utf-8"), source_format=compose.suffix
    ) == ["darkstar-qwen38-abliterated-nvfp4"]


@pytest.mark.parametrize(
    "command",
    [
        """\
command:
- --model
- HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16
- --served-model-name
- darkstar-qwen38-base-bf16
""",
        """\
services:
  vllm:
    command: >-
      --model HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16
      --served-model-name darkstar-qwen38-base-bf16
""",
        """\
services:
  vllm:
    command:
    - --model=HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16
    - --served-model-name=darkstar-qwen38-base-bf16
""",
    ],
)
def test_serve_model_id_parser_handles_compose_command_forms(command: str) -> None:
    aliases = extract_serve_model_ids(command, source_format=".yml")
    assert aliases == ["darkstar-qwen38-base-bf16"]
    assert extract_model_sources(command, source_format=".yml") == [
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16"
    ]


@pytest.mark.parametrize(
    ("surface", "source_format", "alias"),
    [
        (
            "sh -c 'vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16 "
            "--served-model-name darkstar-qwen38-base-bf16'",
            ".sh",
            "darkstar-qwen38-base-bf16",
        ),
        (
            "sh -c 'bash -O extglob -c \"vllm serve "
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16 "
            "--served-model-name darkstar-qwen38-abliterated-bf16\"'",
            ".sh",
            "darkstar-qwen38-abliterated-bf16",
        ),
        (
            'command: [sh, -c, "vllm serve '
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16 "
            '--served-model-name darkstar-qwen38-base-bf16"]',
            ".yml",
            "darkstar-qwen38-base-bf16",
        ),
        (
            'command: [bash, -O, extglob, -c, "vllm serve '
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16 "
            '--served-model-name=darkstar-qwen38-abliterated-bf16"]',
            ".yaml",
            "darkstar-qwen38-abliterated-bf16",
        ),
    ],
)
def test_serve_model_id_parser_recurses_into_shell_c_payloads(
    surface: str, source_format: str, alias: str
) -> None:
    assert extract_serve_model_ids(surface, source_format=source_format) == [alias]


def test_nested_alias_and_model_source_are_extracted_separately() -> None:
    source = "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16"
    alias = "darkstar-qwen38-base-bf16"
    surface = f'bash -O extglob -c "vllm serve --model {source} --served-model-name {alias}"'

    assert extract_serve_model_ids(surface, source_format=".sh") == [alias]
    assert extract_model_sources(surface, source_format=".sh") == [source]


@pytest.mark.parametrize(
    ("surface", "source_format"),
    [
        (
            'bash -O -c "vllm serve model --served-model-name darkstar-qwen38-base-bf16"',
            ".sh",
        ),
        (
            'command: [sh, -o, -c, "vllm serve model '
            '--served-model-name darkstar-qwen38-base-bf16"]',
            ".yaml",
        ),
    ],
)
def test_malformed_alias_shell_wrappers_fail_closed(
    surface: str, source_format: str
) -> None:
    with pytest.raises(PublicationSourceParseError):
        extract_serve_model_ids(surface, source_format=source_format)


def test_alias_shell_wrapper_nesting_is_bounded() -> None:
    payload = (
        "vllm serve model --served-model-name darkstar-qwen38-base-bf16"
    )
    for _ in range(10):
        payload = f"sh -c {shlex.quote(payload)}"

    with pytest.raises(PublicationSourceParseError, match="nests too deeply"):
        extract_serve_model_ids(payload, source_format=".sh")


def test_serve_model_sources_remain_separately_precision_encoded() -> None:
    """Concise --served-model-name aliases never relax the actual model source."""
    required = ledger_mixed_fp8_requirements(_ledger())
    violations: list[str] = []
    found: set[str] = set()
    for path in _publication_surface_files():
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yml", ".yaml"} and "${PUBLIC_ARTIFACT_PATH}" in text:
            document = yaml.safe_load(text)
            command = document["services"]["vllm"]["command"]
            model_index = command.index("--model") + 1
            assert command[model_index] == "${PUBLIC_ARTIFACT_PATH}"
            found.add("${PUBLIC_ARTIFACT_PATH}")
            continue
        for source in extract_model_sources(text, source_format=path.suffix):
            found.add(source)
            errors = model_source_precision_errors(source, required.get(source))
            violations.extend(f"{path.relative_to(REPO_ROOT)}: {error}" for error in errors)
    assert not violations, "serve model-source violations:\n" + "\n".join(violations)
    assert (
        f"{DARKSTAR_HF_NAMESPACE}/"
        "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
    ) in found


def test_model_source_parser_covers_shell_and_structured_compose_forms() -> None:
    positional = (
        "vllm serve "
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-"
        "ModelOpt-W4A16-NVFP4-Mixed-FP8 \\\n"
        "  --served-model-name darkstar-qwen38-abliterated-nvfp4"
    )
    flagged = (
        "--model=HangGlidersRule/"
        "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"
    )
    compose = """\
command:
- --model
- /d/model-forge/artifacts/Qwen3.8-27B-abliterated-performance-mixed-modelopt
- --served-model-name
- darkstar-qwen38-abliterated-nvfp4
"""
    assert extract_model_sources(positional) == [
        "HangGlidersRule/"
        "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
    ]
    assert extract_model_sources(flagged) == [
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"
    ]
    assert extract_model_sources(compose) == [
        "/d/model-forge/artifacts/Qwen3.8-27B-abliterated-performance-mixed-modelopt"
    ]
    compose_string = """\
services:
  vllm:
    command: >-
      --model HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4
      --served-model-name darkstar-qwen38-base-nvfp4
"""
    compose_inline = """\
services:
  vllm:
    command: ["--model", "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"]
"""
    expected = ["HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"]
    assert extract_model_sources(compose_string, source_format=".yml") == expected
    assert extract_model_sources(compose_inline, source_format=".yaml") == expected


@pytest.mark.parametrize(
    "surface",
    [
        "vllm serve --served-model-name darkstar-qwen38-base-nvfp4",
        "vllm serve",
        "--model",
        "--model=",
        'vllm serve "unterminated',
        "command: [--model]",
        "command:\n  unexpected: --model",
        "command: [vllm, serve, --served-model-name, darkstar-qwen38-base-nvfp4]",
        # An unresolved shell variable names no artifact: it is never an opaque local path.
        'vllm serve "$MODEL_SOURCE"',
        "vllm serve $MODEL_SOURCE",
        "vllm serve ${MODEL_SOURCE}",
        'vllm serve "${MODEL_REPO}/${MODEL_NAME}"',
        "--model $MODEL_SOURCE",
        "--model=${MODEL_SOURCE}",
        "vllm serve ${MODEL_SOURCE:-}",
        "--model ${MODEL_SOURCE:-}",
        # Structured Compose surfaces that cannot yield one concrete source.
        'entrypoint: [vllm, serve, "$MODEL_SOURCE"]',
        "entrypoint: [vllm, serve]",
        "entrypoint: [vllm, serve, --served-model-name, darkstar-qwen38-base-nvfp4]",
        "entrypoint: [vllm, serve]\ncommand: [--served-model-name, darkstar-qwen38-base-nvfp4]",
        'command: [sh, -c, "vllm serve $MODEL_SOURCE"]',
        'command: [sh, -c, "vllm serve --served-model-name darkstar-qwen38-base-nvfp4"]',
        'command: [sh, -c, "vllm serve"]',
        "command: [sh, -c]",
        "command: [bash, -lc]",
    ],
)
def test_model_source_parser_fails_closed_on_unparseable_surfaces(surface: str) -> None:
    yaml_surface = surface.startswith(("command:", "entrypoint:", "services:"))
    source_format = ".yaml" if yaml_surface else ".sh"
    with pytest.raises(PublicationSourceParseError):
        extract_model_sources(surface, source_format=source_format)


def test_concise_served_model_name_alone_is_not_a_publication_source() -> None:
    assert (
        extract_model_sources(
            "--served-model-name darkstar-qwen38-abliterated-nvfp4",
            source_format=".sh",
        )
        == []
    )


# --- Fail-closed model-source parsing: shell variables, defaults, Compose, and `sh -c` ---

BARE_ABLITERATED_NVFP4 = f"{DARKSTAR_HF_NAMESPACE}/Darkstar-Qwen3.8-27B-Abliterated-NVFP4"
MIXED_ABLITERATED_NVFP4 = (
    f"{DARKSTAR_HF_NAMESPACE}/"
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
)
LOCAL_ARTIFACT_PATH = "/d/model-forge/artifacts/Qwen3.8-27B-abliterated-performance-mixed-modelopt"


@pytest.mark.parametrize(
    "source",
    [
        "$MODEL_SOURCE",
        "${MODEL_SOURCE}",
        "${MODEL_ROOT}/current",
        "${MODEL_REPO}/${MODEL_NAME}",
    ],
)
def test_unresolved_shell_variable_source_is_a_validation_error_not_a_local_path(
    source: str,
) -> None:
    """A source still carrying an expansion resolves to nothing, so it may never pass silently."""
    assert model_source_precision_errors(source)


def test_shell_parameter_default_is_parsed_and_precision_validated() -> None:
    surface = f"vllm serve ${{MODEL_SOURCE:-{BARE_ABLITERATED_NVFP4}}}"
    assert extract_model_sources(surface, source_format=".sh") == [BARE_ABLITERATED_NVFP4]
    assert model_source_precision_errors(BARE_ABLITERATED_NVFP4)


@pytest.mark.parametrize(
    ("surface", "source"),
    [
        (f"vllm serve ${{MODEL_SOURCE:-{MIXED_ABLITERATED_NVFP4}}}", MIXED_ABLITERATED_NVFP4),
        (f'vllm serve "${{MODEL_SOURCE:-{MIXED_ABLITERATED_NVFP4}}}"', MIXED_ABLITERATED_NVFP4),
        (f"--model ${{MODEL_SOURCE:-{MIXED_ABLITERATED_NVFP4}}}", MIXED_ABLITERATED_NVFP4),
        (f"--model=${{MODEL_SOURCE:={MIXED_ABLITERATED_NVFP4}}}", MIXED_ABLITERATED_NVFP4),
        (f"vllm serve ${{MODEL_SOURCE:-{LOCAL_ARTIFACT_PATH}}}", LOCAL_ARTIFACT_PATH),
        (f'vllm serve "{LOCAL_ARTIFACT_PATH}"', LOCAL_ARTIFACT_PATH),
        (f"vllm serve '{LOCAL_ARTIFACT_PATH}'", LOCAL_ARTIFACT_PATH),
    ],
)
def test_resolvable_shell_model_sources_are_extracted_and_accepted(
    surface: str, source: str
) -> None:
    assert extract_model_sources(surface, source_format=".sh") == [source]
    assert model_source_precision_errors(source, requires_mixed_fp8=True) == []


def test_compose_entrypoint_carries_the_model_source() -> None:
    surface = f"""\
services:
  vllm:
    entrypoint: [vllm, serve, {BARE_ABLITERATED_NVFP4}]
"""
    assert extract_model_sources(surface, source_format=".yml") == [BARE_ABLITERATED_NVFP4]
    assert model_source_precision_errors(BARE_ABLITERATED_NVFP4)


def test_compose_entrypoint_and_command_combine_into_one_serve_command() -> None:
    """Compose appends `command` to `entrypoint`, so the source may live in either half."""
    split = f"""\
services:
  vllm:
    entrypoint: [vllm, serve]
    command: [{BARE_ABLITERATED_NVFP4}, --served-model-name, darkstar-qwen38-abliterated-nvfp4]
"""
    assert extract_model_sources(split, source_format=".yml") == [BARE_ABLITERATED_NVFP4]
    assert model_source_precision_errors(BARE_ABLITERATED_NVFP4)

    encoded = f"""\
services:
  vllm:
    entrypoint: vllm serve
    command:
      - {MIXED_ABLITERATED_NVFP4}
      - --served-model-name
      - darkstar-qwen38-abliterated-nvfp4
"""
    assert extract_model_sources(encoded, source_format=".yaml") == [MIXED_ABLITERATED_NVFP4]
    assert model_source_precision_errors(MIXED_ABLITERATED_NVFP4, requires_mixed_fp8=True) == []

    entrypoint_flag = f"""\
services:
  vllm:
    entrypoint: [vllm, serve]
    command: ["--model", "{LOCAL_ARTIFACT_PATH}"]
"""
    assert extract_model_sources(entrypoint_flag, source_format=".yml") == [LOCAL_ARTIFACT_PATH]


def test_compose_shell_c_payload_is_parsed_recursively_as_shell_text() -> None:
    surface = f"""\
services:
  vllm:
    command: [sh, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]
"""
    assert extract_model_sources(surface, source_format=".yml") == [BARE_ABLITERATED_NVFP4]
    assert model_source_precision_errors(BARE_ABLITERATED_NVFP4)


@pytest.mark.parametrize(
    "surface",
    [
        f'command: [sh, -c, "vllm serve {MIXED_ABLITERATED_NVFP4}"]',
        f'command: [bash, -lc, "vllm serve {MIXED_ABLITERATED_NVFP4}"]',
        f'entrypoint: [/bin/sh, -c, "vllm serve {MIXED_ABLITERATED_NVFP4}"]',
        f'command: [sh, -c, "vllm serve ${{MODEL_SOURCE:-{MIXED_ABLITERATED_NVFP4}}}"]',
    ],
)
def test_precision_encoded_shell_c_payloads_are_extracted_and_accepted(surface: str) -> None:
    assert extract_model_sources(surface, source_format=".yaml") == [MIXED_ABLITERATED_NVFP4]
    assert model_source_precision_errors(MIXED_ABLITERATED_NVFP4, requires_mixed_fp8=True) == []


@pytest.mark.parametrize(
    "surface",
    [
        # A shell option that consumes the following token used to hide the `-c` behind its
        # argument, so the payload was never read and a bare NVFP4 source passed silently.
        f'sh -o errexit -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash -O extglob -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash +O extglob -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash -eo pipefail -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash --rcfile /dev/null -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'sh -o errexit -c "vllm serve --model {BARE_ABLITERATED_NVFP4}"',
        # The same launchers written as Compose token lists.
        f'command: [sh, -o, errexit, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        f'command: [bash, -O, extglob, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        f'entrypoint: [/bin/sh, -o, errexit, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        (
            "entrypoint: [bash, -O, extglob, -c]\n"
            f'command: ["vllm serve {BARE_ABLITERATED_NVFP4}"]'
        ),
    ],
)
def test_shell_option_arguments_cannot_hide_a_c_payload(surface: str) -> None:
    """An option argument before `-c` must not smuggle a bare NVFP4 source past the parser."""
    source_format = ".yaml" if surface.startswith(("command:", "entrypoint:")) else ".sh"
    assert extract_model_sources(surface, source_format=source_format) == [
        BARE_ABLITERATED_NVFP4
    ]
    assert model_source_precision_errors(BARE_ABLITERATED_NVFP4)


@pytest.mark.parametrize(
    "surface",
    [
        f'sh -o errexit -c "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        f'bash -O extglob -c "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        f'bash -O extglob -O globstar -lc "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        f'bash --init-file /dev/null -c "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        # `-c` does not end option parsing in any real shell: the payload is the first
        # non-option token, or whatever follows `--`.
        f'bash -c -x "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        f'bash -co errexit "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        f'bash -c -- "vllm serve {MIXED_ABLITERATED_NVFP4}"',
        f'command: [sh, -o, errexit, -c, "vllm serve {MIXED_ABLITERATED_NVFP4}"]',
        f'command: [bash, -O, extglob, -c, "vllm serve {MIXED_ABLITERATED_NVFP4}"]',
        (
            "command: [bash, -O, extglob, -c, "
            f'"vllm serve ${{MODEL_SOURCE:-{MIXED_ABLITERATED_NVFP4}}}"]'
        ),
    ],
)
def test_precision_encoded_payloads_behind_shell_options_are_extracted_and_accepted(
    surface: str,
) -> None:
    source_format = ".yaml" if surface.startswith("command:") else ".sh"
    assert extract_model_sources(surface, source_format=source_format) == [
        MIXED_ABLITERATED_NVFP4
    ]
    assert model_source_precision_errors(MIXED_ABLITERATED_NVFP4, requires_mixed_fp8=True) == []


@pytest.mark.parametrize(
    "surface",
    [
        # An option argument that is missing or is itself an option makes the launcher malformed:
        # where its command text starts is unknowable, so it may not be waved through.
        f'sh -o -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash -O -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash --rcfile -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        # Compact option arguments are not a shell spelling: `sh -oerrexit` still consumes the
        # next token as its option name, which here is the `-c` itself.
        f'sh -oerrexit -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        # Unknown option arity leaves the payload position undecidable.
        f'bash -Z -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'bash --unknown-option -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'sh -- -c "vllm serve {BARE_ABLITERATED_NVFP4}"',
        f'command: [sh, -o, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        f'command: [bash, -O, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        f'command: [sh, -oerrexit, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        f'command: [bash, --unknown-option, -c, "vllm serve {BARE_ABLITERATED_NVFP4}"]',
        "command: [sh, -o, errexit, -c]",
        "command: [bash, -O, extglob, -lc]",
    ],
)
def test_malformed_shell_option_arguments_fail_closed(surface: str) -> None:
    source_format = ".yaml" if surface.startswith("command:") else ".sh"
    with pytest.raises(PublicationSourceParseError):
        extract_model_sources(surface, source_format=source_format)


def test_shell_launchers_without_serve_content_stay_readable() -> None:
    """Fail-closed option scanning only rejects launchers that could still carry a serve command."""
    assert (
        extract_model_sources("command: [bash, -O, extglob, entry.sh]", source_format=".yaml") == []
    )
    assert (
        extract_model_sources(
            "command: [bash, --unknown-option, entry.sh]", source_format=".yaml"
        )
        == []
    )
    assert extract_model_sources(
        f'command: [bash, -O, extglob, entry.sh, --model, "{LOCAL_ARTIFACT_PATH}"]',
        source_format=".yaml",
    ) == [LOCAL_ARTIFACT_PATH]


def test_shell_c_payload_keeps_served_model_name_aliases_excluded() -> None:
    surface = (
        'command: [sh, -c, "vllm serve '
        f'{LOCAL_ARTIFACT_PATH} --served-model-name darkstar-qwen38-abliterated-nvfp4"]'
    )
    assert extract_model_sources(surface, source_format=".yaml") == [LOCAL_ARTIFACT_PATH]


@pytest.mark.parametrize(
    ("surface", "source"),
    [
        (
            "vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Base-NVFP4",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-NVFP4",
        ),
        (
            "vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4",
        ),
        (
            "--model Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4",
            "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4",
        ),
        (
            "--model=HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4",
        ),
        (
            "command: [\"--model\", \"HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4\"]",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4",
        ),
        (
            "command:\n- --model\n- HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-NVFP4",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-NVFP4",
        ),
    ],
)
def test_every_shortened_nvfp4_command_form_is_extracted_and_rejected(
    surface: str, source: str
) -> None:
    source_format = ".yaml" if surface.startswith("command:") else ".sh"
    assert extract_model_sources(surface, source_format=source_format) == [source]
    assert model_source_precision_errors(source, requires_mixed_fp8=True)


@pytest.mark.parametrize(
    "source",
    [
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-NVFP4",
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4",
        "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4",
    ],
)
def test_bare_or_short_nvfp4_server_model_sources_are_rejected(source: str) -> None:
    assert model_source_precision_errors(source)


def test_shortened_mixed_server_source_is_rejected_from_precision_map() -> None:
    source = (
        "HangGlidersRule/"
        "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4"
    )
    errors = model_source_precision_errors(source, requires_mixed_fp8=True)
    assert any("must encode Mixed-FP8" in error for error in errors)


def test_recipe_publication_targets_are_precision_encoded_or_unresolved() -> None:
    required = ledger_mixed_fp8_requirements(_ledger())
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "recipes").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"huggingface_nvfp4:\s*(\S+)", text):
            value = match.group(1)
            if value == "null":
                continue
            rel = path.relative_to(REPO_ROOT)
            errors = precision_encoded_id_errors(value, required.get(value))
            violations.extend(f"{rel}: {error}" for error in errors)
    assert not violations, "recipe publication target violations:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "identifier",
    [
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-NVFP4",
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-NVFP4",
        "Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4",
        "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4",
    ],
)
def test_bare_nvfp4_ids_are_rejected(identifier: str) -> None:
    assert is_bare_nvfp4_id(identifier)
    errors = precision_encoded_id_errors(identifier)
    assert any("precision class" in error for error in errors)


@pytest.mark.parametrize(
    ("identifier", "expected_error"),
    [
        ("Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-W4A4-NVFP4", "conflates W4A16 and W4A4"),
        ("Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8", "must not encode FP8"),
        ("Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-FP8", "spelled Mixed-FP8"),
        ("Darkstar-Qwen3.8-27B-Base-W4A16-NVFP4-Mixed-FP8", "must encode ModelOpt"),
    ],
)
def test_conflated_precision_ids_are_rejected(identifier: str, expected_error: str) -> None:
    errors = precision_encoded_id_errors(identifier)
    assert any(expected_error in error for error in errors), errors


def test_precision_encoded_ids_are_accepted() -> None:
    for identifier in (
        f"{DARKSTAR_HF_NAMESPACE}/{MIXED_BASE_CANDIDATE}",
        f"{DARKSTAR_HF_NAMESPACE}/{W4A4_BASE_CANDIDATE}",
        f"{DARKSTAR_HF_NAMESPACE}/Darkstar-Qwen3.8-27B-Abliterated-BF16",
    ):
        assert not precision_encoded_id_errors(identifier)
        assert not is_bare_nvfp4_id(identifier)


def test_schema_rejects_a_bare_nvfp4_target_repository() -> None:
    ledger = _ledger()
    base = next(p for p in ledger["products"] if p["role"] == "base-modelopt-nvfp4")
    base["target_repository"] = f"{DARKSTAR_HF_NAMESPACE}/Darkstar-Qwen3.8-27B-Base-NVFP4"
    with pytest.raises(jsonschema.ValidationError):
        validate_ledger_schema(ledger, _schema())


def test_schema_requires_every_candidate_to_reserve_a_repository() -> None:
    ledger = _ledger()
    base = next(p for p in ledger["products"] if p["role"] == "base-modelopt-nvfp4")
    del base["candidates"][0]["target_repository"]
    with pytest.raises(jsonschema.ValidationError):
        validate_ledger_schema(ledger, _schema())


def test_target_repository_errors_reject_slot_ids_and_rejected_targets() -> None:
    ledger = _ledger()
    base = next(p for p in ledger["products"] if p["role"] == "base-modelopt-nvfp4")

    slot_target = dict(base)
    slot_target["target_repository"] = (
        f"{DARKSTAR_HF_NAMESPACE}/Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4"
    )
    assert any("precision class" in e for e in target_repository_errors(slot_target))

    # The uniform W4A4 base candidate was built and rejected, so it may never become the target even
    # though it exists.
    rejected_target = dict(base)
    rejected_target["target_repository"] = f"{DARKSTAR_HF_NAMESPACE}/{W4A4_BASE_CANDIDATE}"
    assert any(
        "must name a built, non-rejected candidate" in e
        for e in target_repository_errors(rejected_target)
    )

    # An unresolved ModelOpt product must keep its target null.
    unresolved = dict(base)
    unresolved["target_repository_status"] = "unresolved_pending_precision_winner"
    unresolved["target_repository"] = f"{DARKSTAR_HF_NAMESPACE}/{MIXED_BASE_CANDIDATE}"
    assert any("must be null" in e for e in target_repository_errors(unresolved))


# --- Mixed-FP8 is derived from the precision map, never trusted from the declared names ---


def _mixed_base_product(ledger: dict) -> dict:
    return next(p for p in ledger["products"] if p["role"] == "base-modelopt-nvfp4")


def _shorten_mixed_ids_to_w4a16(ledger: dict) -> dict:
    """Reproduce the reviewer mutation: keep FP8 in the map, drop Mixed-FP8 from the names.

    The candidate keeps serving FP8 self-attention and GDN weights, but its precision class,
    candidate id, reserved repository id, and the product target that resolves to it are all
    shortened to `W4A16-NVFP4`. Every validator must reject this.
    """
    product = _mixed_base_product(ledger)
    candidate = next(
        c for c in product_candidates(product) if c["candidate_id"] == MIXED_BASE_CANDIDATE
    )
    shortened = MIXED_BASE_CANDIDATE.removesuffix("-Mixed-FP8")
    candidate["candidate_id"] = shortened
    candidate["precision_class"] = "W4A16-NVFP4"
    candidate["target_repository"] = f"{DARKSTAR_HF_NAMESPACE}/{shortened}"
    product["target_repository"] = f"{DARKSTAR_HF_NAMESPACE}/{shortened}"
    assert "FP8" in candidate["precision_map"]["self_attention"]
    return ledger


def test_mixed_fp8_requirement_is_derived_from_served_weight_paths() -> None:
    assert RUNTIME_PRECISION_MAP_COMPONENTS == {"kv_cache"}
    assert WEIGHT_PRECISION_MAP_COMPONENTS | RUNTIME_PRECISION_MAP_COMPONENTS == set(
        REQUIRED_PRECISION_MAP_COMPONENTS
    )

    candidates = _all_candidates()
    for candidate_id, candidate in candidates.items():
        assert candidate_requires_mixed_fp8(candidate) == ("Mixed-FP8" in candidate_id)


def test_runtime_kv_fp8_alone_never_implies_mixed_fp8() -> None:
    # Runtime KV metadata is not a served weight path: an FP8 KV cache neither creates the
    # Mixed-FP8 naming requirement nor satisfies it. Runtime BF16 KV stays a separate concern.
    uniform_with_fp8_kv = {
        "language_mlp": "W4A4-NVFP4-g16",
        "lm_head": "BF16",
        "self_attention": "W4A4-NVFP4-g16",
        "gdn_projections": "W4A4-NVFP4-g16",
        "kv_cache": "FP8-e4m3",
        "protected": "BF16",
    }
    assert not precision_map_requires_mixed_fp8(uniform_with_fp8_kv)
    assert not candidate_precision_errors(
        {
            "candidate_id": W4A4_BASE_CANDIDATE,
            "precision_class": "W4A4-NVFP4",
            "precision_map": uniform_with_fp8_kv,
        }
    )


def test_shortened_mixed_candidate_names_are_rejected_by_the_validators() -> None:
    ledger = _shorten_mixed_ids_to_w4a16(_ledger())
    product = _mixed_base_product(ledger)
    candidate = next(
        c for c in product_candidates(product) if c["candidate_id"].endswith("W4A16-NVFP4")
    )

    assert candidate_requires_mixed_fp8(candidate)
    candidate_errors = candidate_precision_errors(candidate)
    assert any("precision_class" in e and "omits Mixed-FP8" in e for e in candidate_errors)
    assert any("candidate_id omits Mixed-FP8" in e for e in candidate_errors)
    assert ledger_candidate_precision_errors(ledger)

    target_errors = ledger_target_repository_errors(ledger)
    assert any("must encode Mixed-FP8" in e for e in target_errors), target_errors
    assert any("must encode Mixed-FP8" in e for e in target_repository_errors(product))


def test_schema_rejects_shortened_mixed_candidate_names() -> None:
    ledger = _shorten_mixed_ids_to_w4a16(_ledger())
    with pytest.raises(jsonschema.ValidationError):
        validate_ledger_schema(ledger, _schema())


def test_shortened_mixed_card_identity_and_serve_ids_are_rejected() -> None:
    ledger = _shorten_mixed_ids_to_w4a16(_ledger())
    required = ledger_mixed_fp8_requirements(ledger)
    product = _mixed_base_product(ledger)
    shortened_repository = str(product["target_repository"])
    shortened_name = shortened_repository.split("/")[-1]

    # The mutated card identity heading and serve example would publish the shortened id.
    assert expected_card_identity_ids(product) == [shortened_name]
    for identifier in (shortened_name, shortened_repository):
        errors = precision_encoded_id_errors(identifier, required.get(identifier))
        assert any("must encode Mixed-FP8" in e for e in errors), errors

    # Absence of the requirement is never silent: the id alone is not enough to clear the rule.
    assert not precision_encoded_id_errors(shortened_name)
    assert precision_encoded_id_errors(shortened_name, requires_mixed_fp8=True)


def test_w4a4_candidate_must_reject_mixed_fp8_naming() -> None:
    ledger = _ledger()
    product = _mixed_base_product(ledger)
    candidate = next(
        c for c in product_candidates(product) if c["candidate_id"] == W4A4_BASE_CANDIDATE
    )
    mixed_name = f"{W4A4_BASE_CANDIDATE}-Mixed-FP8"
    candidate["candidate_id"] = mixed_name
    candidate["precision_class"] = "W4A4-NVFP4-Mixed-FP8"
    candidate["target_repository"] = f"{DARKSTAR_HF_NAMESPACE}/{mixed_name}"

    assert not candidate_requires_mixed_fp8(candidate)
    errors = candidate_precision_errors(candidate)
    assert any("claims FP8 but no served weight path is FP8" in e for e in errors), errors
    assert any("must not encode Mixed-FP8" in e for e in ledger_target_repository_errors(ledger))
    with pytest.raises(jsonschema.ValidationError):
        validate_ledger_schema(ledger, _schema())


def test_precision_encoded_ids_reject_mixed_fp8_when_no_weight_path_is_fp8() -> None:
    errors = precision_encoded_id_errors(
        f"{DARKSTAR_HF_NAMESPACE}/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8",
        requires_mixed_fp8=False,
    )
    assert any("must not encode Mixed-FP8" in e for e in errors), errors


def test_contract_and_process_document_state_the_derivation_rule() -> None:
    derivation = _contract()["candidate_precision_encoding"]["mixed_fp8_derivation"]
    assert set(derivation["weight_components"]) == set(WEIGHT_PRECISION_MAP_COMPONENTS)
    assert set(derivation["runtime_components_excluded"]) == set(RUNTIME_PRECISION_MAP_COMPONENTS)
    assert "DERIVED" in derivation["rule"]

    text = (REPO_ROOT / "docs" / "darkstar-four-product-release-process.md").read_text(
        encoding="utf-8"
    )
    assert "`Mixed-FP8` is derived, not declared." in text
    assert "runtime KV metadata is never evidence of the mixed class" in text


def test_process_document_forbids_omitting_precision_wording() -> None:
    text = (REPO_ROOT / "docs" / "darkstar-four-product-release-process.md").read_text(
        encoding="utf-8"
    )
    assert "No actual identifier may drop precision wording" in text
    assert "may omit redundant format wording" not in text

    contract = _contract()
    actual = contract["candidate_precision_encoding"]["actual_id_encoding"]
    assert set(actual["target_repository_states"]) == {
        "resolved",
        "unresolved_pending_precision_winner",
    }
    assert "model_card_identity_heading" in actual["applies_to"]
    assert "serve_model_source" in actual["applies_to"]


# --- Mixed W4A16/FP8 GPQA evidence: verified full-denominator, selected and promoted ---

GPQA_EVIDENCE_PATH = (
    REPO_ROOT
    / "models"
    / "qwen3.8-27b-r3"
    / "results"
    / "gpqa-base-modelopt-w4a16-nvfp4-mixed-fp8.json"
)

_REQUIRED_HASH_KEYS = (
    "summary_sha256",
    "journal_sha256",
    "artifact_success_sha256",
    "recipe_sha256",
    "operator_snapshot_compose_sha256",
    "vllm_image",
)


@pytest.mark.private_source_only
def test_mixed_candidate_gpqa_evidence_recorded() -> None:
    assert GPQA_EVIDENCE_PATH.exists()
    evidence = load_json(GPQA_EVIDENCE_PATH)
    result = evidence["result"]
    assert result["correct"] == 153
    assert result["denominator"] == 198
    assert result["terminal_parseable"] == 198
    assert result["timeouts"] == 0
    assert result["parse_errors"] == 0
    assert result["errors"] == 0
    assert abs(result["accuracy_full_denominator"] - 153 / 198) < 1e-6
    protocol = evidence["protocol"]
    assert protocol["thinking"] is False
    assert protocol["temperature"] == 1.0
    assert protocol["top_p"] == 0.95
    assert protocol["top_k"] == 20
    assert protocol["workers"] == 4
    assert protocol["output_cap"] is None
    for key in _REQUIRED_HASH_KEYS:
        assert evidence["immutable_hashes"][key]


@pytest.mark.private_source_only
def test_mixed_candidate_gpqa_verified_and_promoted() -> None:
    ledger = _ledger()
    base = next(p for p in ledger["products"] if p["role"] == "base-modelopt-nvfp4")
    statuses = product_gate_statuses(base)
    assert statuses["gpqa_matched_full_denominator"] == "verified"

    mixed = next(
        c
        for c in product_candidates(base)
        if c["candidate_id"] == "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8"
    )
    # The mixed candidate is the selected/promoted Base ModelOpt product (chosen on throughput).
    assert not candidate_promotion_blocked(mixed)
    assert candidate_promoted(mixed)
    assert candidate_selection(mixed) == "selected"
    assert mixed["immutable_hashes"] == load_json(GPQA_EVIDENCE_PATH)["immutable_hashes"]
    assert mixed["gpqa"]["correct"] == 153
    # The base product is build-complete and publication-ready after the immutable tag is cut.
    assert product_build_complete(base)
    assert product_publication_ready(base)
    # The gate points at the committed external operator evidence.
    gate = next(g for g in base["gates"] if g["id"] == "gpqa_matched_full_denominator")
    assert (REPO_ROOT / gate["evidence_path"]).exists()


def _all_keys(obj: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(str(key).lower())
            keys |= _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


@pytest.mark.private_source_only
def test_gpqa_evidence_excludes_restricted_question_data() -> None:
    evidence = load_json(GPQA_EVIDENCE_PATH)
    # Aggregates and hashes only: no key may carry per-question text/answers/journal payload.
    banned_keys = {
        "question_text",
        "questions_text",
        "answer_key",
        "answer_keys",
        "choices",
        "responses",
        "per_question",
        "journal_lines",
        "journal",
    }
    assert not (_all_keys(evidence) & banned_keys)
    # The journal is referenced only by hash, never inlined.
    assert "journal_sha256" in evidence["immutable_hashes"]
    assert "excluded_from_commit" in evidence


# --- Exact per-product completed metrics (regression against value drift) ---

RESULTS_DIR = REPO_ROOT / "models" / "qwen3.8-27b-r3" / "results"


@pytest.mark.private_source_only
def test_base_bf16_exact_metrics() -> None:
    ev = load_json(RESULTS_DIR / "gpqa-base-bf16.json")
    r = ev["result"]
    assert (r["correct"], r["denominator"], r["terminal_parseable"]) == (157, 198, 198)
    assert (r["timeouts"], r["parse_errors"], r["errors"]) == (0, 0, 0)
    assert abs(r["accuracy_full_denominator"] - 157 / 198) < 1e-6
    assert ev["protocol"]["thinking"] is False

    perf = load_json(RESULTS_DIR / "performance-base-bf16.json")
    assert perf["winner"]["tok_s"] == 130.158
    assert perf["winner"]["mtp_depth"] == 8
    assert perf["harness"]["scheduler_max_num_batched_tokens"] == 65536


@pytest.mark.private_source_only
def test_abliterated_bf16_exact_metrics() -> None:
    ev = load_json(RESULTS_DIR / "gpqa-abliterated-bf16.json")
    r = ev["result"]
    assert (r["correct"], r["denominator"], r["terminal_parseable"]) == (146, 198, 198)
    assert (r["timeouts"], r["parse_errors"], r["errors"]) == (0, 0, 0)
    assert abs(r["accuracy_full_denominator"] - 146 / 198) < 1e-6
    assert ev["edit_delta_vs_base_bf16"]["delta_questions"] == -11

    ab = load_json(RESULTS_DIR / "abliteration-eval-abliterated-bf16.json")["result"]
    assert ab["harmful_prompts"] == 200
    assert ab["harmful_compliance"] == 200
    assert ab["harmful_refusals"] == 0
    assert ab["safe_over_refusals"] == 0
    assert ab["errors"] == 0

    perf = load_json(RESULTS_DIR / "performance-abliterated-bf16.json")
    assert perf["winner"]["tok_s"] == 144.502
    assert perf["winner"]["mtp_depth"] == 11
    assert perf["harness"]["scheduler_max_num_batched_tokens"] == 16384


@pytest.mark.private_source_only
def test_base_modelopt_exact_metrics_and_candidate_comparison() -> None:
    ev = load_json(RESULTS_DIR / "gpqa-base-modelopt-w4a16-nvfp4-mixed-fp8.json")
    r = ev["result"]
    assert (r["correct"], r["denominator"]) == (153, 198)

    perf = load_json(RESULTS_DIR / "performance-nvfp4.json")
    assert perf["winner"]["tok_s"] == 203.636
    assert perf["winner"]["mtp_depth"] == 4
    assert perf["harness"]["scheduler_max_num_batched_tokens"] == 32768

    comparison = {c["candidate_id"]: c for c in perf["candidate_comparison"]}
    mixed = comparison["Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8"]
    w4a4 = comparison["Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4"]
    assert mixed["selection"] == "selected" and mixed["tok_s"] == 203.636
    assert w4a4["selection"] == "rejected" and w4a4["tok_s"] == 129.441
    assert perf["quantization_delta_vs_base_bf16"]["delta_questions"] == -4


@pytest.mark.private_source_only
def test_abliterated_modelopt_exact_metrics_and_runtime_evidence() -> None:
    gpqa = load_json(
        RESULTS_DIR / "gpqa-abliterated-modelopt-w4a16-nvfp4-mixed-fp8.json"
    )
    result = gpqa["result"]
    assert (result["correct"], result["denominator"], result["terminal_parseable"]) == (
        148,
        198,
        198,
    )
    assert (result["timeouts"], result["parse_errors"], result["errors"]) == (0, 0, 0)
    assert abs(result["accuracy_full_denominator"] - 148 / 198) < 1e-10
    assert gpqa["protocol"]["thinking"] is False
    assert gpqa["immutable_hashes"]["summary_sha256"] == (
        "d8d0b5c0de686846338ce89e9a55456baec0550bbad765ccc65e9fa57380b818"
    )
    assert gpqa["immutable_hashes"]["journal_sha256"] == (
        "9bb4913202977bad204ebde8d2e31e8357a3308f3b77c44539ef3977a2c6e813"
    )

    behavior_evidence = load_json(
        RESULTS_DIR
        / "abliteration-eval-abliterated-modelopt-w4a16-nvfp4-mixed-fp8.json"
    )
    behavior = behavior_evidence["result"]
    assert (behavior["terminal_responses"], behavior["terminal_denominator"]) == (283, 283)
    assert behavior["harmful_compliance"] == 200
    assert behavior["safe_over_refusals"] == 0
    assert behavior["errors"] == 0
    assert behavior_evidence["immutable_hashes"]["summary_sha256"] == (
        "d814eac6eef86cb32c891d5c3b1765be806cb0fb634173080cd5df46ea9f9233"
    )
    assert behavior_evidence["immutable_hashes"]["journal_sha256"] == (
        "7b6ddf556ab3afc1f8582041d7b723dbfeaeb24b02b8bd27562b0c9928a37d4f"
    )

    perf = load_json(
        RESULTS_DIR / "performance-abliterated-modelopt-w4a16-nvfp4-mixed-fp8.json"
    )
    assert perf["mtp_sweep"]["headline_peak"] == {"mtp_depth": 8, "tok_s": 251.316}
    confirmation = perf["mtp_sweep"]["confirmation"]
    assert confirmation["sequence"] == "10->8->8->10"
    assert confirmation["mtp8_mean_tok_s"] == 250.86164296979206
    assert confirmation["mtp10_mean_tok_s"] == 251.8893355735767
    assert perf["winner"]["mtp_depth"] == 10
    assert (
        confirmation["confirmation_evidence_sha256"]
        == "6e52a5ad4f87a8b12866e0939c2d2024701172d8b5a56a7839ce00738f1a3ac9"
    )

    snapshots = {
        product["product_id"]: product
        for product in load_json(RESULTS_DIR / "serving-runtime-snapshots.json")["products"]
    }
    snapshot = snapshots["Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4"]
    assert snapshot["inspect_sha256"] == (
        "f1e94a763d5bb71de0a9991b2c1211db6ae6c3294dbf8a03b38f53af95235222"
    )
    assert snapshot["logs_sha256"] == (
        "4ffe8666015e4c4f4f45cd31c84e4a2e056a0575dfe3aa45e32e0011012d2582"
    )
    assert snapshot["operator_snapshot_compose_sha256"] == (
        "85ba68155418dad7387219f62889def88c62a0e2ca35d15e3f83d62879077088"
    )
    assert snapshot["tracked_serve_compose_sha256"] == (
        "5434c2a99bdadce512bd87b65c30f830c21fc2eae647182ffa89e77b174833cc"
    )

    manifest = load_json(
        RESULTS_DIR / "manifests" / "abliterated-modelopt-mixed-manifest.json"
    )
    assert manifest["artifact_path"] == (
        "/d/model-forge/artifacts/Qwen3.8-27B-abliterated-performance-mixed-modelopt"
    )
    assert manifest["success_marker_sha256"] == (
        "3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79"
    )
    assert manifest["manifest_sha256"] == (
        "642dbbe89b085a2daf5119c37c0496576a475ed64c36653fc993c04abaf2ca9f"
    )


CAPACITY_PATH = RESULTS_DIR / "serving-capacity-profiles.json"
PRODUCT4_CONFIRMATION_SHA256 = (
    "6e52a5ad4f87a8b12866e0939c2d2024701172d8b5a56a7839ce00738f1a3ac9"
)
PRODUCT4_CORRECTNESS_SHA256 = (
    "4c88632efcfc736518a66351c95735eb2f9ff7ce79496d050a53d171beaf4613"
)


def _product4_capacity() -> dict:
    products = load_json(CAPACITY_PATH)["products"]
    return next(
        p
        for p in products
        if p["product_id"] == "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
    )


def test_product4_capacity_evidence_is_a_real_per_concurrency_sweep() -> None:
    """The serving_capacity_profile gate must rest on measured per-concurrency cells.

    Correctness gates alone do not satisfy the gate, so the committed evidence carries the matched
    4K/16K/48K cells at concurrency 1 and 2 from the selected MTP10 confirmation, with the request
    counts, repeats, aggregate throughput, latency, and zero-failure facts the operator measured.
    """
    sweep = _product4_capacity()["concurrency_sweep"]
    assert sweep["source_evidence_sha256"] == PRODUCT4_CONFIRMATION_SHA256
    assert sweep["selected_mtp_depth"] == 10
    assert sweep["concurrency_levels"] == [1, 2]
    assert sweep["prompt_chars"] == [4000, 16000, 48000]
    assert sweep["generated_tokens_per_request"] == 512
    assert sweep["repeats_per_cell"] == 2
    assert sweep["finish_reasons"] == ["length"]
    assert sweep["failed_requests"] == 0
    assert sweep["fatal_markers"] == 0

    passes = sweep["passes"]
    assert [p["pass"] for p in passes] == [1, 4]
    for entry in passes:
        assert entry["mtp_depth"] == 10
        assert entry["fatal_markers"] == 0
        cells = entry["cells"]
        assert {(c["prompt_chars"], c["concurrency"]) for c in cells} == {
            (chars, concurrency) for chars in (4000, 16000, 48000) for concurrency in (1, 2)
        }
        for cell in cells:
            repeats = cell["repeats"]
            concurrency = cell["concurrency"]
            assert repeats == sweep["repeats_per_cell"]
            assert cell["requests_per_repeat"] == concurrency
            assert cell["requests_total"] == repeats * concurrency
            assert len(cell["aggregate_tps_per_repeat"]) == repeats
            assert len(cell["wall_seconds_per_repeat"]) == repeats
            assert len(cell["request_latency_seconds"]) == cell["requests_total"]
            assert cell["failed_requests"] == 0
            assert all(value > 0 for value in cell["aggregate_tps_per_repeat"])
            assert all(value > 0 for value in cell["request_latency_seconds"])
            measured = cell["aggregate_tps_per_repeat"]
            assert abs(cell["mean_aggregate_tps"] - sum(measured) / len(measured)) < 1e-9

    prompt_tokens = {
        (cell["prompt_chars"], cell["prompt_tokens"])
        for entry in passes
        for cell in entry["cells"]
    }
    assert prompt_tokens == {(4000, 738), (16000, 2653), (48000, 7758)}


def test_product4_capacity_exact_measured_throughput() -> None:
    passes = {p["pass"]: p for p in _product4_capacity()["concurrency_sweep"]["passes"]}
    assert passes[1]["single_stream_mean_tok_s"] == 250.22256467790066
    assert passes[4]["single_stream_mean_tok_s"] == 253.5561064692527

    def mean(pass_number: int, prompt_chars: int, concurrency: int) -> float:
        cell = next(
            c
            for c in passes[pass_number]["cells"]
            if c["prompt_chars"] == prompt_chars and c["concurrency"] == concurrency
        )
        return float(cell["mean_aggregate_tps"])

    assert mean(1, 4000, 1) == 193.41082122336513
    assert mean(1, 4000, 2) == 352.31310359430586
    assert mean(1, 16000, 1) == 183.61103127076737
    assert mean(1, 16000, 2) == 328.39312383448373
    assert mean(1, 48000, 1) == 157.57874524151805
    assert mean(1, 48000, 2) == 286.02856290296376
    assert mean(4, 4000, 1) == 195.485500541494
    assert mean(4, 4000, 2) == 356.56438426714027
    assert mean(4, 16000, 1) == 183.13440752399822
    assert mean(4, 16000, 2) == 343.7986885780714
    assert mean(4, 48000, 1) == 157.88871081639138
    assert mean(4, 48000, 2) == 284.464947088421


def test_product4_capacity_semantic_tests_and_gate_evidence_agree() -> None:
    product = _product4_capacity()
    correctness = product["serving_correctness"]
    gates = correctness["gates"]
    assert correctness["evidence_sha256"] == PRODUCT4_CORRECTNESS_SHA256
    for name in ("tools", "strict_json", "vision", "prefix_cache", "sustained_load"):
        assert gates[name] == "pass"
    assert gates["sustained_load_requests"] == 20
    assert gates["sustained_load_concurrency"] == 2
    assert gates["sustained_load_failures"] == 0
    assert gates["sustained_load_mean_latency_seconds"] > 0
    assert gates["sustained_load_p95_latency_seconds"] >= gates["sustained_load_mean_latency_seconds"]
    assert product["fatal_markers"] == 0

    gate = next(
        g
        for g in _product_by_role("abliterated-modelopt-nvfp4")["gates"]
        if g["id"] == "serving_capacity_profile"
    )
    assert gate["status"] == "verified"
    assert gate["evidence_path"] == "models/qwen3.8-27b-r3/results/serving-capacity-profiles.json"
    # The gate may no longer describe correctness gates alone: it must name the concurrency evidence.
    assert "concurrency 1 and 2" in gate["evidence"]
    assert PRODUCT4_CONFIRMATION_SHA256[:8] in gate["evidence"]


def test_product4_model_card_renders_every_measured_capacity_cell_to_three_decimals() -> None:
    card = (
        REPO_ROOT / "models" / "qwen3.8-27b-r3" / "model-card" / "nvfp4.md"
    ).read_text(encoding="utf-8")
    for entry in _product4_capacity()["concurrency_sweep"]["passes"]:
        for cell in entry["cells"]:
            displayed_tps = f"{cell['mean_aggregate_tps']:.3f}"
            assert displayed_tps in card, (
                f"nvfp4.md omits the measured {cell['prompt_chars']}-char "
                f"c{cell['concurrency']} cell from pass {entry['pass']}"
            )


def _iter_hash_prefix_values(obj: object) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and str(key).endswith("_prefix"):
                out.append((str(key), value))
            out.extend(_iter_hash_prefix_values(value))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_hash_prefix_values(item))
    return out


@pytest.mark.parametrize(
    "filename",
    [
        "serving-runtime-snapshots.json",
        "gpqa-base-bf16.json",
        "gpqa-abliterated-bf16.json",
        "abliteration-eval-abliterated-bf16.json",
    ],
)
@pytest.mark.private_source_only
def test_hash_prefixes_are_labelled_and_never_padded_to_a_full_digest(filename: str) -> None:
    """A recorded prefix must be a short hex prefix, never a fabricated full 64-char digest."""
    data = load_json(RESULTS_DIR / filename)
    prefixes = _iter_hash_prefix_values(data)
    assert prefixes, f"{filename} declares no *_prefix hash fields"
    for key, value in prefixes:
        assert re.fullmatch(r"[0-9a-f]+", value), f"{filename}:{key}={value!r} is not lowercase hex"
        assert len(value) < 64, f"{filename}:{key} is a full digest, not a prefix"


@pytest.mark.private_source_only
def test_snapshot_prefixes_match_the_immutable_facts() -> None:
    snaps = {p["product_id"]: p for p in load_json(RESULTS_DIR / "serving-runtime-snapshots.json")["products"]}
    base = snaps["Darkstar-Qwen3.8-27B-Base-BF16"]["snapshot_hash_prefixes"]
    assert base["inspect_sha256_prefix"] == "c70d19ec"
    assert base["logs_sha256_prefix"] == "6731e6f1"
    assert base["operator_snapshot_compose_sha256_prefix"] == "16936193"

    abl = snaps["Darkstar-Qwen3.8-27B-Abliterated-BF16"]["snapshot_hash_prefixes"]
    assert abl["inspect_sha256_prefix"] == "fca3"
    assert abl["logs_sha256_prefix"] == "7b126"
    assert abl["operator_snapshot_compose_sha256_prefix"] == "4665"


def test_no_evidence_field_conflates_snapshot_and_tracked_compose_hashes() -> None:
    """Compose digests must say which file they hash.

    Product 4 owns two different Compose files — the operator's live host snapshot and this
    repository's deterministic rendering — have different bytes and meanings. An unqualified
    `serve_compose_sha256`/`compose_sha256` field cannot say which of them it means, so it may not
    reappear anywhere in the evidence.
    """
    banned = {"serve_compose_sha256", "compose_sha256", "compose_sha256_prefix"}
    offenders: list[str] = []
    for path in sorted(RESULTS_DIR.rglob("*.json")) + [LEDGER_PATH]:
        keys = _all_keys(load_json(path))
        for key in sorted(banned & keys):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {key}")
    assert not offenders, "ambiguous compose hash fields:\n" + "\n".join(offenders)


def test_tracked_serve_compose_hash_is_verified_from_current_bytes() -> None:
    """Every recorded ``tracked_serve_compose_sha256`` must hash the checked-in Compose right now.

    The tracked digest is only meaningful if it is verified against the actual file bytes rather than
    copied as a literal. It must also stay distinct from the operator snapshot digest, which hashes a
    different (host-private) file.
    """
    import hashlib

    compose = REPO_ROOT / "containers" / "serve" / "darkstar-qwen38-abliterated-nvfp4.yml"
    actual = hashlib.sha256(compose.read_bytes()).hexdigest()
    public_attestation = REPO_ROOT / "PUBLIC_EXPORT_MANIFEST.json"
    if public_attestation.is_file():
        records = load_json(public_attestation)["files"]
        record = next(
            item
            for item in records
            if item["output_path"]
            == "containers/serve/darkstar-qwen38-abliterated-nvfp4.yml"
        )
        assert record["transform_id"].startswith(
            "sanitize_and_validate_serve_profile:"
        )
        assert record["output_sha256"] == actual
        assert "${PUBLIC_ARTIFACT_PATH}" in compose.read_text(encoding="utf-8")
        return

    sha256_re = re.compile(r"^[0-9a-f]{64}$")

    def _dicts_with_tracked_hash(obj: object) -> list[dict]:
        found: list[dict] = []
        if isinstance(obj, dict):
            value = obj.get("tracked_serve_compose_sha256")
            # Skip the glossary entry whose value documents the field rather than recording a digest.
            if isinstance(value, str) and sha256_re.fullmatch(value):
                found.append(obj)
            for child in obj.values():
                found.extend(_dicts_with_tracked_hash(child))
        elif isinstance(obj, list):
            for item in obj:
                found.extend(_dicts_with_tracked_hash(item))
        return found

    carriers = []
    for path in (RESULTS_DIR / "serving-runtime-snapshots.json", LEDGER_PATH):
        carriers.extend(_dicts_with_tracked_hash(load_json(path)))

    assert carriers, "no tracked_serve_compose_sha256 recorded anywhere"
    for carrier in carriers:
        assert carrier["tracked_serve_compose_sha256"] == actual, (
            "recorded tracked_serve_compose_sha256 does not hash the current Compose bytes"
        )
        # Wherever both digests are recorded together they must stay distinct.
        if "operator_snapshot_compose_sha256" in carrier:
            assert carrier["operator_snapshot_compose_sha256"] != actual
