"""Loaders and validators for the Darkstar four-product release contract and per-family
publication-readiness ledgers.

The ledger JSON is the machine-readable source of truth. Human-readable statuses are rendered
into the Git-tracked benchmark matrices and model cards; this module provides the deterministic
checks that keep those renderings honest and prevent gate weakening.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jsonschema
import yaml

CANONICAL_ROLES: tuple[str, ...] = (
    "base-bf16",
    "base-modelopt-nvfp4",
    "abliterated-bf16",
    "abliterated-modelopt-nvfp4",
)

VALID_STATUSES: tuple[str, ...] = (
    "verified",
    "in_progress",
    "missing",
    "rejected_historical",
    "not_applicable",
)

# Product lifecycle states. These describe *where a product sits between build and public release*
# and are orthogonal to the per-gate status vocabulary above. A product is `locally_complete_unpublished`
# when every applicable *build* gate is verified but at least one *publication-only* gate is still
# open; `in_progress` while any build gate is still missing/in_progress; `published` once every
# applicable gate (build and publication) is verified.
LIFECYCLE_STATES: tuple[str, ...] = (
    "in_progress",
    "locally_complete_unpublished",
    "published",
)

# Publication-only gates gate *public release*, not the local build. A locally complete product may
# legitimately leave these open (nothing has been uploaded, no release tag cut, and license/commit/tag
# placeholders in the card are only resolved at publication). Every other gate describes the built
# artifact itself and must be verified before a product is build-complete.
PUBLICATION_ONLY_GATES: frozenset[str] = frozenset(
    {
        "model_card_final",
        "publication_targets_hf_ghcr",
        "clean_download_boot_smoke",
        "release_tag",
    }
)

# Roles whose candidates/products must encode an explicit NVFP4 activation/recipe precision class.
MODELOPT_NVFP4_ROLES: tuple[str, ...] = (
    "base-modelopt-nvfp4",
    "abliterated-modelopt-nvfp4",
)

# A ModelOpt NVFP4 candidate must declare a complete precision map: no component may be omitted.
REQUIRED_PRECISION_MAP_COMPONENTS: frozenset[str] = frozenset(
    {
        "language_mlp",
        "lm_head",
        "self_attention",
        "gdn_projections",
        "kv_cache",
        "protected",
    }
)

# Components whose precision describes a served *weight* path. Whether an artifact is mixed FP8 is
# derived from these alone. `kv_cache` is runtime state, not a weight path: a BF16 or FP8 KV cache
# never makes an artifact Mixed-FP8, so runtime KV metadata can neither create nor satisfy the
# naming requirement.
WEIGHT_PRECISION_MAP_COMPONENTS: frozenset[str] = frozenset(
    {
        "language_mlp",
        "lm_head",
        "self_attention",
        "gdn_projections",
        "protected",
    }
)
RUNTIME_PRECISION_MAP_COMPONENTS: frozenset[str] = (
    REQUIRED_PRECISION_MAP_COMPONENTS - WEIGHT_PRECISION_MAP_COMPONENTS
)

# Diffusion/DiT architecture (contract v2). A DiT composite has a different anatomy: the
# transformer blocks, the (possibly quantized) attention projections inside them, the text
# encoder, and the VAE are the component boundaries. There is no lm_head, language MLP, GDN
# projection, or KV-cache component; prefix-KV reuse is runtime state on the text encoder and is
# deliberately not a declared component.
DIFFUSION_PRECISION_MAP_COMPONENTS: frozenset[str] = frozenset(
    {
        "dit_blocks",
        "protected_blocks",
        "attention",
        "text_encoder",
        "vae",
    }
)
DIFFUSION_WEIGHT_PRECISION_MAP_COMPONENTS: frozenset[str] = frozenset(
    {
        "dit_blocks",
        "attention",
        "text_encoder",
        "vae",
    }
)

# Hugging Face namespace that owns every actual Darkstar weight repository.
DARKSTAR_HF_NAMESPACE = "HangGlidersRule"

# Unchanged upstream checkpoints a family may name as its resolved base-bf16 or image-base-bf16
# reference. The reference is measured, never owned or republished; every derivative still follows
# the namespace and precision rules. Each entry appears in the v3 ledger schema's target_repository
# pattern. Entries that predate v3 also appear in the v2 pattern.
UPSTREAM_BF16_REFERENCES: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Reranker-4B",
        "Qwen/Qwen-Image-2.1",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "nvidia/Nemotron-3.5-Rerank-8B-BF16",
        "nvidia/Nemotron-3.5-Embed-8B-BF16",
    }
)
_UPSTREAM_BF16_ROLES: frozenset[str] = frozenset({"base-bf16", "image-base-bf16"})

# A product-level publication target is either resolved to one concrete, precision-encoded
# repository id, or explicitly unresolved until a precision winner is selected. There is no third
# option: a bare release-slot id may never stand in for an unresolved target.
TARGET_REPOSITORY_STATUSES: tuple[str, ...] = (
    "resolved",
    "unresolved_pending_precision_winner",
)

_DARKSTAR_ID = re.compile(r"Darkstar-[A-Za-z0-9._-]*[A-Za-z0-9]")
_DARKSTAR_REPOSITORY_ID = re.compile(
    rf"{DARKSTAR_HF_NAMESPACE}/Darkstar-[A-Za-z0-9._-]*[A-Za-z0-9]"
)
_SERVE_MODEL_ALIAS = re.compile(
    r"darkstar-[a-z0-9]+-(?:base|abliterated)-(?:bf16|nvfp4)"
)
_SERVED_MODEL_NAME_FLAG = "--served-model-name"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_ROOT = _REPO_ROOT / "contracts" / "darkstar-release"
_CONTRACT_MAJORS = frozenset({1, 2, 3})
_KV_CACHE_EFFECT_GATE = "kv_cache_effect_measured"
_FP8_KV_STATUSES = frozenset({"verified", "in_progress", "missing"})
_QUALITY_GATE = "quality_matched_full_denominator"
_MARKDOWN_FENCE = re.compile(r"```(?P<language>[A-Za-z0-9_-]*)\n(?P<body>.*?)```", re.DOTALL)
_MARKDOWN_H1 = re.compile(r"^#\s+(.*)$", re.MULTILINE)

# Frozen minimum gate set. A new contract version may add or strengthen gates but must never drop
# one of these. tests/test_release_contract.py asserts the contract remains a superset.
FROZEN_REQUIRED_GATES: frozenset[str] = frozenset(
    {
        "provenance_ownership",
        "artifact_manifest",
        "recipe_edit_manifest",
        "artifact_validation",
        "abliteration_pass",
        "modelopt_candidate_comparison",
        "nvfp4_tensor_scale_validation",
        "performance_profile",
        "serving_capacity_profile",
        "gpqa_matched_full_denominator",
        "behavior_refusal_eval",
        "serve_profile_frozen",
        "model_card_final",
        "publication_targets_hf_ghcr",
        "clean_download_boot_smoke",
        "release_tag",
        "no_inherited_unverified_results",
    }
)


def load_json(path: Path) -> dict[str, Any]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def contract_paths_for(ledger: dict[str, Any]) -> tuple[Path, Path]:
    """Return the contract and ledger schema for a ledger's ``contract_version``.

    Major 1 selects ``v1/``, major 2 selects ``v2/``, and major 3 selects ``v3/``. Any other
    major version is an error.
    """
    version = ledger["contract_version"]
    if not isinstance(version, str):
        raise ValueError(f"Unknown contract version {version!r}")
    major_text = version.split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError as exc:
        raise ValueError(f"Unknown contract version {version!r}") from exc
    if major not in _CONTRACT_MAJORS:
        raise ValueError(f"Unknown contract version {version!r}")
    directory = _CONTRACT_ROOT / f"v{major}"
    return directory / "contract.json", directory / "ledger.schema.json"


def validate_ledger_schema(ledger: dict[str, Any], schema: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the ledger does not match the schema."""
    jsonschema.validate(instance=ledger, schema=schema)


def contract_gate_ids(contract: dict[str, Any]) -> list[str]:
    return [str(gate["id"]) for gate in contract["gates"]]


def gate_applies_to(contract: dict[str, Any], gate_id: str, role: str) -> bool:
    for gate in contract["gates"]:
        if gate["id"] == gate_id:
            return role in gate["applies_to"]
    raise KeyError(f"Unknown gate id: {gate_id}")


def product_gate_statuses(product: dict[str, Any]) -> dict[str, str]:
    return {str(gate["id"]): str(gate["status"]) for gate in product["gates"]}


def product_publication_ready(product: dict[str, Any]) -> bool:
    """A product is publication-ready only when every applicable gate is verified."""
    applicable = [
        str(gate["status"])
        for gate in product["gates"]
        if gate["status"] != "not_applicable"
    ]
    return bool(applicable) and all(status == "verified" for status in applicable)


def product_build_gate_statuses(product: dict[str, Any]) -> list[str]:
    """Applicable *build* gate statuses (publication-only gates excluded)."""
    return [
        str(gate["status"])
        for gate in product["gates"]
        if gate["status"] != "not_applicable" and str(gate["id"]) not in PUBLICATION_ONLY_GATES
    ]


def product_build_complete(product: dict[str, Any]) -> bool:
    """A product is build-complete (a finished local artifact) when every applicable build gate is
    verified. Publication-only gates are ignored, so a locally complete but unpublished product is
    build-complete even though it is not yet publication-ready."""
    build = product_build_gate_statuses(product)
    return bool(build) and all(status == "verified" for status in build)


def expected_lifecycle(product: dict[str, Any]) -> str:
    """The lifecycle a product's gate statuses imply, independent of its declared `lifecycle`."""
    if product_publication_ready(product):
        return "published"
    if product_build_complete(product):
        return "locally_complete_unpublished"
    return "in_progress"


def product_lifecycle_errors(product: dict[str, Any]) -> list[str]:
    """Return inconsistencies between a product's declared `lifecycle` and its gate statuses.

    The declared lifecycle is not free text: it must match what the gates prove. A product that
    claims `locally_complete_unpublished` while a build gate is still `in_progress`, or that claims
    `in_progress` while every build gate is verified, is a contract violation. `publication_claim`
    is only ever true for a `published` lifecycle.
    """
    errors: list[str] = []
    pid = str(product.get("product_id"))
    declared = str(product.get("lifecycle", ""))
    if declared not in LIFECYCLE_STATES:
        errors.append(f"{pid}: lifecycle {declared!r} is not one of {LIFECYCLE_STATES}")
        return errors
    implied = expected_lifecycle(product)
    if declared != implied:
        errors.append(
            f"{pid}: declared lifecycle {declared!r} disagrees with gate statuses "
            f"(implied {implied!r})"
        )
    if bool(product.get("publication_claim")) and declared != "published":
        errors.append(f"{pid}: publication_claim is true but lifecycle is {declared!r}")
    return errors


def ledger_lifecycle_errors(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for product in ledger["products"]:
        errors.extend(product_lifecycle_errors(product))
    return errors


def product_candidates(product: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = product.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"candidates must be a list in {product.get('product_id')}")
    return list(candidates)


def precision_map_requires_mixed_fp8(
    precision_map: dict[str, Any],
    weight_components: frozenset[str] | None = None,
) -> bool:
    """True when a precision map puts FP8 on any served weight path.

    Only the weight-path components of the candidate's architecture are consulted, so the answer
    is a property of the served weights and never of runtime KV metadata. This is the authority
    for the `Mixed-FP8` naming requirement: enforcement derives the requirement from the map
    instead of trusting whatever the declared precision class or ids happen to spell, which is
    what makes it fail closed. `weight_components` defaults to the text-model set; diffusion
    candidates pass the DiT set.
    """
    if weight_components is None:
        weight_components = WEIGHT_PRECISION_MAP_COMPONENTS
    weights = " ".join(
        str(value)
        for component, value in precision_map.items()
        if component in weight_components
    )
    return "FP8" in weights


def candidate_architecture(candidate: dict[str, Any]) -> str:
    """Architecture family of a candidate: `text` (default, causal LM) or `diffusion` (DiT).

    The field is optional in the ledger: its absence means text, so every v1 ledger and every
    existing caller keeps working unchanged. A candidate whose precision_map carries DiT
    anatomy (e.g. `dit_blocks`) without declaring `architecture: "diffusion"` is rejected
    downstream by `candidate_precision_errors` rather than silently re-validated as text with
    a misleading error message.
    """
    arch = str(candidate.get("architecture", "text"))
    if arch not in ("text", "diffusion"):
        raise ValueError(
            f"{candidate.get('candidate_id', '<unnamed>')}: unknown architecture {arch!r} "
            "(expected 'text' or 'diffusion')"
        )
    return arch


def diffusion_anatomy_undeclared(candidate: dict[str, Any]) -> bool:
    """True when a candidate's precision_map looks DiT-shaped but no architecture is declared.

    Fails closed: an undeclared DiT candidate cannot masquerade as a text candidate and pass
    validation on a technically-satisfying component subset.
    """
    if "architecture" in candidate:
        return False
    return "dit_blocks" in candidate.get("precision_map", {})


def candidate_weight_components(candidate: dict[str, Any]) -> frozenset[str]:
    """Served-weight precision-map components for the candidate's architecture."""
    if candidate_architecture(candidate) == "diffusion":
        return DIFFUSION_WEIGHT_PRECISION_MAP_COMPONENTS
    return WEIGHT_PRECISION_MAP_COMPONENTS


def candidate_required_components(candidate: dict[str, Any]) -> frozenset[str]:
    """Required (complete) precision-map components for the candidate's architecture."""
    if candidate_architecture(candidate) == "diffusion":
        return DIFFUSION_PRECISION_MAP_COMPONENTS
    return REQUIRED_PRECISION_MAP_COMPONENTS


def candidate_requires_mixed_fp8(candidate: dict[str, Any]) -> bool:
    """Whether this candidate's every actual id must carry `Mixed-FP8`, per its precision map."""
    return precision_map_requires_mixed_fp8(
        candidate["precision_map"], candidate_weight_components(candidate)
    )


def _contract_at_least(version: str | None, major: int, minor: int) -> bool:
    """True when a dotted contract version is at least major.minor (unknown or malformed versions are not)."""
    if not version:
        return False
    try:
        parts = [int(p) for p in str(version).split(".")[:2]]
    except ValueError:
        return False
    parts += [0] * (2 - len(parts))
    return (parts[0], parts[1]) >= (major, minor)


# Contract 3.1: a W4A4 build may serve an NVFP4 weight-only (W4A16) lm_head, declared on the lm_head component. This is
# NVIDIA's own ModelOpt layout for hybrid MoE checkpoints; the output projection's activations stay BF16 because nothing
# after it absorbs quantization error. Every other weight path keeps the original rule.
_W4A4_WEIGHT_ONLY_LM_HEAD = re.compile(r"^W4A16-NVFP4\b")


def candidate_precision_errors(candidate: dict[str, Any], *, contract_version: str | None = None) -> list[str]:
    """Return violations of the candidate precision-encoding rules.

    The four-product family slot may be labelled `ModelOpt-NVFP4`, but every candidate id must
    encode the real activation/recipe precision class (W4A16 vs W4A4, plus mixed FP8 where present)
    and carry a complete precision map. This prevents precision-map omission or conflation — for
    example, naming a mixed W4A16+FP8 artifact as if it were uniform W4A4.

    The `Mixed-FP8` rule is enforced from the precision map outwards: a map with FP8 on any served
    weight path must be named `Mixed-FP8` in its precision class and candidate id, and a map without
    it must not be. Shortening a mixed candidate's class/id to `W4A16-NVFP4` while FP8 stays in the
    map is therefore an error rather than a silent pass. Naming rules read served weight paths only,
    so runtime KV dtype neither triggers nor excuses any of them.
    """
    errors: list[str] = []
    cid = str(candidate["candidate_id"])
    pclass = str(candidate["precision_class"])
    pmap = candidate["precision_map"]
    is_diffusion = candidate_architecture(candidate) == "diffusion"
    weight_components = candidate_weight_components(candidate)

    if diffusion_anatomy_undeclared(candidate):
        errors.append(
            f"{cid}: precision_map carries DiT anatomy but architecture is not declared; "
            "add architecture: 'diffusion'"
        )

    missing = candidate_required_components(candidate) - set(pmap)
    if missing:
        errors.append(f"{cid}: precision_map omits components {sorted(missing)}")

    if pclass not in cid:
        errors.append(f"{cid}: candidate_id does not encode precision_class {pclass!r}")
    if not is_diffusion and "W4A16" not in cid and "W4A4" not in cid:
        errors.append(f"{cid}: candidate_id must encode W4A16 or W4A4")
    if is_diffusion:
        # DiT recipes quantize linear weights to NVFP4 (W4A4-style fp4 GEMM) or FP8; the
        # vocabulary is the activation/recipe class the recipe actually applies, spelled the
        # same way: NVFP4 (uniform fp4), Mixed-FP8 (fp4 + fp8 weights), or FP8 (uniform fp8).
        if not any(tok in cid for tok in ("NVFP4", "FP8")):
            errors.append(
                f"{cid}: diffusion candidate_id must encode NVFP4 or FP8"
            )

    weights = " ".join(
        str(value)
        for component, value in pmap.items()
        if component in weight_components
    )
    has_fp8 = "FP8" in weights
    has_w4a16 = "W4A16" in weights
    requires_mixed_fp8 = precision_map_requires_mixed_fp8(pmap, weight_components)

    # W4A4 never serves W4A16 weights. It may serve FP8 weights only as the declared Mixed-FP8 class
    # (W4A4 NVFP4 on some linears plus FP8 W8A8 on others, e.g. NVIDIA's Qwen3.8-27B layout).
    # From contract 3.1 the one exception is an NVFP4 weight-only lm_head declared on the lm_head component.
    w4a16_paths = [c for c in weight_components & set(pmap) if "W4A16" in str(pmap[c])]
    if (
        _contract_at_least(contract_version, 3, 1)
        and w4a16_paths == ["lm_head"]
        and _W4A4_WEIGHT_ONLY_LM_HEAD.match(str(pmap["lm_head"]))
    ):
        w4a16_paths = []
    if pclass.startswith("W4A4") and w4a16_paths:
        errors.append(f"{cid}: named W4A4 but precision_map serves W4A16 weights (conflation)")
    if pclass.startswith("W4A4") and has_fp8 and "Mixed-FP8" not in pclass:
        errors.append(
            f"{cid}: named uniform W4A4 but precision_map serves FP8 weights (conflation)"
        )

    if requires_mixed_fp8:
        fp8_weights = sorted(
            component
            for component in weight_components & set(pmap)
            if "FP8" in str(pmap[component])
        )
        if "Mixed-FP8" not in pclass:
            errors.append(
                f"{cid}: precision_map serves FP8 weights {fp8_weights} but precision_class "
                f"{pclass!r} omits Mixed-FP8"
            )
        if "Mixed-FP8" not in cid:
            errors.append(
                f"{cid}: precision_map serves FP8 weights {fp8_weights} but candidate_id omits "
                f"Mixed-FP8"
            )
    else:
        if "FP8" in pclass:
            errors.append(
                f"{cid}: precision_class {pclass!r} claims FP8 but no served weight path is FP8"
            )
        if "FP8" in cid:
            errors.append(f"{cid}: candidate_id claims FP8 but no served weight path is FP8")

    if is_diffusion:
        # Diffusion Mixed-FP8 means a DiT recipe mixing NVFP4 and FP8 served weights.
        if "Mixed-FP8" in pclass and not (has_fp8 and "NVFP4" in weights):
            errors.append(
                f"{cid}: named Mixed-FP8 but precision_map is not NVFP4+FP8"
            )
    elif "Mixed-FP8" in pclass and not has_fp8:
        errors.append(f"{cid}: named Mixed-FP8 but precision_map serves no FP8 weights")
    elif "Mixed-FP8" in pclass and not pclass.startswith(("W4A16", "W4A4")):
        errors.append(
            f"{cid}: text-model Mixed-FP8 precision_class {pclass!r} must start with W4A16 or W4A4"
        )
    elif "Mixed-FP8" in pclass and pclass.startswith("W4A16") and not has_w4a16:
        errors.append(f"{cid}: named W4A16 Mixed-FP8 but precision_map is not W4A16+FP8")
    elif "Mixed-FP8" in pclass and pclass.startswith("W4A4") and "W4A4" not in weights:
        errors.append(f"{cid}: named W4A4 Mixed-FP8 but precision_map is not W4A4+FP8")

    return errors


def candidate_promotion_blocked(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("promotion_status", "none")) == "blocked"


def candidate_promoted(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("promotion_status", "none")) == "promoted"


def candidate_selection(candidate: dict[str, Any]) -> str:
    return str(candidate.get("selection", ""))


def precision_encoded_id_errors(
    identifier: str, requires_mixed_fp8: bool | None = None
) -> list[str]:
    """Return violations for an artifact repository, model-card, or server model source.

    The bare `ModelOpt-NVFP4` release slot may be used as an abstract label in the contract and the
    process document, where it names a slot rather than an artifact. It may never be used as an id
    that something concrete resolves to: a Hugging Face repository, a model-card identity, or the
    model source passed to a server. Every such id that mentions NVFP4 must encode ModelOpt plus its
    real activation class (`W4A16` or `W4A4`), and `Mixed-FP8` when FP8 is part of the recipe. The
    concise ``--served-model-name`` API alias is validated separately.

    `requires_mixed_fp8` carries the requirement derived from the artifact's precision map (see
    `precision_map_requires_mixed_fp8`). When it is `True` the id must spell `Mixed-FP8`; when it is
    `False` the id must not mention FP8 at all. `None` means the caller has no precision map for
    this id and only the internal consistency rules apply — it must never be used for an id that a
    ledger candidate exists for.
    """
    name = identifier.split("/")[-1]
    if "NVFP4" not in name:
        return []

    errors: list[str] = []
    has_w4a16 = "W4A16" in name
    has_w4a4 = "W4A4" in name
    has_mixed_fp8 = "Mixed-FP8" in name

    if "ModelOpt" not in name:
        errors.append(f"{identifier}: NVFP4 id must encode ModelOpt")
    if not has_w4a16 and not has_w4a4:
        errors.append(
            f"{identifier}: bare NVFP4 id must encode its precision class (W4A16 or W4A4)"
        )
    if has_w4a16 and has_w4a4:
        errors.append(f"{identifier}: id conflates W4A16 and W4A4")
    if "FP8" in name and not has_mixed_fp8:
        errors.append(f"{identifier}: FP8 in an id must be spelled Mixed-FP8")
    # Mixed-FP8 is valid on a W4A16 id (W4A16 NVFP4 + FP8) and on a W4A4 id (W4A4 NVFP4 + FP8), but
    # on a W4A4 id only when the caller derived it from a precision map that serves FP8 weights: a
    # W4A4 id with no such evidence reads as uniform W4A4 and must not encode FP8.
    if has_w4a4 and has_mixed_fp8 and requires_mixed_fp8 is not True:
        errors.append(
            f"{identifier}: uniform W4A4 id must not encode FP8; Mixed-FP8 on a W4A4 id requires a "
            "precision map that serves FP8 weights"
        )
    elif has_w4a4 and "FP8" in name and not has_mixed_fp8:
        errors.append(f"{identifier}: uniform W4A4 id must not encode FP8")

    if requires_mixed_fp8 is True and not has_mixed_fp8:
        errors.append(
            f"{identifier}: precision map serves FP8 weights, so this id must encode Mixed-FP8"
        )
    if requires_mixed_fp8 is False and "FP8" in name:
        errors.append(
            f"{identifier}: no served weight path is FP8, so this id must not encode Mixed-FP8"
        )

    return errors


def is_bare_nvfp4_id(identifier: str) -> bool:
    """True when an id names NVFP4 without encoding W4A16 or W4A4."""
    name = identifier.split("/")[-1]
    return "NVFP4" in name and "W4A16" not in name and "W4A4" not in name


def extract_repository_ids(text: str) -> list[str]:
    """Every `HangGlidersRule/Darkstar-*` id in a text: these are always actual repository ids."""
    return _DARKSTAR_REPOSITORY_ID.findall(text)


class PublicationSourceParseError(ValueError):
    """A publication/serve command was present but its model source was ambiguous."""


# Shells whose ``-c`` argument is command *text*: its payload is parsed as a nested shell command
# rather than treated as an opaque argument, so a serve command hidden inside one is still read.
_SHELL_EXECUTABLES: frozenset[str] = frozenset({"sh", "bash", "dash", "ash", "zsh"})

# Shell invocation options, split by arity, because arity is what decides where a ``-c`` payload
# starts. Option *letters* may be written compactly in one group (``-ec``, ``-lc``), but an option
# that takes an argument always takes the following *token*: `sh -oerrexit` is not a compact
# spelling of `sh -o errexit`, it makes the shell read the next token as the option name. So
# ``-o``/``-O``/``+O`` and the long options below consume one token each, whether they stand alone
# or end a group, and any letter or long option not listed here has unknown arity.
_SHELL_FLAG_LETTERS: frozenset[str] = frozenset("abcefhiklmnprstuvxBCDEHPT")
_SHELL_OPTION_ARGUMENT_LETTERS: frozenset[str] = frozenset("oO")
_SHELL_COMMAND_LETTER = "c"
_SHELL_LONG_OPTIONS: frozenset[str] = frozenset(
    {
        "--debug",
        "--debugger",
        "--dump-po-strings",
        "--dump-strings",
        "--help",
        "--login",
        "--noediting",
        "--noprofile",
        "--norc",
        "--posix",
        "--pretty-print",
        "--protected",
        "--restricted",
        "--verbose",
        "--version",
    }
)
_SHELL_LONG_OPTIONS_WITH_ARGUMENT: frozenset[str] = frozenset(
    {"--init-file", "--option", "--rcfile"}
)

# What makes a shell launcher worth failing closed over: content that could still resolve to a
# model source once the shell runs it.
_SHELL_SERVE_CONTENT = re.compile(r"vllm|--model")

# Commands that only print their arguments. No shell runs a quoted argument of one as command text, so
# the argument is prose (`say "$c serve finished"`), not a serve command. `say`, `log`, `die`, `info`
# and the like are the script-local logging helpers the runbooks print with; they count only as the
# command word (first word of a simple command, after any assignments). A command substitution inside
# such an argument is still run by the shell that runs the script, so it is still read.
_OUTPUT_ONLY_COMMANDS = frozenset(
    {"echo", "printf", "say", "log", "info", "warn", "error", "die", "note", "msg", "logger"}
)
_SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")

_SHELL_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_SHELL_PARAMETER_DEFAULT_OPERATORS: tuple[str, ...] = (":-", ":=", "-", "=")

# ``vllm serve`` written as a structured sequence (``[vllm, serve, ...]``) as well as shell text.
_VLLM_SERVE_TEXT = re.compile(r"vllm[\s,\"']+serve")

# Nested `sh -c` payloads and nested parameter defaults are both bounded: an unbounded nest is
# itself an unreadable surface rather than something to keep unwrapping.
_MAX_SHELL_NESTING = 8

# Compose runs `entrypoint` followed by `command`, so a model source may be split across the two.
_COMPOSE_COMMAND_KEYS: tuple[str, ...] = ("entrypoint", "command")

# Surfaces whose text is shell: an untyped surface, a shell script, or a markdown fence a reader
# pastes into a terminal (unlabelled and `text` fences included). In .py/.json/.toml text without
# a shell shebang, parentheses and backquotes are Python and RST syntax, not shell.
_SHELL_LANGUAGES = frozenset(
    {"", "sh", "bash", "zsh", "ksh", "dash", "ash", "shell", "console", "shell-session", "text"}
)

_SHELL_SHEBANG = re.compile(r"#!\s*(?:\S*/)?(?:env\s+(?:-\S+\s+)*)?(?P<interpreter>[^\s/]+)")
_SOURCE_PLACEHOLDER = re.compile(r"<[^<>\s]+>")

_SHELL_BLANKS = frozenset(" \t\r\n")
# Unquoted, these end a word wherever they appear: `true;vllm serve ...` is two commands. The
# redirection characters stay in their word so a `<placeholder>` in a documented command reads whole.
# Parentheses only separate words in shell text: elsewhere they are a call such as `f("--model")`.
_SHELL_SEPARATOR_CHARACTERS = frozenset(";&|")
_SHELL_OPERATOR_CHARACTERS = _SHELL_SEPARATOR_CHARACTERS | frozenset("()")
# A word holding any of these may read differently once a shell parses it as command text.
_SHELL_SYNTAX_CHARACTERS = frozenset(" \t\r\n'\"\\$`;&|()<>")
_ANSI_C_ESCAPES: dict[str, str] = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}
_OCTAL_DIGITS = frozenset("01234567")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


class _ShellSyntaxError(ValueError):
    """Shell text that cannot be read the way bash reads it; `offset` is where reading failed."""

    def __init__(self, message: str, offset: int) -> None:
        super().__init__(message)
        self.offset = offset


def _take_digits(text: str, index: int, digits: frozenset[str], limit: int) -> str:
    end = index
    while end < len(text) and end - index < limit and text[end] in digits:
        end += 1
    return text[index:end]


def _decode_ansi_c(content: str) -> str:
    """The text bash produces for the body of a ``$'...'`` quote."""
    decoded = bytearray()
    index = 0
    while index < len(content):
        character = content[index]
        if character != "\\" or index + 1 >= len(content):
            decoded += character.encode("utf-8", "surrogateescape")
            index += 1
            continue
        escape = content[index + 1]
        index += 2
        if escape in _ANSI_C_ESCAPES:
            decoded += _ANSI_C_ESCAPES[escape].encode()
        elif escape in _OCTAL_DIGITS:
            digits = escape + _take_digits(content, index, _OCTAL_DIGITS, 2)
            index += len(digits) - 1
            decoded.append(int(digits, 8) & 0xFF)
        elif escape == "x" and (digits := _take_digits(content, index, _HEX_DIGITS, 2)):
            index += len(digits)
            decoded.append(int(digits, 16))
        elif escape in "uU" and (
            digits := _take_digits(content, index, _HEX_DIGITS, 4 if escape == "u" else 8)
        ):
            index += len(digits)
            codepoint = int(digits, 16)
            valid = codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF
            decoded += (chr(codepoint) if valid else "\ufffd").encode()
        elif escape == "c" and index < len(content):
            control = content[index]
            index += 1
            if control == "\\" and content.startswith("\\", index):
                index += 1
            decoded.append(0x7F if control == "?" else ord(control.upper()) & 0x1F)
        else:
            decoded += ("\\" + escape).encode("utf-8", "surrogateescape")
    # Bash strings end at NUL, so `$'vllm\0...'` is just `vllm`.
    return bytes(decoded).split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


class _ShellLexer:
    """Split shell text into commands and words with bash's quoting rules, not POSIX ``shlex``'s.

    ``shlex`` keeps ``$'...'`` as a literal dollar plus an ordinary quote, ends a word at ``#``,
    glues control operators onto words, and keeps an empty ```` `` ```` or ``$()`` glued to a word,
    and each of those lets bash run a serve command the gate never reads. Here ANSI-C quotes are
    decoded, control operators separate words, ``#`` starts a comment only at the start of a word,
    empty command substitutions expand to nothing, and a quote left open at a newline continues
    onto the next line. Other expansions (``$name``, ``${...}``, ``$(...)``, backquotes) stay
    verbatim in their word, so a source that depends on one is still reported as unresolved; the
    bodies of command substitutions are collected in `substitutions` because bash runs them as
    commands too. ``$"..."`` is translated through the locale when bash runs it, so its text is
    unknowable and strict reading rejects it.

    Read leniently (``strict=False``) nothing is rejected: an unterminated construct runs to the end
    of the text and ``$"..."`` reads as a plain double quote. That reading is only for deciding
    whether text that cannot be read strictly could still hold a serve command. Without
    `shell_syntax`, backquotes and parentheses are ordinary characters, as they are in the RST
    prose and Python calls of surfaces that are not shell.
    """

    def __init__(
        self, text: str, *, strict: bool, start: int = 0, shell_syntax: bool = True
    ) -> None:
        self.text = text
        self.strict = strict
        self.shell_syntax = shell_syntax
        self.operators = (
            _SHELL_OPERATOR_CHARACTERS if shell_syntax else _SHELL_SEPARATOR_CHARACTERS
        )
        self.index = start
        self.command_start = start
        self.commands: list[tuple[int, list[str]]] = []
        self.substitutions: list[str] = []

    def _reject(self, message: str, offset: int) -> None:
        if self.strict:
            raise _ShellSyntaxError(message, offset)

    def read(self, *, single_command: bool = False) -> list[tuple[int, list[str]]]:
        """Every command as (offset of its first character, words), split at unquoted newlines.

        With `single_command`, reading stops at the first unquoted newline.
        """
        text = self.text
        words: list[str] = []
        word: list[str] = []
        in_word = False
        quoted = False
        while self.index < len(text):
            character = text[self.index]
            following = text[self.index + 1] if self.index + 1 < len(text) else ""
            if character in _SHELL_BLANKS or character in self.operators:
                if in_word:
                    self._end_word(words, word, quoted)
                word, in_word, quoted = [], False, False
                if character == "\n":
                    if words:
                        self.commands.append((self.command_start, words))
                    words = []
                    self.index += 1
                    self.command_start = self.index
                    if single_command:
                        return self.commands
                elif character in self.operators:
                    end = self.index
                    while end < len(text) and text[end] in self.operators:
                        end += 1
                    words.append(text[self.index : end])
                    self.index = end
                else:
                    self.index += 1
                continue
            if character == "#" and not in_word:
                newline = text.find("\n", self.index)
                self.index = len(text) if newline < 0 else newline
                continue
            if character == "\\" and following == "\n":
                self.index += 2
                continue
            in_word = True
            if character == "\\":
                if not following:
                    self._reject("trailing backslash escapes nothing", self.index)
                    word.append("\\")
                    self.index += 1
                    continue
                word.append(following)
                self.index += 2
            elif character == "'":
                quoted = True
                word.append(self._single_quoted())
            elif character == "$" and following == "'":
                quoted = True
                word.append(self._ansi_c_quoted())
            elif character == "$" and following == '"':
                self._reject(
                    'locale-translated $"..." text is unknown until bash runs it', self.index
                )
                quoted = True
                self.index += 1
                word.append(self._double_quoted())
            elif character == '"':
                quoted = True
                word.append(self._double_quoted())
            elif self._starts_expansion(character, following):
                word.append(self._expansion())
            else:
                word.append(character)
                self.index += 1
        if in_word:
            self._end_word(words, word, quoted)
        if words:
            self.commands.append((self.command_start, words))
        return self.commands

    @staticmethod
    def _end_word(words: list[str], word: list[str], quoted: bool) -> None:
        # An unquoted word that expands to nothing (``` `` ```, `$()`) is no argument at all.
        value = "".join(word)
        if value or quoted:
            words.append(value)

    def _starts_expansion(self, character: str, following: str) -> bool:
        return (character == "`" and self.shell_syntax) or (
            character == "$" and following in ("(", "{")
        )

    def _single_quoted(self) -> str:
        opening = self.index
        closing = self.text.find("'", opening + 1)
        if closing < 0:
            self._reject("unterminated single quote", opening)
            self.index = len(self.text)
            return self.text[opening + 1 :]
        self.index = closing + 1
        return self.text[opening + 1 : closing]

    def _ansi_c_quoted(self) -> str:
        opening = self.index
        index = opening + 2
        while index < len(self.text) and self.text[index] != "'":
            index += 2 if self.text[index] == "\\" else 1
        if index >= len(self.text):
            self._reject("unterminated $'...' quote", opening)
            self.index = len(self.text)
            return _decode_ansi_c(self.text[opening + 2 :])
        self.index = index + 1
        return _decode_ansi_c(self.text[opening + 2 : index])

    def _double_quoted(self) -> str:
        opening = self.index
        text = self.text
        out: list[str] = []
        self.index += 1
        while self.index < len(text):
            character = text[self.index]
            following = text[self.index + 1] if self.index + 1 < len(text) else ""
            if character == '"':
                self.index += 1
                return "".join(out)
            if character == "\\" and following == "\n":
                self.index += 2
            elif character == "\\" and following in ("$", "`", '"', "\\"):
                out.append(following)
                self.index += 2
            elif self._starts_expansion(character, following):
                out.append(self._expansion())
            else:
                out.append(character)
                self.index += 1
        self._reject("unterminated double quote", opening)
        return "".join(out)

    def _expansion(self) -> str:
        """A ``$(...)``, ``${...}``, or backquoted expansion, verbatim, starting at `index`."""
        text = self.text
        start = self.index
        if text[start] == "`":
            body: list[str] = []
            self.index += 1
            while self.index < len(text):
                character = text[self.index]
                if character == "\\" and self.index + 1 < len(text):
                    escaped = text[self.index + 1]
                    body.append(escaped if escaped in ("$", "`", "\\") else character + escaped)
                    self.index += 2
                    continue
                self.index += 1
                if character == "`":
                    command = "".join(body)
                    self.substitutions.append(command)
                    return text[start : self.index] if command.strip() else ""
                body.append(character)
            self._reject("unterminated backquote", start)
            self.substitutions.append("".join(body))
            return text[start:]
        opener = text[start + 1]
        closer = ")" if opener == "(" else "}"
        depth = 1
        self.index = start + 2
        while self.index < len(text):
            character = text[self.index]
            following = text[self.index + 1] if self.index + 1 < len(text) else ""
            if character == "\\":
                self.index += 2
            elif character == "'":
                closing = text.find("'", self.index + 1)
                if closing < 0:
                    self._reject("unterminated single quote", self.index)
                    self.index = len(text)
                    break
                self.index = closing + 1
            elif character == '"':
                self._double_quoted()
            elif self._starts_expansion(character, following):
                self._expansion()
            else:
                self.index += 1
                if character == opener:
                    depth += 1
                elif character == closer:
                    depth -= 1
                    if depth == 0:
                        if opener == "(":
                            command = text[start + 2 : self.index - 1]
                            self.substitutions.append(command)
                            if not command.strip():
                                return ""
                        return text[start : self.index]
        self._reject(f"unterminated {text[start : start + 2]}", start)
        self.index = len(text)
        if opener == "(":
            self.substitutions.append(text[start + 2 :])
        return text[start:]


class _ShellCommand(NamedTuple):
    """One command read from shell text, or the line and the reason it cannot be read."""

    line: int
    words: list[str] | None
    substitutions: list[str]
    text: str
    error: str


def _shell_commands(text: str, *, shell_syntax: bool) -> list[_ShellCommand]:
    """The commands of shell text, read line by line and also the way bash reads across lines.

    Each line is read on its own, the way a reader copying it would run it, and a line that cannot
    be read on its own is returned without words. When a line leaves a quote open, the command bash
    reads from that line through the closing quote is returned as well, so a payload split over
    several lines is not lost. Text that is not shell only gets that reading for a line that names
    a shell, because elsewhere its quotes are not shell quotes and the joined text means nothing.
    """
    commands: list[_ShellCommand] = []
    offset = 0
    for number, line in _logical_lines(text):
        lexer = _ShellLexer(line, strict=True, shell_syntax=shell_syntax)
        try:
            read = lexer.read()
        except _ShellSyntaxError as error:
            commands.append(_ShellCommand(number, None, [], line, str(error)))
            read = []
            if shell_syntax or _names_shell(line):
                lexer = _ShellLexer(text, strict=True, start=offset, shell_syntax=shell_syntax)
                try:
                    read = lexer.read(single_command=True)
                except _ShellSyntaxError:
                    read = []
        commands.extend(
            _ShellCommand(number, words, lexer.substitutions, "", "") for _, words in read
        )
        offset += len(line) + 1
    return commands


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Physical lines joined across a trailing unescaped backslash, with their first line number.

    The newline stays in the joined text: the lexer decides what it means, because a backslash and
    newline continue the line outside quotes but are kept literally inside single and ANSI-C quotes.
    """
    physical = text.split("\n")
    lines: list[tuple[int, str]] = []
    pending: list[str] = []
    first = 1
    for number, line in enumerate(physical, start=1):
        if not pending:
            first = number
        pending.append(line)
        trailing = len(line) - len(line.rstrip("\\"))
        if trailing % 2 == 1 and number < len(physical):
            continue
        lines.append((first, "\n".join(pending)))
        pending = []
    return lines


def _is_shell_surface(format_name: str, text: str) -> bool:
    """Whether a surface is shell text: by its declared format, or by a shebang naming a shell."""
    if format_name in _SHELL_LANGUAGES:
        return True
    shebang = _SHELL_SHEBANG.match(text)
    return shebang is not None and shebang.group("interpreter") in _SHELL_EXECUTABLES


def _names_shell(text: str) -> bool:
    return any(
        word.rpartition("/")[2] in _SHELL_EXECUTABLES
        for _, words in _ShellLexer(text, strict=False).read()
        for word in words
    )


def _shell_words(text: str) -> list[str]:
    """All words of shell text read strictly as one argument vector."""
    return [word for _, words in _ShellLexer(text, strict=True).read() for word in words]


def _mentions_serve_content(words: list[str]) -> bool:
    return any(_SHELL_SERVE_CONTENT.search(word) for word in words)


def _mentions_served_model_name(words: list[str]) -> bool:
    return any(_SERVED_MODEL_NAME_FLAG in word for word in words)


def _is_model_option(word: str) -> bool:
    return word == "--model" or word.startswith("--model=")


def _starts_serve_command(words: list[str]) -> bool:
    return any(
        _is_model_option(word)
        or (word.rpartition("/")[2] == "vllm" and words[index + 1 : index + 2] == ["serve"])
        or (word == "serve" and _starts_with_expansion_command(words, index))
        for index, word in enumerate(words)
    )


def _shell_text_matches(
    text: str,
    matches: Callable[[list[str]], bool],
    depth: int = 0,
    *,
    shell_syntax: bool = True,
) -> bool:
    """Whether shell text, as written or read leniently through nested quoting, `matches`."""
    if matches([text]):
        return True
    lexer = _ShellLexer(text, strict=False, shell_syntax=shell_syntax)
    commands = lexer.read()
    if any(matches(words) for _, words in commands):
        return True
    if depth >= _MAX_SHELL_NESTING:
        return False
    nested = [
        word
        for _, words in commands
        for word in words
        if word != text and _SHELL_SYNTAX_CHARACTERS.intersection(word)
    ]
    return any(
        _shell_text_matches(inner, matches, depth + 1, shell_syntax=shell_syntax)
        for inner in [*nested, *(lexer.substitutions if shell_syntax else ())]
    )


def _starts_with_expansion_command(
    tokens: list[str], index: int, *, backquotes: bool = True
) -> bool:
    """Whether the word before `index` is a command name that only an expansion decides."""
    if index == 0:
        return False
    previous = tokens[index - 1]
    return "$" in previous or (backquotes and "`" in previous)


def _shell_parameter_default(token: str) -> str | None:
    """The default text of a whole-token ``${NAME:-default}`` expansion, else None.

    Only an expansion that spans the entire token has a single default. A concatenation such as
    ``${A}/${B}`` or a bare ``$NAME``/``${NAME}`` has none, and is therefore unresolved.
    """
    if not token.startswith("${") or not token.endswith("}"):
        return None
    inner = token[2:-1]
    depth = 0
    for character in inner:
        if character == "{":
            depth += 1
        elif character == "}":
            if depth == 0:
                return None
            depth -= 1
    if depth != 0:
        return None
    name = _SHELL_PARAMETER_NAME.match(inner)
    if name is None:
        return None
    rest = inner[name.end() :]
    for operator in _SHELL_PARAMETER_DEFAULT_OPERATORS:
        if rest.startswith(operator):
            return rest[len(operator) :]
    return None


def _resolve_model_source(token: str, *, context: str, depth: int = 0) -> str:
    """Resolve a model-source token to one concrete source, or fail closed.

    A shell parameter default (``${MODEL_SOURCE:-HangGlidersRule/...}``) names a concrete source
    that ships in the surface, so it is unwrapped and validated like any other. Anything else that
    still carries a shell expansion — ``$MODEL_SOURCE``, ``${MODEL_SOURCE}``, ``${A}/${B}`` — names
    nothing concrete and is an error rather than an opaque local path.
    """
    if "$" not in token:
        return token
    if depth >= _MAX_SHELL_NESTING:
        raise PublicationSourceParseError(
            f"{context}: model source {token!r} nests shell parameter defaults too deeply"
        )
    default = _shell_parameter_default(token)
    if default is None:
        raise PublicationSourceParseError(
            f"{context}: model source {token!r} is an unresolved shell variable, not a concrete "
            f"artifact path or repository id"
        )
    try:
        default_words = _shell_words(default)
    except _ShellSyntaxError as error:
        raise PublicationSourceParseError(
            f"{context}: model source {token!r} has an unreadable default: {error}"
        ) from error
    if len(default_words) > 1:
        raise PublicationSourceParseError(
            f"{context}: model source {token!r} defaults to more than one word"
        )
    resolved = _resolve_model_source(
        default_words[0] if default_words else "", context=context, depth=depth + 1
    )
    if not resolved:
        raise PublicationSourceParseError(
            f"{context}: model source {token!r} defaults to an empty source"
        )
    return resolved


def _shell_option_arity(token: str) -> tuple[int, bool] | None:
    """How many following tokens a shell option consumes and whether it carries ``-c``.

    Returns None when the option's arity is unknown, which makes the whole invocation unreadable:
    without arity there is no way to tell an option's argument from the ``-c`` that follows it.
    """
    if token.startswith("--"):
        if token in _SHELL_LONG_OPTIONS_WITH_ARGUMENT:
            return 1, False
        if token in _SHELL_LONG_OPTIONS:
            return 0, False
        return None
    letters = token[1:]
    if any(
        letter not in _SHELL_FLAG_LETTERS and letter not in _SHELL_OPTION_ARGUMENT_LETTERS
        for letter in letters
    ):
        return None
    carries_command = _SHELL_COMMAND_LETTER in letters
    if carries_command and token.startswith("+"):
        return None
    arguments = sum(letter in _SHELL_OPTION_ARGUMENT_LETTERS for letter in letters)
    return arguments, carries_command


def _unreadable_shell_invocation(
    tokens: list[str], index: int, *, context: str, launcher: str, reason: str
) -> None:
    """Fail closed on a shell launcher whose options cannot be read.

    An unreadable launcher may be hiding a serve command in the payload position, so it may only be
    treated as a plain (non ``-c``) invocation when nothing serve-like remains in it to hide, even
    once its words are read again as shell text.
    """
    if any(_shell_text_matches(token, _mentions_serve_content) for token in tokens[index:]):
        raise PublicationSourceParseError(f"{context}: {launcher} {reason}")


def _shell_command_payload_index(
    tokens: list[str], start: int, *, context: str
) -> int | None:
    """Index of the command payload of the shell invoked at `start`, or None when it has none.

    Options are scanned with their real arity so an option argument can never disguise the ``-c``
    behind it: `sh -o errexit -c "vllm serve ..."` and `bash -O extglob -c "vllm serve ..."` yield
    their payload rather than reading as an option-only invocation. Like the shells themselves, this
    keeps parsing options past ``-c`` and takes the payload to be the first non-option token (or
    whatever follows ``--``). The returned index may equal ``len(tokens)``, meaning ``-c`` was given
    no payload at all.
    """
    launcher = tokens[start]
    wants_command = False
    index = start + 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            if wants_command:
                return index + 1
            # Options are terminated, so what follows is a script to run. An option-looking
            # "script" is a launcher no shell can run, and its later tokens may still hold the
            # command text a `-c` would have taken.
            if index + 1 < len(tokens) and tokens[index + 1].startswith(("-", "+")):
                _unreadable_shell_invocation(
                    tokens,
                    index + 1,
                    context=context,
                    launcher=launcher,
                    reason=(
                        f"runs {tokens[index + 1]!r} as a script after --, so its command text "
                        f"cannot be located"
                    ),
                )
            return None
        if not token.startswith(("-", "+")) or token in ("-", "+"):
            return index if wants_command else None
        arity = _shell_option_arity(token)
        if arity is None:
            _unreadable_shell_invocation(
                tokens,
                index,
                context=context,
                launcher=launcher,
                reason=f"option {token!r} has unknown arity, so its -c payload cannot be located",
            )
            return None
        arguments, carries_command = arity
        wants_command = wants_command or carries_command
        index += 1
        for _ in range(arguments):
            if index >= len(tokens) or tokens[index].startswith(("-", "+")):
                _unreadable_shell_invocation(
                    tokens,
                    index,
                    context=context,
                    launcher=launcher,
                    reason=f"option {token!r} is missing the argument it consumes",
                )
                return None
            index += 1
    return len(tokens) if wants_command else None


def _concrete_model_source(token: str, *, context: str) -> str:
    source = _resolve_model_source(token, context=context)
    if not source:
        raise PublicationSourceParseError(f"{context}: model source is empty")
    # No repository id or path holds `<`/`>`: one glued to a source (`MODEL>log`, `2>/dev/null`)
    # is a redirection, so the text is not the model the shell serves. A whole `<placeholder>`
    # in a documented command is no redirection a shell could run.
    if ("<" in source or ">" in source) and not _SOURCE_PLACEHOLDER.fullmatch(source):
        raise PublicationSourceParseError(
            f"{context}: model source {source!r} carries a shell redirection"
        )
    return source


def _sources_from_tokens(
    tokens: list[str], *, context: str, depth: int = 0, shell_surface: bool = False
) -> list[str]:
    sources: list[str] = []
    saw_model_option = any(_is_model_option(token) for token in tokens)
    printed, piped = _output_only_arguments(tokens)
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.rpartition("/")[2] in _SHELL_EXECUTABLES:
            payload_index = _shell_command_payload_index(tokens, index, context=context)
            if payload_index is not None:
                if payload_index >= len(tokens):
                    raise PublicationSourceParseError(
                        f"{context}: {token} -c has no command payload to parse"
                    )
                sources.extend(
                    _sources_from_shell(
                        tokens[payload_index],
                        context=f"{context}:{token}-c",
                        depth=depth + 1,
                        shell_surface=True,
                    )
                )
                index = payload_index + 1
                continue

        if token == "--model":
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise PublicationSourceParseError(f"{context}: --model has no parseable value")
            sources.append(_concrete_model_source(tokens[index + 1], context=context))
            index += 2
            continue

        if token.startswith("--model="):
            value = token.partition("=")[2]
            if not value:
                raise PublicationSourceParseError(f"{context}: --model has no parseable value")
            sources.append(_concrete_model_source(value, context=context))
            index += 1
            continue

        if token.rpartition("/")[2] == "vllm" and tokens[index + 1 : index + 2] == ["serve"]:
            if index + 2 >= len(tokens):
                raise PublicationSourceParseError(f"{context}: vllm serve has no model source")
            positional = tokens[index + 2]
            if positional.startswith("-"):
                if not saw_model_option:
                    raise PublicationSourceParseError(
                        f"{context}: vllm serve positional model source is missing or ambiguous"
                    )
                index += 2
                continue
            sources.append(_concrete_model_source(positional, context=context))
            index += 3
            continue

        if token == "serve" and _starts_with_expansion_command(
            tokens, index, backquotes=shell_surface
        ):
            raise PublicationSourceParseError(
                f"{context}: {tokens[index - 1]!r} serve names its command through a shell "
                f"expansion, so whether it starts vLLM cannot be read"
            )

        if token == "eval":
            payload, index = _eval_payload(tokens, index)
            sources.extend(
                _sources_from_shell(
                    payload, context=f"{context}:eval", depth=depth + 1, shell_surface=True
                )
            )
            continue

        if _substitutes_serve_text(token, shell_syntax=shell_surface):
            raise PublicationSourceParseError(
                f"{context}: argument {token!r} holds a command substitution that mentions a serve "
                f"command; a shell or remote command that re-reads it serves whatever it prints"
            )

        if index in piped and "\\" in token:
            raise PublicationSourceParseError(
                f"{context}: argument {token!r} is printed into a command that runs text and holds a "
                f"backslash escape, which printf and echo -e decode, so what runs cannot be read"
            )

        if index not in printed and _embeds_shell_text(
            token, _starts_serve_command, shell_syntax=shell_surface
        ):
            sources.extend(
                _sources_from_shell(
                    token,
                    context=f"{context}:argument",
                    depth=depth + 1,
                    shell_surface=shell_surface,
                )
            )

        index += 1
    return sources


def _embeds_shell_text(
    token: str, matches: Callable[[list[str]], bool], *, shell_syntax: bool
) -> bool:
    """Whether one argument reads as further shell words that could `matches` a serve command.

    Command text passed as a single argument (`ssh host '...'`, `su -c '...'`) is run by a shell all
    the same, so it is read again rather than skipped.
    """
    if not _SHELL_SYNTAX_CHARACTERS.intersection(token):
        return False
    lexer = _ShellLexer(token, strict=False, shell_syntax=shell_syntax)
    words = [word for _, line in lexer.read() for word in line]
    return words != [token] and _shell_text_matches(token, matches, shell_syntax=shell_syntax)


# Words that run text they are given: a shell, `eval`, `source`/`.`, and the launchers that hand text to one.
_TEXT_EXECUTORS = frozenset({"eval", "source", ".", "xargs", "ssh", "su"})


def _is_text_executor(word: str) -> bool:
    name = word.rpartition("/")[2]
    return name in _SHELL_EXECUTABLES or word in _TEXT_EXECUTORS


def _is_operator(token: str) -> bool:
    return bool(token) and all(character in _SHELL_OPERATOR_CHARACTERS for character in token)


def _pipelines(tokens: list[str]) -> list[list[list[int]]]:
    """The word indexes of `tokens`, split into pipelines of stages.

    `|` and `|&` separate the stages of a pipeline; any other control operator (`;`, `&`, `&&`,
    `||`) ends it. Parentheses alone only group, and operator words themselves are not returned.
    """
    pipelines: list[list[list[int]]] = [[[]]]
    for index, token in enumerate(tokens):
        if not _is_operator(token):
            pipelines[-1][-1].append(index)
            continue
        core = token.strip("()")
        if core in ("|", "|&"):
            pipelines[-1].append([])
        elif core:
            pipelines.append([[]])
    return [[stage for stage in pipeline if stage] for pipeline in pipelines if any(pipeline)]


def _output_only_arguments(tokens: list[str]) -> tuple[set[int], set[int]]:
    """Indexes of the arguments of output-only commands (`echo`, `say`, ...) among one command's words.

    Returns (prose, piped). A command starts at the first word of a pipeline stage; leading
    assignments are skipped. Only the command word itself decides: `ssh host echo x` is `ssh`'s
    argument list. When the printed output is piped into a later stage of the same pipeline that
    runs text (a shell, `eval`, `source`, `xargs`, `ssh`, `su`), what is printed is the command
    that runs, so those arguments are not prose: they are returned in `piped`, and the caller reads
    them as shell text.
    """
    prose: set[int] = set()
    piped: set[int] = set()
    for pipeline in _pipelines(tokens):
        for position, stage in enumerate(pipeline):
            command = next(
                (index for index in stage if not _SHELL_ASSIGNMENT.match(tokens[index])), None
            )
            if command is None or tokens[command] not in _OUTPUT_ONLY_COMMANDS:
                continue
            executes = any(
                _is_text_executor(tokens[index])
                for later in pipeline[position + 1 :]
                for index in later
            )
            (piped if executes else prose).update(index for index in stage if index > command)
    return prose, piped


def _substitutes_serve_text(token: str, *, shell_syntax: bool) -> bool:
    """Whether one argument holds a ``$(...)``, ``<(...)`` or backquoted substitution near serve text.

    A shell that re-reads the argument (`ssh host '$(...)'`, `su -c '$(...)'`, `bash -c '$(...)'`)
    runs the substitution and then runs what it prints, so the command it serves is only known once
    it has run. Backquotes count only in shell text, where they are not prose markup.
    """
    openers = ("$(", "<(", "`") if shell_syntax else ("$(", "<(")
    return any(opener in token for opener in openers) and _shell_text_matches(
        token, _mentions_serve_content
    )


def _eval_payload(tokens: list[str], index: int) -> tuple[str, int]:
    """The command text `eval` at `index` runs, and the index of the word after its arguments."""
    end = index + 1
    while end < len(tokens) and not all(
        character in _SHELL_OPERATOR_CHARACTERS for character in tokens[end]
    ):
        end += 1
    return " ".join(tokens[index + 1 : end]), end


def _sources_from_shell(
    text: str, *, context: str, depth: int = 0, shell_surface: bool = False
) -> list[str]:
    if depth > _MAX_SHELL_NESTING:
        raise PublicationSourceParseError(f"{context}: shell command text nests too deeply")
    sources: list[str] = []
    for command in _shell_commands(text, shell_syntax=shell_surface):
        line_context = f"{context}:{command.line}"
        if command.words is None:
            if _shell_text_matches(command.text, _mentions_serve_content):
                raise PublicationSourceParseError(
                    f"{line_context}: cannot parse serve/model command: {command.error}"
                )
            continue
        sources.extend(
            _sources_from_tokens(
                command.words, context=line_context, depth=depth, shell_surface=shell_surface
            )
        )
        for body in command.substitutions if shell_surface else ():
            sources.extend(
                _sources_from_shell(
                    body,
                    context=f"{line_context}:substitution",
                    depth=depth + 1,
                    shell_surface=True,
                )
            )
    return sources


def _command_tokens(value: object, *, context: str) -> list[str]:
    if isinstance(value, str):
        try:
            return _shell_words(value)
        except _ShellSyntaxError as error:
            raise PublicationSourceParseError(
                f"{context}: cannot parse Compose command: {error}"
            ) from error
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return [str(item) for item in value]
    raise PublicationSourceParseError(f"{context}: Compose command must be a string or scalar list")


def _serve_model_ids_from_tokens(
    tokens: list[str], *, context: str, depth: int = 0, shell_surface: bool = False
) -> list[str]:
    aliases: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.rpartition("/")[2] in _SHELL_EXECUTABLES:
            payload_index = _shell_command_payload_index(tokens, index, context=context)
            if payload_index is not None:
                if payload_index >= len(tokens):
                    raise PublicationSourceParseError(
                        f"{context}: {token} -c has no command payload to parse"
                    )
                aliases.extend(
                    _serve_model_ids_from_shell(
                        tokens[payload_index],
                        context=f"{context}:{token}-c",
                        depth=depth + 1,
                        shell_surface=True,
                    )
                )
                index = payload_index + 1
                continue

        if token == "eval":
            payload, index = _eval_payload(tokens, index)
            aliases.extend(
                _serve_model_ids_from_shell(
                    payload, context=f"{context}:eval", depth=depth + 1, shell_surface=True
                )
            )
            continue

        value: str | None = None
        if token == _SERVED_MODEL_NAME_FLAG and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith(_SERVED_MODEL_NAME_FLAG + "="):
            value = token.partition("=")[2]
        if value is not None:
            aliases.append(value)
        elif shell_surface and _embeds_shell_text(
            token, _mentions_served_model_name, shell_syntax=True
        ):
            aliases.extend(
                _serve_model_ids_from_shell(
                    token, context=f"{context}:argument", depth=depth + 1, shell_surface=True
                )
            )
        index += 1
    return aliases


def _serve_model_ids_from_shell(
    text: str, *, context: str = "<shell>", depth: int = 0, shell_surface: bool = False
) -> list[str]:
    if depth > _MAX_SHELL_NESTING:
        raise PublicationSourceParseError(f"{context}: shell command text nests too deeply")
    aliases: list[str] = []
    for command in _shell_commands(text, shell_syntax=shell_surface):
        if command.words is None:
            continue
        line_context = f"{context}:{command.line}"
        aliases.extend(
            _serve_model_ids_from_tokens(
                command.words, context=line_context, depth=depth, shell_surface=shell_surface
            )
        )
        for body in command.substitutions if shell_surface else ():
            aliases.extend(
                _serve_model_ids_from_shell(
                    body,
                    context=f"{line_context}:substitution",
                    depth=depth + 1,
                    shell_surface=True,
                )
            )
    return aliases


def _serve_model_ids_from_yaml(text: str) -> list[str]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return []

    aliases: list[str] = []

    def visit(value: object, location: str) -> None:
        if isinstance(value, dict):
            declared = [key for key in _COMPOSE_COMMAND_KEYS if key in value]
            if declared:
                tokens: list[str] = []
                for key in declared:
                    try:
                        tokens.extend(_command_tokens(value[key], context=f"{location}.{key}"))
                    except PublicationSourceParseError:
                        return
                aliases.extend(
                    _serve_model_ids_from_tokens(
                        tokens, context=f"{location}.{'+'.join(declared)}", shell_surface=True
                    )
                )
            for key, child in value.items():
                if key not in _COMPOSE_COMMAND_KEYS:
                    visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(document, "<yaml>")
    return aliases


def extract_serve_model_ids(text: str, *, source_format: str | None = None) -> list[str]:
    """Extract every value passed through ``--served-model-name``.

    Compose YAML is parsed structurally so command scalar lists and strings have the same option
    semantics. Other surfaces are tokenized as shell text. Values passed through ``--model`` are
    deliberately ignored because publication model sources and runtime API aliases are separate
    identities. Registration against a ledger alias and the concise alias pattern is a separate
    check; this extractor does not drop values that fall outside that pattern.
    """
    format_name = (source_format or "").lower().lstrip(".")
    aliases: list[str] = []
    if format_name in {"yaml", "yml"} or (
        not format_name and re.search(r"(?m)^\s*(?:services|command|entrypoint):", text)
    ):
        aliases.extend(_serve_model_ids_from_yaml(text))
    elif format_name in {"md", "markdown"}:
        for match in _MARKDOWN_FENCE.finditer(text):
            language = match.group("language").lower()
            body = match.group("body")
            if language in {"yaml", "yml"}:
                aliases.extend(_serve_model_ids_from_yaml(body))
            else:
                aliases.extend(
                    _serve_model_ids_from_shell(body, shell_surface=_is_shell_surface(language, body))
                )
    else:
        aliases.extend(
            _serve_model_ids_from_shell(text, shell_surface=_is_shell_surface(format_name, text))
        )

    return list(dict.fromkeys(aliases))


def _sources_from_yaml(text: str, *, context: str) -> list[str]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PublicationSourceParseError(f"{context}: invalid YAML serve surface: {error}") from error

    sources: list[str] = []
    command_count = 0
    serve_like_scalar = False

    def visit(value: object, location: str) -> None:
        nonlocal command_count, serve_like_scalar
        if isinstance(value, dict):
            declared = [key for key in _COMPOSE_COMMAND_KEYS if key in value]
            if declared:
                command_count += 1
                tokens: list[str] = []
                for key in declared:
                    tokens.extend(_command_tokens(value[key], context=f"{location}.{key}"))
                sources.extend(
                    _sources_from_tokens(
                        tokens, context=f"{location}.{'+'.join(declared)}", shell_surface=True
                    )
                )
            for key, child in value.items():
                if key in _COMPOSE_COMMAND_KEYS:
                    continue
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")
        elif isinstance(value, str) and not serve_like_scalar:
            serve_like_scalar = _shell_text_matches(value, _starts_serve_command)

    visit(document, context)
    if command_count == 0 and (
        serve_like_scalar
        or _VLLM_SERVE_TEXT.search(text)
        or re.search(r"(?<![\w-])--model(?:=|\s)", text)
    ):
        raise PublicationSourceParseError(
            f"{context}: serve/model syntax exists outside a parseable YAML command"
        )
    return sources


def extract_model_sources(text: str, *, source_format: str | None = None) -> list[str]:
    """Parse actual vLLM model sources and fail closed on ambiguous serve surfaces.

    YAML is parsed structurally, over both Compose `entrypoint` and `command` and over their
    combined execution semantics. Markdown code fences are parsed according to their language, and
    shell commands are read with bash's quoting rules (ANSI-C ``$'...'`` quotes decoded, control
    operators and continuations honoured) and support positional ``vllm serve`` sources, both
    model-option spellings, ``sh``/``bash -c`` and ``eval`` payloads (parsed recursively as shell
    text, including when shell options and their arguments precede the ``-c``), and shell parameter
    defaults (whose default source is extracted and validated). On shell surfaces, command
    substitutions and command text passed as one argument are read as well. Any serve surface that
    cannot yield one concrete source — an unresolved shell variable, a locale-translated
    ``$"..."`` string, or a command name that only an expansion decides especially — raises
    `PublicationSourceParseError`. Concise ``--served-model-name`` aliases are intentionally
    ignored.
    """
    format_name = (source_format or "").lower().lstrip(".")
    sources: list[str] = []
    if format_name in {"yaml", "yml"} or (
        not format_name and re.search(r"(?m)^\s*(?:services|command|entrypoint):", text)
    ):
        sources.extend(_sources_from_yaml(text, context=source_format or "<yaml>"))
    elif format_name in {"md", "markdown"}:
        for fence_index, match in enumerate(_MARKDOWN_FENCE.finditer(text), start=1):
            language = match.group("language").lower()
            body = match.group("body")
            context = f"{source_format or '<markdown>'}:fence-{fence_index}"
            if language in {"yaml", "yml"}:
                sources.extend(_sources_from_yaml(body, context=context))
            else:
                sources.extend(
                    _sources_from_shell(
                        body, context=context, shell_surface=_is_shell_surface(language, body)
                    )
                )
    else:
        sources.extend(
            _sources_from_shell(
                text,
                context=source_format or "<text>",
                shell_surface=_is_shell_surface(format_name, text),
            )
        )

    deduplicated: list[str] = []
    for source in sources:
        if source and source not in deduplicated:
            deduplicated.append(source)
    return deduplicated


def model_source_precision_errors(
    source: str, requires_mixed_fp8: bool | None = None
) -> list[str]:
    """Validate a concrete vLLM model source without conflating it with the API alias.

    A source that still carries a shell expansion is not concrete and is rejected outright: it names
    no artifact, so it can never be waved through as a local path. Absolute local paths are
    validated by their owning profile, which must carry immutable artifact identity and precision
    metadata. Repository-like sources fail closed: Darkstar sources must use the owned namespace,
    and any NVFP4 repository source must encode ModelOpt plus W4A16/W4A4 and the derived Mixed-FP8
    class.
    """
    if "$" in source:
        return [
            f"{source}: model source is an unresolved shell variable, not a concrete artifact "
            f"path or repository id"
        ]
    if source.startswith("/"):
        return []

    errors = precision_encoded_id_errors(source, requires_mixed_fp8)
    if "Darkstar-" in source and not source.startswith(f"{DARKSTAR_HF_NAMESPACE}/"):
        errors.append(
            f"{source}: Darkstar model source must be a full "
            f"{DARKSTAR_HF_NAMESPACE}/Darkstar-* repository id"
        )
    return errors


def model_card_identity_ids(text: str) -> list[str]:
    """The Darkstar ids declared in a model card's first level-1 heading."""
    heading = _MARKDOWN_H1.search(text)
    if heading is None:
        return []
    return _DARKSTAR_ID.findall(heading.group(1))


def candidate_target_repository(candidate: dict[str, Any]) -> str:
    return str(candidate["target_repository"])


def candidate_target_repository_errors(candidate: dict[str, Any]) -> list[str]:
    """A candidate's reserved repository id must be exactly its precision-encoded candidate id."""
    cid = str(candidate["candidate_id"])
    target = candidate_target_repository(candidate)
    errors = [
        f"{cid}: {error}"
        for error in precision_encoded_id_errors(
            target, requires_mixed_fp8=candidate_requires_mixed_fp8(candidate)
        )
    ]
    if target != f"{DARKSTAR_HF_NAMESPACE}/{cid}":
        errors.append(
            f"{cid}: target_repository {target!r} must be "
            f"{DARKSTAR_HF_NAMESPACE}/{cid} so the repository id carries the candidate precision"
        )
    return errors


def target_repository_errors(product: dict[str, Any]) -> list[str]:
    """Return violations of the actual-publication-target rules for one product.

    A ModelOpt NVFP4 product may only resolve its target repository to a candidate that has been
    built; until a precision winner exists the target stays explicitly unresolved and only
    candidate-specific ids are reserved.
    """
    errors: list[str] = []
    pid = str(product["product_id"])
    role = str(product["role"])
    status = str(product.get("target_repository_status", ""))
    target = product.get("target_repository")
    candidates = product_candidates(product)

    # The evaluation matrix retains a BF16 control for matched deltas, but that cell is the
    # unchanged upstream reference rather than an owned Darkstar publication. Its resolved target is
    # therefore the exact pinned upstream repository; every derivative still follows the namespace
    # and precision-encoding rules below. Image families use the same exception for image-base-bf16.
    if role in _UPSTREAM_BF16_ROLES and status == "resolved" and target in UPSTREAM_BF16_REFERENCES:
        if candidates:
            errors.append(f"{pid}: the upstream BF16 reference must not declare Darkstar candidates")
        return errors

    if status not in TARGET_REPOSITORY_STATUSES:
        errors.append(
            f"{pid}: target_repository_status {status!r} is not one of "
            f"{TARGET_REPOSITORY_STATUSES}"
        )

    for candidate in candidates:
        errors.extend(candidate_target_repository_errors(candidate))

    if status == "unresolved_pending_precision_winner":
        if role not in MODELOPT_NVFP4_ROLES:
            errors.append(f"{pid}: only ModelOpt NVFP4 roles may leave the target unresolved")
        if target is not None:
            errors.append(
                f"{pid}: target_repository must be null while the precision winner is unselected, "
                f"got {target!r}"
            )
        if not candidates:
            errors.append(f"{pid}: an unresolved target requires candidate-specific repository ids")
        return errors

    if not isinstance(target, str) or not target:
        errors.append(f"{pid}: a resolved target_repository must be a non-empty repository id")
        return errors

    named = next(
        (c for c in candidates if candidate_target_repository(c) == target),
        None,
    )
    requires_mixed_fp8 = None if named is None else candidate_requires_mixed_fp8(named)
    errors.extend(
        f"{pid}: {error}"
        for error in precision_encoded_id_errors(target, requires_mixed_fp8=requires_mixed_fp8)
    )
    if not target.startswith(f"{DARKSTAR_HF_NAMESPACE}/"):
        errors.append(
            f"{pid}: target_repository {target!r} must live under {DARKSTAR_HF_NAMESPACE}/"
        )

    if role in MODELOPT_NVFP4_ROLES:
        # A resolved target must name a candidate that is genuinely in play: `selected` (a promoted
        # release winner) or `under_evaluation` (built and being tuned). A `not_built` candidate has
        # no artifact, and a `rejected` candidate lost its comparison; neither may become the target.
        eligible = {
            candidate_target_repository(c)
            for c in candidates
            if str(c["selection"]) in ("selected", "under_evaluation")
        }
        if target not in eligible:
            errors.append(
                f"{pid}: resolved target_repository {target!r} must name a built, non-rejected "
                f"candidate (selected or under_evaluation); eligible targets are {sorted(eligible)}"
            )

    return errors


def ledger_target_repository_errors(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for product in ledger["products"]:
        errors.extend(target_repository_errors(product))
    return errors


def ledger_candidates(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for product in ledger["products"] for c in product_candidates(product)]


def ledger_candidate_precision_errors(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    version = ledger.get("contract_version")
    for candidate in ledger_candidates(ledger):
        errors.extend(candidate_precision_errors(candidate, contract_version=str(version) if version else None))
    return errors


def ledger_mixed_fp8_requirements(ledger: dict[str, Any]) -> dict[str, bool]:
    """Map every id a ledger candidate owns to whether that id must encode `Mixed-FP8`.

    Both the bare candidate id (as a model card declares it in its identity heading) and the fully
    qualified repository id (as a serve example passes it) resolve to the same derived requirement,
    so any surface that publishes an id can be checked against its artifact's precision map.
    """
    requirements: dict[str, bool] = {}
    for candidate in ledger_candidates(ledger):
        required = candidate_requires_mixed_fp8(candidate)
        requirements[str(candidate["candidate_id"])] = required
        requirements[candidate_target_repository(candidate)] = required
    return requirements


def expected_card_identity_ids(product: dict[str, Any]) -> list[str]:
    """The ids a product's model card must declare in its identity heading.

    A resolved product declares exactly its target repository name. An unresolved ModelOpt product
    declares each candidate id instead, so the card never presents a bare slot id as its identity.
    """
    if str(product.get("target_repository_status", "")) == "resolved":
        return [str(product["target_repository"]).split("/")[-1]]
    return [str(c["candidate_id"]) for c in product_candidates(product)]


def _is_bf16_kv(value: str) -> bool:
    return value.strip().casefold().startswith("bf16")


def _is_fp8_kv(value: str) -> bool:
    return "fp8" in value.casefold()


def kv_cache_gate_errors(product: dict[str, Any]) -> list[str]:
    """Check ``kv_cache_effect_measured`` against the selected candidate's KV precision.

    The gate applies to ModelOpt NVFP4 roles. A selected candidate whose ``kv_cache`` starts with
    ``BF16`` (case-insensitive) must mark the gate ``not_applicable``. A selected candidate that
    serves FP8 KV must mark it ``verified``, ``in_progress``, or ``missing``.
    """
    role = str(product.get("role", ""))
    if role not in MODELOPT_NVFP4_ROLES:
        return []
    errors: list[str] = []
    pid = str(product.get("product_id", "<unknown>"))
    gates = product.get("gates", [])
    gate: dict[str, Any] | None = None
    if isinstance(gates, list):
        found = next(
            (
                item
                for item in gates
                if isinstance(item, dict) and item.get("id") == _KV_CACHE_EFFECT_GATE
            ),
            None,
        )
        if isinstance(found, dict):
            gate = found
    if gate is None:
        errors.append(f"{pid}: missing gate {_KV_CACHE_EFFECT_GATE}")
        return errors
    status = str(gate.get("status", ""))
    selected = [
        candidate
        for candidate in product_candidates(product)
        if str(candidate.get("selection", "")) == "selected"
    ]
    if not selected:
        if status != "not_applicable":
            errors.append(
                f"{pid}: {_KV_CACHE_EFFECT_GATE} must be not_applicable when no candidate is selected "
                f"(nothing is served), got {status!r}"
            )
        return errors
    for candidate in selected:
        cid = str(candidate.get("candidate_id", "<unnamed>"))
        precision_map = candidate.get("precision_map")
        if not isinstance(precision_map, dict) or "kv_cache" not in precision_map:
            errors.append(f"{pid}: selected candidate {cid} has no kv_cache precision")
            continue
        kv = str(precision_map["kv_cache"])
        if _is_bf16_kv(kv):
            if status != "not_applicable":
                errors.append(
                    f"{pid}: kv_cache {kv!r} is BF16 so {_KV_CACHE_EFFECT_GATE} "
                    f"must be not_applicable, got {status!r}"
                )
        elif _is_fp8_kv(kv):
            if status not in _FP8_KV_STATUSES:
                errors.append(
                    f"{pid}: FP8 kv_cache {kv!r} requires {_KV_CACHE_EFFECT_GATE} "
                    f"verified, in_progress, or missing, got {status!r}"
                )
        else:
            errors.append(f"{pid}: selected candidate {cid} has unrecognized kv_cache {kv!r}")
    return errors


def _duplicate_messages(field: str, values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    for value in values:
        seen[value] = seen.get(value, 0) + 1
    return [f"duplicate {field} {value!r}" for value, count in seen.items() if count > 1]


def ledger_uniqueness_errors(ledger: dict[str, Any]) -> list[str]:
    """Require unique product ids, resolved targets, serve aliases, and container names.

    A ledger may contain several products with the same role. Unresolved targets are null and are
    not compared. Duplicate null targets are not a collision.
    """
    products = ledger.get("products", [])
    if not isinstance(products, list):
        raise TypeError("products must be a list")
    records = [product for product in products if isinstance(product, dict)]
    errors = _duplicate_messages(
        "product_id", [str(product.get("product_id")) for product in records]
    )
    targets: list[str] = []
    aliases: list[str] = []
    containers: list[str] = []
    for product in records:
        status = str(product.get("target_repository_status", ""))
        target = product.get("target_repository")
        if status == "resolved" and isinstance(target, str) and target:
            targets.append(target)
        alias = product.get("served_model_alias")
        if isinstance(alias, str):
            aliases.append(alias)
        container = product.get("runtime_container_name")
        if isinstance(container, str):
            containers.append(container)
    errors.extend(_duplicate_messages("target_repository", targets))
    errors.extend(_duplicate_messages("served_model_alias", aliases))
    errors.extend(_duplicate_messages("runtime_container_name", containers))
    return errors


def quality_gate_errors(product: dict[str, Any]) -> list[str]:
    """Require the task-scoped full-denominator quality gate declared by contract v3.

    Applicability is read from each gate's ``applies_to_tasks`` in the v3 contract. A task listed
    there must carry the gate; any other task must not. ``quality_matched_full_denominator`` also
    requires non-empty ``metric`` and ``protocol`` strings.
    """
    errors: list[str] = []
    pid = str(product.get("product_id", "<unknown>"))
    task = product.get("task")
    if not isinstance(task, str) or not task:
        errors.append(f"{pid}: task is required")
        return errors
    contract = _v3_contract()
    gates = product.get("gates", [])
    if not isinstance(gates, list):
        errors.append(f"{pid}: gates must be a list")
        return errors
    present: dict[str, dict[str, Any]] = {}
    for gate in gates:
        if isinstance(gate, dict) and "id" in gate:
            present[str(gate["id"])] = gate
    for gate_def in contract["gates"]:
        if "applies_to_tasks" not in gate_def:
            continue
        gate_id = str(gate_def["id"])
        applies = task in gate_def["applies_to_tasks"]
        recorded = present.get(gate_id)
        if applies and recorded is None:
            errors.append(f"{pid}: task {task!r} requires gate {gate_id}")
        elif not applies and recorded is not None:
            errors.append(f"{pid}: task {task!r} must not carry gate {gate_id}")
        if recorded is not None and gate_id == _QUALITY_GATE:
            for field in ("metric", "protocol"):
                value = recorded.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{pid}: gate {gate_id} requires non-empty {field}")
    return errors


def _extend_ledger_errors(errors: list[str], path: Path, call: Callable[[], list[str]]) -> None:
    try:
        errors.extend(call())
    except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")


@functools.lru_cache(maxsize=1)
def _v3_contract() -> dict[str, Any]:
    """The v3 contract, read once per process (the every-ledger CI test calls it for every product)."""
    return load_json(contract_paths_for({"contract_version": "3.0.0"})[0])


def publication_claim_flag_errors(ledger: dict[str, Any]) -> list[str]:
    """``any_publication_claim`` must equal whether any product claims publication."""
    products = ledger.get("products", [])
    if not isinstance(products, list):
        return []
    derived = any(bool(p.get("publication_claim")) for p in products if isinstance(p, dict))
    declared = ledger.get("any_publication_claim")
    if bool(declared) != derived:
        return [f"any_publication_claim is {declared!r} but the products' publication_claim values imply {derived!r}"]
    return []


# Gates whose evidence must be a measurement or check of the exact product. Family-wide gates (provenance, release tag,
# card, recipe) legitimately share evidence across products.
PER_ARTIFACT_GATES: frozenset[str] = frozenset(
    {
        "artifact_manifest",
        "artifact_validation",
        "nvfp4_tensor_scale_validation",
        "kv_cache_effect_measured",
        "performance_profile",
        "serving_capacity_profile",
        "gpqa_matched_full_denominator",
        "quality_matched_full_denominator",
        "behavior_refusal_eval",
        "clean_download_boot_smoke",
    }
)


def inherited_evidence_errors(ledger: dict[str, Any]) -> list[str]:
    """A per-artifact gate verified on two products with the same evidence text and path is an inherited result."""
    seen: dict[tuple[str, str, str], list[str]] = {}
    products = ledger.get("products", [])
    if not isinstance(products, list):
        return []
    for product in products:
        if not isinstance(product, dict):
            continue
        pid = str(product.get("product_id", "<unknown>"))
        for gate in product.get("gates", []) or []:
            if not isinstance(gate, dict) or gate.get("status") != "verified":
                continue
            gate_id = str(gate.get("id", ""))
            if gate_id not in PER_ARTIFACT_GATES:
                continue
            key = (gate_id, str(gate.get("evidence", "")), str(gate.get("evidence_path", "")))
            seen.setdefault(key, []).append(pid)
    return [
        f"{gate_id}: identical verified evidence on {', '.join(pids)} (an inherited result; measure each product)"
        for (gate_id, _evidence, _path), pids in sorted(seen.items())
        if len(pids) > 1
    ]


def _schema_errors(path: Path, ledger: dict[str, Any], schema_path: Path) -> list[str]:
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    found = sorted(validator.iter_errors(ledger), key=lambda err: (err.json_path, err.message))
    return [f"{path}: {err.json_path}: {err.message}" for err in found]


def validate_ledger(path: Path) -> list[str]:
    """Validate one publication-readiness ledger and return every error string.

    Runs schema, lifecycle, target-repository, and candidate-precision checks. Contract v3 also
    runs uniqueness, KV-cache, and task quality-gate checks. A malformed ledger, including a legacy
    free-form file, becomes a list of error strings rather than an exception.
    """
    errors: list[str] = []
    try:
        ledger = load_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return [f"{path}: {type(exc).__name__}: {exc}"]

    major: int | None = None
    schema_path: Path | None = None
    try:
        _contract_path, schema_path = contract_paths_for(ledger)
        version = str(ledger["contract_version"])
        major = int(version.split(".", 1)[0])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        schema_path = None
        major = None

    if schema_path is not None:
        try:
            errors.extend(_schema_errors(path, ledger, schema_path))
        except (
            OSError,
            ValueError,
            TypeError,
            jsonschema.ValidationError,
            jsonschema.SchemaError,
        ) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    _extend_ledger_errors(errors, path, lambda: ledger_lifecycle_errors(ledger))
    _extend_ledger_errors(errors, path, lambda: ledger_target_repository_errors(ledger))
    _extend_ledger_errors(errors, path, lambda: ledger_candidate_precision_errors(ledger))
    if major == 3:
        _extend_ledger_errors(errors, path, lambda: ledger_uniqueness_errors(ledger))
        _extend_ledger_errors(errors, path, lambda: publication_claim_flag_errors(ledger))
        _extend_ledger_errors(errors, path, lambda: inherited_evidence_errors(ledger))
        products = ledger.get("products", [])
        if isinstance(products, list):
            for index, product in enumerate(products):
                if not isinstance(product, dict):
                    errors.append(f"{path}: products[{index}] is not an object")
                    continue

                def _kv(product: dict[str, Any] = product) -> list[str]:
                    return kv_cache_gate_errors(product)

                def _quality(product: dict[str, Any] = product) -> list[str]:
                    return quality_gate_errors(product)

                _extend_ledger_errors(errors, path, _kv)
                _extend_ledger_errors(errors, path, _quality)
        else:
            errors.append(f"{path}: TypeError: products is not a list")
    return errors
