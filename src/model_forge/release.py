"""Loaders and validators for the Darkstar four-product release contract and per-family
publication-readiness ledgers.

The ledger JSON is the machine-readable source of truth. Human-readable statuses are rendered
into the Git-tracked benchmark matrices and model cards; this module provides the deterministic
checks that keep those renderings honest and prevent gate weakening.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

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

# Hugging Face namespace that owns every actual Darkstar weight repository.
DARKSTAR_HF_NAMESPACE = "HangGlidersRule"

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


def precision_map_requires_mixed_fp8(precision_map: dict[str, Any]) -> bool:
    """True when a precision map puts FP8 on any served weight path.

    Only `WEIGHT_PRECISION_MAP_COMPONENTS` are consulted, so the answer is a property of the served
    weights and never of runtime KV metadata. This is the authority for the `Mixed-FP8` naming
    requirement: enforcement derives the requirement from the map instead of trusting whatever the
    declared precision class or ids happen to spell, which is what makes it fail closed.
    """
    weights = " ".join(
        str(value)
        for component, value in precision_map.items()
        if component in WEIGHT_PRECISION_MAP_COMPONENTS
    )
    return "FP8" in weights


def candidate_requires_mixed_fp8(candidate: dict[str, Any]) -> bool:
    """Whether this candidate's every actual id must carry `Mixed-FP8`, per its precision map."""
    return precision_map_requires_mixed_fp8(candidate["precision_map"])


def candidate_precision_errors(candidate: dict[str, Any]) -> list[str]:
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

    missing = REQUIRED_PRECISION_MAP_COMPONENTS - set(pmap)
    if missing:
        errors.append(f"{cid}: precision_map omits components {sorted(missing)}")

    if pclass not in cid:
        errors.append(f"{cid}: candidate_id does not encode precision_class {pclass!r}")
    if "W4A16" not in cid and "W4A4" not in cid:
        errors.append(f"{cid}: candidate_id must encode W4A16 or W4A4")

    weights = " ".join(
        str(value)
        for component, value in pmap.items()
        if component in WEIGHT_PRECISION_MAP_COMPONENTS
    )
    has_fp8 = "FP8" in weights
    has_w4a16 = "W4A16" in weights
    requires_mixed_fp8 = precision_map_requires_mixed_fp8(pmap)

    if pclass.startswith("W4A4") and (has_fp8 or has_w4a16):
        errors.append(
            f"{cid}: named W4A4 but precision_map serves FP8/W4A16 weights (conflation)"
        )

    if requires_mixed_fp8:
        fp8_weights = sorted(
            component
            for component in WEIGHT_PRECISION_MAP_COMPONENTS & set(pmap)
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

    if "Mixed-FP8" in pclass and not (has_w4a16 and has_fp8):
        errors.append(f"{cid}: named Mixed-FP8 but precision_map is not W4A16+FP8")

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
    if has_mixed_fp8 and not has_w4a16:
        errors.append(f"{identifier}: Mixed-FP8 is only valid on a W4A16 id")
    if has_w4a4 and "FP8" in name:
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

_SHELL_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_SHELL_PARAMETER_DEFAULT_OPERATORS: tuple[str, ...] = (":-", ":=", "-", "=")

# ``vllm serve`` written as a structured sequence (``[vllm, serve, ...]``) as well as shell text.
_VLLM_SERVE_TEXT = re.compile(r"vllm[\s,\"']+serve")

# Nested `sh -c` payloads and nested parameter defaults are both bounded: an unbounded nest is
# itself an unreadable surface rather than something to keep unwrapping.
_MAX_SHELL_NESTING = 8

# Compose runs `entrypoint` followed by `command`, so a model source may be split across the two.
_COMPOSE_COMMAND_KEYS: tuple[str, ...] = ("entrypoint", "command")


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
    resolved = _resolve_model_source(default, context=context, depth=depth + 1)
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
    treated as a plain (non ``-c``) invocation when nothing serve-like remains in it to hide.
    """
    if any(_SHELL_SERVE_CONTENT.search(token) for token in tokens[index:]):
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


def _sources_from_tokens(tokens: list[str], *, context: str, depth: int = 0) -> list[str]:
    sources: list[str] = []
    saw_model_option = any(
        token == "--model" or token.startswith("--model=") for token in tokens
    )
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
                    )
                )
                index = payload_index + 1
                continue

        if token == "--model":
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise PublicationSourceParseError(f"{context}: --model has no parseable value")
            sources.append(_resolve_model_source(tokens[index + 1], context=context))
            index += 2
            continue

        if token.startswith("--model="):
            value = token.partition("=")[2]
            if not value:
                raise PublicationSourceParseError(f"{context}: --model has no parseable value")
            sources.append(_resolve_model_source(value, context=context))
            index += 1
            continue

        if tokens[index : index + 2] == ["vllm", "serve"]:
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
            sources.append(_resolve_model_source(positional, context=context))
            index += 3
            continue

        index += 1
    return sources


def _sources_from_shell(text: str, *, context: str, depth: int = 0) -> list[str]:
    if depth > _MAX_SHELL_NESTING:
        raise PublicationSourceParseError(f"{context}: shell command text nests too deeply")
    normalized = text.replace("\\\n", " ")
    sources: list[str] = []
    for line_number, line in enumerate(normalized.splitlines(), start=1):
        if "vllm" not in line and "--model" not in line:
            continue
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise PublicationSourceParseError(
                f"{context}:{line_number}: cannot parse serve/model command: {error}"
            ) from error
        sources.extend(
            _sources_from_tokens(tokens, context=f"{context}:{line_number}", depth=depth)
        )
    return sources


def _command_tokens(value: object, *, context: str) -> list[str]:
    if isinstance(value, str):
        try:
            return shlex.split(value.replace("\\\n", " "), comments=True, posix=True)
        except ValueError as error:
            raise PublicationSourceParseError(
                f"{context}: cannot parse Compose command: {error}"
            ) from error
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return [str(item) for item in value]
    raise PublicationSourceParseError(f"{context}: Compose command must be a string or scalar list")


def _serve_model_ids_from_tokens(
    tokens: list[str], *, context: str, depth: int = 0
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
                    )
                )
                index = payload_index + 1
                continue

        value: str | None = None
        if token == "--served-model-name" and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith("--served-model-name="):
            value = token.partition("=")[2]
        if value is not None and _SERVE_MODEL_ALIAS.fullmatch(value):
            aliases.append(value)
        index += 1
    return aliases


def _serve_model_ids_from_shell(
    text: str, *, context: str = "<shell>", depth: int = 0
) -> list[str]:
    if depth > _MAX_SHELL_NESTING:
        raise PublicationSourceParseError(f"{context}: shell command text nests too deeply")
    normalized = text.replace("\\\n", " ")
    aliases: list[str] = []
    for line_number, line in enumerate(normalized.splitlines(), start=1):
        if "--served-model-name" not in line:
            continue
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        aliases.extend(
            _serve_model_ids_from_tokens(
                tokens, context=f"{context}:{line_number}", depth=depth
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
                        tokens, context=f"{location}.{'+'.join(declared)}"
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
    """Extract concise API aliases passed through ``--served-model-name``.

    Compose YAML is parsed structurally so command scalar lists and strings have the same option
    semantics. Other surfaces are tokenized as shell text. Values passed through ``--model`` are
    deliberately ignored because publication model sources and runtime API aliases are separate
    identities.
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
                aliases.extend(_serve_model_ids_from_shell(body))
    else:
        aliases.extend(_serve_model_ids_from_shell(text))

    return list(dict.fromkeys(aliases))


def _sources_from_yaml(text: str, *, context: str) -> list[str]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PublicationSourceParseError(f"{context}: invalid YAML serve surface: {error}") from error

    sources: list[str] = []
    command_count = 0

    def visit(value: object, location: str) -> None:
        nonlocal command_count
        if isinstance(value, dict):
            declared = [key for key in _COMPOSE_COMMAND_KEYS if key in value]
            if declared:
                command_count += 1
                tokens: list[str] = []
                for key in declared:
                    tokens.extend(_command_tokens(value[key], context=f"{location}.{key}"))
                sources.extend(
                    _sources_from_tokens(tokens, context=f"{location}.{'+'.join(declared)}")
                )
            for key, child in value.items():
                if key in _COMPOSE_COMMAND_KEYS:
                    continue
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(document, context)
    if command_count == 0 and (
        _VLLM_SERVE_TEXT.search(text) or re.search(r"(?<![\w-])--model(?:=|\s)", text)
    ):
        raise PublicationSourceParseError(
            f"{context}: serve/model syntax exists outside a parseable YAML command"
        )
    return sources


def extract_model_sources(text: str, *, source_format: str | None = None) -> list[str]:
    """Parse actual vLLM model sources and fail closed on ambiguous serve surfaces.

    YAML is parsed structurally, over both Compose `entrypoint` and `command` and over their
    combined execution semantics. Markdown code fences are parsed according to their language, and
    shell commands support continuations, positional ``vllm serve`` sources, both model-option
    spellings, ``sh``/``bash -c`` payloads (parsed recursively as shell text, including when shell
    options and their arguments precede the ``-c``), and shell parameter defaults (whose default
    source is extracted and validated). Any serve surface that cannot yield
    one concrete source — an unresolved shell variable especially — raises
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
                sources.extend(_sources_from_shell(body, context=context))
    else:
        sources.extend(_sources_from_shell(text, context=source_format or "<text>"))

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

    # The four-cell evaluation matrix retains Base BF16 for matched deltas, but that cell is the
    # unchanged upstream reference rather than an owned Darkstar publication. Its resolved target is
    # therefore the exact pinned upstream repository; every derivative still follows the namespace
    # and precision-encoding rules below.
    if role == "base-bf16" and status == "resolved" and target == "Qwen/Qwen3.8-27B":
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
    for candidate in ledger_candidates(ledger):
        errors.extend(candidate_precision_errors(candidate))
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
