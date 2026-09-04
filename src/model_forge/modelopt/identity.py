"""Complete build identity for a ModelOpt quantization artifact.

Idempotency decisions compare this identity, never the mere presence of a
``_SUCCESS.json``. The identity covers everything that changes the produced
checkpoint: the pin (version / commit / wheel digest), the selected candidate
and its recipe digest, the calibration contract, and the source identity. A
recipe edit under the same artifact name therefore invalidates the marker
instead of being masked by a stale success.

The identity dict doubles as the provenance record written next to the export,
so what is hashed is exactly what is recorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from model_forge.modelopt.calibration import CalibrationContract
from model_forge.modelopt.pin import ModelOptPin
from model_forge.pipeline import SUCCESS_MARKER, SuccessManifest, config_hash

IDENTITY_SCHEMA_VERSION = "1"

# Outcomes of comparing an existing artifact against the current identity.
ARTIFACT_ABSENT = "absent"
ARTIFACT_MATCH = "match"
ARTIFACT_NO_MARKER = "no_marker"
ARTIFACT_UNREADABLE_MARKER = "unreadable_marker"
ARTIFACT_IDENTITY_MISMATCH = "identity_mismatch"

CANONICAL_SOURCE_KINDS: tuple[str, ...] = ("clean", "abliterated")

# Pre-migration spellings of a source kind, mapped to the canonical term. Kept
# so callers and artifacts written before the terminology change stay usable;
# every new record is written with the canonical term.
LEGACY_SOURCE_KIND_ALIASES: Mapping[str, str] = MappingProxyType({"darkstar": "abliterated"})


def build_identity(
    *,
    pin: ModelOptPin,
    candidate: str,
    recipe: Path,
    recipe_sha256: str,
    calibration: CalibrationContract,
    source_kind: str,
    source_dir: Path,
) -> dict[str, Any]:
    """Return the full build identity / provenance record for one quantization."""
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "pin": pin.provenance(),
        "git_commit": pin.git_commit,
        "candidate": candidate,
        "recipe": str(recipe),
        "recipe_sha256": recipe_sha256,
        "calibration": calibration.to_dict(),
        "source_kind": source_kind,
        "source_dir": str(source_dir),
    }


def build_identity_sha(identity: Mapping[str, Any]) -> str:
    """Canonical SHA-256 over the whole identity record."""
    return config_hash(dict(identity))


def normalize_source_kind(source_kind: str) -> str:
    """Map a pre-migration source kind onto its canonical term."""
    return LEGACY_SOURCE_KIND_ALIASES.get(source_kind, source_kind)


def legacy_source_kind_identity_shas(identity: Mapping[str, Any]) -> tuple[str, ...]:
    """SHAs of this identity as it would have been recorded before the rename.

    Only the ``source_kind`` term is substituted. Pin, candidate, recipe path
    and digest, calibration, and source directory are all held fixed, so an
    artifact that differs in any of those still fails to match and is refused.
    """
    canonical = identity.get("source_kind")
    shas = []
    for legacy, current in LEGACY_SOURCE_KIND_ALIASES.items():
        if current == canonical:
            shas.append(build_identity_sha({**identity, "source_kind": legacy}))
    return tuple(shas)


def classify_artifact(
    export_dir: Path,
    expected_identity_sha: str,
    *,
    equivalent_identity_shas: Sequence[str] = (),
) -> tuple[str, str]:
    """Compare an existing artifact against the current build identity.

    Returns ``(outcome, detail)``. Only ``ARTIFACT_MATCH`` justifies an
    idempotent skip; every other non-absent outcome must refuse the build
    rather than reuse or clobber what is already there.

    ``equivalent_identity_shas`` lists digests of the *same* identity written
    under superseded terminology (see ``legacy_source_kind_identity_shas``).
    They are accepted as a match so a pure rename does not strand artifacts
    whose every other pinned input is unchanged.
    """
    if not export_dir.exists():
        return ARTIFACT_ABSENT, f"no existing artifact at {export_dir}"
    marker = export_dir / SUCCESS_MARKER
    if not marker.exists():
        return ARTIFACT_NO_MARKER, f"existing directory carries no {SUCCESS_MARKER}"
    try:
        manifest = SuccessManifest.from_json(marker.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return ARTIFACT_UNREADABLE_MARKER, f"unreadable {SUCCESS_MARKER}"
    if manifest.config_sha == expected_identity_sha:
        return ARTIFACT_MATCH, f"build identity {expected_identity_sha} matches"
    if manifest.config_sha and manifest.config_sha in equivalent_identity_shas:
        return ARTIFACT_MATCH, (
            f"build identity {manifest.config_sha} matches {expected_identity_sha} "
            "under superseded source-kind terminology"
        )
    recorded = manifest.config_sha or "<none>"
    return (
        ARTIFACT_IDENTITY_MISMATCH,
        f"build identity changed: recorded {recorded} != current {expected_identity_sha}",
    )
