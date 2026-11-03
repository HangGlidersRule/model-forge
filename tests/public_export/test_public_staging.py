from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage_public_root.sh"


def test_staging_script_is_bounded_and_uses_canonical_export_and_verify() -> None:
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"$model_forge" public-export' in text
    assert '$source_repo/.venv/bin/model-forge' in text
    assert 'PYTHONPATH="$source_repo/src"' in text
    assert "verify_public_export.sh" in text
    assert "--source-sha" in text
    assert "PUBLIC_EXPORT_MANIFEST.json" in text
    assert ".git" not in text
    assert ".hermes" not in text
    assert "curl " not in text
    assert "wget " not in text


def test_staging_script_requires_a_fleet_hostname_denylist() -> None:
    """Issue #29: the fleet-hostname detector and the host-name sanitizer only run with a denylist. The staging script
    must pass one to both the export and the verify step, and refuse to stage without it, so an export can never again
    run with the check silently disabled."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "MODEL_FORGE_FLEET_HOSTNAMES" in text
    assert '--fleet-hostname-denylist "$fleet_hostnames"' in text
    assert "fleet hostname denylist is required" in text
    verifier = (ROOT / "scripts/verify_public_export.sh").read_text(encoding="utf-8")
    assert '--fleet-hostname-denylist "$fleet_hostnames"' in verifier
    assert "fleet hostname denylist is required" in verifier


def test_staging_script_refuses_to_run_without_a_denylist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tools/public_export").mkdir(parents=True)
    (source / "tools/public_export/public-files.yaml").write_text("version: 1\nrules: []\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("MODEL_FORGE_FLEET_HOSTNAMES", None)
    environment["MODEL_FORGE_COMMAND"] = "/bin/true"
    proc = subprocess.run(
        [SCRIPT, tmp_path / "public", tmp_path / "summary.json", source, "a" * 40],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "fleet hostname denylist is required" in proc.stderr


def test_staging_script_writes_deterministic_summary_after_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "scripts").mkdir()
    manifest = source / "tools/public_export/public-files.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version: 1\nrules: []\n", encoding="utf-8")
    calls = tmp_path / "calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    output = tmp_path / "public"
    summary = tmp_path / "public.lock.json"
    sha = "a" * 40
    digest = "b" * 64

    model_forge = bin_dir / "model-forge"
    model_forge.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'export\n' >> "$CALLS"
while (( $# )); do
  if [[ "$1" == "--output" ]]; then output=$2; shift 2; else shift; fi
done
mkdir -p "$output"
cat > "$output/PUBLIC_EXPORT_MANIFEST.json" <<EOF
{"source_sha":"%s","payload_tree_sha256":"%s","files":[{},{}]}
EOF
"""
        % (sha, digest),
        encoding="utf-8",
    )
    model_forge.chmod(0o755)
    verifier = source / "scripts/verify_public_export.sh"
    verifier.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'verify\\n' >> \"$CALLS\"\n",
        encoding="utf-8",
    )
    verifier.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["CALLS"] = str(calls)
    denylist = tmp_path / "fleet-hostnames.txt"
    denylist.write_text("forge-worker-07.private\n", encoding="utf-8")
    environment["MODEL_FORGE_FLEET_HOSTNAMES"] = str(denylist)
    subprocess.run(
        [SCRIPT, output, summary, source, sha, tmp_path / "wheelhouse"],
        check=True,
        env=environment,
    )

    assert calls.read_text(encoding="utf-8").splitlines() == ["export", "verify"]
    expected = {
        "schema": "model-forge-public-staging/v1",
        "source_sha": sha,
        "export_digest": digest,
        "file_count": 2,
    }
    assert json.loads(summary.read_text(encoding="utf-8")) == expected
    first = summary.read_bytes()
    summary.unlink()
    subprocess.run(
        [SCRIPT, output, summary, source, sha, tmp_path / "wheelhouse"],
        check=True,
        env=environment,
    )
    assert summary.read_bytes() == first
