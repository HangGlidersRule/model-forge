"""Symlink materialization in Lightning config restore, validation, and splits.

These tests fail while the tools follow a symlink and write its target into an
export or release directory. A passing run refuses the symlink before creating
or replacing any file, including when the path is replaced by a symlink after a
lookup and before the bytes are read. A clean copy of a regular file still
matches the pre-swap result.

Not exercised here:
make_variants.py links companion files with os.link. On this interpreter that
leaves the destination as a symlink, so it does not write a regular file.
quantize_ne8.py and quantize_nr8.py call shutil.copyfile only after a full
model export, which these tests do not run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "models" / "nemotron-3.5-lightning-r1" / "eval"
RESTORE = EVAL / "restore_config.py"
VALIDATE = EVAL / "validate_release.py"
VALIDATE_NVL = EVAL / "validate_release_nvl.py"
MAKE_SPLIT = EVAL / "make_split.py"
SPLIT_MSE = EVAL / "split_mse.py"
RACE_HOOK = Path(__file__).resolve().parent / "support" / "symlink_race"

MARKER = b"symlink-target"
PLACEHOLDER = b'{"placeholder":"symlink-target"}\n'
SOURCE_GENERATION = b'{"bos_token_id":1}\n'
EXPORT_GENERATION = b'{"bos_token_id":2}\n'
KEPT_NOTE = b"kept-regular\n"
COMPANIONS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
    "bias.md",
    "explainability.md",
    "privacy.md",
    "safety.md",
    "accuracy_plot.png",
    "agentic_coding_benchmarks.png",
    ".gitattributes",
)
SOURCE_CONFIG = {"model_type": "kept-source", "vocab_size": 3}
EXPORT_CONFIG = {
    "model_type": "export-before-restore",
    "vocab_size": 9,
    "quantization_config": {"quant_algo": "NVFP4", "ignore": []},
}

Snapshot = dict[str, tuple[object, ...]]


def _dumps(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode()


def _snapshot(root: Path) -> Snapshot:
    state: Snapshot = {}
    if not root.exists():
        return state
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            state[rel] = ("symlink", os.readlink(path))
        elif path.is_file():
            state[rel] = ("file", path.read_bytes())
        elif path.is_dir():
            state[rel] = ("dir",)
    return state


def _marker_hits(root: Path) -> list[str]:
    if not root.exists():
        return []
    hits: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if MARKER in path.read_bytes():
            hits.append(path.relative_to(root).as_posix())
    return hits


def _run(script: Path, args: list[Path], race: tuple[Path, Path] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("RESTORE_SYMLINK_RACE", "RESTORE_SYMLINK_RACE_PATH", "RESTORE_SYMLINK_RACE_TARGET"):
        env.pop(key, None)
    if race is not None:
        hook = str(RACE_HOOK)
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = hook if not previous else hook + os.pathsep + previous
        env["RESTORE_SYMLINK_RACE"] = "1"
        env["RESTORE_SYMLINK_RACE_PATH"] = str(race[0])
        env["RESTORE_SYMLINK_RACE_TARGET"] = str(race[1])
    return subprocess.run(
        [sys.executable, str(script), *[str(arg) for arg in args]],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _problem_text(proc: subprocess.CompletedProcess[str]) -> str:
    return f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _write_release_sides(src: Path, exp: Path) -> None:
    for name in COMPANIONS:
        (src / name).write_bytes(b"same-companion")
        (exp / name).write_bytes(b"same-companion")
    index = '{"weight_map":{}}\n'
    (src / "model.safetensors.index.json").write_text(index)
    (exp / "model.safetensors.index.json").write_text(index)
    quant = {
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
        }
    }
    (exp / "hf_quant_config.json").write_text(json.dumps(quant) + "\n")


def _prepare_restore(root: Path) -> tuple[Path, Path, Path]:
    src = root / "src"
    exp = root / "exp"
    outside = root / "outside"
    src.mkdir(parents=True)
    exp.mkdir()
    outside.mkdir()
    (src / "config.json").write_bytes(_dumps(SOURCE_CONFIG))
    (exp / "config.json").write_bytes(_dumps(EXPORT_CONFIG))
    (src / "generation_config.json").write_bytes(SOURCE_GENERATION)
    (exp / "generation_config.json").write_bytes(EXPORT_GENERATION)
    _write_release_sides(src, exp)
    return src, exp, outside


def _link_placeholder(directory: Path, name: str, outside: Path, payload: bytes = PLACEHOLDER) -> Path:
    target = outside / name
    target.write_bytes(payload)
    link = directory / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    return target


def _variant_file(root: Path) -> Path:
    path = root / "variants.json"
    path.write_text('{"mse":{}}\n')
    return path


def _validator_reports(art: Path, src: Path, root: Path) -> list[tuple[str, dict]]:
    variant = _variant_file(root)
    v110 = root / "v110"
    v110.mkdir()
    (v110 / "model.safetensors.index.json").write_text('{"weight_map":{}}\n')
    reports: list[tuple[str, dict]] = []
    commands = (
        ("validate_release", [art, src, variant, Path("mse"), root / "validate_release.json"]),
        ("validate_release_nvl", [art, src, v110, variant, root / "validate_release_nvl.json"]),
    )
    scripts = {"validate_release": VALIDATE, "validate_release_nvl": VALIDATE_NVL}
    for name, args in commands:
        proc = _run(scripts[name], args)
        out = args[-1]
        assert out.is_file(), f"{name} wrote no report\n{_problem_text(proc)}"
        reports.append((name, json.loads(out.read_text())))
    return reports


def _accepts_placeholder(report: dict, digest: str) -> bool:
    failures = [str(item) for item in report["failures"]]
    differs = any("generation_config.json differs" in item for item in failures)
    listed = report["release_files"].get("generation_config.json")
    return listed == digest and not differs


def _assert_refused(
    proc: subprocess.CompletedProcess[str],
    exp: Path,
    before: Snapshot,
    outside_files: list[tuple[Path, bytes]],
    reports: list[tuple[str, dict]] | None = None,
) -> None:
    problems: list[str] = []
    if proc.returncode == 0:
        problems.append(_problem_text(proc))
    after = _snapshot(exp)
    if after != before:
        problems.append(f"export changed\nbefore {sorted(before)}\nafter {sorted(after)}")
    hits = _marker_hits(exp)
    if hits:
        problems.append(f"placeholder bytes written into {hits}")
    for path, original in outside_files:
        if path.read_bytes() != original:
            problems.append(f"symlink target {path.name} was modified")
    if reports is not None:
        digest = hashlib.sha256(PLACEHOLDER).hexdigest()
        for name, report in reports:
            if _accepts_placeholder(report, digest):
                problems.append(f"{name} hashed the placeholder into release_files")
    assert not problems, "\n".join(problems)


def _without(snapshot: Snapshot, name: str | None) -> Snapshot:
    if name is None:
        return snapshot
    return {key: value for key, value in snapshot.items() if key != name}


def _assert_race_safe(
    proc: subprocess.CompletedProcess[str],
    before: Snapshot,
    after: Snapshot,
    golden: Snapshot,
    outside: Path,
    outside_before: bytes,
    output: Path,
    watched_name: str | None = None,
) -> None:
    # The harness itself replaces the watched path after a lookup. A refusal
    # may leave that path as a symlink and must not create or edit anything else.
    refused = proc.returncode != 0 and _without(after, watched_name) == _without(before, watched_name)
    copied_regular = proc.returncode == 0 and after == golden
    problems: list[str] = []
    if outside.read_bytes() != outside_before:
        problems.append("symlink target was modified")
    if not refused and not copied_regular:
        problems.append(
            _problem_text(proc) + f"\nbefore {sorted(before)}\nafter {sorted(after)}\ngolden {sorted(golden)}"
        )
    hits = _marker_hits(output)
    if hits:
        problems.append(f"placeholder bytes written into {hits}")
    assert not problems, "\n".join(problems)


def test_regular_configs_restore_into_the_export(tmp_path: Path) -> None:
    src, exp, _outside = _prepare_restore(tmp_path)
    proc = _run(RESTORE, [src, exp])
    assert proc.returncode == 0, _problem_text(proc)
    merged = dict(SOURCE_CONFIG)
    merged["quantization_config"] = EXPORT_CONFIG["quantization_config"]
    assert (exp / "config.json").read_bytes() == _dumps(merged)
    assert (exp / "generation_config.json").read_bytes() == SOURCE_GENERATION
    assert (exp / "generation_config.json").is_symlink() is False
    assert (exp / "config.transformers5.json").read_bytes() == _dumps(EXPORT_CONFIG)
    assert (exp / "generation_config.transformers5.json").read_bytes() == EXPORT_GENERATION


def test_source_generation_config_symlink_is_not_shipped(tmp_path: Path) -> None:
    src, exp, outside = _prepare_restore(tmp_path)
    target = _link_placeholder(src, "generation_config.json", outside)
    original = target.read_bytes()
    before = _snapshot(exp)
    proc = _run(RESTORE, [src, exp])
    reports = _validator_reports(exp, src, tmp_path)
    _assert_refused(proc, exp, before, [(target, original)], reports)


def test_source_generation_config_non_json_symlink_is_not_copied(tmp_path: Path) -> None:
    src, exp, outside = _prepare_restore(tmp_path)
    payload = b"placeholder-symlink-target\n"
    target = _link_placeholder(src, "generation_config.json", outside, payload)
    original = target.read_bytes()
    before = _snapshot(exp)
    proc = _run(RESTORE, [src, exp])
    _assert_refused(proc, exp, before, [(target, original)])


def test_source_config_json_symlink_is_not_copied(tmp_path: Path) -> None:
    src, exp, outside = _prepare_restore(tmp_path)
    payload = _dumps({"placeholder": "symlink-target", "model_type": "from-link"})
    target = _link_placeholder(src, "config.json", outside, payload)
    original = target.read_bytes()
    before = _snapshot(exp)
    proc = _run(RESTORE, [src, exp])
    _assert_refused(proc, exp, before, [(target, original)])


def test_export_generation_config_symlink_is_not_copied(tmp_path: Path) -> None:
    src, exp, outside = _prepare_restore(tmp_path)
    target = _link_placeholder(exp, "generation_config.json", outside)
    original = target.read_bytes()
    before = _snapshot(exp)
    proc = _run(RESTORE, [src, exp])
    _assert_refused(proc, exp, before, [(target, original)])


def test_export_config_json_symlink_is_not_copied(tmp_path: Path) -> None:
    src, exp, outside = _prepare_restore(tmp_path)
    payload = _dumps(
        {
            "placeholder": "symlink-target",
            "model_type": "from-link",
            "quantization_config": {"quant_algo": "NVFP4", "ignore": []},
        }
    )
    target = _link_placeholder(exp, "config.json", outside, payload)
    original = target.read_bytes()
    before = _snapshot(exp)
    proc = _run(RESTORE, [src, exp])
    _assert_refused(proc, exp, before, [(target, original)])


def _race_restore(tmp_path: Path, relative: str, payload: bytes = PLACEHOLDER) -> None:
    clean_root = tmp_path / "clean"
    raced_root = tmp_path / "raced"
    clean_src, clean_exp, _clean_outside = _prepare_restore(clean_root)
    raced_src, raced_exp, raced_outside = _prepare_restore(raced_root)
    in_export = relative.startswith("exp/")
    watched = (raced_exp if in_export else raced_src) / Path(relative).name
    target = raced_outside / "placeholder.json"
    target.write_bytes(payload)
    original = target.read_bytes()
    before = _snapshot(raced_exp)
    golden_proc = _run(RESTORE, [clean_src, clean_exp])
    assert golden_proc.returncode == 0, _problem_text(golden_proc)
    proc = _run(RESTORE, [raced_src, raced_exp], race=(watched, target))
    watched_name = Path(relative).name if in_export else None
    _assert_race_safe(
        proc,
        before,
        _snapshot(raced_exp),
        _snapshot(clean_exp),
        target,
        original,
        raced_exp,
        watched_name,
    )


def test_source_generation_config_swap_after_lookup_is_not_copied(tmp_path: Path) -> None:
    _race_restore(tmp_path, "src/generation_config.json")


def test_source_config_json_swap_after_lookup_is_not_copied(tmp_path: Path) -> None:
    _race_restore(tmp_path, "src/config.json")


def test_export_generation_config_swap_after_lookup_is_not_copied(tmp_path: Path) -> None:
    _race_restore(tmp_path, "exp/generation_config.json")


def test_export_config_json_swap_after_lookup_is_not_copied(tmp_path: Path) -> None:
    payload = _dumps(
        {
            "placeholder": "symlink-target",
            "model_type": "from-link",
            "quantization_config": {"quant_algo": "NVFP4", "ignore": []},
        }
    )
    _race_restore(tmp_path, "exp/config.json", payload)


def _expert_scale_names() -> list[str]:
    return [f"backbone.layers.0.mixer.experts.{index}.up_proj.input_scale" for index in range(5934)]


def _write_expert_shard(directory: Path) -> None:
    import torch
    from safetensors.torch import save_file

    names = _expert_scale_names()
    save_file({name: torch.ones(1) for name in names}, directory / "model.safetensors")
    weight_map = {name: "model.safetensors" for name in names}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _prepare_split_input(directory: Path, *, with_scales: bool) -> tuple[Path, Path]:
    directory.mkdir()
    outside = directory.parent / (directory.name + "-outside")
    outside.mkdir()
    if with_scales:
        _write_expert_shard(directory)
    else:
        (directory / "model.safetensors.index.json").write_text('{"weight_map":{}}\n')
    (directory / "config.json").write_bytes(_dumps(SOURCE_CONFIG))
    (directory / "generation_config.json").write_bytes(SOURCE_GENERATION)
    (directory / "notes.txt").write_bytes(KEPT_NOTE)
    return directory, outside


def _assert_output_absent(
    proc: subprocess.CompletedProcess[str],
    output: Path,
    outside_files: list[tuple[Path, bytes]],
) -> None:
    problems: list[str] = []
    if proc.returncode == 0:
        problems.append(_problem_text(proc))
    if output.exists():
        problems.append(f"output was created: {sorted(_snapshot(output))}")
    hits = _marker_hits(output)
    if hits:
        problems.append(f"placeholder bytes written into {hits}")
    for path, original in outside_files:
        if path.read_bytes() != original:
            problems.append(f"symlink target {path.name} was modified")
    assert not problems, "\n".join(problems)


def test_make_split_does_not_materialize_symlinks(tmp_path: Path) -> None:
    exp, outside = _prepare_split_input(tmp_path / "exp", with_scales=False)
    targets = [
        _link_placeholder(exp, "generation_config.json", outside),
        _link_placeholder(exp, "config.json", outside, _dumps({"placeholder": "symlink-target"})),
        _link_placeholder(exp, "notes.txt", outside, b"note-symlink-target\n"),
    ]
    originals = [(path, path.read_bytes()) for path in targets]
    output = tmp_path / "out"
    proc = _run(MAKE_SPLIT, [exp, output])
    _assert_output_absent(proc, output, originals)


def test_make_split_copies_regular_companion_files(tmp_path: Path) -> None:
    exp, _outside = _prepare_split_input(tmp_path / "exp", with_scales=False)
    output = tmp_path / "out"
    proc = _run(MAKE_SPLIT, [exp, output])
    assert proc.returncode == 0, _problem_text(proc)
    assert (output / "generation_config.json").is_symlink() is False
    assert (output / "generation_config.json").read_bytes() == SOURCE_GENERATION
    assert (output / "config.json").read_bytes() == _dumps(SOURCE_CONFIG)
    assert (output / "notes.txt").read_bytes() == KEPT_NOTE


def test_split_mse_does_not_materialize_symlinks(tmp_path: Path) -> None:
    exp, outside = _prepare_split_input(tmp_path / "exp", with_scales=True)
    targets = [
        _link_placeholder(exp, "generation_config.json", outside),
        _link_placeholder(exp, "config.json", outside, _dumps({"placeholder": "symlink-target"})),
        _link_placeholder(exp, "notes.txt", outside, b"note-symlink-target\n"),
    ]
    originals = [(path, path.read_bytes()) for path in targets]
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"mse": {"L0.up": 2688.0}}))
    output = tmp_path / "out"
    proc = _run(SPLIT_MSE, [exp, rules, output])
    _assert_output_absent(proc, output, originals)


def test_split_mse_copies_regular_companion_files(tmp_path: Path) -> None:
    exp, _outside = _prepare_split_input(tmp_path / "exp", with_scales=True)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"mse": {"L0.up": 2688.0}}))
    output = tmp_path / "out"
    proc = _run(SPLIT_MSE, [exp, rules, output])
    assert proc.returncode == 0, _problem_text(proc)
    assert (output / "generation_config.json").is_symlink() is False
    assert (output / "generation_config.json").read_bytes() == SOURCE_GENERATION
    assert (output / "config.json").read_bytes() == _dumps(SOURCE_CONFIG)
    assert (output / "notes.txt").read_bytes() == KEPT_NOTE


def _assert_split_race(proc: subprocess.CompletedProcess[str], output: Path, golden: Path, target: Path, original: bytes) -> None:
    refused = proc.returncode != 0 and not output.exists()
    copied = proc.returncode == 0 and output.is_dir() and _snapshot(output) == _snapshot(golden)
    problems: list[str] = []
    if target.read_bytes() != original:
        problems.append("symlink target was modified")
    if not refused and not copied:
        shown = sorted(_snapshot(output)) if output.exists() else None
        problems.append(_problem_text(proc) + f"\noutput {shown}")
    if _marker_hits(output):
        problems.append(f"placeholder bytes written into {_marker_hits(output)}")
    assert not problems, "\n".join(problems)


def test_make_split_swap_after_lookup_is_not_copied(tmp_path: Path) -> None:
    clean, _clean_outside = _prepare_split_input(tmp_path / "clean", with_scales=False)
    raced, outside = _prepare_split_input(tmp_path / "raced", with_scales=False)
    target = outside / "placeholder.json"
    target.write_bytes(PLACEHOLDER)
    original = target.read_bytes()
    golden = tmp_path / "golden"
    raced_out = tmp_path / "out"
    golden_proc = _run(MAKE_SPLIT, [clean, golden])
    assert golden_proc.returncode == 0, _problem_text(golden_proc)
    watched = raced / "generation_config.json"
    proc = _run(MAKE_SPLIT, [raced, raced_out], race=(watched, target))
    _assert_split_race(proc, raced_out, golden, target, original)


def test_split_mse_swap_after_lookup_is_not_copied(tmp_path: Path) -> None:
    clean, _clean_outside = _prepare_split_input(tmp_path / "clean", with_scales=True)
    raced, outside = _prepare_split_input(tmp_path / "raced", with_scales=True)
    target = outside / "placeholder.json"
    target.write_bytes(PLACEHOLDER)
    original = target.read_bytes()
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"mse": {"L0.up": 2688.0}}))
    golden = tmp_path / "golden"
    raced_out = tmp_path / "out"
    golden_proc = _run(SPLIT_MSE, [clean, rules, golden])
    assert golden_proc.returncode == 0, _problem_text(golden_proc)
    watched = raced / "generation_config.json"
    proc = _run(SPLIT_MSE, [raced, rules, raced_out], race=(watched, target))
    _assert_split_race(proc, raced_out, golden, target, original)
