"""Symlink publication checks for release-layout copies.

These tests fail while a symlink is followed and its target bytes are written
into the output as a normal file. They pass once those copies reject symlinks
before creating or replacing any output file, including when the path is
replaced by a symlink after a stat and before the open.

The race hook in tests/relayout_symlink_race/sitecustomize.py performs that
replacement inside open and link, after any earlier lstat.

Not covered here, and why:
- make_variants.py hardlinks with os.link. On this interpreter link is not in
  os.supports_follow_symlinks, and linking a symlink creates another symlink
  to the same target rather than a regular file of the target bytes.
- make_split.py and split_mse.py do not copy directory symlinks: is_file() is
  false, so the copy loop never runs for them.
- A non-JSON model.safetensors.index.json symlink raises in json.loads before
  either script writes a release file.
- quantize_nr8.py and quantize_ne8.py copyfile after ModelOpt export. Reaching
  that call needs a loaded model and ModelOpt, which this run does not execute.
- Hash walks (pipeline.sha256_tree, quantize output hashes, the inline manifest
  in scripts/qwen3_8/gpu_host/run_ablit_full_w4a4.sh) read file bytes but do
  not copy them into a new release file.
- apply_qwen38_abliteration.py skips names that start with ".".
"""
from __future__ import annotations

import ctypes
import json
import os
import stat
import struct
import subprocess
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file
if not sys.platform.startswith("linux") or not os.path.isdir("/proc/self/fd"):
    pytest.skip(
        "the output watch uses Linux inotify and the scripts bind through /proc/self/fd",
        allow_module_level=True,
    )

REPO = Path(__file__).resolve().parents[1]
LIGHTNING = REPO / "models" / "nemotron-3.5-lightning-r1" / "eval"
MAKE_SPLIT = LIGHTNING / "make_split.py"
SPLIT_MSE = LIGHTNING / "split_mse.py"
RESTORE_CONFIG = LIGHTNING / "restore_config.py"
MERGE_LORA = REPO / "models" / "jev-27b" / "scripts" / "merge_lora.py"
APPLY_ABLIT = REPO / "scripts" / "qwen3_8" / "apply_qwen38_abliteration.py"
HOOK = Path(__file__).resolve().parent / "relayout_symlink_race"
SRC = REPO / "src"

PLACEHOLDER = b"placeholder-secret-value"
BENIGN = b"benign-companion\n"
INSIDE = b"inside-companion\n"

IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
IN_NONBLOCK = 0o0004000
IN_CLOEXEC = 0o2000000
_WATCH_MASK = IN_CREATE | IN_CLOSE_WRITE | IN_MOVED_TO
_HEADER = struct.calcsize("iIII")

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.inotify_init1.argtypes = [ctypes.c_int]
_libc.inotify_init1.restype = ctypes.c_int
_libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
_libc.inotify_add_watch.restype = ctypes.c_int

_LAYERS = 1978
_TENSOR = "model.language_model.layers.0.mlp.down_proj.weight"
_SELECTOR = _TENSOR


def read_nofollow(path: Path, limit: int = 8 * 1024 * 1024) -> bytes | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    chunks: list[bytes] = []
    remaining = limit
    try:
        while remaining > 0:
            block = os.read(fd, min(remaining, 65536))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
    finally:
        os.close(fd)
    return b"".join(chunks)


def symlink_to(link: Path, target: Path, kind: str) -> None:
    if kind == "absolute":
        link.symlink_to(target)
        return
    if kind == "relative":
        link.symlink_to(Path(os.path.relpath(target, link.parent)))
        return
    raise AssertionError(kind)


def _walk(root: Path) -> list[tuple[Path, str]]:
    if root.is_symlink() or not root.exists():
        return []
    found: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        if base != root:
            found.append((base, base.relative_to(root).as_posix()))
        kept: list[str] = []
        for name in dirnames:
            path = base / name
            if path.is_symlink():
                found.append((path, path.relative_to(root).as_posix()))
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            path = base / name
            found.append((path, path.relative_to(root).as_posix()))
    return found


def node_id(path: Path) -> tuple[object, ...]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return ("missing",)
    if stat.S_ISLNK(info.st_mode):
        return ("symlink", info.st_ino, os.readlink(path))
    if stat.S_ISREG(info.st_mode):
        return ("file", info.st_ino, info.st_size, read_nofollow(path))
    if stat.S_ISDIR(info.st_mode):
        return ("dir", info.st_ino)
    return ("other", info.st_mode, info.st_ino)


def fingerprint(root: Path) -> tuple[tuple[str, tuple[object, ...]], ...]:
    return tuple((rel, node_id(path)) for path, rel in _walk(root))


def write_checkpoint(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "model-00001.safetensors")
    payload = {
        "metadata": {"format": "pt"},
        "weight_map": {name: "model-00001.safetensors" for name in tensors},
    }
    (directory / "model.safetensors.index.json").write_text(json.dumps(payload) + "\n")


def small_export(directory: Path) -> None:
    write_checkpoint(
        directory,
        {
            "backbone.embed.weight": torch.zeros(2, dtype=torch.float32),
            "backbone.layers.0.mixer.experts.0.up_proj.input_scale": torch.ones(
                1, dtype=torch.float32
            ),
        },
    )


def mse_names() -> list[str]:
    names: list[str] = []
    for layer in range(_LAYERS):
        base = f"backbone.layers.{layer}.mixer"
        names.append(f"{base}.shared_experts.up_proj.input_scale")
        names.append(f"{base}.shared_experts.down_proj.input_scale")
        names.append(f"{base}.experts.0.down_proj.input_scale")
    if len(names) != 5934:
        raise AssertionError(len(names))
    return names


def mse_export(directory: Path) -> Path:
    tensors = {name: torch.ones(1, dtype=torch.float32) for name in mse_names()}
    tensors["backbone.embed.weight"] = torch.zeros(2, dtype=torch.float32)
    write_checkpoint(directory, tensors)
    streams: dict[str, float] = {}
    for layer in range(_LAYERS):
        streams[f"L{layer}.up"] = 2688.0
        streams[f"L{layer}.sdown"] = 2688.0
        streams[f"L{layer}.rdown"] = 2688.0
    gpath = directory.parent / "variants-G.json"
    gpath.write_text(json.dumps({"mse": streams}))
    return gpath


def relocate_index(directory: Path, outside: Path) -> None:
    payload = json.loads((directory / "model.safetensors.index.json").read_text())
    payload["note"] = PLACEHOLDER.decode()
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text(json.dumps(payload) + "\n")
    index = directory / "model.safetensors.index.json"
    index.unlink()
    symlink_to(index, outside, "relative")


def lora_dirs(tmp: Path) -> tuple[Path, Path, Path]:
    base = tmp / "base"
    adapter = tmp / "adapter"
    write_checkpoint(base, {_TENSOR: torch.zeros((4, 4), dtype=torch.float32)})
    adapter.mkdir(parents=True)
    save_file(
        {
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros(
                (2, 4), dtype=torch.float32
            ),
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros(
                (4, 2), dtype=torch.float32
            ),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}) + "\n")
    (adapter / "decision_head.json").write_bytes(BENIGN)
    return base, adapter, tmp / "out"


def ablit_config(directory: Path) -> Path:
    path = directory / "experiment.yaml"
    path.write_text(
        """
schema_version: "1.0"
name: symlink-spec
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
abliteration:
  layer: 1
  seed: 1
  harmful_prompts: 1
  harmless_prompts: 1
  orthogonalize_harmless: false
  target_selectors: ["model.language_model.layers.0.mlp.down_proj.weight"]
  expected_target_count: 1
datasets:
  harmful:
    source: demo/harmful
    revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  calibration_samples: 1
  max_sequence_length: 8
  calibration_dataset: demo/cal
  calibration_config: LLM
  pipeline: basic
  shard_size: 5GB
  ignore: [lm_head, mtp]
  keep_bf16: [mtp]
validation:
  max_refusal_leakage: 0.01
  max_benign_kl_divergence: 0.05
  max_perplexity_delta_pct: 5.0
  vision_byte_identical: true
  mtp_present: true
runtime:
  kv_dtype: bf16
  context_length: 128
  compiled_mode: false
  flash_attention: false
  mtp_depth_initial: 1
  mtp_sweep_range: [1]
performance:
  target_tok_s: 1
  minimum_tok_s: 1
  warmup_repeats: 1
  measure_repeats: 1
  prompt_lengths: [8]
"""
    )
    return path


def ablit_model(directory: Path) -> tuple[Path, Path]:
    model = directory / "model"
    write_checkpoint(model, {_SELECTOR: torch.zeros((4, 4), dtype=torch.float32)})
    direction = directory / "direction"
    direction.mkdir()
    save_file(
        {"direction": torch.zeros(4, dtype=torch.float32)},
        direction / "direction.safetensors",
    )
    return model, direction


def restore_trees(tmp: Path) -> tuple[Path, Path]:
    source = tmp / "source"
    export = tmp / "export"
    source.mkdir()
    export.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "demo", "n": 1}) + "\n")
    (source / "generation_config.json").write_bytes(BENIGN)
    (export / "config.json").write_text(
        json.dumps({"model_type": "demo", "quantization_config": {"quant_algo": "NVFP4"}}) + "\n"
    )
    (export / "generation_config.json").write_bytes(BENIGN)
    return source, export


def run(
    script: Path,
    args: list[Path | str],
    race: tuple[Path, Path] | None = None,
    marker: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    parts = [str(HOOK), str(SRC)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.pop("RELAYOUT_RACE_SOURCE", None)
    env.pop("RELAYOUT_RACE_TARGET", None)
    env.pop("RELAYOUT_RACE_MARKER", None)
    if race is not None:
        env["RELAYOUT_RACE_SOURCE"] = str(race[0])
        env["RELAYOUT_RACE_TARGET"] = str(race[1])
        if marker is not None:
            env["RELAYOUT_RACE_MARKER"] = str(marker)
    return subprocess.run(
        [sys.executable, str(script), *[str(arg) for arg in args]],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _redact(text: str) -> str:
    return text.replace(PLACEHOLDER.decode(), "[placeholder]")


class OutputWatch:
    """Record files created or rewritten under output while a script runs."""

    def __init__(self, output: Path, ignore: frozenset[str] = frozenset()) -> None:
        self.output = output
        self.parent = output.parent
        self.ignore = ignore
        self.observed: set[str] = set()
        self.saw_placeholder = False
        self._baseline: dict[str, tuple[object, ...]] = {}
        self._stop = threading.Event()
        self._fd = _libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self._watches: dict[int, Path] = {}
        self._watched: set[Path] = set()
        self._thread = threading.Thread(target=self._loop, name="relayout-watch", daemon=True)

    def __enter__(self) -> OutputWatch:
        self._baseline = self._snapshot()
        self._add(self.parent)
        if self._is_real_dir(self.output):
            self._add(self.output)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self._scan()
        except FileNotFoundError:
            pass
        os.close(self._fd)

    def _is_real_dir(self, path: Path) -> bool:
        try:
            info = path.lstat()
        except OSError:
            return False
        return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)

    def _add(self, path: Path) -> None:
        if path in self._watched or not self._is_real_dir(path):
            return
        watch = _libc.inotify_add_watch(self._fd, os.fsencode(path), _WATCH_MASK)
        if watch < 0:
            return
        self._watches[watch] = path
        self._watched.add(path)

    def _snapshot(self) -> dict[str, tuple[object, ...]]:
        return {rel: node_id(path) for path, rel in _walk(self.output)}

    def _contains_placeholder(self, ident: tuple[object, ...]) -> bool:
        return bool(
            ident
            and ident[0] == "file"
            and isinstance(ident[-1], bytes)
            and PLACEHOLDER in ident[-1]
        )

    def _skip(self, rel: str) -> bool:
        name = rel.split("/")[-1]
        return name.endswith(".race-tmp") or name.endswith(".race-tmp (removed)")

    def _mark(self, rel: str, ident: tuple[object, ...]) -> None:
        if self._skip(rel):
            return
        if rel in self.ignore and not self._contains_placeholder(ident):
            return
        self.observed.add(rel)
        if self._contains_placeholder(ident):
            self.saw_placeholder = True

    def _scan(self) -> None:
        if self._is_real_dir(self.output) and self.output not in self._watched:
            self._add(self.output)
        current = self._snapshot()
        for rel, ident in current.items():
            if self._baseline.get(rel) != ident:
                self._mark(rel, ident)
            if self._is_real_dir(self.output / rel):
                self._add(self.output / rel)
        for rel in self._baseline:
            if rel not in current and not self._skip(rel):
                self.observed.add(f"{rel} (removed)")

    def _on_event(self, watch: int, mask: int, name: str) -> None:
        base = self._watches.get(watch)
        if base is None or mask & IN_IGNORED:
            self._watches.pop(watch, None)
            return
        full = base / name if name else base
        if mask & IN_ISDIR:
            self._add(full)
        if full == self.output:
            return
        try:
            rel = full.relative_to(self.output).as_posix()
        except ValueError:
            return
        if rel == ".":
            return
        if mask & IN_ISDIR:
            self.observed.add(rel)
            return
        if mask & (IN_CREATE | IN_CLOSE_WRITE | IN_MOVED_TO):
            if self._skip(rel):
                return
            if full.exists() or full.is_symlink():
                self._mark(rel, node_id(full))
            else:
                self.observed.add(rel)

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self._fd, 65536)
            except BlockingIOError:
                return
            offset = 0
            while offset + _HEADER <= len(data):
                watch, mask, _cookie, length = struct.unpack_from("iIII", data, offset)
                raw = data[offset + _HEADER : offset + _HEADER + length]
                offset += _HEADER + length
                name = raw.split(b"\x00", 1)[0].decode("utf-8", "surrogateescape")
                self._on_event(watch, mask, name)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan()
                self._drain()
            except FileNotFoundError:
                pass
            self._stop.wait(0.001)


def assert_refused(
    proc: subprocess.CompletedProcess[str],
    output: Path,
    watched: OutputWatch,
    export: Path | None,
    export_before: tuple[tuple[str, tuple[object, ...]], ...] | None,
    secret: Path | None,
    secret_before: tuple[object, ...] | None,
    marker: Path | None = None,
) -> None:
    problems: list[str] = []
    if marker is not None and not marker.is_file():
        problems.append("race hook did not run")
    if proc.returncode == 0:
        problems.append("exited 0")
    if watched.saw_placeholder:
        problems.append("placeholder bytes were written into the output")
    if watched.saw_placeholder or watched.observed:
        shown = ", ".join(sorted(watched.observed)) or "placeholder bytes"
        problems.append(f"output changed during the run: {shown}")
    if secret is not None and secret_before is not None and node_id(secret) != secret_before:
        problems.append("placeholder file was modified")
    if export is not None and export_before is not None and fingerprint(export) != export_before:
        problems.append("input tree changed")
    if problems:
        detail = _redact(proc.stderr[-2000:] + proc.stdout[-1000:])
        raise AssertionError("; ".join(problems) + "\n" + detail)


def race_marker(tmp: Path) -> Path:
    return tmp / "race-marker"


def test_make_split_copies_regular_companion(tmp_path: Path) -> None:
    export = tmp_path / "export"
    small_export(export)
    (export / "tokenizer.json").write_bytes(BENIGN)
    out = tmp_path / "out"
    proc = run(MAKE_SPLIT, [export, out])
    assert proc.returncode == 0, proc.stderr
    copied = out / "tokenizer.json"
    assert read_nofollow(copied) == BENIGN
    assert not copied.is_symlink()
    assert (out / "model.safetensors.index.json").is_file()
    assert (out / "model-00001.safetensors").is_file()
    assert (out / "model-input-scales.safetensors").is_file()


@pytest.mark.parametrize("name", ["tokenizer.json", "privacy.md"])
@pytest.mark.parametrize("kind", ["absolute", "relative"])
def test_make_split_rejects_symlink_companion(tmp_path: Path, name: str, kind: str) -> None:
    export = tmp_path / "export"
    small_export(export)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    symlink_to(export / name, secret, kind)
    before = fingerprint(export)
    secret_before = node_id(secret)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(MAKE_SPLIT, [export, out])
    assert_refused(proc, out, watched, export, before, secret, secret_before)


def test_make_split_rejects_internal_symlink(tmp_path: Path) -> None:
    export = tmp_path / "export"
    small_export(export)
    (export / "notes.txt").write_bytes(INSIDE)
    (export / "tokenizer.json").symlink_to(Path("notes.txt"))
    before = fingerprint(export)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(MAKE_SPLIT, [export, out])
    assert_refused(proc, out, watched, export, before, None, None)


def test_make_split_rejects_symlink_swapped_at_open(tmp_path: Path) -> None:
    export = tmp_path / "export"
    small_export(export)
    companion = export / "tokenizer.json"
    companion.write_bytes(BENIGN)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(MAKE_SPLIT, [export, out], race=(companion, secret), marker=marker)
    assert_refused(proc, out, watched, None, None, secret, secret_before, marker=marker)


def test_make_split_rejects_symlink_index(tmp_path: Path) -> None:
    export = tmp_path / "export"
    small_export(export)
    outside = tmp_path / "outside" / "index.json"
    relocate_index(export, outside)
    before = fingerprint(export)
    secret_before = node_id(outside)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(MAKE_SPLIT, [export, out])
    assert_refused(proc, out, watched, export, before, outside, secret_before)


def test_split_mse_copies_regular_companion(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = mse_export(export)
    (export / "tokenizer.json").write_bytes(BENIGN)
    out = tmp_path / "out"
    proc = run(SPLIT_MSE, [export, gpath, out])
    assert proc.returncode == 0, proc.stderr
    assert read_nofollow(out / "tokenizer.json") == BENIGN
    assert not (out / "tokenizer.json").is_symlink()


def test_split_mse_rejects_symlink_companion(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = mse_export(export)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    symlink_to(export / "tokenizer.json", secret, "relative")
    before = fingerprint(export)
    secret_before = node_id(secret)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(SPLIT_MSE, [export, gpath, out])
    assert_refused(proc, out, watched, export, before, secret, secret_before)


def test_split_mse_rejects_symlink_swapped_at_open(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = mse_export(export)
    companion = export / "tokenizer.json"
    companion.write_bytes(BENIGN)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(SPLIT_MSE, [export, gpath, out], race=(companion, secret), marker=marker)
    assert_refused(proc, out, watched, None, None, secret, secret_before, marker=marker)


def test_split_mse_rejects_symlink_index(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = mse_export(export)
    outside = tmp_path / "outside" / "index.json"
    relocate_index(export, outside)
    before = fingerprint(export)
    secret_before = node_id(outside)
    out = tmp_path / "out"
    with OutputWatch(out) as watched:
        proc = run(SPLIT_MSE, [export, gpath, out])
    assert_refused(proc, out, watched, export, before, outside, secret_before)


def test_restore_config_copies_regular_files(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    proc = run(RESTORE_CONFIG, [source, export])
    assert proc.returncode == 0, proc.stderr
    assert read_nofollow(export / "generation_config.json") == BENIGN
    assert (export / "generation_config.transformers5.json").is_file()
    assert (export / "config.transformers5.json").is_file()
    written = json.loads((export / "config.json").read_text())
    assert written["quantization_config"]["quant_algo"] == "NVFP4"
    assert written["n"] == 1


def test_restore_config_rejects_export_generation_symlink(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    generation = export / "generation_config.json"
    generation.unlink()
    symlink_to(generation, secret, "relative")
    before = fingerprint(export)
    secret_before = node_id(secret)
    with OutputWatch(export) as watched:
        proc = run(RESTORE_CONFIG, [source, export])
    assert_refused(proc, export, watched, export, before, secret, secret_before)


def test_restore_config_rejects_source_generation_symlink(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    generation = source / "generation_config.json"
    generation.unlink()
    symlink_to(generation, secret, "absolute")
    source_before = fingerprint(source)
    export_before = fingerprint(export)
    secret_before = node_id(secret)
    with OutputWatch(export) as watched:
        proc = run(RESTORE_CONFIG, [source, export])
    assert_refused(proc, export, watched, export, export_before, secret, secret_before)
    assert fingerprint(source) == source_before


def test_restore_config_rejects_export_generation_swapped_at_open(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    companion = export / "generation_config.json"
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    with OutputWatch(export, ignore=frozenset({"generation_config.json"})) as watched:
        proc = run(RESTORE_CONFIG, [source, export], race=(companion, secret), marker=marker)
    assert_refused(proc, export, watched, None, None, secret, secret_before, marker=marker)


def test_restore_config_rejects_source_generation_swapped_at_open(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    companion = source / "generation_config.json"
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    with OutputWatch(export) as watched:
        proc = run(RESTORE_CONFIG, [source, export], race=(companion, secret), marker=marker)
    assert_refused(proc, export, watched, None, None, secret, secret_before, marker=marker)


def test_restore_config_rejects_source_config_symlink(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    outside = tmp_path / "outside" / "config.json"
    outside.parent.mkdir()
    outside.write_text(
        json.dumps({"model_type": "demo", "n": 1, "note": PLACEHOLDER.decode()}) + "\n"
    )
    config = source / "config.json"
    config.unlink()
    symlink_to(config, outside, "relative")
    source_before = fingerprint(source)
    export_before = fingerprint(export)
    secret_before = node_id(outside)
    with OutputWatch(export) as watched:
        proc = run(RESTORE_CONFIG, [source, export])
    assert_refused(proc, export, watched, export, export_before, outside, secret_before)
    assert fingerprint(source) == source_before


def test_restore_config_rejects_export_config_symlink(tmp_path: Path) -> None:
    source, export = restore_trees(tmp_path)
    outside = tmp_path / "outside" / "config.json"
    outside.parent.mkdir()
    outside.write_text(
        json.dumps(
            {
                "model_type": "demo",
                "quantization_config": {"quant_algo": "NVFP4"},
                "note": PLACEHOLDER.decode(),
            }
        )
        + "\n"
    )
    config = export / "config.json"
    config.unlink()
    symlink_to(config, outside, "absolute")
    before = fingerprint(export)
    secret_before = node_id(outside)
    with OutputWatch(export) as watched:
        proc = run(RESTORE_CONFIG, [source, export])
    assert_refused(proc, export, watched, export, before, outside, secret_before)


def test_merge_lora_copies_regular_companions(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    (base / "tokenizer.json").write_bytes(BENIGN)
    (tmp_path / "calibration.json").write_bytes(BENIGN)
    proc = run(MERGE_LORA, ["--base", base, "--adapter", adapter, "--out", out])
    assert proc.returncode == 0, proc.stderr
    assert read_nofollow(out / "tokenizer.json") == BENIGN
    assert read_nofollow(out / "decision_head.json") == BENIGN
    assert read_nofollow(out / "calibration.json") == BENIGN
    assert not (out / "tokenizer.json").is_symlink()


def test_merge_lora_rejects_base_symlink(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    symlink_to(base / "tokenizer.json", secret, "relative")
    before = fingerprint(base)
    secret_before = node_id(secret)
    with OutputWatch(out) as watched:
        proc = run(MERGE_LORA, ["--base", base, "--adapter", adapter, "--out", out])
    assert_refused(proc, out, watched, base, before, secret, secret_before)


def test_merge_lora_rejects_base_symlink_swapped_at_open(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    companion = base / "tokenizer.json"
    companion.write_bytes(BENIGN)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    with OutputWatch(out) as watched:
        proc = run(
            MERGE_LORA,
            ["--base", base, "--adapter", adapter, "--out", out],
            race=(companion, secret),
            marker=marker,
        )
    assert_refused(proc, out, watched, None, None, secret, secret_before, marker=marker)


def test_merge_lora_rejects_decision_head_symlink(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    decision = adapter / "decision_head.json"
    decision.unlink()
    symlink_to(decision, secret, "absolute")
    before = fingerprint(adapter)
    secret_before = node_id(secret)
    with OutputWatch(out) as watched:
        proc = run(MERGE_LORA, ["--base", base, "--adapter", adapter, "--out", out])
    assert_refused(proc, out, watched, adapter, before, secret, secret_before)


def test_merge_lora_rejects_decision_head_swapped_at_open(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    companion = adapter / "decision_head.json"
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    with OutputWatch(out) as watched:
        proc = run(
            MERGE_LORA,
            ["--base", base, "--adapter", adapter, "--out", out],
            race=(companion, secret),
            marker=marker,
        )
    assert_refused(proc, out, watched, None, None, secret, secret_before, marker=marker)


def test_merge_lora_rejects_calibration_symlink(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    calibration = tmp_path / "calibration.json"
    symlink_to(calibration, secret, "relative")
    secret_before = node_id(secret)
    calib_before = node_id(calibration)
    with OutputWatch(out) as watched:
        proc = run(MERGE_LORA, ["--base", base, "--adapter", adapter, "--out", out])
    assert node_id(calibration) == calib_before
    assert_refused(proc, out, watched, None, None, secret, secret_before)


def test_merge_lora_rejects_calibration_swapped_at_open(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    companion = tmp_path / "calibration.json"
    companion.write_bytes(BENIGN)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    with OutputWatch(out) as watched:
        proc = run(
            MERGE_LORA,
            ["--base", base, "--adapter", adapter, "--out", out],
            race=(companion, secret),
            marker=marker,
        )
    assert_refused(proc, out, watched, None, None, secret, secret_before, marker=marker)


def test_merge_lora_rejects_symlink_index(tmp_path: Path) -> None:
    base, adapter, out = lora_dirs(tmp_path)
    outside = tmp_path / "outside" / "index.json"
    relocate_index(base, outside)
    before = fingerprint(base)
    secret_before = node_id(outside)
    with OutputWatch(out) as watched:
        proc = run(MERGE_LORA, ["--base", base, "--adapter", adapter, "--out", out])
    assert_refused(proc, out, watched, base, before, outside, secret_before)


def test_apply_abliteration_copies_regular_companion(tmp_path: Path) -> None:
    config = ablit_config(tmp_path)
    model, direction = ablit_model(tmp_path)
    (model / "tokenizer.json").write_bytes(BENIGN)
    run_root = tmp_path / "run"
    proc = run(
        APPLY_ABLIT,
        ["--config", config, "--run-root", run_root, "--direction-dir", direction, "--source-model", model],
    )
    assert proc.returncode == 0, proc.stderr
    copied = run_root / "apply_abliteration" / "tokenizer.json"
    assert read_nofollow(copied) == BENIGN
    assert not copied.is_symlink()


def test_apply_abliteration_rejects_symlink_companion(tmp_path: Path) -> None:
    config = ablit_config(tmp_path)
    model, direction = ablit_model(tmp_path)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    symlink_to(model / "tokenizer.json", secret, "relative")
    before = fingerprint(model)
    secret_before = node_id(secret)
    run_root = tmp_path / "run"
    with OutputWatch(run_root) as watched:
        proc = run(
            APPLY_ABLIT,
            [
                "--config",
                config,
                "--run-root",
                run_root,
                "--direction-dir",
                direction,
                "--source-model",
                model,
            ],
        )
    assert_refused(proc, run_root, watched, model, before, secret, secret_before)


def test_apply_abliteration_rejects_symlink_swapped_at_open(tmp_path: Path) -> None:
    config = ablit_config(tmp_path)
    model, direction = ablit_model(tmp_path)
    companion = model / "tokenizer.json"
    companion.write_bytes(BENIGN)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    secret_before = node_id(secret)
    marker = race_marker(tmp_path)
    run_root = tmp_path / "run"
    with OutputWatch(run_root) as watched:
        proc = run(
            APPLY_ABLIT,
            [
                "--config",
                config,
                "--run-root",
                run_root,
                "--direction-dir",
                direction,
                "--source-model",
                model,
            ],
            race=(companion, secret),
            marker=marker,
        )
    assert_refused(proc, run_root, watched, None, None, secret, secret_before, marker=marker)


def test_apply_abliteration_rejects_symlinked_directory(tmp_path: Path) -> None:
    config = ablit_config(tmp_path)
    model, direction = ablit_model(tmp_path)
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    secret = outside / "token"
    secret.write_bytes(PLACEHOLDER)
    (model / "assets").symlink_to(Path(os.path.relpath(outside, model)))
    before = fingerprint(model)
    secret_before = node_id(secret)
    run_root = tmp_path / "run"
    with OutputWatch(run_root) as watched:
        proc = run(
            APPLY_ABLIT,
            [
                "--config",
                config,
                "--run-root",
                run_root,
                "--direction-dir",
                direction,
                "--source-model",
                model,
            ],
        )
    assert_refused(proc, run_root, watched, model, before, secret, secret_before)


def test_checkpoint_assets_write_the_files_they_opened(tmp_path: Path) -> None:
    """Assets are opened before the run root exists and written later: the bytes written are the
    bytes of the file that was opened then, even if a link is swapped in at that path meanwhile."""
    sys.path.insert(0, str(SRC))
    try:
        from model_forge.pipeline import CheckpointAssets
    finally:
        sys.path.remove(str(SRC))

    model = tmp_path / "model"
    model.mkdir()
    companion = model / "tokenizer.json"
    companion.write_bytes(BENIGN)
    secret = tmp_path / "outside" / "token"
    secret.parent.mkdir()
    secret.write_bytes(PLACEHOLDER)
    assets = CheckpointAssets(model, skip=lambda name: False)
    companion.unlink()
    symlink_to(companion, secret, "relative")
    dest = tmp_path / "dest"
    dest.mkdir()
    with assets:
        written = assets.write(dest)
    assert written == ["tokenizer.json"]
    assert read_nofollow(dest / "tokenizer.json") == BENIGN
