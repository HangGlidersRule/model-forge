"""Symlinks in an export must not be stored as ordinary files in a shipped layout.

The packaging scripts below copy companion files with shutil.copy2 / shutil.copyfile,
which follow links, or they hardlink after a following stat. These tests drive those
scripts. They fail while a symlink target is written into the output as a regular
file, including when the path is replaced by a symlink after the first lookup.

Untested lookalikes, and why:
- models/nemotron-3.5-lightning-r1/eval/make_variants.py uses os.link. On this
  machine that calls link(), which keeps a symlink as a symlink, so it does not
  store the target as a regular file.
- models/nemotron-3.5-embed-8b/scripts/quantize_ne8.py and
  models/nemotron-3.5-rerank-8b/scripts/quantize_nr8.py copy missing source files
  with shutil.copyfile, but only after a ModelOpt GPU export. That export cannot
  be run here, so those loops are not covered.
"""

from __future__ import annotations

import ctypes
import json
import os
import stat
import struct
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
SPLIT_MSE = REPO / "models/nemotron-3.5-lightning-r1/eval/split_mse.py"
MAKE_SPLIT = REPO / "models/nemotron-3.5-lightning-r1/eval/make_split.py"
RESTORE_CONFIG = REPO / "models/nemotron-3.5-lightning-r1/eval/restore_config.py"
MERGE_LORA = REPO / "models/jev-27b/scripts/merge_lora.py"
APPLY_ABLIT = REPO / "scripts/qwen3_8/apply_qwen38_abliteration.py"

SECRET = b"PLACEHOLDER_NOT_A_SECRET\n"
REGULAR = b'{"companion":"regular"}\n'
EXPERT_SCALE_COUNT = 5934

# inotify flags from /usr/include/linux/inotify.h and bits/inotify.h.
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_ISDIR = 0x40000000
_IN_CLOEXEC = 0o2000000
_IN_WATCH = _IN_CREATE | _IN_CLOSE_WRITE | _IN_MOVED_TO

_RACE_HOOK = textwrap.dedent(
    """\
    import os

    _race = os.environ["MF_RACE_PATH"]
    _target = os.environ["MF_RACE_TARGET"]
    _state = {"n": 0, "busy": False}
    _logfd = os.open(
        os.environ["MF_RACE_LOG"],
        os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
        0o644,
    )
    os.write(os.open(os.environ["MF_RACE_MARKER"], os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644), b"armed\\n")

    def _norm(path):
        if isinstance(path, int):
            return None
        try:
            return os.path.abspath(os.fsdecode(path))
        except (TypeError, ValueError):
            return None

    def _note(path):
        if _state["busy"] or _norm(path) != _race:
            return
        _state["n"] += 1
        os.write(_logfd, f"{_state['n']}\\n".encode())
        if _state["n"] >= 2:
            _state["busy"] = True
            os.unlink(_race)
            os.symlink(_target, _race)

    _real_stat = os.stat

    def _stat(path, *args, **kwargs):
        _note(path)
        return _real_stat(path, *args, **kwargs)

    os.stat = _stat

    _real_lstat = os.lstat

    def _lstat(path, *args, **kwargs):
        _note(path)
        return _real_lstat(path, *args, **kwargs)

    os.lstat = _lstat

    _real_open = os.open

    def _open(path, flags, *args, **kwargs):
        _note(path)
        return _real_open(path, flags, *args, **kwargs)

    os.open = _open

    _real_link = os.link

    def _link(src, dst, *args, **kwargs):
        _note(src)
        return _real_link(src, dst, *args, **kwargs)

    os.link = _link

    import builtins

    _real_bopen = builtins.open

    def _bopen(file, mode="r", *args, **kwargs):
        _note(file)
        return _real_bopen(file, mode, *args, **kwargs)

    builtins.open = _bopen

    import io

    _real_io_open = io.open

    def _io_open(file, mode="r", *args, **kwargs):
        _note(file)
        return _real_io_open(file, mode, *args, **kwargs)

    io.open = _io_open
    """
)


class _OutputWatch:
    """Record secret bytes, hardlinks, or symlinks created under output roots."""

    def __init__(
        self,
        roots: list[Path],
        secret: bytes,
        secret_ino: int,
        ignore_symlinks: set[Path] | None = None,
    ) -> None:
        self.roots = roots
        self.secret = secret
        self.secret_ino = secret_ino
        self.ignore_symlinks = ignore_symlinks or set()
        self.hits: list[str] = []
        self._wds: dict[int, Path] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        libc = _libc()
        self._libc = libc
        self._fd = libc.inotify_init1(_IN_CLOEXEC)
        if self._fd < 0:
            err = os.strerror(__import__("ctypes").get_errno())
            raise OSError(f"inotify_init1 failed: {err}")

    def __enter__(self) -> _OutputWatch:
        parents: set[Path] = set()
        for root in self.roots:
            root.parent.mkdir(parents=True, exist_ok=True)
            parents.add(root.parent)
            if root.exists():
                self._arm(root)
        for parent in parents:
            self._arm(parent)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        os.close(self._fd)
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _arm(self, path: Path) -> None:
        wd = self._libc.inotify_add_watch(self._fd, os.fsencode(path), _IN_WATCH)
        if wd >= 0:
            self._wds[wd] = path

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = os.read(self._fd, 4096)
            except OSError:
                return
            offset = 0
            while offset + 16 <= len(chunk):
                wd, mask, _cookie, name_len = struct.unpack_from("iIII", chunk, offset)
                offset += 16
                raw = chunk[offset : offset + name_len].split(b"\x00", 1)[0]
                offset += name_len
                base = self._wds.get(wd)
                if base is None or not raw:
                    continue
                path = base / os.fsdecode(raw)
                if mask & _IN_CREATE and mask & _IN_ISDIR:
                    self._arm(path)
                if self._inside(path):
                    self._inspect(path)

    def _inside(self, path: Path) -> bool:
        for root in self.roots:
            if path == root or root in path.parents:
                return True
        return False

    def _inspect(self, path: Path) -> None:
        try:
            info = path.lstat()
        except OSError:
            return
        if stat.S_ISLNK(info.st_mode):
            if path in self.ignore_symlinks:
                return
            self.hits.append(f"symlink {path}")
            return
        if not stat.S_ISREG(info.st_mode):
            return
        if info.st_ino == self.secret_ino:
            self.hits.append(f"hardlink {path}")
        try:
            data = path.read_bytes()
        except OSError:
            return
        if self.secret in data:
            self.hits.append(f"bytes {path}")


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    return libc


def _secret(tmp: Path) -> Path:
    path = tmp / "outside" / "credential-file"
    path.parent.mkdir(parents=True)
    path.write_bytes(SECRET)
    return path


def _plant_symlink(link: Path, secret: Path) -> None:
    link.symlink_to(secret)


def _snapshot(path: Path) -> dict[str, object]:
    info = path.lstat()
    return {
        "data": path.read_bytes(),
        "ino": info.st_ino,
        "nlink": info.st_nlink,
        "mode": info.st_mode,
    }


def _run(
    args: list[str], tmp: Path, race: Path | None = None, race_target: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = [str(REPO / "src")]
    if race is not None:
        assert race_target is not None
        hook = tmp / "race-hook"
        hook.mkdir()
        (hook / "sitecustomize.py").write_text(_RACE_HOOK)
        env["MF_RACE_PATH"] = str(race.resolve())
        env["MF_RACE_TARGET"] = str(race_target.resolve())
        env["MF_RACE_LOG"] = str(tmp / "race-lookups.log")
        env["MF_RACE_MARKER"] = str(tmp / "race-armed.txt")
        pythonpath.insert(0, str(hook))
    env["PYTHONPATH"] = os.pathsep.join([*pythonpath, env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_secret_stays(secret: Path, before: dict[str, object]) -> None:
    after = secret.lstat()
    assert secret.read_bytes() == before["data"]
    assert after.st_ino == before["ino"]
    assert after.st_nlink == before["nlink"]
    assert stat.S_ISREG(after.st_mode)


def _assert_tree_has_no_secret(
    roots: list[Path],
    secret: Path,
    before: dict[str, object],
    watch: _OutputWatch,
    preexisting_symlinks: set[Path] | None = None,
) -> None:
    allowed = preexisting_symlinks or set()
    assert watch.hits == []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                if path in allowed:
                    continue
                raise AssertionError(f"output contains a symlink: {path} -> {os.readlink(path)}")
            if not path.is_file():
                continue
            info = path.lstat()
            assert info.st_ino != before["ino"]
            assert SECRET not in path.read_bytes()
            assert not stat.S_ISLNK(info.st_mode)
    outside = sorted(p.name for p in secret.parent.iterdir())
    assert outside == [secret.name]


def _assert_race_hook_loaded(tmp: Path) -> None:
    marker = tmp / "race-armed.txt"
    log = tmp / "race-lookups.log"
    assert marker.read_text() == "armed\n"
    assert log.read_text().strip() != ""


def _write_scale_shard(export: Path, *, expert_scales: int) -> None:
    shard = "model-00001.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    for index in range(expert_scales):
        key = f"backbone.layers.0.mixer.experts.{index}.up_proj.input_scale"
        tensors[key] = torch.zeros(1, dtype=torch.float32)
        weight_map[key] = shard
    other = "backbone.embed_tokens.weight"
    tensors[other] = torch.ones(1, dtype=torch.float32)
    weight_map[other] = shard
    save_file(tensors, export / shard, metadata={"format": "pt"})
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "pt"}, "weight_map": weight_map}) + "\n"
    )


def _g_file(path: Path) -> None:
    path.write_text(json.dumps({"mse": {"L0.up": 2688.0}}) + "\n")


def _finished_without_publishing(
    proc: subprocess.CompletedProcess[str], out: Path, copied_name: str, copied: bytes
) -> None:
    if proc.returncode == 0:
        assert (out / copied_name).is_file()
        assert not (out / copied_name).is_symlink()
        assert (out / copied_name).read_bytes() == copied
        return
    detail = proc.stderr[-2000:] or proc.stdout[-2000:]
    assert proc.returncode != 0, detail


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    return _secret(tmp_path)


def test_split_mse_does_not_publish_symlink_target(tmp_path: Path, secret_file: Path) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _write_scale_shard(export, expert_scales=EXPERT_SCALE_COUNT)
    (export / "config.json").write_bytes(REGULAR)
    _plant_symlink(export / "tokenizer.json", secret_file)
    gpath = tmp_path / "variants-G.json"
    _g_file(gpath)
    before = _snapshot(secret_file)
    with _OutputWatch([out], SECRET, int(before["ino"])) as watch:
        proc = _run([str(SPLIT_MSE), str(export), str(gpath), str(out)], tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([out], secret_file, before, watch)
    assert not (out / "tokenizer.json").exists()
    _finished_without_publishing(proc, out, "config.json", REGULAR)


def test_split_mse_does_not_publish_symlink_swapped_after_first_lookup(
    tmp_path: Path, secret_file: Path
) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _write_scale_shard(export, expert_scales=EXPERT_SCALE_COUNT)
    (export / "config.json").write_bytes(REGULAR)
    raced = export / "tokenizer.json"
    raced.write_bytes(REGULAR)
    gpath = tmp_path / "variants-G.json"
    _g_file(gpath)
    before = _snapshot(secret_file)
    with _OutputWatch([out], SECRET, int(before["ino"])) as watch:
        proc = _run(
            [str(SPLIT_MSE), str(export), str(gpath), str(out)],
            tmp_path,
            race=raced,
            race_target=secret_file,
        )
    _assert_race_hook_loaded(tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([out], secret_file, before, watch)
    copied = out / "tokenizer.json"
    if copied.exists():
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_bytes() == REGULAR
    _finished_without_publishing(proc, out, "config.json", REGULAR)


def test_make_split_does_not_publish_symlink_target(tmp_path: Path, secret_file: Path) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _write_scale_shard(export, expert_scales=1)
    (export / "config.json").write_bytes(REGULAR)
    _plant_symlink(export / "tokenizer.json", secret_file)
    before = _snapshot(secret_file)
    with _OutputWatch([out], SECRET, int(before["ino"])) as watch:
        proc = _run([str(MAKE_SPLIT), str(export), str(out)], tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([out], secret_file, before, watch)
    assert not (out / "tokenizer.json").exists()
    _finished_without_publishing(proc, out, "config.json", REGULAR)


def test_make_split_does_not_publish_symlink_swapped_after_first_lookup(
    tmp_path: Path, secret_file: Path
) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _write_scale_shard(export, expert_scales=1)
    (export / "config.json").write_bytes(REGULAR)
    raced = export / "tokenizer.json"
    raced.write_bytes(REGULAR)
    before = _snapshot(secret_file)
    with _OutputWatch([out], SECRET, int(before["ino"])) as watch:
        proc = _run(
            [str(MAKE_SPLIT), str(export), str(out)], tmp_path, race=raced, race_target=secret_file
        )
    _assert_race_hook_loaded(tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([out], secret_file, before, watch)
    copied = out / "tokenizer.json"
    if copied.exists():
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_bytes() == REGULAR
    _finished_without_publishing(proc, out, "config.json", REGULAR)


def test_restore_config_does_not_publish_export_symlink(tmp_path: Path, secret_file: Path) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["Demo"]}) + "\n")
    (source / "generation_config.json").write_bytes(REGULAR)
    (export / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}) + "\n"
    )
    planted = export / "generation_config.json"
    _plant_symlink(planted, secret_file)
    before = _snapshot(secret_file)
    with _OutputWatch([export], SECRET, int(before["ino"]), ignore_symlinks={planted}) as watch:
        _run([str(RESTORE_CONFIG), str(source), str(export)], tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([export], secret_file, before, watch, preexisting_symlinks={planted})


def test_restore_config_does_not_publish_source_symlink(tmp_path: Path, secret_file: Path) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["Demo"]}) + "\n")
    _plant_symlink(source / "generation_config.json", secret_file)
    (export / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}) + "\n"
    )
    (export / "generation_config.json").write_bytes(REGULAR)
    before = _snapshot(secret_file)
    with _OutputWatch([export], SECRET, int(before["ino"])) as watch:
        _run([str(RESTORE_CONFIG), str(source), str(export)], tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([export], secret_file, before, watch)


def test_restore_config_does_not_publish_symlink_swapped_after_first_lookup(
    tmp_path: Path, secret_file: Path
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["Demo"]}) + "\n")
    raced = source / "generation_config.json"
    raced.write_bytes(REGULAR)
    (export / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}) + "\n"
    )
    (export / "generation_config.json").write_bytes(REGULAR)
    before = _snapshot(secret_file)
    with _OutputWatch([export], SECRET, int(before["ino"])) as watch:
        _run(
            [str(RESTORE_CONFIG), str(source), str(export)],
            tmp_path,
            race=raced,
            race_target=secret_file,
        )
    _assert_race_hook_loaded(tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([export], secret_file, before, watch)


def _lora_fixture(
    tmp: Path, secret: Path, *, base_symlink: bool, adapter_symlink: bool, calib_symlink: bool
) -> tuple[Path, Path, Path]:
    base = tmp / "base"
    adapter = tmp / "adapter-repo" / "adapter"
    out = tmp / "merged"
    base.mkdir()
    adapter.mkdir(parents=True)
    weight = "model.language_model.layers.0.mlp.down_proj.weight"
    shard = "model-00001.safetensors"
    save_file({weight: torch.zeros(2, 3, dtype=torch.float32)}, base / shard)
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight: shard}}) + "\n"
    )
    (base / "config.json").write_bytes(REGULAR)
    if base_symlink:
        _plant_symlink(base / "tokenizer.json", secret)
    else:
        (base / "tokenizer.json").write_bytes(REGULAR)
    save_file(
        {
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros(1, 3),
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros(2, 1),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}) + "\n")
    if adapter_symlink:
        _plant_symlink(adapter / "decision_head.json", secret)
    else:
        (adapter / "decision_head.json").write_bytes(REGULAR)
    calib = adapter.parent / "calibration.json"
    if calib_symlink:
        _plant_symlink(calib, secret)
    else:
        calib.write_bytes(REGULAR)
    return base, adapter, out


def test_merge_lora_does_not_publish_symlink_targets(tmp_path: Path, secret_file: Path) -> None:
    base, adapter, out = _lora_fixture(
        tmp_path, secret_file, base_symlink=True, adapter_symlink=True, calib_symlink=True
    )
    before = _snapshot(secret_file)
    with _OutputWatch([out], SECRET, int(before["ino"])) as watch:
        proc = _run(
            [str(MERGE_LORA), "--base", str(base), "--adapter", str(adapter), "--out", str(out)],
            tmp_path,
        )
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([out], secret_file, before, watch)
    for name in ("tokenizer.json", "decision_head.json", "calibration.json"):
        assert not (out / name).exists() or (out / name).read_bytes() == REGULAR
    if proc.returncode == 0:
        assert (out / "config.json").read_bytes() == REGULAR


def test_merge_lora_does_not_publish_symlink_swapped_after_first_lookup(
    tmp_path: Path, secret_file: Path
) -> None:
    base, adapter, out = _lora_fixture(
        tmp_path, secret_file, base_symlink=False, adapter_symlink=False, calib_symlink=False
    )
    raced = base / "tokenizer.json"
    before = _snapshot(secret_file)
    with _OutputWatch([out], SECRET, int(before["ino"])) as watch:
        proc = _run(
            [str(MERGE_LORA), "--base", str(base), "--adapter", str(adapter), "--out", str(out)],
            tmp_path,
            race=raced,
            race_target=secret_file,
        )
    _assert_race_hook_loaded(tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([out], secret_file, before, watch)
    copied = out / "tokenizer.json"
    if copied.exists():
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_bytes() == REGULAR
    if proc.returncode == 0:
        assert (out / "config.json").read_bytes() == REGULAR


def _abliteration_fixture(tmp: Path) -> tuple[Path, Path, Path]:
    model = tmp / "model"
    model.mkdir()
    weight = "model.layers.0.mlp.down_proj.weight"
    save_file({weight: torch.zeros(4, 4)}, model / "model-00001.safetensors")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight: "model-00001.safetensors"}}) + "\n"
    )
    (model / "config.json").write_bytes(REGULAR)
    direction = tmp / "direction"
    direction.mkdir()
    save_file({"direction": torch.zeros(4)}, direction / "direction.safetensors")
    config = tmp / "experiment.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            schema_version: "1.0"
            name: symlink-copy
            source:
              model_id: org/model
              revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
            abliteration:
              layer: 1
              seed: 0
              harmful_prompts: 1
              harmless_prompts: 1
              orthogonalize_harmless: false
              target_selectors: ["model.layers.0.mlp.down_proj.weight"]
              expected_target_count: 1
            datasets:
              harmful:
                source: demo/harmful
                revision: 01cead01398926d81f7c52bdb790ee8cf77ebba7
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
              ignore: ["re:.*mtp.*"]
              keep_bf16: [mtp]
            validation:
              max_refusal_leakage: 0.01
              max_benign_kl_divergence: 0.05
              max_perplexity_delta_pct: 5.0
              vision_byte_identical: true
              mtp_present: true
            runtime:
              kv_dtype: bf16
              context_length: 8
              compiled_mode: false
              flash_attention: false
              mtp_depth_initial: 1
              mtp_sweep_range: [1]
            performance:
              target_tok_s: 1
              minimum_tok_s: 1
              warmup_repeats: 1
              measure_repeats: 1
              prompt_lengths: [1]
            """
        )
    )
    return model, direction, config


def test_apply_abliteration_does_not_publish_symlink_targets(
    tmp_path: Path, secret_file: Path
) -> None:
    model, direction, config = _abliteration_fixture(tmp_path)
    _plant_symlink(model / "tokenizer.json", secret_file)
    sidecar = tmp_path / "outside-sidecar"
    sidecar.mkdir()
    (sidecar / "notes.txt").write_bytes(SECRET)
    (model / "sidecar").symlink_to(sidecar)
    run_root = tmp_path / "run"
    before = _snapshot(secret_file)
    with _OutputWatch([run_root], SECRET, int(before["ino"])) as watch:
        proc = _run(
            [
                str(APPLY_ABLIT),
                "--config",
                str(config),
                "--run-root",
                str(run_root),
                "--direction-dir",
                str(direction),
                "--source-model",
                str(model),
            ],
            tmp_path,
        )
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([run_root], secret_file, before, watch)
    if proc.returncode == 0:
        stage = run_root / "apply_abliteration"
        assert (stage / "config.json").read_bytes() == REGULAR
        assert not (stage / "tokenizer.json").exists()
        assert not (stage / "sidecar").exists()


def test_apply_abliteration_does_not_publish_symlink_swapped_after_first_lookup(
    tmp_path: Path, secret_file: Path
) -> None:
    model, direction, config = _abliteration_fixture(tmp_path)
    raced = model / "tokenizer.json"
    raced.write_bytes(REGULAR)
    run_root = tmp_path / "run"
    before = _snapshot(secret_file)
    with _OutputWatch([run_root], SECRET, int(before["ino"])) as watch:
        proc = _run(
            [
                str(APPLY_ABLIT),
                "--config",
                str(config),
                "--run-root",
                str(run_root),
                "--direction-dir",
                str(direction),
                "--source-model",
                str(model),
            ],
            tmp_path,
            race=raced,
            race_target=secret_file,
        )
    _assert_race_hook_loaded(tmp_path)
    _assert_secret_stays(secret_file, before)
    _assert_tree_has_no_secret([run_root], secret_file, before, watch)
    if proc.returncode == 0:
        copied = run_root / "apply_abliteration" / "tokenizer.json"
        if copied.exists():
            assert copied.is_file() and not copied.is_symlink()
            assert copied.read_bytes() == REGULAR


def test_split_mse_still_copies_regular_companions(tmp_path: Path) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _write_scale_shard(export, expert_scales=EXPERT_SCALE_COUNT)
    (export / "config.json").write_bytes(REGULAR)
    gpath = tmp_path / "variants-G.json"
    _g_file(gpath)
    proc = _run([str(SPLIT_MSE), str(export), str(gpath), str(out)], tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "config.json").is_file()
    assert not (out / "config.json").is_symlink()
    assert (out / "config.json").read_bytes() == REGULAR
    assert (out / "model-input-scales.safetensors").is_file()
