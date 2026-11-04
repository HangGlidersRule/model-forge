"""Packagers must refuse checkpoint symlinks instead of publishing the target bytes.

These tests fail while the copy loops follow symlinks. They pass once a run with a
symlink, or a file that becomes a symlink before it is read, stops without writing
that target into the output.

``quantize_nr8.py`` loads a GPU model before its sidecar copy. The quantize tests
execute the sidecar-copy statements in that file, from ``shards = set`` through the
copy loop, and every function defined in the file. The other tests call the real
script entry points.

``make_variants.py`` is not covered. ``os.link`` hardlinks a symlink inode, so a
relative link in the variant directory does not resolve to the outside file and the
credential bytes are not written there. ``run_ablit_full_w4a4.sh`` hashes an export
with ``Path.read_bytes`` and does not copy sidecars into a new artifact.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

QUANTIZE_NR8 = REPO_ROOT / "models/nemotron-3.5-rerank-8b/scripts/quantize_nr8.py"
MERGE_LORA = REPO_ROOT / "models/jev-27b/scripts/merge_lora.py"
MAKE_SPLIT = REPO_ROOT / "models/nemotron-3.5-lightning-r1/eval/make_split.py"
SPLIT_MSE = REPO_ROOT / "models/nemotron-3.5-lightning-r1/eval/split_mse.py"
RESTORE_CONFIG = REPO_ROOT / "models/nemotron-3.5-lightning-r1/eval/restore_config.py"
APPLY_ABLITERATION = REPO_ROOT / "scripts/qwen3_8/apply_qwen38_abliteration.py"

# Obvious placeholder. This is not a credential.
SENTINEL = b"placeholder-local-credential-value"
BENIGN = b"benign-sidecar"

_ARMED: dict[str, tuple[Path, Path]] = {}
_REMOVALS: list[str] = []
_HOOK = False


def _audit(event: str, args: tuple[object, ...]) -> None:
    """Swap an armed regular file for a relative symlink just before it is used."""
    if not args:
        return
    try:
        raw = os.fspath(args[0])  # type: ignore[arg-type]
    except TypeError:
        return
    path = os.path.abspath(raw)
    if event == "os.remove":
        _REMOVALS.append(path)
        return
    if event not in {"open", "os.link"}:
        return
    armed = _ARMED.pop(path, None)
    if armed is None:
        return
    decoy, secret = armed
    if decoy.is_symlink() or not decoy.exists():
        return
    decoy.unlink()
    decoy.symlink_to(os.path.relpath(secret, decoy.parent))


def _install_hook() -> None:
    global _HOOK
    if not _HOOK:
        sys.addaudithook(_audit)
        _HOOK = True


_install_hook()


@pytest.fixture
def credential(tmp_path: Path) -> Path:
    path = tmp_path / "outside" / "local_credential"
    path.parent.mkdir(parents=True)
    path.write_bytes(SENTINEL)
    return path


def _plant(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def _arm(decoy: Path, secret: Path) -> None:
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_bytes(BENIGN)
    _ARMED[os.path.abspath(decoy)] = (decoy, secret)


def _read_nofollow(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1 << 16):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _inspect(root: Path, sentinel: bytes) -> list[str]:
    if root.is_symlink():
        return [f"symlink:{root}"]
    if not root.exists():
        return []
    problems: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            path = Path(dirpath) / name
            if path.is_symlink():
                problems.append(f"symlink:{path.relative_to(root)}")
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root)
            if path.is_symlink():
                problems.append(f"symlink:{rel}")
                continue
            if path.is_file() and sentinel in _read_nofollow(path):
                problems.append(f"bytes:{rel}")
    return problems


def _assert_stopped(call: Callable[[], object], roots: list[Path], sentinel: bytes, secret: Path) -> None:
    """The run must fail, and the output must not gain the outside bytes or a link."""
    _REMOVALS.clear()
    watched = {root.resolve() for root in roots}
    try:
        call()
    except (SystemExit, OSError):
        refused = True
    else:
        refused = False
    problems: list[str] = []
    for root in roots:
        problems.extend(_inspect(root, sentinel))
    for removed in _REMOVALS:
        removed_path = Path(removed)
        if removed_path.name in {"NOTICE", "config.json", "tokenizer.json", "generation_config.json",
                                 "decision_head.json", "calibration.json", "note.txt"} and any(
            removed_path.is_relative_to(root) for root in watched
        ):
            problems.append(f"removed:{removed}")
    if secret.read_bytes() != SENTINEL:
        problems.append("outside file changed")
    assert not problems, problems
    assert refused, "packager completed without refusing the symlink"


def _run_path(script: Path, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(script), run_name=f"{script.stem}_under_test")
    finally:
        sys.argv = old


def _invoke(main: Callable[[], object], argv: list[str]) -> None:
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MERGE = _load("merge_lora_under_test", MERGE_LORA)
_ABLIT = _load("apply_abliteration_under_test", APPLY_ABLITERATION)


def _quantize_sidecar_copy(src: Path, partial: Path) -> None:
    """Run quantize_nr8.py's sidecar copy, including helpers defined in that file."""
    source = QUANTIZE_NR8.read_text()
    tree = ast.parse(source)
    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            functions.append(segment)
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("shards = set("))
    end = next(i for i, line in enumerate(lines) if line.startswith("cfg_src = json.loads"))
    block = "\n\n".join(functions) + "\n" + "\n".join(lines[start:end]) + "\n"
    namespace = {
        "src": src,
        "partial": partial,
        "json": json,
        "shutil": shutil,
        "os": os,
        "sys": sys,
        "Path": Path,
    }
    exec(compile(block, str(QUANTIZE_NR8), "exec"), namespace)


def _index(weight_map: dict[str, str]) -> str:
    return json.dumps({"weight_map": weight_map})


def _write_merge_pair(base: Path, adapter: Path) -> None:
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    weight = "model.language_model.layers.0.self_attn.q_proj.weight"
    shard = "model-00001.safetensors"
    save_file({weight: torch.ones(4, 4)}, base / shard)
    (base / "model.safetensors.index.json").write_text(_index({weight: shard}))
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.ones(2, 4),
            f"{prefix}.lora_B.weight": torch.ones(4, 2),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2.0}))


def _merge(base: Path, adapter: Path, out: Path) -> None:
    _invoke(
        _MERGE.main,  # type: ignore[attr-defined]
        ["merge_lora.py", "--base", str(base), "--adapter", str(adapter), "--out", str(out)],
    )


def _scale_checkpoint(root: Path, *, expert_count: int) -> None:
    tensors: dict[str, torch.Tensor] = {"lm_head.weight": torch.ones(1)}
    for index in range(expert_count):
        for proj in ("up", "down"):
            key = f"backbone.layers.0.mixer.experts.{index}.{proj}_proj.input_scale"
            tensors[key] = torch.zeros(1)
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        _index({key: "model.safetensors" for key in tensors})
    )


def _experiment(path: Path) -> None:
    path.write_text(
        """
schema_version: "1.0"
name: symlink-packaging
source:
  model_id: Qwen/Qwen3.8-27B
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
abliteration:
  layer: 38
  seed: 42
  harmful_prompts: 1
  harmless_prompts: 1
  orthogonalize_harmless: false
  target_selectors: ["model.layers.0.mlp.down_proj.weight"]
  expected_target_count: 1
datasets:
  harmful: {source: advbench, revision: 0123456789abcdef0123456789abcdef01234567}
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  calibration_samples: 1
  max_sequence_length: 8
  calibration_dataset: none
  calibration_config: none
  pipeline: basic
  shard_size: 5GB
  ignore: []
  keep_bf16: []
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


def _abliteration_model(model: Path, direction: Path) -> None:
    model.mkdir()
    direction.mkdir()
    weight = "model.layers.0.mlp.down_proj.weight"
    save_file({weight: torch.ones(4, 4)}, model / "model-00001.safetensors")
    (model / "model.safetensors.index.json").write_text(_index({weight: "model-00001.safetensors"}))
    save_file({"direction": torch.tensor([1.0, 0.0, 0.0, 0.0])}, direction / "direction.safetensors")


def _apply(config: Path, run_root: Path, direction: Path, model: Path) -> None:
    _invoke(
        _ABLIT.main,  # type: ignore[attr-defined]
        [
            "apply_qwen38_abliteration.py",
            "--config",
            str(config),
            "--run-root",
            str(run_root),
            "--direction-dir",
            str(direction),
            "--source-model",
            str(model),
        ],
    )


def _quantize_tree(src: Path) -> None:
    src.mkdir(parents=True)
    (src / "model.safetensors.index.json").write_text(_index({"w": "model.safetensors"}))


def test_quantize_nr8_copies_relative_symlink_outside_the_checkpoint(
    tmp_path: Path, credential: Path
) -> None:
    src = tmp_path / "source"
    partial = tmp_path / "export.partial"
    _quantize_tree(src)
    _plant(src / "extra" / "NOTICE", credential)
    partial.mkdir()
    _assert_stopped(lambda: _quantize_sidecar_copy(src, partial), [partial], SENTINEL, credential)


def test_quantize_nr8_refuses_symlink_whose_target_stays_inside_the_checkpoint(tmp_path: Path) -> None:
    src = tmp_path / "source"
    partial = tmp_path / "export.partial"
    _quantize_tree(src)
    inside = src / "README.txt"
    inside.write_bytes(BENIGN)
    _plant(src / "extra" / "NOTICE", inside)
    partial.mkdir()
    secret = tmp_path / "outside" / "local_credential"
    secret.parent.mkdir()
    secret.write_bytes(SENTINEL)
    _assert_stopped(lambda: _quantize_sidecar_copy(src, partial), [partial], SENTINEL, secret)


def test_quantize_nr8_open_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, credential: Path
) -> None:
    src = tmp_path / "source"
    partial = tmp_path / "export.partial"
    _quantize_tree(src)
    _arm(src / "extra" / "NOTICE", credential)
    partial.mkdir()
    _assert_stopped(lambda: _quantize_sidecar_copy(src, partial), [partial], SENTINEL, credential)


def test_merge_lora_copies_base_sidecar_symlink(tmp_path: Path, credential: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    out = tmp_path / "out"
    _write_merge_pair(base, adapter)
    (adapter / "decision_head.json").write_text("{}")
    _plant(base / "config.json", credential)
    _assert_stopped(lambda: _merge(base, adapter, out), [out], SENTINEL, credential)


def test_merge_lora_copies_decision_head_symlink(tmp_path: Path, credential: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    out = tmp_path / "out"
    _write_merge_pair(base, adapter)
    (base / "config.json").write_text("{}")
    _plant(adapter / "decision_head.json", credential)
    _assert_stopped(lambda: _merge(base, adapter, out), [out], SENTINEL, credential)


def test_merge_lora_copies_calibration_symlink(tmp_path: Path, credential: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "parent" / "adapter"
    out = tmp_path / "out"
    _write_merge_pair(base, adapter)
    (base / "config.json").write_text("{}")
    (adapter / "decision_head.json").write_text("{}")
    _plant(adapter.parent / "calibration.json", credential)
    _assert_stopped(lambda: _merge(base, adapter, out), [out], SENTINEL, credential)


def test_merge_lora_open_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, credential: Path
) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    out = tmp_path / "out"
    _write_merge_pair(base, adapter)
    (adapter / "decision_head.json").write_text("{}")
    _arm(base / "config.json", credential)
    _assert_stopped(lambda: _merge(base, adapter, out), [out], SENTINEL, credential)


def test_make_split_copies_sidecar_symlink(tmp_path: Path, credential: Path) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _scale_checkpoint(export, expert_count=1)
    _plant(export / "tokenizer.json", credential)
    _assert_stopped(
        lambda: _run_path(MAKE_SPLIT, ["make_split.py", str(export), str(out)]),
        [out],
        SENTINEL,
        credential,
    )


def test_make_split_open_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, credential: Path
) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _scale_checkpoint(export, expert_count=1)
    _arm(export / "tokenizer.json", credential)
    _assert_stopped(
        lambda: _run_path(MAKE_SPLIT, ["make_split.py", str(export), str(out)]),
        [out],
        SENTINEL,
        credential,
    )


def test_split_mse_copies_sidecar_symlink(tmp_path: Path, credential: Path) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _scale_checkpoint(export, expert_count=2967)
    _plant(export / "tokenizer.json", credential)
    variants = tmp_path / "variants.json"
    variants.write_text(json.dumps({"mse": {"L0.up": 2688.0, "L0.rdown": 2688.0}}))
    _assert_stopped(
        lambda: _run_path(SPLIT_MSE, ["split_mse.py", str(export), str(variants), str(out)]),
        [out],
        SENTINEL,
        credential,
    )


def test_split_mse_open_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, credential: Path
) -> None:
    export = tmp_path / "export"
    out = tmp_path / "out"
    export.mkdir()
    _scale_checkpoint(export, expert_count=2967)
    _arm(export / "tokenizer.json", credential)
    variants = tmp_path / "variants.json"
    variants.write_text(json.dumps({"mse": {"L0.up": 2688.0, "L0.rdown": 2688.0}}))
    _assert_stopped(
        lambda: _run_path(SPLIT_MSE, ["split_mse.py", str(export), str(variants), str(out)]),
        [out],
        SENTINEL,
        credential,
    )


def test_restore_config_copies_generation_config_symlink(tmp_path: Path, credential: Path) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["M"]}))
    _plant(source / "generation_config.json", credential)
    (export / "config.json").write_text(json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}))
    (export / "generation_config.json").write_text("{}")
    _assert_stopped(
        lambda: _run_path(RESTORE_CONFIG, ["restore_config.py", str(source), str(export)]),
        [export],
        SENTINEL,
        credential,
    )


def test_restore_config_rewrites_symlinked_config_json(tmp_path: Path, credential: Path) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    outside_config = credential.parent / "config.json"
    outside_config.write_text(json.dumps({"marker": SENTINEL.decode()}))
    _plant(source / "config.json", outside_config)
    (source / "generation_config.json").write_text("{}")
    (export / "config.json").write_text(json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}))
    (export / "generation_config.json").write_text("{}")
    _assert_stopped(
        lambda: _run_path(RESTORE_CONFIG, ["restore_config.py", str(source), str(export)]),
        [export],
        SENTINEL,
        credential,
    )


def test_restore_config_open_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, credential: Path
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["M"]}))
    _arm(source / "generation_config.json", credential)
    (export / "config.json").write_text(json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}))
    (export / "generation_config.json").write_text("{}")
    _assert_stopped(
        lambda: _run_path(RESTORE_CONFIG, ["restore_config.py", str(source), str(export)]),
        [export],
        SENTINEL,
        credential,
    )


def test_abliteration_copies_top_level_file_symlink(tmp_path: Path, credential: Path) -> None:
    model = tmp_path / "model"
    direction = tmp_path / "direction"
    run_root = tmp_path / "run"
    config = tmp_path / "experiment.yaml"
    _abliteration_model(model, direction)
    _plant(model / "config.json", credential)
    _experiment(config)
    _assert_stopped(
        lambda: _apply(config, run_root, direction, model),
        [run_root],
        SENTINEL,
        credential,
    )


def test_abliteration_copytree_follows_directory_symlink(tmp_path: Path, credential: Path) -> None:
    model = tmp_path / "model"
    direction = tmp_path / "direction"
    run_root = tmp_path / "run"
    config = tmp_path / "experiment.yaml"
    _abliteration_model(model, direction)
    (model / "config.json").write_text("{}")
    outside = credential.parent / "sidecar_dir"
    outside.mkdir()
    (outside / "note.txt").write_bytes(SENTINEL)
    _plant(model / "assets", outside)
    _experiment(config)
    _assert_stopped(
        lambda: _apply(config, run_root, direction, model),
        [run_root],
        SENTINEL,
        credential,
    )


def test_abliteration_copytree_follows_symlink_inside_directory(tmp_path: Path, credential: Path) -> None:
    model = tmp_path / "model"
    direction = tmp_path / "direction"
    run_root = tmp_path / "run"
    config = tmp_path / "experiment.yaml"
    _abliteration_model(model, direction)
    (model / "config.json").write_text("{}")
    _plant(model / "assets" / "NOTICE", credential)
    _experiment(config)
    _assert_stopped(
        lambda: _apply(config, run_root, direction, model),
        [run_root],
        SENTINEL,
        credential,
    )


def test_abliteration_open_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, credential: Path
) -> None:
    model = tmp_path / "model"
    direction = tmp_path / "direction"
    run_root = tmp_path / "run"
    config = tmp_path / "experiment.yaml"
    _abliteration_model(model, direction)
    _arm(model / "config.json", credential)
    _experiment(config)
    _assert_stopped(
        lambda: _apply(config, run_root, direction, model),
        [run_root],
        SENTINEL,
        credential,
    )
