"""Failing spec: quantize post-export copy must not publish symlink targets.

``quantize_ne8.py`` and ``quantize_nr8.py`` walk the BF16 source checkpoint and
``shutil.copyfile`` every non-shard file the exporter left out. ``copyfile``
follows symlinks, so a symlink is stored as a regular file of the target's
bytes and the partial directory is renamed to the release output.

These tests run that script path. They fail while the copy follows symlinks and
pass when every symlink is rejected before any output path for it is created.
A check that looks the path up and then opens it again still fails: the
check-then-use case swaps the regular file for a symlink on the read open.

Other copy sites are not this walk and are not covered here:
``scripts/qwen3_8/apply_qwen38_abliteration.py``, ``models/jev-27b/scripts/merge_lora.py``,
``models/nemotron-3.5-lightning-r1/eval/make_split.py``,
``models/nemotron-3.5-lightning-r1/eval/split_mse.py``,
``models/nemotron-3.5-lightning-r1/eval/restore_config.py``, and the quantize
scripts' later ``copyfile`` of ``partial/config.json`` onto
``config.transformers5.json``.
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NE8 = REPO_ROOT / "models" / "nemotron-3.5-embed-8b" / "scripts" / "quantize_ne8.py"
NR8 = REPO_ROOT / "models" / "nemotron-3.5-rerank-8b" / "scripts" / "quantize_nr8.py"
SCRIPTS = (NE8, NR8)

# Obvious placeholder. Not a credential.
OUTSIDE_MARKER = b"placeholder-outside-file"
INTERNAL_MARKER = b"placeholder-internal-target"
BENIGN_BYTES = b"placeholder-benign-regular"
TOKENIZER_BYTES = b'{"placeholder":"tokenizer"}'

NR8_MODEL_PY = """\
class Mistral3ForSequenceClassification:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _Model()


class _Model:
    def cuda(self):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def named_modules(self):
        return []

    def forward(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        return None
"""


class _Flex:
    """Stand-in tensor so the real quantize script can reach its copy loop."""

    def cuda(self) -> _Flex:
        return self

    def clone(self) -> _Flex:
        return _Flex()

    def float(self) -> _Flex:
        return self

    def abs(self) -> _Flex:
        return self

    def max(self) -> _Flex:
        return self

    def item(self) -> float:
        return 1.0

    def sum(self, *args: object, **kwargs: object) -> _Flex:
        return self

    def cpu(self) -> _Flex:
        return self

    def tolist(self) -> list[float]:
        return [0.25, 0.75]

    @property
    def shape(self) -> tuple[int, int]:
        return (1, 3)

    def __getitem__(self, key: object) -> _Flex:
        return _Flex()

    def __setitem__(self, key: object, value: object) -> None:
        return None

    def __sub__(self, other: object) -> _Flex:
        return _Flex()

    def __mul__(self, other: object) -> _Flex:
        return _Flex()

    def __truediv__(self, other: object) -> _Flex:
        return _Flex()


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False


class _Output:
    last_hidden_state = _Flex()


class _Param:
    def numel(self) -> int:
        return 8


class Mistral3Model:
    def cuda(self) -> Mistral3Model:
        return self

    def eval(self) -> Mistral3Model:
        return self

    def parameters(self) -> list[_Param]:
        return [_Param()]

    def named_modules(self) -> list[object]:
        return []

    def forward(self, *args: object, **kwargs: object) -> _Output:
        return _Output()

    def __call__(self, *args: object, **kwargs: object) -> _Output:
        return _Output()


class _TextConfig:
    is_causal = False


class _Config:
    is_causal = False
    text_config = _TextConfig()


class _Tokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, _Flex]:
        return {"input_ids": _Flex(), "attention_mask": _Flex()}


class _QuantizeConfig:
    def model_dump(self) -> dict[str, object]:
        return {"algorithm": {"method": "placeholder"}}


class _Recipe:
    def __init__(self) -> None:
        self.quantize = _QuantizeConfig()


def _ensure_module(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        setattr(_ensure_module(parent_name), child, module)
    return module


def _install_import_fakes() -> None:
    torch = _ensure_module("torch")

    class Tensor:
        pass

    torch.Tensor = Tensor  # type: ignore[attr-defined]
    torch.bfloat16 = object()
    torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    torch.ones_like = lambda tensor: _Flex()  # type: ignore[attr-defined]
    functional = _ensure_module("torch.nn.functional")
    functional.normalize = lambda tensor, dim=-1: tensor  # type: ignore[attr-defined]

    quant = _ensure_module("modelopt.torch.quantization")

    def quantize(model: object, qcfg: object, forward_loop: object = None) -> object:
        if callable(forward_loop):
            forward_loop(model)
        return model

    quant.quantize = quantize  # type: ignore[attr-defined]
    nn = _ensure_module("modelopt.torch.quantization.nn")

    class TensorQuantizer:
        pass

    nn.TensorQuantizer = TensorQuantizer  # type: ignore[attr-defined]
    tensor_quant = _ensure_module("modelopt.torch.quantization.tensor_quant")
    tensor_quant.DISABLE_TRITON_KERNEL = False  # type: ignore[attr-defined]

    export = _ensure_module("modelopt.torch.export")

    def export_hf_checkpoint(model: object, export_dir: str) -> None:
        dest = Path(export_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_algo": "NVFP4", "ignore": []}}),
            encoding="utf-8",
        )
        (dest / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"exclude_modules": []}}),
            encoding="utf-8",
        )

    export.export_hf_checkpoint = export_hf_checkpoint  # type: ignore[attr-defined]
    recipe = _ensure_module("modelopt.recipe")
    recipe.load_recipe = lambda path: _Recipe()  # type: ignore[attr-defined]

    transformers = _ensure_module("transformers")
    transformers.AutoConfig = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: _Config()
    )
    transformers.AutoModel = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: Mistral3Model()
    )
    transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: _Tokenizer()
    )


def _publish_roots(out: Path) -> tuple[Path, Path, Path]:
    return (
        out,
        out.with_name(out.name + ".partial"),
        out.with_name(out.name + ".ref.json"),
    )


def _under_publish(path: Path, roots: tuple[Path, Path, Path]) -> bool:
    absolute = Path(os.path.abspath(path))
    out, partial, ref = roots
    if absolute == Path(os.path.abspath(ref)):
        return True
    for root in (out, partial):
        root_abs = Path(os.path.abspath(root))
        if absolute == root_abs or root_abs in absolute.parents:
            return True
    return False


class _RecordingIO:
    def __init__(self, raw: Any, path: str, writes: list[dict[str, str]]) -> None:
        self._raw = raw
        self._path = path
        self._writes = writes

    def write(self, data: object) -> int:
        if isinstance(data, str):
            payload = data.encode()
        else:
            payload = bytes(data)  # type: ignore[arg-type]
        self._writes.append({"path": self._path, "b64": base64.b64encode(payload).decode("ascii")})
        return int(self._raw.write(data))

    def __enter__(self) -> _RecordingIO:
        self._raw.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self._raw.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw, name)


def _execute(spec: dict[str, Any]) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    out = Path(spec["out"])
    roots = _publish_roots(out)
    writes: list[dict[str, str]] = []
    creates: list[str] = []
    mkdirs: list[str] = []
    symlinks: list[dict[str, str]] = []
    swap_target = spec.get("swap_source")
    swapped = {"done": False}
    real_open = io.open
    real_os_open = os.open
    real_mkdir = os.mkdir
    real_symlink = os.symlink

    def _same(path: object, expected: str | None) -> bool:
        if expected is None:
            return False
        try:
            return os.path.abspath(os.fspath(path)) == os.path.abspath(expected)
        except TypeError:
            return False

    def _swap() -> None:
        if swap_target is None or swapped["done"]:
            return
        swapped["done"] = True
        os.unlink(swap_target)
        real_symlink(spec["secret"], swap_target)

    def _write_mode(mode: str) -> bool:
        return any(flag in mode for flag in ("w", "a", "x", "+"))

    def open_hook(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        if _same(file, swap_target) and not _write_mode(mode):
            _swap()
        handle = real_open(file, mode, *args, **kwargs)
        try:
            path = Path(os.fspath(file))
        except TypeError:
            return handle
        if _write_mode(mode) and _under_publish(path, roots):
            absolute = os.path.abspath(path)
            creates.append(absolute)
            return _RecordingIO(handle, absolute, writes)
        return handle

    def os_open_hook(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        readable = not (flags & (os.O_WRONLY | os.O_RDWR))
        if readable and _same(path, swap_target):
            _swap()
        if flags & os.O_CREAT:
            try:
                as_path = Path(os.fspath(path))
            except TypeError:
                as_path = None
            if as_path is not None and _under_publish(as_path, roots):
                creates.append(os.path.abspath(as_path))
        if dir_fd is None:
            return real_os_open(path, flags, mode)
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    def mkdir_hook(path: object, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        try:
            as_path = Path(os.fspath(path))
        except TypeError:
            as_path = None
        if as_path is not None and _under_publish(as_path, roots):
            mkdirs.append(os.path.abspath(as_path))
        if dir_fd is None:
            real_mkdir(path, mode)
            return
        real_mkdir(path, mode, dir_fd=dir_fd)

    def symlink_hook(
        src: object, dst: object, target_is_directory: bool = False, *, dir_fd: int | None = None
    ) -> None:
        try:
            as_path = Path(os.fspath(dst))
        except TypeError:
            as_path = None
        if as_path is not None and _under_publish(as_path, roots):
            symlinks.append({"path": os.path.abspath(as_path), "target": os.fspath(src)})
        if dir_fd is None:
            real_symlink(src, dst, target_is_directory)
            return
        real_symlink(src, dst, target_is_directory, dir_fd=dir_fd)

    io.open = open_hook  # type: ignore[assignment]
    builtins.open = open_hook  # type: ignore[assignment]
    os.open = os_open_hook  # type: ignore[assignment]
    os.mkdir = mkdir_hook  # type: ignore[assignment]
    os.symlink = symlink_hook  # type: ignore[assignment]

    _install_import_fakes()
    sys.argv = [
        spec["script"],
        spec["recipe"],
        spec["src"],
        spec["calib"],
        "2",
        "16",
        spec["out"],
    ]
    stdout = io.StringIO()
    code = 0
    error = ""
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            runpy.run_path(spec["script"], run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            code = 0
        elif isinstance(exc.code, int):
            code = exc.code
        else:
            code = 1
            error = str(exc.code)
    except Exception:
        code = 1
        error = traceback.format_exc()
    return {
        "code": code,
        "error": error,
        "stdout": stdout.getvalue()[-4000:],
        "writes": writes,
        "creates": creates,
        "mkdirs": mkdirs,
        "symlinks": symlinks,
        "swapped": swapped["done"],
    }


def _run_child(spec: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(spec["result"])
    spec_path = result_path.with_name("spec.json")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(spec_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    if not result_path.exists():
        raise AssertionError(
            f"harness wrote no result (exit {completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["harness_exit"] = completed.returncode
    return payload


def _write_checkpoint(src: Path, *, rerank: bool) -> None:
    src.mkdir(parents=True, exist_ok=True)
    (src / "config.json").write_text(
        json.dumps({"model_type": "mistral3", "architectures": ["Mistral3Model"]}),
        encoding="utf-8",
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"model.embed": "model.safetensors"}}),
        encoding="utf-8",
    )
    (src / "model.safetensors").write_bytes(b"placeholder-shard")
    if rerank:
        (src / "model.py").write_text(NR8_MODEL_PY, encoding="utf-8")


def _write_calib(path: Path) -> None:
    rows = [{"text": "query: one"}, {"text": "passage: two"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _base_spec(tmp: Path, script: Path) -> dict[str, Any]:
    src = tmp / "src"
    out = tmp / "out"
    _write_checkpoint(src, rerank=script == NR8)
    _write_calib(tmp / "calib.jsonl")
    (tmp / "recipe.txt").write_text("placeholder\n", encoding="utf-8")
    secret = tmp / "outside-file"
    secret.write_bytes(OUTSIDE_MARKER)
    return {
        "script": str(script),
        "src": str(src),
        "out": str(out),
        "recipe": str(tmp / "recipe.txt"),
        "calib": str(tmp / "calib.jsonl"),
        "secret": str(secret),
        "result": str(tmp / "result.json"),
    }


def _parent_dirs(relative: str) -> list[str]:
    parts = Path(relative).parts
    if len(parts) < 2:
        return []
    parents: list[str] = []
    acc = Path()
    for part in parts[:-1]:
        acc = acc / part
        parents.append(acc.as_posix())
    return parents


def _scan_publish(out: Path) -> list[tuple[str, str]]:
    """Return (kind, absolute path) for files and symlinks under the export."""
    found: list[tuple[str, str]] = []
    out_root, partial, ref = _publish_roots(out)
    for root in (out_root, partial):
        if not root.exists() and not root.is_symlink():
            continue
        if root.is_symlink():
            found.append(("symlink", str(root)))
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            for name in dirnames:
                candidate = base / name
                if candidate.is_symlink():
                    found.append(("symlink", str(candidate)))
            for name in filenames:
                candidate = base / name
                if candidate.is_symlink():
                    found.append(("symlink", str(candidate)))
                elif candidate.is_file():
                    found.append(("file", str(candidate)))
    if ref.is_symlink():
        found.append(("symlink", str(ref)))
    elif ref.is_file():
        found.append(("file", str(ref)))
    return found


def _read_nofollow(path: Path) -> bytes | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _problems(
    payload: dict[str, Any],
    *,
    out: Path,
    relative: str,
    marker: bytes,
    marker_anywhere: bool,
) -> list[str]:
    problems: list[str] = []
    if payload["code"] == 0:
        problems.append("quantize script exited 0")
    rel = Path(relative)
    roots = _publish_roots(out)[:2]
    for root in roots:
        for parent in _parent_dirs(relative):
            created = root / parent
            if created.exists() or created.is_symlink():
                problems.append(f"created {created}")
        destination = root / rel
        if destination.exists() or destination.is_symlink():
            problems.append(f"created {destination}")
    for recorded in payload["mkdirs"]:
        for root in roots:
            for parent in _parent_dirs(relative):
                if Path(recorded) == Path(os.path.abspath(root / parent)):
                    problems.append(f"mkdir {recorded}")
    interesting = {Path(os.path.abspath(root / rel)) for root in roots}
    for created in payload["creates"]:
        if Path(created) in interesting:
            problems.append(f"opened for write {created}")
    for link in payload["symlinks"]:
        if Path(link["path"]) in interesting:
            problems.append(f"symlinked {link['path']} -> {link['target']}")
    for entry in payload["writes"]:
        data = base64.b64decode(entry["b64"])
        path = Path(entry["path"])
        if marker_anywhere and marker in data:
            problems.append(f"wrote outside-target bytes to {path}")
        elif path in interesting and marker in data:
            problems.append(f"wrote target bytes to {path}")
    for kind, raw in _scan_publish(out):
        path = Path(raw)
        if kind == "symlink" and path in interesting:
            problems.append(f"left a symlink at {path}")
        if kind != "file":
            continue
        data = _read_nofollow(path)
        if data is None:
            continue
        if marker_anywhere and marker in data:
            problems.append(f"published outside-target bytes at {path}")
        elif path in interesting and marker in data:
            problems.append(f"published target bytes at {path}")
    return problems


def _assert_rejected(
    payload: dict[str, Any], spec: dict[str, Any], relative: str, marker: bytes, *, anywhere: bool
) -> None:
    problems = _problems(
        payload,
        out=Path(spec["out"]),
        relative=relative,
        marker=marker,
        marker_anywhere=anywhere,
    )
    assert not problems, (
        "symlink copy was not rejected before publication:\n"
        + "\n".join(problems)
        + "\n--- script output ---\n"
        + payload.get("error", "")
        + "\n"
        + payload.get("stdout", "")
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=("ne8", "nr8"))
def test_regular_sidecar_is_copied(tmp_path: Path, script: Path) -> None:
    spec = _base_spec(tmp_path, script)
    tokenizer = Path(spec["src"]) / "tokenizer.json"
    tokenizer.write_bytes(TOKENIZER_BYTES)
    payload = _run_child(spec)
    assert payload["code"] == 0, payload.get("error", "") + payload.get("stdout", "")
    copied = Path(spec["out"]) / "tokenizer.json"
    assert copied.is_file() and not copied.is_symlink()
    assert _read_nofollow(copied) == TOKENIZER_BYTES
    assert not Path(spec["out"]).with_name(Path(spec["out"]).name + ".partial").exists()


@pytest.mark.parametrize("script", SCRIPTS, ids=("ne8", "nr8"))
@pytest.mark.parametrize(
    "case",
    (
        "absolute",
        "relative-escape",
        "internal",
        "directory",
        "hidden",
        "shard-name",
        "broken",
        "replace-on-open",
    ),
)
def test_symlink_is_not_published(tmp_path: Path, script: Path, case: str) -> None:
    spec = _base_spec(tmp_path, script)
    src = Path(spec["src"])
    secret = Path(spec["secret"])
    relative = "1_Pooling/config.json"
    marker = OUTSIDE_MARKER
    anywhere = True

    if case == "absolute":
        link = src / relative
        link.parent.mkdir(parents=True)
        link.symlink_to(secret)
    elif case == "relative-escape":
        link = src / relative
        link.parent.mkdir(parents=True)
        link.symlink_to(os.path.relpath(secret, link.parent))
    elif case == "internal":
        relative = "readme.md"
        marker = INTERNAL_MARKER
        anywhere = False
        (src / "notes.txt").write_bytes(marker)
        (src / relative).symlink_to("notes.txt")
    elif case == "directory":
        relative = "linked-dir"
        outside = tmp_path / "outside-dir"
        outside.mkdir()
        (outside / "nested.txt").write_bytes(marker)
        (src / relative).symlink_to(outside, target_is_directory=True)
    elif case == "hidden":
        relative = ".hidden/payload.json"
        link = src / relative
        link.parent.mkdir(parents=True)
        link.symlink_to(secret)
    elif case == "shard-name":
        relative = "nested/model.safetensors"
        link = src / relative
        link.parent.mkdir(parents=True)
        link.symlink_to(secret)
    elif case == "broken":
        link = src / relative
        link.parent.mkdir(parents=True)
        link.symlink_to(link.parent / "missing-target")
        anywhere = False
        marker = b""
    elif case == "replace-on-open":
        link = src / relative
        link.parent.mkdir(parents=True)
        link.write_bytes(BENIGN_BYTES)
        spec["swap_source"] = str(link)
    else:
        raise AssertionError(case)

    payload = _run_child(spec)
    if case == "replace-on-open":
        assert payload["swapped"] is True or payload["code"] != 0
    _assert_rejected(payload, spec, relative, marker, anywhere=anywhere)


def _main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result_path = Path(spec["result"])
    payload: dict[str, Any] = {
        "code": 1,
        "error": "",
        "stdout": "",
        "writes": [],
        "creates": [],
        "mkdirs": [],
        "symlinks": [],
        "swapped": False,
    }
    try:
        payload.update(_execute(spec))
    except Exception:
        payload["error"] = traceback.format_exc()
    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    _main()
