"""Test helper (not a test module): run models/ripple-65b's build (scripts/quantize_ripple.py, its GPU path) or the BF16
teacher's Streamer (eval/bf16_teacher_cpu.py) for real, with stand-ins for torch, transformers, safetensors and ModelOpt
(none is installed in CI). Stdlib only; each run is its own process.

Usage: python ripple_stub_runner.py SPEC_JSON, SPEC {"mode": "build" | "teacher", "script", "eval_dir", "pins" (file
name: sha256, replacing ripple_layout.SOURCE_FILE_PINS values so a synthetic source passes), build: "argv",
"calib_sha256" (replaces ripple_layout.CALIB_SHA256), "export_files" ({name: text}) and "export_links" ({name: target})
the stand-in exporter leaves besides its own files; teacher: "src" and one "tensor" to read}.
Prints RUNNER-RESULT {"exit", "out" (the output tail), "opened" (files opened for reading by path, as given and
resolved), "lib" (one {"call", "root", "after_check"} per library load; a load before checkpoint_files.read_source
returned raises instead)}.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import runpy
import sys
import types
from pathlib import Path

LIB: list[dict] = []  # one entry per library load
CHECKED: list[str] = []  # the roots checkpoint_files.read_source returned for


def _lib(call: str, root: str) -> None:
    """Record a library load; before read_source has returned, raise instead (no library may read the source first)."""
    LIB.append({"call": call, "root": root, "after_check": bool(CHECKED)})
    if not CHECKED:
        raise RuntimeError(f"stand-in {call} called before checkpoint_files.read_source returned ({root})")


class _Anything:
    """A permissive stand-in: every unknown attribute, call, item and arithmetic result is another stand-in."""

    shape = (1, 4)
    device = types.SimpleNamespace(type="cpu")

    def __getattr__(self, name: str) -> _Anything:
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()

    def __call__(self, *a: object, **k: object) -> _Anything:
        return _Anything()

    def __getitem__(self, key: object) -> _Anything:
        return _Anything()

    def __iter__(self):
        return iter(())


def _module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = mod
    if "." in name:
        parent, _, child = name.rpartition(".")
        setattr(_module(parent), child, mod)
    return mod


class _Tok:
    pad_token_id = 0

    def __call__(self, text: str, **k: object) -> dict:
        return {"input_ids": [1, 2, 3]}


def _install(spec: dict) -> None:
    tq_cls = type("TensorQuantizer", (), {"is_enabled": True})

    class _LmHead:
        def __init__(self) -> None:
            self.weight_quantizer = tq_cls()
            self.weight = types.SimpleNamespace(is_meta=False)

    class _Model(_Anything):
        hf_device_map: dict = {}

        def named_modules(self) -> list:
            return [("lm_head", _LmHead())]

        def eval(self) -> _Model:  # torch's nn.Module.eval() (train/eval mode switch), not Python's eval
            return self

    torch = _module("torch")
    torch.bfloat16 = torch.float32 = object()  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True, max_memory_allocated=lambda *a: 0, memory_allocated=lambda *a: 0,
        set_per_process_memory_fraction=lambda *a: None)
    torch.no_grad = lambda: contextlib.nullcontext()  # type: ignore[attr-defined]
    torch.device = lambda *a: contextlib.nullcontext()  # type: ignore[attr-defined]
    torch.tensor = lambda *a, **k: _Anything()  # type: ignore[attr-defined]

    def safe_open(path: object, framework: str) -> _Anything:
        _lib("safe_open", str(path))
        return _Anything()

    _module("safetensors").safe_open = safe_open  # type: ignore[attr-defined]

    def config_from_pretrained(path: object, *a: object, **k: object) -> _Anything:
        _lib("AutoConfig.from_pretrained", str(path))
        return _Anything()

    def model_from_pretrained(path: object, *a: object, **k: object) -> _Model:
        _lib("AutoModelForCausalLM.from_pretrained", str(path))
        return _Model()

    def model_from_config(cfg: object, *a: object, **k: object) -> _Model:
        _lib("AutoModelForCausalLM.from_config", "")
        return _Model()

    def tokenizer_from_pretrained(path: object, *a: object, **k: object) -> _Tok:
        _lib("AutoTokenizer.from_pretrained", str(path))
        return _Tok()

    tf = _module("transformers")
    tf.AutoConfig = types.SimpleNamespace(from_pretrained=config_from_pretrained)  # type: ignore[attr-defined]
    tf.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=model_from_pretrained, from_config=model_from_config)
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer_from_pretrained)  # type: ignore[attr-defined]

    class _Recipe:
        def __init__(self, path: object) -> None:
            body = {"quant_cfg": [{"quantizer_name": "*", "cfg": {"num_bits": (4, 3)}}], "algorithm": {"method": "x"}}
            self.quantize = types.SimpleNamespace(model_dump=lambda: json.loads(json.dumps(body)))

    _module("modelopt.recipe").load_recipe = _Recipe  # type: ignore[attr-defined]
    mtq = _module("modelopt.torch.quantization")
    mtq.quantize = lambda m, c, forward_loop=None: m  # type: ignore[attr-defined]
    mtq.calibrate = lambda *a, **k: None  # type: ignore[attr-defined]
    _module("modelopt.torch.quantization.tensor_quant").DISABLE_TRITON_KERNEL = False  # type: ignore[attr-defined]
    _module("modelopt.torch.quantization.conversion").set_quantizer_by_cfg = lambda *a: None  # type: ignore[attr-defined]
    _module("modelopt.torch.quantization.nn").TensorQuantizer = tq_cls  # type: ignore[attr-defined]
    lwc = _module("modelopt.torch.quantization.utils.layerwise_calib")
    lwc._CheckpointState = type("_CheckpointState", (), {"save": lambda *a, **k: None})  # type: ignore[attr-defined]
    lwc._read_manifest = lambda d: None  # type: ignore[attr-defined]
    lwc.detect_resume_point = lambda d: None  # type: ignore[attr-defined]
    lwc.LayerActivationCollector = types.SimpleNamespace(get_decoder_layers=lambda m: [])  # type: ignore[attr-defined]
    cu = _module("modelopt.torch.quantization.utils.core_utils")
    cu.module_name_maps = lambda m: {}  # type: ignore[attr-defined]
    cu.requires_weight_materialization = lambda *a: False  # type: ignore[attr-defined]
    _module("modelopt.torch.models").hf_model_type = lambda m: "ricursive"  # type: ignore[attr-defined]
    _module("modelopt.torch.export.registry").ExportContext = lambda **k: None  # type: ignore[attr-defined]
    ueh = _module("modelopt.torch.export.unified_export_hf")
    ueh._dispatch_export_handler = lambda *a: None  # type: ignore[attr-defined]
    ueh._resolve_export_dtype = lambda *a: None  # type: ignore[attr-defined]
    streaming = _module("modelopt.torch.export.unified_export_hf_streaming")
    streaming.get_quant_config = lambda *a, **k: {}  # type: ignore[attr-defined]

    def export_hf_checkpoint(model: object, export_dir: str) -> None:
        """What a ModelOpt export leaves, plus the spec's extra files and links."""
        streaming.get_quant_config(model)  # the build packs the resident lm_head inside this call
        d = Path(export_dir)
        d.mkdir(parents=True, exist_ok=True)
        qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {}}
        (d / "config.json").write_text(json.dumps({"quantization_config": qc, "dtype": "bfloat16"}))
        (d / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "MIXED_PRECISION"}}))
        (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 1}))
        (d / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
        for name, text in spec.get("export_files", {}).items():
            (d / name).write_text(text)
        for name, target in spec.get("export_links", {}).items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).unlink(missing_ok=True)
            os.symlink(target, d / name)

    _module("modelopt.torch.export").export_hf_checkpoint = export_hf_checkpoint  # type: ignore[attr-defined]


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec["eval_dir"])
    import ripple_layout as lay  # the family's own layout module, imported before the script so the pins can be set

    for name, value in spec.get("pins", {}).items():
        if name in lay.SOURCE_FILE_PINS:
            lay.SOURCE_FILE_PINS[name] = value
    if "calib_sha256" in spec:
        lay.CALIB_SHA256 = spec["calib_sha256"]
    lay.expected_quantizer_total = lambda *a, **k: 0  # the stand-in model enables no quantizer
    import checkpoint_files as cf  # the module the scripts import (the same sys.modules entry)

    real_read_source = cf.read_source

    def read_source(root: object, pins: object) -> dict:
        got = real_read_source(root, pins)  # type: ignore[arg-type]
        CHECKED.append(str(root))
        return got

    cf.read_source = read_source  # type: ignore[assignment]
    opened: list[str] = []
    _install(spec)
    real_open, real_os_open = builtins.open, os.open

    def open_hook(file: object, mode: str = "r", *a: object, **k: object) -> object:
        if isinstance(file, (str, os.PathLike)) and not any(c in mode for c in "wax+"):
            opened.extend([os.path.abspath(os.fspath(file)), os.path.realpath(os.fspath(file))])
        return real_open(file, mode, *a, **k)  # type: ignore[call-overload]

    def os_open_hook(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None and isinstance(path, (str, os.PathLike)) and not flags & (os.O_WRONLY | os.O_RDWR) \
                and not flags & os.O_DIRECTORY:
            opened.extend([os.path.abspath(os.fspath(path)), os.path.realpath(os.fspath(path))])
        return real_os_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    out = io.StringIO()
    code = 0
    builtins.open = open_hook  # type: ignore[assignment]
    io.open = open_hook  # type: ignore[assignment]
    os.open = os_open_hook  # type: ignore[assignment]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            if spec["mode"] == "build":
                sys.argv = [spec["script"], *spec["argv"]]
                runpy.run_path(spec["script"], run_name="__main__")
            else:
                streamer = runpy.run_path(spec["script"], run_name="ripple_teacher_under_test")["Streamer"]
                streamer(Path(spec["src"])).get(spec["tensor"])
                print("TEACHER-STREAMER-OK")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if e.code is not None and not isinstance(e.code, int):
            out.write(str(e.code))
    except BaseException as e:  # noqa: BLE001 - the test reads the type and message
        code = 99
        out.write(f"{type(e).__name__}: {e}")
    finally:
        builtins.open = real_open  # type: ignore[assignment]
        io.open = real_open  # type: ignore[assignment]
        os.open = real_os_open  # type: ignore[assignment]
    print("RUNNER-RESULT " + json.dumps({"exit": code, "out": out.getvalue()[-4000:], "opened": opened, "lib": LIB}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
