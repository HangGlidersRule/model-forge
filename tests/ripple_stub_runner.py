"""Test helper (not a test module): run models/ripple-65b's build (scripts/quantize_ripple.py) or BF16 teacher
(eval/bf16_teacher_cpu.py) for real, with stand-ins for torch, transformers, safetensors and ModelOpt (none is installed
in CI), so tests/test_ripple_65b_kit.py reaches every file the scripts read or write. Stdlib only. Each run is its own
process: the build runs at import time, and the stand-ins replace modules.

Usage: python ripple_stub_runner.py SPEC_JSON
SPEC keys:
  mode           "build" (runs the script as __main__) or "teacher" (constructs the teacher's Streamer on the source and
                 reads one tensor through it)
  script         the script's path; eval_dir: the family's eval/ directory (ripple_layout is imported from there first)
  argv           the script's arguments (build)
  src, tensor    the source directory and one tensor name (teacher)
  pins           {file name: sha256}: replaces the value of each pin the code under test already has
                 (ripple_layout.SOURCE_FILE_PINS, and the teacher's own remote-code pins where the script has them), so a
                 synthetic source passes the pin checks the test does not exercise; a file the code does not pin stays
                 unpinned
  calib_sha256   replaces ripple_layout.CALIB_SHA256 (build)
  export_files   {name: text} the stand-in exporter writes besides its own files; a name may hold a subdirectory,
                 created as needed (build)
  export_links   {name: target} the stand-in exporter leaves as symbolic links (replacing any file it wrote); a name
                 may hold a subdirectory, created as needed (build)
  export_sparse  {name: size} files the stand-in exporter then extends (or creates) as sparse files of that size (build)
  swaps          [{"at": "load_recipe" | "config" | "export", "path": p, "target": t}]: at the first call of that
                 stand-in (ModelOpt's load_recipe, transformers' AutoConfig.from_pretrained, the exporter), replace the
                 file p with a symbolic link to t, as a source changed after the script first read it
Prints one line: RUNNER-RESULT {"exit", "out" (the script's output tail), "opened", "safe_open"}. "opened" lists every
file the script opened for reading by its path (open(), io.open(), or os.open() without dir_fd; directories excluded),
both as given and as resolved (os.path.realpath: where a link at that path leads); "safe_open" every path the script
gave to safetensors.safe_open.
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


class _Anything:
    """A permissive stand-in: every unknown attribute, call, item and arithmetic result is another stand-in."""

    shape = (1, 4)
    device = types.SimpleNamespace(type="cpu")
    is_meta = False

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

    def __int__(self) -> int:
        return 0

    def __float__(self) -> float:
        return 0.0

    def __enter__(self) -> _Anything:
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def _op(self, *a: object) -> _Anything:
        return _Anything()

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __matmul__ = _op


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
    chat_template = ""

    def __call__(self, text: str, **k: object) -> dict:
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids: list) -> str:
        return " Paris"


TINY_CONFIG = {"num_hidden_layers": 2, "model_type": "ricursive", "hidden_size": 256}


def _install(spec: dict, safe_open_log: list[str]) -> None:
    swaps = list(spec.get("swaps", []))

    def swap(at: str) -> None:
        for s in [s for s in swaps if s["at"] == at]:
            swaps.remove(s)
            os.unlink(s["path"])
            os.symlink(s["target"], s["path"])

    tq_cls = type("TensorQuantizer", (), {"is_enabled": True})

    class _LmHead:
        def __init__(self) -> None:
            self.weight_quantizer = tq_cls()
            self.weight = types.SimpleNamespace(is_meta=False)

    class _Model(_Anything):
        hf_device_map: dict = {}

        def named_modules(self) -> list:
            return [("lm_head", _LmHead())]

        def named_parameters(self) -> list:
            return []

        def to(self, *a: object, **k: object) -> _Model:
            return self

        def eval(self) -> _Model:  # torch's nn.Module.eval() (train/eval mode switch), not Python's eval
            return self

        def save_pretrained(self, path: object, **k: object) -> None:
            d = Path(str(path))
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(TINY_CONFIG))
            (d / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    torch = _module("torch")
    torch.bfloat16 = torch.float32 = object()  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True, max_memory_allocated=lambda *a: 0, memory_allocated=lambda *a: 0,
        set_per_process_memory_fraction=lambda *a: None)
    torch.no_grad = lambda: contextlib.nullcontext()  # type: ignore[attr-defined]
    torch.device = lambda *a: contextlib.nullcontext()  # type: ignore[attr-defined]
    torch.set_num_threads = lambda n: None  # type: ignore[attr-defined]
    torch.Tensor = _Anything  # type: ignore[attr-defined]
    torch.__version__ = "stand-in"  # type: ignore[attr-defined]
    for name in ("tensor", "manual_seed", "load", "save", "arange", "log_softmax", "topk", "rsqrt", "isfinite"):
        setattr(torch, name, lambda *a, **k: _Anything())

    def safe_open(path: object, framework: str) -> _Anything:
        safe_open_log.append(str(path))
        return _Anything()

    _module("safetensors").safe_open = safe_open  # type: ignore[attr-defined]

    def config_from_pretrained(*a: object, **k: object) -> _Anything:
        swap("config")
        return _Anything()

    tf = _module("transformers")
    tf.__version__ = "stand-in"  # type: ignore[attr-defined]
    tf.AutoConfig = types.SimpleNamespace(from_pretrained=config_from_pretrained)  # type: ignore[attr-defined]
    tf.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **k: _Model(), from_config=lambda *a, **k: _Model())
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())  # type: ignore[attr-defined]

    class _Recipe:
        def __init__(self, path: object) -> None:
            swap("load_recipe")
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
        swap("export")
        d = Path(export_dir)
        d.mkdir(parents=True, exist_ok=True)
        qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {}}
        (d / "config.json").write_text(json.dumps({"quantization_config": qc, "dtype": "bfloat16"}))
        (d / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "MIXED_PRECISION"}}))
        (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 1}))
        (d / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
        for name, text in spec.get("export_files", {}).items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(text)
        for name, target in spec.get("export_links", {}).items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            (d / name).unlink(missing_ok=True)
            os.symlink(target, d / name)
        for name, size in spec.get("export_sparse", {}).items():
            (d / name).touch(exist_ok=True)
            os.truncate(d / name, size)

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
    opened: list[str] = []
    safe_open_log: list[str] = []
    _install(spec, safe_open_log)
    real_open, real_os_open = builtins.open, os.open

    def open_hook(file: object, mode: str = "r", *a: object, **k: object) -> object:
        if isinstance(file, (str, os.PathLike)) and not any(c in mode for c in "wax+"):
            opened.append(os.path.abspath(os.fspath(file)))
            opened.append(os.path.realpath(os.fspath(file)))  # where a link at that path leads
        return real_open(file, mode, *a, **k)  # type: ignore[call-overload]

    def os_open_hook(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None and isinstance(path, (str, os.PathLike)) and not flags & (os.O_WRONLY | os.O_RDWR) \
                and not flags & os.O_DIRECTORY:
            opened.append(os.path.abspath(os.fspath(path)))
            opened.append(os.path.realpath(os.fspath(path)))
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
                g = runpy.run_path(spec["script"], run_name="ripple_teacher_under_test")
                streamer = g["Streamer"]
                fglobals = streamer.__init__.__globals__
                for const, name in (("MODELING_SHA256", "modeling_ricursive.py"),
                                    ("CONFIGURATION_SHA256", "configuration_ricursive.py")):
                    if const in fglobals and name in spec.get("pins", {}):
                        fglobals[const] = spec["pins"][name]
                s = streamer(Path(spec["src"]))
                s.get(spec["tensor"])
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
    print("RUNNER-RESULT " + json.dumps({"exit": code, "out": out.getvalue()[-4000:], "opened": opened,
                                         "safe_open": safe_open_log}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
