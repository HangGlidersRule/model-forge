"""Shared fakes for ModelOpt execution/runner tests.

Provides a minimal safetensors writer, a builder for a *valid* exported
checkpoint (one that passes the fail-closed validators), and a self-contained
fake ``docker`` executable that materializes such an export on ``run`` so the
whole runner path can be exercised without a GPU.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

TRACKED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[float]]]) -> None:
    """Write a minimal safetensors file (F32 / BF16 tensors only)."""
    header: dict[str, object] = {}
    blob = b""
    offset = 0
    for name, (dtype, values) in tensors.items():
        if dtype == "F32":
            data = struct.pack("<%df" % len(values), *values)
        elif dtype == "BF16":
            data = b"".join(struct.pack("<f", value)[2:4] for value in values)
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        header[name] = {
            "dtype": dtype,
            "shape": [len(values)],
            "data_offsets": [offset, offset + len(data)],
        }
        blob += data
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)


def write_fake_source(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_8_moe",
                "architectures": ["Qwen38ForCausalLM"],
                "torch_dtype": "bfloat16",
                "vocab_size": 152064,
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (source_dir / "tokenizer_config.json").write_text('{"model_max_length":126144}', encoding="utf-8")


def build_fake_export(export_dir: Path, source_dir: Path) -> None:
    """Create a checkpoint that passes validators: 15 BF16 MTP + finite scales."""
    export_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, tuple[str, list[float]]] = {}
    weight_map: dict[str, str] = {}
    for i in range(15):
        name = f"mtp.layers.0.block_{i}.weight"
        tensors[name] = ("BF16", [0.5, -0.25])
        weight_map[name] = "model.safetensors"

    scale_name = "model.language_model.layers.0.mlp.down_proj.weight_scale"
    tensors[scale_name] = ("F32", [0.5, 1.0, 0.125, 2.0])
    weight_map[scale_name] = "model.safetensors"

    write_safetensors(export_dir / "model.safetensors", tensors)
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (export_dir / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None}}),
        encoding="utf-8",
    )
    # Tokenizer/config assets must match source exactly (no drift).
    for name in ("config.json", *TRACKED_TOKENIZER_FILES):
        src = source_dir / name
        if src.exists():
            (export_dir / name).write_bytes(src.read_bytes())


FAKE_DOCKER_TEMPLATE = '''#!/usr/bin/env python3
"""Self-contained fake `docker` for runner tests."""
import json
import struct
import sys
from pathlib import Path

FAKE_CONTAINER_ID = "runtimecafe0001"


def _write_safetensors(path, tensors):
    header = {}
    blob = b""
    offset = 0
    for name, (dtype, values) in tensors.items():
        if dtype == "F32":
            data = struct.pack("<%df" % len(values), *values)
        else:
            data = b"".join(struct.pack("<f", v)[2:4] for v in values)
        header[name] = {"dtype": dtype, "shape": [len(values)],
                        "data_offsets": [offset, offset + len(data)]}
        blob += data
        offset += len(data)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)


def _build_export(export_dir, source_dir):
    export_dir = Path(export_dir)
    source_dir = Path(source_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    weight_map = {}
    for i in range(15):
        n = "mtp.layers.0.block_%d.weight" % i
        tensors[n] = ("BF16", [0.5, -0.25])
        weight_map[n] = "model.safetensors"
    scale = "model.language_model.layers.0.mlp.down_proj.weight_scale"
    tensors[scale] = ("F32", [0.5, 1.0, 0.125, 2.0])
    weight_map[scale] = "model.safetensors"
    _write_safetensors(export_dir / "model.safetensors", tensors)
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
    (export_dir / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None}}))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        src = source_dir / name
        if src.exists():
            (export_dir / name).write_bytes(src.read_bytes())


def _inspect():
    return [{
        "Name": "/vllm-serve",
        "Config": {
            "Image": "vllm/vllm-openai:v0.27.1",
            "Env": ["VLLM_MODEL_PATH=/models/current", "VLLM_PORT=8000"],
            "Cmd": ["--model", "/models/current", "--tensor-parallel-size", "1"],
        },
        "HostConfig": {
            "Binds": ["/mnt/d/model-forge/artifacts:/models:ro"],
            "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8000"}]},
            "RestartPolicy": {"Name": "unless-stopped"},
            "Runtime": "nvidia",
            "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
        },
    }]


def main(argv):
    if not argv:
        return 0
    sub = argv[0]
    if sub == "ps":
        print(FAKE_CONTAINER_ID)
        return 0
    if sub == "inspect":
        print(json.dumps(_inspect()))
        return 0
    if sub == "logs":
        sys.stderr.write("fake vllm logs\\n")
        return 0
    if sub == "run":
        binds = {}
        i = 0
        while i < len(argv):
            if argv[i] == "-v" and i + 1 < len(argv):
                spec = argv[i + 1]
                parts = spec.split(":")
                if len(parts) >= 2:
                    host, container = parts[0], parts[1]
                    binds[container] = host
                i += 2
                continue
            i += 1
        export_host = binds.get("/mnt/export")
        source_host = binds.get("/mnt/source")
        if not export_host or not source_host:
            sys.stderr.write("fake docker run: missing export/source mounts\\n")
            return 3
        _build_export(export_host, source_host)
        return 0
    sys.stderr.write("fake docker: unhandled subcommand %s\\n" % sub)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def write_fake_docker(path: Path) -> Path:
    path.write_text(FAKE_DOCKER_TEMPLATE, encoding="utf-8")
    path.chmod(0o755)
    return path
