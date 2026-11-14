"""Issue #39 (Ripple-65B Preview) kit: the pure-Python pieces the GPU window relies on, tested without the model, the GPU
or the network. Covers the candidate layouts and the vLLM modelopt_mixed resolution re-implementation, the calibration
row rule (rank key, smoke take), the export validator (its self-test on synthetic exports with planted faults), the
gate (bars, bootstrap, selection rule) on synthetic result files, the BFCL AST checker re-implementation, the synthetic
tool-call tasks (reference solutions pass and no-ops fail in the local sandbox backend) and the KL estimator.
The last section tests each rule the record PR's review rounds added, directly: the confined checkpoint reads
(eval/checkpoint_files.py) of the validator, the build and the BF16 teacher, and the record's provenance and GPQA
wording. The build and the teacher run for real under stand-ins for torch, transformers, safetensors and ModelOpt
(tests/ripple_stub_runner.py; none is installed in CI)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "models" / "ripple-65b"
sys.path.insert(0, str(FAMILY / "eval"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lay = _load(FAMILY / "eval" / "ripple_layout.py", "ripple_layout")
ve = _load(FAMILY / "eval" / "validate_export.py", "validate_export")
vst = _load(FAMILY / "eval" / "validate_selftest.py", "validate_selftest")
gate = _load(FAMILY / "eval" / "gate_ripple39.py", "gate_ripple39")
syn = _load(FAMILY / "eval" / "tasks_synthetic.py", "tasks_synthetic")
proxy = _load(FAMILY / "eval" / "toolcall_proxy.py", "toolcall_proxy")
calib = _load(FAMILY / "calibration" / "build_ripple_calib.py", "build_ripple_calib")
sizes_mod = _load(FAMILY / "eval" / "checkpoint_sizes.py", "checkpoint_sizes")
cf = ve.cf  # eval/checkpoint_files.py, as the validator imported it


# ---- layouts ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("letter", "counts", "quantizers"), [
    ("A", {"FP8": 400, "NVFP4": 240, "W4A16_NVFP4": 1}, 1281),
    ("B", {"FP8": 400, "W4A16_NVFP4": 241}, 1041),
    ("C", {"NVFP4": 240, "W4A16_NVFP4": 1}, 481),
])
def test_quantized_layers_match_the_posted_dry_run_counts(letter: str, counts: dict, quantizers: int) -> None:
    assert lay.expected_algo_counts(letter) == counts
    assert lay.expected_quantizer_total(letter) == quantizers == lay.QUANTIZER_TOTALS[letter]
    ql = lay.quantized_layers(letter)
    assert "model.embed_tokens" not in ql and ql["lm_head"] == {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    assert all(v.get("group_size") == 16 for v in ql.values() if v["quant_algo"] != "FP8")


def test_letters_recipes_and_product_ids() -> None:
    assert sorted(lay.PLANS) == ["A", "B", "C"]
    with pytest.raises(ValueError):
        lay.check_letter("D")
    for letter, (name, pin) in lay.RECIPES.items():
        path = ROOT / lay.RECIPE_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin, letter
    lm = ROOT / lay.RECIPE_DIR / lay.LM_HEAD_RECIPE[0]
    assert hashlib.sha256(lm.read_bytes()).hexdigest() == lay.LM_HEAD_RECIPE[1]
    assert lay.PRODUCT_IDS["A"].endswith("-W4A4-NVFP4-Mixed-FP8") and lay.PRODUCT_IDS["C"].endswith("-W4A4-NVFP4")


@pytest.mark.parametrize("letter", ["A", "B", "C"])
def test_vllm_resolution_of_each_plan(letter: str) -> None:
    ql = lay.quantized_layers(letter)
    for prefix, want in lay.runtime_plan(letter).items():
        assert lay.vllm_resolve(ql, ["model.embed_tokens"], prefix) == want, prefix


def test_vllm_resolution_rejects_a_mixed_fused_group_and_honours_exclusions() -> None:
    ql = lay.quantized_layers("A")
    ql["model.layers.3.self_attn.k_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    with pytest.raises(ValueError, match="mixed"):
        lay.vllm_resolve(ql, [], "model.layers.3.self_attn.qkg_proj")
    assert lay.vllm_resolve(lay.quantized_layers("A"), ["model.layers.3.mlp*"], "model.layers.3.mlp.down_proj") is None
    assert lay.vllm_resolve(lay.quantized_layers("A"), ["model.embed_tokens"], "model.embed_tokens") is None


def test_linear_of() -> None:
    assert lay.linear_of("model.layers.7.mlp.up_proj.weight_scale_2") == "model.layers.7.mlp.up_proj"
    assert lay.linear_of("lm_head.weight") == "lm_head"
    assert lay.linear_of("model.layers.7.self_attn.q_norm.weight") is None
    assert lay.linear_of("model.layers.7.self_attn.value_res_lambda") is None


# ---- calibration rule -------------------------------------------------------------------------------------------------------
def test_rank_key_is_the_posted_rule() -> None:
    assert calib.rank_key("func-calling.json", 7) == hashlib.sha256(b"1234:func-calling.json:7").hexdigest()
    order = sorted(range(50), key=lambda i: calib.rank_key("x.parquet", i))
    assert order == sorted(range(50), key=lambda i: hashlib.sha256(f"1234:x.parquet:{i}".encode()).hexdigest())


def test_smoke_take_is_two_rows_per_source_in_file_order() -> None:
    rows = [{"source": s, "i": i} for s, n in zip(lay.SOURCES_IN_FILE_ORDER, (128, 128, 192, 64)) for i in range(n)]
    take = lay.calib_take(rows, 8)
    assert [(r["source"], r["i"]) for r in take] == [(s, i) for s in lay.SOURCES_IN_FILE_ORDER for i in (0, 1)]
    assert lay.calib_take(rows, 512) == rows


def test_hermes_conversion_and_swe_arguments() -> None:
    row = {"tools": json.dumps([{"type": "function", "function": {"name": "f", "parameters": {}}}]),
           "conversations": [{"from": "system", "value": "preamble"}, {"from": "human", "value": "hi"},
                             {"from": "gpt", "value": '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'},
                             {"from": "tool", "value": "<tool_response>\nok\n</tool_response>"},
                             {"from": "gpt", "value": "done"}]}
    msgs, tools = calib.conv_hermes(row)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "f", "arguments": {"a": 1}} and tools[0]["function"]["name"] == "f"
    bad = dict(row, conversations=row["conversations"][:2] + [{"from": "gpt", "value": "<tool_call>\n{'name': 1}\n</tool_call>"}])
    with pytest.raises(calib.Ineligible):
        calib.conv_hermes(bad)
    swe = {"messages": json.dumps([{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "u"}]},
                                   {"role": "assistant", "content": "c", "tool_calls": [
                                       {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
                                   {"role": "tool", "content": [{"type": "text", "text": "out"}]}])}
    msgs, tools = calib.conv_swe(swe)
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"} and tools == calib.SWE_TOOLS


# ---- validator ----------------------------------------------------------------------------------------------------------------
def test_validator_selftest_catches_every_planted_fault(tmp_path: Path, capsys) -> None:
    sys.argv = ["validate_selftest.py", str(tmp_path / "vs")]
    vst.main()
    out = capsys.readouterr().out
    assert "VALIDATOR-SELFTEST-OK" in out and "A: clean PASS, 24 planted faults caught" in out


# ---- gate -----------------------------------------------------------------------------------------------------------------------
def _gpqa(path: Path, correct: set[int]) -> None:
    runs = [{"item_id": f"q{i:03d}", "phase": "profile", "ok": True, "correct": i in correct} for i in range(198)]
    path.write_text(json.dumps({"accuracy": {"correct": len(correct), "unparseable": 0}, "runs": runs}))


def _results(tmp: Path, cand_correct: int, cand_succ: float, cand_b1: float, cand_noise: float) -> Path:
    d = tmp / "eval"
    (d / "speed").mkdir(parents=True)
    rng = random.Random(1)
    for cell, n in (("vendor", 150), ("A", cand_correct)):
        for r in (1, 2):
            _gpqa(d / f"gpqa-{cell}-r{r}.json", set(rng.sample(range(198), n)))
    for cell, succ in (("vendor", 0.86), ("A", cand_succ)):
        s = {"items": 500, "parse_rate": 0.99, "success_rate": succ, "errors": 0}
        (d / f"proxy-{cell}.json").write_text(json.dumps({"summary": {"overall": s}}))
    rows = [[{str(t): -2.0 - 0.5 * t for t in range(20)} for _ in range(10)] for _ in range(8)]  # top 20 sum 0.34
    (d / "teacher.json").write_text(json.dumps({"heldout_sha256": "h", "rows": rows,
                                                "prompts": [{"span_start": 4}] + [{"span_start": None}] * 7}))
    for cell, noise in (("vendor", 0.30), ("A", cand_noise)):
        r2 = random.Random(cell)
        out = [[{t: v + r2.uniform(-noise, noise) for t, v in p.items()} for p in row] for row in rows]
        (d / f"kl-{cell}.json").write_text(json.dumps({"heldout_sha256": "h", "rows": out}))
    for cell, b1 in (("vendor", 30.0), ("A", cand_b1)):
        for s in (1, 2):
            c = {"decode_b1": {"tok_s_median": b1 + 0.2 * s, "tok_s_range": [b1, b1 + 0.5]}}
            (d / "speed" / f"{cell}-s{s}.json").write_text(json.dumps({"cells": c}))
    (tmp / "out").mkdir()
    (tmp / "out" / "A.validation.json").write_text(json.dumps({"verdict": "PASS"}))
    return d


def test_gate_passes_and_ships_a_faster_candidate(tmp_path: Path) -> None:
    d = _results(tmp_path, 145, 0.855, 37.0, 0.25)
    g = gate.evaluate(d, ["vendor", "A"], tmp_path / "out")
    a = g["candidates"]["A"]
    assert a["gpqa"]["pass"] and a["proxy"]["pass"] and a["artifact_pass"] and a["passes_bars"]
    assert a["speed_win"] and g["ships"] == "A"
    lo, hi = a["gpqa"]["bootstrap95_questions"]
    assert lo <= a["gpqa"]["diff_questions"] <= hi


def test_gate_gpqa_bar_is_minus_11_88_on_unrounded_means(tmp_path: Path) -> None:
    d = _results(tmp_path, 138, 0.86, 37.0, 0.25)  # 150 - 138 = 12 > 11.88
    a = gate.evaluate(d, ["vendor", "A"], tmp_path / "out")["candidates"]["A"]
    assert a["gpqa"]["diff_questions"] == -12.0 and not a["gpqa"]["pass"] and not a["passes_bars"] and not a["eligible"]
    d2 = _results(tmp_path / "b", 139, 0.86, 37.0, 0.25)  # -11 passes
    assert gate.evaluate(d2, ["vendor", "A"], tmp_path / "b" / "out")["candidates"]["A"]["gpqa"]["pass"]


def test_gate_proxy_bars(tmp_path: Path) -> None:
    v = {"parse_rate": 0.99, "success_rate": 0.86}
    assert gate.proxy_compare({"parse_rate": 0.97, "success_rate": 0.8428}, v)["pass"]  # -2 pt, 0.98x exactly
    assert not gate.proxy_compare({"parse_rate": 0.969, "success_rate": 0.86}, v)["pass"]
    assert not gate.proxy_compare({"parse_rate": 0.99, "success_rate": 0.842}, v)["pass"]


def test_gate_speed_win_needs_5_percent_and_both_vendor_serves() -> None:
    v = {"decode_b1": [30.0, 31.0], "decode_b1_mean": 30.5}
    assert gate.speed_compare({"decode_b1": [32.1, 32.1], "decode_b1_mean": 32.1}, v)["speed_win"]  # 1.052x
    assert not gate.speed_compare({"decode_b1": [31.9, 31.9], "decode_b1_mean": 31.9}, v)["speed_win"]  # 1.046x
    v2 = {"decode_b1": [25.0, 36.0], "decode_b1_mean": 30.5}
    assert not gate.speed_compare({"decode_b1": [32.1, 32.1], "decode_b1_mean": 32.1}, v2)["speed_win"]  # inside band


def test_gate_quality_win_and_no_eligible_candidate(tmp_path: Path) -> None:
    d = _results(tmp_path, 145, 0.855, 30.5, 0.05)  # no speed win, far lower KL
    g = gate.evaluate(d, ["vendor", "A"], tmp_path / "out")
    a = g["candidates"]["A"]
    assert a["quality_win"] and not a["speed_win"] and a["eligible"] and g["ships"] == "A"
    d2 = _results(tmp_path / "x", 145, 0.855, 30.5, 0.35)  # neither win (higher KL than the vendor)
    g2 = gate.evaluate(d2, ["vendor", "A"], tmp_path / "x" / "out")
    assert not g2["candidates"]["A"]["eligible"] and g2["ships"] is None and "vendor" in g2["recommendation"]


def test_gate_bars_only_exit_codes(tmp_path: Path, monkeypatch) -> None:
    d = _results(tmp_path, 130, 0.86, 37.0, 0.25)
    monkeypatch.setattr(sys, "argv", ["gate", str(d), "--bars-only", "A", "--validation-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        gate.main()
    assert e.value.code == 3
    (d / "kl-A.json").unlink()  # tracked only: the bars still decide
    with pytest.raises(SystemExit) as e:
        gate.main()
    assert e.value.code == 3
    (d / "proxy-A.json").unlink()
    with pytest.raises(SystemExit) as e:
        gate.main()
    assert e.value.code == 2


def test_kl_estimator_is_kl_probes() -> None:
    p = {"1": -0.1, "2": -2.5}
    assert gate.kl_position(p, dict(p))[0] == pytest.approx(0.0, abs=1e-12)
    kl, same = gate.kl_position(p, {"1": -0.5, "2": -1.2})
    assert kl > 0 and same
    kl2, same2 = gate.kl_position(p, {"3": -0.05, "4": -4.0})  # the ref's tokens fall to the candidate's floor
    assert kl2 > kl and not same2


# ---- tool-call proxy ------------------------------------------------------------------------------------------------------------
def test_bfcl_checker_matches_upstream_semantics() -> None:
    funcs = [{"name": "spotify.play", "parameters": {"type": "dict", "properties": {
        "artist": {"type": "string"}, "duration": {"type": "integer"}}, "required": ["artist", "duration"]}}]
    gt = [{"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
          {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}}]
    ok = [{"spotify_play": {"artist": "maroon 5", "duration": 15}}, {"spotify_play": {"artist": "Taylor Swift", "duration": 20}}]
    assert proxy.ast_checker(funcs, ok, gt, "parallel")["valid"]
    assert not proxy.ast_checker(funcs, ok[:1], gt, "parallel")["valid"]
    assert not proxy.ast_checker(funcs, [{"spotify_play": {"artist": "Taylor Swift", "duration": "20"}}], gt[:1],
                                 "simple")["valid"]
    f2 = [{"name": "area", "parameters": {"type": "dict", "properties": {"base": {"type": "float"},
                                                                        "unit": {"type": "string"}}, "required": ["base"]}}]
    g2 = [{"area": {"base": [10.0], "unit": ["units", ""]}}]
    assert proxy.ast_checker(f2, [{"area": {"base": 10}}], g2, "simple")["valid"]  # int -> float, optional omitted
    assert not proxy.ast_checker(f2, [{"area": {"base": 10, "color": "red"}}], g2, "simple")["valid"]


def test_bfcl_tool_conversion() -> None:
    t = proxy.bfcl_tools([{"name": "a.b", "description": "d", "parameters": {"type": "dict", "properties": {
        "x": {"type": "float", "description": "x"}, "y": {"type": "tuple", "items": {"type": "integer"}}}, "required": []}}])
    fn = t[0]["function"]
    assert fn["name"] == "a_b" and fn["parameters"]["type"] == "object"
    assert fn["parameters"]["properties"]["x"]["type"] == "number" and fn["parameters"]["properties"]["y"]["type"] == "array"
    assert fn["description"].endswith("Python 3 syntax.")


def test_parse_calls() -> None:
    declared = {"bash"}
    msg = {"tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    assert proxy.parse_calls(msg, declared) == ([{"bash": {"command": "ls"}}], None)
    assert proxy.parse_calls({"content": "no"}, declared)[1] == "no tool call"
    assert proxy.parse_calls({"tool_calls": [{"function": {"name": "rm", "arguments": "{}"}}]}, declared)[0] is None
    assert proxy.parse_calls({"tool_calls": [{"function": {"name": "bash", "arguments": "[1]"}}]}, declared)[0] is None


def test_synthetic_tasks_are_deterministic_and_balanced() -> None:
    a, b = syn.generate(100, 1234), syn.generate(100, 1234)
    assert a == b and len(a) == 100 and len({t["id"] for t in a}) == 100
    assert {k: sum(1 for t in a if t["kind"] == k) for k in syn.KINDS} == {k: 10 for k in syn.KINDS}
    frozen = (FAMILY / "results" / "proxy-record.json")
    if frozen.exists():
        data = "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in a).encode()
        assert hashlib.sha256(data).hexdigest() == json.loads(frozen.read_text())["files"]["proxy_synthetic.jsonl"]


@pytest.mark.parametrize("idx", list(range(0, 100, 7)))
def test_synthetic_reference_passes_and_noop_fails_in_the_sandbox(idx: int) -> None:
    task = syn.generate(100, 1234)[idx]
    sbx = proxy.Sandbox("unused", backend="local")
    good = sbx.run(task["fixture"], task["reference_calls"])
    assert good["ok"], good
    assert syn.check(task, good), (task["id"], good)
    noop = sbx.run(task["fixture"], [{"name": "bash", "arguments": {"command": "true"}}])
    assert noop["ok"] and not syn.check(task, noop)


def test_sandbox_refuses_paths_outside_work() -> None:
    task = syn.generate(10, 1234)[0]
    res = proxy.Sandbox("unused", backend="local").run(task["fixture"], [
        {"name": "str_replace_editor", "arguments": {"command": "create", "path": "/etc/x", "file_text": "x"}},
        {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/work/../etc/passwd"}}])
    assert res["ok"] and all(o.startswith("ERROR") for o in res["outputs"])


# ---- the record PR's review rounds: one direct test per rule -------------------------------------------------------------
# The confined checkpoint reads (eval/checkpoint_files.py) of the validator, the size report, the build and the BF16
# teacher, and the record's provenance and GPQA wording. Synthetic checkpoints come from validate_selftest's own helpers
# (2 layers, candidate A's layout); every "outside" file holds valid bytes, so a read that left its directory would
# succeed where it must be refused.
STUB_RUNNER = Path(__file__).resolve().parent / "ripple_stub_runner.py"
BUILD = FAMILY / "scripts" / "quantize_ripple.py"
TEACHER = FAMILY / "eval" / "bf16_teacher_cpu.py"
LAST_PACKED = "model.layers.1.mlp.down_proj.weight"  # a packed NVFP4 weight (U8) in candidate A's layout
SMALL_FILE_CAP = 64 * 2**20


@pytest.fixture(scope="module")
def ripple_trees(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic source and its clean candidate-A export, built once; each test copies them."""
    base = tmp_path_factory.mktemp("ripple-trees")
    src_t = vst.make_source(base / "source", random.Random(1234))
    vst.make_export(base / "export", src_t, "A", random.Random(ord("A")))
    return base


def _fresh(trees: Path, tmp_path: Path) -> tuple[Path, Path]:
    for name in ("source", "export"):
        shutil.copytree(trees / name, tmp_path / name)
    return tmp_path / "source", tmp_path / "export"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_tensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    """A single safetensors file's tensors, parsed here (independent of the code under test)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    return {k: (m["dtype"], m["shape"], raw[8 + n + m["data_offsets"][0]:8 + n + m["data_offsets"][1]])
            for k, m in header.items() if k != "__metadata__"}


def _tensor_names(d: Path) -> list[str]:
    return sorted(_read_tensors(d / "model.safetensors"))


def _index(d: Path, weight_map: dict) -> None:
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def _safetensors(header: dict, data: bytes) -> bytes:
    """A safetensors file: the 8-byte header length, the JSON header padded to 8 bytes, then the data."""
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    return struct.pack("<Q", len(h)) + h + data


def _write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]], fault: str = "") -> None:
    """``tensors`` as one shard laid out as an export writes it (data in header order, LAST_PACKED last when present),
    exact or with one fault: "short" (one byte short, inside the last tensor), "long" (one byte after it), "gap" (one
    unused byte before it) or "overlap" (it starts one byte early, inside the tensor before it)."""
    order = sorted(n for n in tensors if n != LAST_PACKED) + [n for n in tensors if n == LAST_PACKED]
    header, data = {}, b""
    for i, name in enumerate(order):
        dt, shape, blob = tensors[name]
        if i == len(order) - 1:
            data = data + b"\x00" if fault == "gap" else data[:-1] if fault == "overlap" else data
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(data), len(data) + len(blob)]}
        data += blob
    raw = _safetensors(header, data)
    path.write_bytes(raw[:-1] if fault == "short" else raw + b"\x00" if fault == "long" else raw)


def _refused(res: dict, fragment: str) -> None:
    assert res["verdict"] == "FAIL" and fragment in " | ".join(res["errors"]), res


# -- the validator: the good cases ------------------------------------------------------------------------------------------
def test_control_clean_synthetic_export_passes(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    res = ve.validate(exp, src, "A")
    assert res["verdict"] == "PASS" and res["errors"] == [], res
    assert res["quantized_layers"] == {"FP8": 10, "NVFP4": 6, "W4A16_NVFP4": 1}


def test_control_sharded_export_with_an_index_passes(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    names = sorted(t)
    parts = {"model-00001-of-00002.safetensors": names[:20], "model-00002-of-00002.safetensors": names[20:]}
    (exp / "model.safetensors").unlink()
    for shard, members in parts.items():
        vst.write_safetensors(exp / shard, {n: t[n] for n in members})
    _index(exp, {n: shard for shard, members in parts.items() for n in members})
    assert ve.validate(exp, src, "A")["verdict"] == "PASS"


# -- checkpoint_files: the shard set ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["dot-dot", "relative", "absolute"])
def test_validator_refuses_an_index_naming_a_shard_outside_the_directory(ripple_trees: Path, tmp_path: Path,
                                                                         kind: str) -> None:
    """The reviewer's repro: an export with no shard of its own whose index points at a decoy outside validated PASS.
    A shard name must be one plain file name: "..", a relative path out and an absolute path are refused, by the
    validator and by the size report."""
    src, exp = _fresh(ripple_trees, tmp_path)
    shutil.copytree(exp, tmp_path / "decoy")
    names = _tensor_names(exp)
    (exp / "model.safetensors").unlink()
    name = {"dot-dot": "..", "relative": "../decoy/model.safetensors",
            "absolute": str(tmp_path / "decoy" / "model.safetensors")}[kind]
    _index(exp, {n: name for n in names})
    _refused(ve.validate(exp, src, "A"), "not a file name in the directory")
    with pytest.raises(SystemExit, match="not a file name in the directory"):
        sizes_mod.sizes(exp)  # the size report reads the same way


def test_validator_refuses_an_unindexed_extra_safetensors(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    _index(exp, {n: "model.safetensors" for n in _tensor_names(exp)})
    shutil.copyfile(exp / "model.safetensors", exp / "model-extra.safetensors")
    _refused(ve.validate(exp, src, "A"), "not exactly the indexed shards: unindexed ['model-extra.safetensors']")


def test_validator_refuses_a_shard_linked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    os.rename(exp / "model.safetensors", tmp_path / "outside.safetensors")
    os.symlink(tmp_path / "outside.safetensors", exp / "model.safetensors")
    _refused(ve.validate(exp, src, "A"), "links are refused")


# -- checkpoint_files: headers and tensor spans -----------------------------------------------------------------------------
@pytest.mark.parametrize("length", ["over-the-cap", "past-the-end"])
def test_validator_refuses_a_header_length_over_the_cap_or_past_the_end(ripple_trees: Path, tmp_path: Path,
                                                                        length: str) -> None:
    """A header length above safetensors' own 100,000,000-byte cap (in a sparse file that long) or longer than its file
    is refused before the header is read."""
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "model.safetensors"
    raw = bytearray(p.read_bytes())
    raw[:8] = struct.pack("<Q", 100_000_001 if length == "over-the-cap" else len(raw) - 7)
    p.write_bytes(bytes(raw))
    if length == "over-the-cap":
        os.truncate(p, 100_000_064)
    _refused(ve.validate(exp, src, "A"), "exceeds the 100,000,000-byte limit" if length == "over-the-cap"
             else "runs past the end of the file")


@pytest.mark.parametrize("shards", [1, 2])
def test_validator_refuses_a_shard_one_byte_short(ripple_trees: Path, tmp_path: Path, shards: int) -> None:
    """The shard holding the packed weight last is one byte short (the validator never reads packed bytes, so only the
    span bound sees it): a single shard, and the smaller of two (each shard is bounded by its own size)."""
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    if shards == 1:
        short = "model.safetensors"
        _write_shard(exp / short, t, "short")
    else:  # the short shard holds the packed weight alone, so it is the smaller one
        short, other = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
        (exp / "model.safetensors").unlink()
        _write_shard(exp / short, {LAST_PACKED: t[LAST_PACKED]}, "short")
        _write_shard(exp / other, {n: v for n, v in t.items() if n != LAST_PACKED})
        _index(exp, {n: short if n == LAST_PACKED else other for n in t})
    _refused(ve.validate(exp, src, "A"), f"/{short}: {LAST_PACKED}: data_offsets")


@pytest.mark.parametrize("fault", ["exact", "gap", "overlap", "long"])
def test_validator_requires_a_shard_to_hold_exactly_its_tensors(ripple_trees: Path, tmp_path: Path,
                                                               fault: str) -> None:
    """safetensors' own rule: in offset order a shard's tensors start at its data's first byte, each where the one before
    it ends, and the last ends at the file's last byte. Exact passes; a gap, an overlap or a byte after the last tensor
    is refused."""
    src, exp = _fresh(ripple_trees, tmp_path)
    _write_shard(exp / "model.safetensors", _read_tensors(exp / "model.safetensors"), "" if fault == "exact" else fault)
    res = ve.validate(exp, src, "A")
    if fault == "exact":
        assert res["verdict"] == "PASS", res
    else:
        _refused(res, "not at its end" if fault == "long" else "(a gap or an overlap)")


def test_validator_refuses_a_tensor_held_by_two_shards(ripple_trees: Path, tmp_path: Path) -> None:
    """The index names the shard holding the real tensor; the other shard also carries an all-zero copy, which a loader
    reading every tensor of every shard could load instead."""
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    rest = sorted(n for n in t if n != LAST_PACKED)
    first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    dt, shape, data = t[LAST_PACKED]
    (exp / "model.safetensors").unlink()
    vst.write_safetensors(exp / first, {**{n: t[n] for n in rest[:20]}, LAST_PACKED: (dt, shape, bytes(len(data)))})
    vst.write_safetensors(exp / second, {n: t[n] for n in [*rest[20:], LAST_PACKED]})
    _index(exp, {**{n: first for n in rest[:20]}, **{n: second for n in [*rest[20:], LAST_PACKED]}})
    _refused(ve.validate(exp, src, "A"), f"{LAST_PACKED} appears in two shards")


def test_validator_refuses_a_malformed_header_entry(ripple_trees: Path, tmp_path: Path) -> None:
    """The review's repro: an export header entry without its shape is refused by name."""
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "model.safetensors"
    raw = p.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = {**json.loads(raw[8:8 + n]), "model.layers.1.extra": {"dtype": "U8", "data_offsets": [0, 0]}}
    p.write_bytes(_safetensors(header, raw[8 + n:]))
    _refused(ve.validate(exp, src, "A"), "model.layers.1.extra: malformed header entry")


def test_validator_refuses_a_small_file_over_the_64_mib_cap(ripple_trees: Path, tmp_path: Path) -> None:
    """Configs, sidecars and the index are read whole, so their size is capped at 64 MiB (the largest such file,
    tokenizer.json, is 7,137,177 bytes): a sparse generation_config.json one byte over it is refused from its size."""
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "generation_config.json"
    p.write_text('{"bos_token_id": 100257}')
    os.truncate(p, SMALL_FILE_CAP + 1)
    _refused(ve.validate(exp, src, "A"), f"{SMALL_FILE_CAP + 1:,} bytes exceeds the {SMALL_FILE_CAP:,}-byte limit")


# -- checkpoint_files: the tree walk and the pinned source reads ---------------------------------------------------------
@pytest.mark.parametrize("entry", ["plain", "dot-named-link", "fifo"])
def test_refuse_links_walks_the_whole_tree(tmp_path: Path, entry: str) -> None:
    """A plain tree's regular files are counted, dot-named ones included; a dot-named link under .cache/huggingface/,
    or a FIFO in a subdirectory, is refused by its path."""
    root = tmp_path / "tree"
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / "inference").mkdir()
    for rel in ("config.json", ".gitattributes", ".cache/huggingface/.gitignore", "inference/model.py"):
        (root / rel).write_text("x\n")
    (tmp_path / "outside").write_text("outside\n")
    if entry == "plain":
        assert cf.refuse_links(root) == 4
        return
    if entry == "fifo":
        rel, want = "inference/pipe", "not a regular file or a directory"
        os.mkfifo(root / rel)
    else:
        rel, want = ".cache/huggingface/.notice", "cannot open it in its directory"
        os.symlink(tmp_path / "outside", root / rel)
    with pytest.raises(cf.ShardError, match=re.escape(f"/{rel}: {want}")):
        cf.refuse_links(root)


def _pins(src: Path) -> dict[str, str]:
    """The run's pins: the synthetic source's own bytes for every pinned name."""
    return {f: _sha((src / f).read_bytes()) for f in lay.SOURCE_FILE_PINS}


@pytest.mark.parametrize("fault", ["pin-mismatch", "pinned-file-missing"])
def test_read_source_refuses_a_pin_mismatch_or_a_missing_pinned_file(ripple_trees: Path, tmp_path: Path,
                                                                     fault: str) -> None:
    """read_source, the build's and the teacher's source read: every pinned file (the four round-1 pins included) must
    be there and equal its sha256 pin."""
    src, _ = _fresh(ripple_trees, tmp_path)
    pins = _pins(src)
    assert {"LICENSE", "tokenizer_config.json", "special_tokens_map.json", "config.json"} <= set(pins)
    assert set(cf.read_source(src, pins)) == set(pins)
    if fault == "pin-mismatch":
        pins["LICENSE"] = "0" * 64
        want = f"/LICENSE: sha256 {_sha((src / 'LICENSE').read_bytes())} is not the pin {'0' * 64}"
    else:
        (src / "tokenizer_config.json").unlink()
        want = "/tokenizer_config.json: missing"
    with pytest.raises(cf.ShardError, match=re.escape(want)):
        cf.read_source(src, pins)


# -- the build and the teacher, run for real under stand-ins (tests/ripple_stub_runner.py) -------------------------------
def _run(spec: dict, tmp_path: Path) -> dict:
    """One run of tests/ripple_stub_runner.py in its own process: the RUNNER-RESULT it prints."""
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"eval_dir": str(FAMILY / "eval"), **spec}))
    r = subprocess.run([sys.executable, str(STUB_RUNNER), str(p)], capture_output=True, text=True, timeout=120,
                       check=False, cwd=tmp_path)
    line = next((x for x in r.stdout.splitlines() if x.startswith("RUNNER-RESULT ")), None)
    assert line is not None, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(line[len("RUNNER-RESULT "):])


def _build(trees: Path, tmp_path: Path, mutate=None, **extra) -> tuple[dict, Path, Path]:
    """The build (candidate A, 4 calibration rows) on a fresh synthetic source pinned for the run; ``mutate(src)``
    changes the source after the pins are taken; ``extra`` goes to the stand-in exporter."""
    src, _ = _fresh(trees, tmp_path)
    pins = _pins(src)
    if mutate is not None:
        mutate(src)
    calib = tmp_path / "calib.jsonl"
    calib.write_text("".join(json.dumps({"text": "x", "source": s}) + "\n" for s in lay.SOURCES_IN_FILE_ORDER))
    out = tmp_path / "out" / "A"
    res = _run({"mode": "build", "script": str(BUILD), "pins": pins, "calib_sha256": _sha(calib.read_bytes()),
                "argv": ["A", str(src), str(calib), str(out), str(tmp_path / "work"), "--rows", "4",
                         "--recipe-dir", str(ROOT / lay.RECIPE_DIR)], **extra}, tmp_path)
    return res, src, out


def _link_outside(path: Path, outside: Path) -> None:
    """Move the file to ``outside`` and leave a link to it: its bytes (and so its pin) are unchanged."""
    os.rename(path, outside)
    os.symlink(outside, path)


def test_build_publishes_the_pinned_sidecars_and_restores_the_dtype(ripple_trees: Path, tmp_path: Path) -> None:
    """The export carries the source's sidecars as regular files with the pinned bytes and no link anywhere, and
    config.json is the source's plus the export's quantization_config and dtype (bfloat16; the source's has none)."""
    res, src, out = _build(ripple_trees, tmp_path)
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    for f in lay.COPIED_FILES:
        assert not (out / f).is_symlink() and (out / f).read_bytes() == (src / f).read_bytes(), f
    assert not [p for p in out.rglob("*") if p.is_symlink()]
    cfg = json.loads((out / "config.json").read_text())
    assert "dtype" not in json.loads((src / "config.json").read_text()) and cfg["dtype"] == "bfloat16"
    assert cfg["quantization_config"]["quant_algo"] == "MIXED_PRECISION"


def test_build_refuses_a_source_license_linked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    """The reviewer's repro: a source LICENSE linked outside was published as a regular file holding the outside
    bytes. Refused before anything reads the source, and nothing is published."""
    outside = tmp_path / "outside-LICENSE"
    res, src, out = _build(ripple_trees, tmp_path, lambda s: _link_outside(s / "LICENSE", outside))
    assert res["exit"] != 0 and "/LICENSE: cannot open it in its directory" in res["out"], res["out"]
    assert not out.exists() and not out.with_name("A.partial").exists()
    assert str(outside) not in res["opened"] and res["lib"] == [], res


def test_build_refuses_a_link_left_in_its_export(ripple_trees: Path, tmp_path: Path) -> None:
    """A link anywhere in the export (here one directory down) stops the build before OUT is published."""
    outside, rel = tmp_path / "outside", "extra/notice"
    outside.write_text("outside\n")
    res, src, out = _build(ripple_trees, tmp_path, export_links={rel: str(outside)})
    assert res["exit"] != 0 and f"/{rel}: cannot open it in its directory" in res["out"], res["out"]
    assert not out.exists()


def test_build_drops_the_store_metadata(ripple_trees: Path, tmp_path: Path) -> None:
    """The model store's own files describe the BF16 source, not the export: dropped before OUT is published."""
    names = ("SHA256SUMS.upstream", "SOURCE_REVISION", "SUBSET")
    res, src, out = _build(ripple_trees, tmp_path, export_files={n: "store metadata\n" for n in names})
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    assert not [n for n in names if (out / n).exists()], sorted(os.listdir(out))


@pytest.mark.parametrize("source", ["clean", "linked"])
def test_teacher_refuses_a_linked_source_before_any_library_load(ripple_trees: Path, tmp_path: Path,
                                                                 source: str) -> None:
    """The teacher's Streamer: on a clean source every library load (each stand-in raises when called before the
    confined check returned) comes after the check; with config.json linked to an outside copy of the same bytes, the
    run is refused with no library load and no outside file opened."""
    src, _ = _fresh(ripple_trees, tmp_path)
    pins = _pins(src)
    outside = tmp_path / "outside-config.json"
    if source == "linked":
        _link_outside(src / "config.json", outside)
    res = _run({"mode": "teacher", "script": str(TEACHER), "src": str(src), "tensor": "model.norm.weight",
                "pins": pins}, tmp_path)
    if source == "clean":
        assert res["exit"] == 0 and "TEACHER-STREAMER-OK" in res["out"], res["out"]
        assert res["lib"] and all(e["after_check"] for e in res["lib"]), res["lib"]
    else:
        assert res["exit"] != 0 and "/config.json: cannot open it in its directory" in res["out"], res["out"]
        assert res["lib"] == [] and str(outside) not in res["opened"], res


# -- the validator: config files and sidecars --------------------------------------------------------------------------
@pytest.mark.parametrize("body", ["match", "mismatch", "empty"])
def test_validator_checks_generation_config_token_ids(ripple_trees: Path, tmp_path: Path, body: str) -> None:
    """A generation_config.json that is there carries config.json's token ids; an empty one is refused, never taken
    for an absent one."""
    src, exp = _fresh(ripple_trees, tmp_path)
    ids = {k: vst.source_config()[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")}
    raw = {"match": json.dumps(ids), "mismatch": json.dumps({**ids, "bos_token_id": 1}), "empty": ""}[body]
    (exp / "generation_config.json").write_text(raw)
    res = ve.validate(exp, src, "A")
    if body == "match":
        assert res["verdict"] == "PASS" and res["generation_config_token_ids"] == ids, res
    else:
        _refused(res, "token ids differ from config.json" if body == "mismatch" else "config files unreadable")


def test_validator_requires_config_dtype_bfloat16(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    cfg = json.loads((exp / "config.json").read_text())
    (exp / "config.json").write_text(json.dumps({**cfg, "dtype": "float16"}))
    _refused(ve.validate(exp, src, "A"), "config.json dtype is 'float16', not 'bfloat16'")


def test_validator_refuses_a_sidecar_one_byte_off(ripple_trees: Path, tmp_path: Path) -> None:
    """The export's LICENSE with one bit of its last byte flipped: the same length, refused."""
    src, exp = _fresh(ripple_trees, tmp_path)
    b = bytearray((exp / "LICENSE").read_bytes())
    b[-1] ^= 1
    (exp / "LICENSE").write_bytes(bytes(b))
    _refused(ve.validate(exp, src, "A"), "LICENSE differs")


# -- the record: provenance and the GPQA cap wording ---------------------------------------------------------------------
def test_provenance_gates_cite_a_file_holding_the_source_revision_and_its_license() -> None:
    ledger = json.loads((FAMILY / "results" / "publication-readiness-ledger.json").read_text())
    gates = [(p["product_id"], g) for p in ledger["products"] for g in p["gates"]
             if g["id"] == "provenance_ownership" and g["status"] == "verified"]
    assert len(gates) == 2
    for pid, gate_ in gates:
        cited = json.loads((ROOT / gate_["evidence_path"]).read_text())
        assert cited["source"] == f"{lay.SOURCE_REPO}@{lay.SOURCE_REV}", pid
        assert cited["source_license"]["license"] == "apache-2.0", pid
        assert cited["source_license"]["license_file_sha256"] == lay.SOURCE_FILE_PINS["LICENSE"], pid
        assert cited["source"] in gate_["evidence"] and "Apache-2.0" in gate_["evidence"], pid


def test_gpqa_cap_counts_are_derived_from_the_runs_and_worded_exactly() -> None:
    g = json.loads((FAMILY / "results" / "gate-summary.json").read_text())
    for cell in ("A", "vendor"):
        runs = [json.loads((FAMILY / "results" / f"gpqa-{cell}-r{r}.json").read_text()) for r in (1, 2)]
        assert g["gpqa"]["hit_max_tokens"][cell] == [r["accuracy"]["hit_max_tokens"] for r in runs]
        assert g["gpqa"]["hit_max_tokens"][cell] == [r["finish_reasons"]["length"] for r in runs]
        assert g["gpqa"]["truncated_no_answer"][cell] == [r["accuracy"]["truncated_no_answer"] for r in runs]
    want = "42 and 48 of A's, 42 and 38 of the vendor's (59 and 57 / 50 and 49 responses hit the cap)"
    assert g["card_strings"]["gpqa_trunc"] == want
    docs = [FAMILY / "model-card" / "base-nvfp4.md", FAMILY / "README.md",
            ROOT / "publication" / "huggingface" / lay.PRODUCT_IDS["A"] / "README.md",
            FAMILY / "results" / "publication-readiness-ledger.json"]
    for doc in docs:
        text = " ".join(doc.read_text().split())
        for old in ("truncated items count as wrong", "GPQA truncations", "truncations 42", "42 and 48 of A's items"):
            assert old not in text, (doc.name, old)
    card = " ".join(docs[0].read_text().split())
    assert f"| Items without an answer at the 65,536-token cap | {want} | counted as wrong;" in card
    readme = " ".join(docs[1].read_text().split())
    assert f"| GPQA items without an answer at the cap | {want} | counted as wrong;" in readme
    assert ("Unparseable items and items without an answer at the token cap count as wrong; a response that hit the "
            "cap with an answer is scored.") in card
