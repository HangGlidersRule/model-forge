"""Issue #39 (Ripple-65B Preview) kit: the pure-Python pieces the GPU window relies on, tested without the model, the GPU
or the network. Covers the candidate layouts and the vLLM modelopt_mixed resolution re-implementation, the calibration
row rule (rank key, smoke take), the export validator (its self-test on synthetic exports with planted faults), the
gate (bars, bootstrap, selection rule) on synthetic result files, the BFCL AST checker re-implementation, the synthetic
tool-call tasks (reference solutions pass and no-ops fail in the local sandbox backend) and the KL estimator.
Review round 1 of the record PR added the confinement tests: the export validator, the size report, the build and the
BF16 teacher read checkpoint files only inside their directories (eval/checkpoint_files.py), and the build publishes
only the pinned source bytes as regular files. The build and the teacher run for real under stand-ins for torch,
transformers, safetensors and ModelOpt (none is installed in CI), each in its own process.
Review round 2 added dot-named links at every depth, tensor spans at the exact end of their file, a tensor held by two
shards, the small-file cap (64 MiB), a pin one digit off and a pinned file removed, every pin's value, and one
negative test per self-check of the record's builders."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tracemalloc
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
    assert "VALIDATOR-SELFTEST-OK" in out and "A: clean PASS, 23 planted faults caught" in out


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



# ---- review round 1: confinement of every checkpoint read ----------------------------------------------------------------
# Synthetic checkpoints from validate_selftest's own helpers. Every "outside" file is a sibling of the directory under
# test holding valid bytes, so a read that leaves the directory succeeds where it should be refused (the reviewer's
# repro: an export with no shard of its own, whose index points at ../decoy/model.safetensors, validated PASS).
STUB_RUNNER = Path(__file__).resolve().parent / "ripple_stub_runner.py"
BUILD = FAMILY / "scripts" / "quantize_ripple.py"
TEACHER = FAMILY / "eval" / "bf16_teacher_cpu.py"
CF_PATH = FAMILY / "eval" / "checkpoint_files.py"
PINNED_SOURCE_FILES = (*lay.COPIED_FILES, "config.json")


def _cf():
    """eval/checkpoint_files.py, the confinement helpers (review round 1); a plain failure where they are missing. The
    module the validator imported is returned (not a second copy), so its classes are the ones the scripts raise."""
    if not CF_PATH.exists():
        pytest.fail("eval/checkpoint_files.py is missing: no confinement helpers")
    return importlib.import_module("checkpoint_files")


@pytest.fixture(scope="module")
def ripple_trees(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic source and its clean candidate-A export, built once; each test copies them."""
    base = tmp_path_factory.mktemp("ripple-trees")
    src_t = vst.make_source(base / "source", random.Random(1234))
    vst.make_export(base / "export", src_t, "A", random.Random(ord("A")))
    return base


def _fresh(trees: Path, tmp_path: Path) -> tuple[Path, Path]:
    for name in ("source", "export"):
        shutil.copytree(trees / name, tmp_path / name, symlinks=True)
    return tmp_path / "source", tmp_path / "export"


def _read_tensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    """A single safetensors file's tensors, parsed here (independent of the validator under test)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    return {k: (m["dtype"], m["shape"], raw[8 + n + m["data_offsets"][0]:8 + n + m["data_offsets"][1]])
            for k, m in header.items() if k != "__metadata__"}


def _tensor_names(d: Path) -> list[str]:
    return sorted(_read_tensors(d / "model.safetensors"))


def _index(d: Path, weight_map: dict) -> None:
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def _validate(exp: Path, src: Path) -> dict:
    """validate_export.validate, with an exception counted as a refusal (the reason is the exception's text)."""
    try:
        return ve.validate(exp, src, "A")
    except Exception as e:  # noqa: BLE001 - the old validator raised on some inputs; a raise is not a PASS
        return {"verdict": "RAISED", "errors": [f"{type(e).__name__}: {e}"]}


def _refused(res: dict, *fragments: str) -> None:
    assert res["verdict"] == "FAIL", res
    text = " | ".join(res["errors"])
    assert any(f in text for f in fragments), text


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _with_header_len(path: Path, n: int, length: int | None = None) -> None:
    raw = bytearray(path.read_bytes())
    raw[:8] = struct.pack("<Q", n)
    path.write_bytes(bytes(raw))
    if length is not None:
        with open(path, "r+b") as f:  # sparse: the file really is this long, so only the cap can refuse the length
            f.truncate(length)


# -- controls: pass on the PR's first head and on the fix -------------------------------------------------------------------
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


# -- validator: shard names and the shard set ----------------------------------------------------------------------------------
def test_validator_refuses_an_index_pointing_at_a_decoy_outside(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    shutil.copytree(exp, tmp_path / "decoy")
    names = _tensor_names(exp)
    (exp / "model.safetensors").unlink()
    _index(exp, {n: "../decoy/model.safetensors" for n in names})
    _refused(_validate(exp, src), "not a file name in the directory")


def test_validator_refuses_an_absolute_shard_name(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    shutil.copytree(exp, tmp_path / "decoy")
    names = _tensor_names(exp)
    (exp / "model.safetensors").unlink()
    _index(exp, {n: str(tmp_path / "decoy" / "model.safetensors") for n in names})
    _refused(_validate(exp, src), "not a file name in the directory")


def test_validator_refuses_a_source_index_pointing_outside(ripple_trees: Path, tmp_path: Path) -> None:
    """The BF16 bytes the export is compared against must be the source's own."""
    src, exp = _fresh(ripple_trees, tmp_path)
    shutil.copytree(src, tmp_path / "other-source")
    names = _tensor_names(src)
    (src / "model.safetensors").unlink()
    _index(src, {n: "../other-source/model.safetensors" for n in names})
    _refused(_validate(exp, src), "not a file name in the directory")


@pytest.mark.parametrize("name", ["sub/model.safetensors", "sub\\model.safetensors", "..", ".", "",
                                  "model\x00.safetensors"])
def test_validator_refuses_a_shard_name_that_is_not_one_plain_file_name(ripple_trees: Path, tmp_path: Path,
                                                                         name: str) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    names = _tensor_names(exp)
    (exp / "sub").mkdir()
    shutil.copyfile(exp / "model.safetensors", exp / "sub" / "model.safetensors")
    (exp / "model.safetensors").unlink()
    _index(exp, {n: name for n in names})
    _refused(_validate(exp, src), "not a file name in the directory")


def test_validator_refuses_a_shard_symlinked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    os.rename(exp / "model.safetensors", tmp_path / "outside.safetensors")
    os.symlink(tmp_path / "outside.safetensors", exp / "model.safetensors")
    _refused(_validate(exp, src), "links are refused")


def test_validator_refuses_a_source_shard_symlinked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    os.rename(src / "model.safetensors", tmp_path / "outside.safetensors")
    os.symlink(tmp_path / "outside.safetensors", src / "model.safetensors")
    _refused(_validate(exp, src), "links are refused")


def test_validator_refuses_a_shard_linked_to_a_file_inside_the_directory(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    names = _tensor_names(exp)
    os.rename(exp / "model.safetensors", exp / "real.bin")
    os.symlink("real.bin", exp / "model-00001-of-00001.safetensors")
    _index(exp, {n: "model-00001-of-00001.safetensors" for n in names})
    _refused(_validate(exp, src), "links are refused")


def test_validator_refuses_a_fifo_shard_without_blocking(ripple_trees: Path, tmp_path: Path) -> None:
    """A FIFO in place of a shard: opening it for reading would wait for a writer. The validator runs in its own
    process here, so a validator that blocks fails this test instead of hanging the suite."""
    src, exp = _fresh(ripple_trees, tmp_path)
    (exp / "model.safetensors").unlink()
    os.mkfifo(exp / "model.safetensors")
    report = tmp_path / "report.json"
    try:
        r = subprocess.run([sys.executable, str(FAMILY / "eval" / "validate_export.py"), str(exp), str(src), "A",
                            str(report)], capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail("the validator blocked opening a FIFO shard")
    assert r.returncode != 0 and "VALIDATE-OK" not in r.stdout, r.stdout + r.stderr
    _refused(json.loads(report.read_text()), "not a regular file")


def test_validator_refuses_an_unindexed_extra_safetensors(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    _index(exp, {n: "model.safetensors" for n in _tensor_names(exp)})
    shutil.copyfile(exp / "model.safetensors", exp / "model-extra.safetensors")
    _refused(_validate(exp, src), "not exactly the indexed shards")


def test_validator_refuses_an_extra_safetensors_without_an_index(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    shutil.copyfile(exp / "model.safetensors", exp / "stray.safetensors")
    _refused(_validate(exp, src), "not exactly the indexed shards")


def test_validator_refuses_a_tensor_held_by_another_shard_than_indexed(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    names = sorted(t)
    (exp / "model.safetensors").unlink()
    vst.write_safetensors(exp / "a.safetensors", {n: t[n] for n in names[:5]})
    vst.write_safetensors(exp / "b.safetensors", {n: t[n] for n in names[5:]})
    wm = {n: ("a.safetensors" if i < 5 else "b.safetensors") for i, n in enumerate(names)}
    wm[names[0]], wm[names[-1]] = "b.safetensors", "a.safetensors"
    _index(exp, wm)
    _refused(_validate(exp, src), "other shards")


# -- validator: header length and tensor spans --------------------------------------------------------------------------------
def test_validator_refuses_a_header_length_over_the_cap_without_allocating_it(ripple_trees: Path,
                                                                              tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    _with_header_len(exp / "model.safetensors", 100_000_001, 100_000_064)
    tracemalloc.start()
    try:
        res = _validate(exp, src)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    _refused(res, "exceeds the 100,000,000-byte limit")
    assert peak < 8 * 1024 * 1024, f"allocated {peak:,} bytes before refusing"


def test_validator_refuses_a_header_length_past_the_end_of_the_file(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "model.safetensors"
    _with_header_len(p, p.stat().st_size - 7)  # under the cap; one byte longer than the file can hold
    _refused(_validate(exp, src), "runs past the end of the file")


def test_header_bounds_are_inclusive(tmp_path: Path) -> None:
    """A header that ends exactly at the end of its file is read; the cap is the safetensors library's 100,000,000."""
    cf = _cf()
    assert cf.MAX_HEADER_BYTES == 100_000_000 and 70_208 < cf.MAX_HEADER_BYTES  # largest real header: 70,208 bytes
    d = tmp_path / "ck"
    d.mkdir()
    header = b'{"__metadata__":{}}'
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    with cf.Shards(d) as s:
        assert s.tensors() == {}
    _with_header_len(d / "model.safetensors", cf.MAX_HEADER_BYTES, 8 + cf.MAX_HEADER_BYTES)
    with cf.Shards(d) as s, pytest.raises(cf.ShardError, match="not valid JSON"):
        s.tensors()  # at the cap exactly, the length passes both bounds and the (blank) header is parsed


@pytest.mark.parametrize("offsets", [[0, 10 ** 9], [-4, 0], [8, 4], [0, "8"], [0.0, 8], [False, True]])
def test_validator_refuses_a_tensor_span_outside_its_shard(ripple_trees: Path, tmp_path: Path, offsets: list) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "model.safetensors"
    raw = p.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    header["model.norm.weight"]["data_offsets"] = offsets
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    p.write_bytes(struct.pack("<Q", len(h)) + h + raw[8 + n:])
    _refused(_validate(exp, src), "fall outside the file")


LAST_PACKED = "model.layers.1.mlp.down_proj.weight"  # a packed NVFP4 weight (U8) in candidate A's layout


def _write_with_last(path: Path, tensors: dict[str, tuple[str, list[int], bytes]], last: str, cut: int = 0) -> None:
    """Write a safetensors file the way an export lays one out (8-byte length, the header padded to 8 bytes, then the
    data in header order) with ``last`` as the last tensor in the file, cut ``cut`` bytes short of its end."""
    order = [n for n in sorted(tensors) if n != last] + [last]
    header, off = {}, 0
    for n in order:
        dt, shape, data = tensors[n]
        header[n] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    raw = struct.pack("<Q", len(h)) + h + b"".join(tensors[n][2] for n in order)
    path.write_bytes(raw[:len(raw) - cut])


def test_a_tensor_ending_exactly_at_the_end_of_its_file_passes(ripple_trees: Path, tmp_path: Path) -> None:
    """The span bound is inclusive: the last tensor in the file, a packed U8 weight, may end at the file's last byte."""
    cf = _cf()
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    assert t[LAST_PACKED][0] == "U8"
    _write_with_last(exp / "model.safetensors", t, LAST_PACKED)
    with cf.Shards(exp) as s:
        _, offset, nbytes, dtype, _ = s.tensors()[LAST_PACKED]
    assert offset + nbytes == (exp / "model.safetensors").stat().st_size and dtype == "U8"
    res = ve.validate(exp, src, "A")
    assert res["verdict"] == "PASS", res


def test_a_tensor_one_byte_past_the_end_of_its_file_is_refused(ripple_trees: Path, tmp_path: Path) -> None:
    """The same file one byte short: the validator never reads a packed weight's bytes, so the span bound is the only
    check that sees the missing byte (safetensors' own reader refuses such a file too)."""
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    _write_with_last(exp / "model.safetensors", t, LAST_PACKED, cut=1)
    res = _validate(exp, src)
    _refused(res, "fall outside the file")
    assert f"{LAST_PACKED}: data_offsets" in " | ".join(res["errors"]), res


@pytest.mark.parametrize("copy_in", ["earlier", "later"])
def test_a_tensor_held_by_two_shards_is_refused(ripple_trees: Path, tmp_path: Path, copy_in: str) -> None:
    """Two shards, the index naming the one that holds the real tensor; the other shard (the earlier or the later one)
    also carries an all-zero copy of it. A loader that reads every tensor of every indexed shard could load either copy,
    so the checkpoint is refused whichever shard is read last."""
    src, exp = _fresh(ripple_trees, tmp_path)
    t = _read_tensors(exp / "model.safetensors")
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    named, other = (shards[1], shards[0]) if copy_in == "earlier" else shards
    rest = [n for n in sorted(t) if n != LAST_PACKED]
    parts = {shards[0]: rest[:20], shards[1]: rest[20:]}
    parts[named] = [*parts[named], LAST_PACKED]
    dt, shape, data = t[LAST_PACKED]
    (exp / "model.safetensors").unlink()
    for shard, members in parts.items():
        extra = {LAST_PACKED: (dt, shape, bytes(len(data)))} if shard == other else {}
        vst.write_safetensors(exp / shard, {**{n: t[n] for n in members}, **extra})
    _index(exp, {n: shard for shard, members in parts.items() for n in members})
    _refused(_validate(exp, src), f"{LAST_PACKED} appears in two shards")


def test_validator_reads_tensors_through_the_descriptors_it_opened(ripple_trees: Path, tmp_path: Path) -> None:
    """Replace the shard after the validator opened it: every read still returns the opened file's bytes."""
    src, exp = _fresh(ripple_trees, tmp_path)
    ck = ve.Checkpoint(exp)
    try:
        name = "model.layers.0.self_attn.value_res_lambda"
        before, digest = ck.raw(name), ck.digest(name)
        size = (exp / "model.safetensors").stat().st_size
        os.rename(exp / "model.safetensors", tmp_path / "moved.safetensors")
        (exp / "model.safetensors").write_bytes(b"\x00" * size)
        assert ck.raw(name) == before and ck.digest(name) == digest
    finally:
        getattr(ck, "close", lambda: None)()


# -- validator: config files and sidecars --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["config.json", "hf_quant_config.json", "generation_config.json"])
def test_validator_refuses_an_export_config_file_linked_outside(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    if name == "generation_config.json":
        (exp / name).write_text(json.dumps({k: vst.source_config()[k] for k in ("bos_token_id", "eos_token_id",
                                                                                  "pad_token_id")}))
    os.rename(exp / name, tmp_path / f"outside-{name}")
    os.symlink(tmp_path / f"outside-{name}", exp / name)
    _refused(_validate(exp, src), "links are refused")


def test_validator_refuses_a_source_config_linked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    os.rename(src / "config.json", tmp_path / "outside-config.json")
    os.symlink(tmp_path / "outside-config.json", src / "config.json")
    _refused(_validate(exp, src), "links are refused")


@pytest.mark.parametrize("side", ["source", "export"])
@pytest.mark.parametrize("name", ["LICENSE", "tokenizer_config.json", "modeling_ricursive.py"])
def test_validator_refuses_a_sidecar_linked_outside(ripple_trees: Path, tmp_path: Path, side: str, name: str) -> None:
    """The reviewer's second repro: a LICENSE linked outside passed the byte-identity check. The other side holds the
    same outside bytes, so the check can only fail by refusing the link."""
    src, exp = _fresh(ripple_trees, tmp_path)
    outside = tmp_path / f"outside-{name}"
    outside.write_text("OUTSIDE THE DIRECTORY\n")
    for d in (src, exp):
        (d / name).unlink()
        if d.name == side:
            os.symlink(outside, d / name)
        else:
            shutil.copyfile(outside, d / name)
    _refused(_validate(exp, src), "links are refused")


def test_validator_skips_a_sidecar_the_source_lacks(ripple_trees: Path, tmp_path: Path) -> None:
    """A sidecar absent from the source is not required in the export (a present one must match byte for byte)."""
    src, exp = _fresh(ripple_trees, tmp_path)
    for d in (src, exp):
        (d / "special_tokens_map.json").unlink()
    res = ve.validate(exp, src, "A")
    assert res["verdict"] == "PASS", res


def test_validator_checks_generation_config_token_ids(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    ids = {k: vst.source_config()[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")}
    (exp / "generation_config.json").write_text(json.dumps({**ids, "_from_model_config": True}))
    res = ve.validate(exp, src, "A")
    assert res["verdict"] == "PASS" and res["generation_config_token_ids"] == ids, res
    for key, bad in (("bos_token_id", 1), ("eos_token_id", [100265]), ("pad_token_id", None)):
        (exp / "generation_config.json").write_text(json.dumps({**ids, key: bad}))
        _refused(_validate(exp, src), f"generation_config.json token ids differ from config.json: ['{key}")
    for key in ids:  # a key dropped from generation_config.json is a difference too
        (exp / "generation_config.json").write_text(json.dumps({k: v for k, v in ids.items() if k != key}))
        _refused(_validate(exp, src), f"generation_config.json token ids differ from config.json: ['{key} None")
    (exp / "generation_config.json").write_text("[1, 2]")
    _refused(_validate(exp, src), "generation_config.json token ids differ")


# -- size report -----------------------------------------------------------------------------------------------------------------
def _sizes(d: Path) -> str:
    try:
        sizes_mod.sizes(d)
    except BaseException as e:  # noqa: BLE001 - SystemExit carries the refusal; anything else counts as one too
        return f"{type(e).__name__}: {e}"
    return "NO-REFUSAL"


def test_sizes_counts_the_indexed_shards_and_refuses_an_index_escape(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    clean = sizes_mod.sizes(exp)
    assert clean["files"] == 1 and clean["tensors"] == len(_tensor_names(exp))
    shutil.copytree(exp, tmp_path / "decoy")
    names = _tensor_names(exp)
    (exp / "model.safetensors").unlink()
    _index(exp, {n: "../decoy/model.safetensors" for n in names})
    assert "not a file name in the directory" in _sizes(exp)


def test_sizes_refuses_an_unindexed_extra_and_a_symlinked_shard(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    shutil.copyfile(exp / "model.safetensors", exp / "stray.safetensors")
    assert "not exactly the indexed shards" in _sizes(exp)
    (exp / "stray.safetensors").unlink()
    os.rename(exp / "model.safetensors", tmp_path / "outside.safetensors")
    os.symlink(tmp_path / "outside.safetensors", exp / "model.safetensors")
    assert "links are refused" in _sizes(exp)


def test_sizes_refuses_a_header_over_the_cap_and_past_the_end(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "model.safetensors"
    good = p.read_bytes()
    _with_header_len(p, 100_000_001, 100_000_064)
    tracemalloc.start()
    try:
        msg = _sizes(exp)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert "exceeds the 100,000,000-byte limit" in msg and peak < 8 * 1024 * 1024, (msg, peak)
    p.write_bytes(good)
    _with_header_len(p, len(good) - 7)
    assert "runs past the end of the file" in _sizes(exp)


# -- pins -------------------------------------------------------------------------------------------------------------------------
def test_every_file_the_build_reads_by_name_is_pinned() -> None:
    """All eight pins with their values (the stand-in runs replace every pin, so only this test keeps them): the first
    four posted on the issue, the other four added in review round 1. Each equals the model store's SHA256SUMS.upstream
    line and the bytes the Hugging Face API serves at the pinned revision."""
    assert set(PINNED_SOURCE_FILES) <= set(lay.SOURCE_FILE_PINS)
    pins = """\
f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3  modeling_ricursive.py
002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6  configuration_ricursive.py
877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3  chat_template.jinja
73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca  tokenizer.json
dbd4722605382159a16f2cf5779f4a8b29c51d8b397d2a05c099c90eeacfe771  tokenizer_config.json
3f7cd7d0cfb20f9faa47c75fd4df91b1c9a78aef0be27ded12cb705b23951341  special_tokens_map.json
58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd  LICENSE
9a5654b256ead01855f0d74cf7a70fa65294267e6b7b754b5e5737457a45e14a  config.json
"""
    want = {line.split()[1]: line.split()[0] for line in pins.splitlines()}
    assert len(want) == 8 and lay.SOURCE_FILE_PINS == want


# -- the build and the teacher, run for real under stand-ins --------------------------------------------------------------------
def _run(spec: dict, tmp_path: Path) -> dict:
    spec = {"eval_dir": str(FAMILY / "eval"), **spec}
    p = tmp_path / f"spec-{len(list(tmp_path.glob('spec-*')))}.json"
    p.write_text(json.dumps(spec))
    r = subprocess.run([sys.executable, str(STUB_RUNNER), str(p)], capture_output=True, text=True, timeout=120,
                       check=False, cwd=tmp_path)
    line = next((x for x in r.stdout.splitlines() if x.startswith("RUNNER-RESULT ")), None)
    assert line is not None, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(line[len("RUNNER-RESULT "):])


def _calib(tmp_path: Path) -> Path:
    p = tmp_path / "calib.jsonl"
    p.write_text("".join(json.dumps({"text": "x", "source": s}) + "\n" for s in lay.SOURCES_IN_FILE_ORDER))
    return p


def _build(trees: Path, tmp_path: Path, mutate=None, pins: dict | None = None, argv: tuple = (),
           **extra) -> tuple[dict, Path, Path]:
    """Run the build on a fresh synthetic source. Its small files get pins for this run, except where ``pins`` says
    otherwise; ``mutate(src)`` changes the source after the pins are taken."""
    src, _ = _fresh(trees, tmp_path)
    run_pins = {f: _sha((src / f).read_bytes()) for f in PINNED_SOURCE_FILES}
    run_pins.update(pins or {})
    if mutate is not None:
        mutate(src)
    calib = _calib(tmp_path)
    out = tmp_path / "out" / "A"
    res = _run({"mode": "build", "script": str(BUILD), "pins": run_pins, "calib_sha256": _sha(calib.read_bytes()),
                "argv": ["A", str(src), str(calib), str(out), str(tmp_path / "work"), "--rows", "4",
                         "--recipe-dir", str(ROOT / lay.RECIPE_DIR), *argv], **extra}, tmp_path)
    return res, src, out


def _outside(tmp_path: Path, name: str = "outside-LICENSE") -> Path:
    p = tmp_path / name
    p.write_text("OUTSIDE THE SOURCE DIRECTORY\n")
    return p


def _link_into(src: Path, name: str, target: Path) -> None:
    (src / name).unlink()
    os.symlink(target, src / name)


def _not_built(res: dict, out: Path, *, before_export: bool = True) -> None:
    """The build refused: nothing is published at OUT. A refusal of the source comes before the export, so not even
    OUT.partial exists; a refusal of the export directory leaves OUT.partial for inspection (the next run clears it)."""
    assert res["exit"] != 0 and "QUANT-OK" not in res["out"], res["out"]
    assert not out.exists()
    if before_export:
        assert not out.with_name(out.name + ".partial").exists()


def test_control_build_publishes_the_pinned_sidecars_as_regular_files(ripple_trees: Path, tmp_path: Path) -> None:
    res, src, out = _build(ripple_trees, tmp_path)
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    for f in lay.COPIED_FILES:
        assert (out / f).is_file() and not (out / f).is_symlink() and (out / f).read_bytes() == (src / f).read_bytes()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quant_algo"] == "MIXED_PRECISION"
    assert {k: v for k, v in cfg.items() if k not in ("quantization_config", "dtype")} == \
        json.loads((src / "config.json").read_text())
    assert json.loads((out / "config.transformers5.json").read_text())["dtype"] == "bfloat16"


def test_build_refuses_a_sidecar_linked_outside_the_source(ripple_trees: Path, tmp_path: Path) -> None:
    """The reviewer's repro through the build: a source LICENSE linked outside was published as a regular file carrying
    the outside bytes. The run's LICENSE pin is the outside file's own hash, so only the link refusal can stop it."""
    outside = _outside(tmp_path)
    res, src, out = _build(ripple_trees, tmp_path, lambda s: _link_into(s, "LICENSE", outside),
                           pins={"LICENSE": _sha(outside.read_bytes())})
    _not_built(res, out)
    assert str(outside) not in res["opened"], res["opened"]


@pytest.mark.parametrize("name", ["LICENSE", "tokenizer_config.json", "special_tokens_map.json", "config.json"])
def test_build_refuses_a_newly_pinned_file_that_misses_its_pin(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    """The run keeps the repository's own pin for ``name``: the synthetic file's bytes miss it."""
    res, src, out = _build(ripple_trees, tmp_path, pins={name: lay.SOURCE_FILE_PINS.get(name, "0" * 64)})
    _not_built(res, out)
    assert name in res["out"], res["out"]


def test_build_refuses_a_sidecar_that_differs_from_its_pin(ripple_trees: Path, tmp_path: Path) -> None:
    def tamper(src: Path) -> None:
        (src / "chat_template.jinja").write_text("{{ tampered }}\n")

    res, src, out = _build(ripple_trees, tmp_path, tamper)
    _not_built(res, out)
    assert "chat_template.jinja" in res["out"], res["out"]


def test_build_refuses_a_config_linked_outside_the_source(ripple_trees: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside-config.json"

    def link(src: Path) -> None:
        shutil.copyfile(src / "config.json", outside)
        _link_into(src, "config.json", outside)

    res, src, out = _build(ripple_trees, tmp_path, link)
    _not_built(res, out)
    assert str(outside) not in res["opened"], res["opened"]


@pytest.mark.parametrize("name", ["model.safetensors", ".cache-note", "inference/.cache-note", ".cache/download/note"])
def test_build_refuses_a_link_anywhere_in_the_source_tree(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    """transformers, accelerate and the exporter read the source by path: a link anywhere in it is refused before they
    run, dot-named entries included, at the top level and in subdirectories (real checkpoints carry .gitattributes and
    a .cache directory)."""
    outside = tmp_path / "outside-file"

    def link(src: Path) -> None:
        if name == "model.safetensors":
            os.rename(src / name, outside)
        else:
            outside.write_text("OUTSIDE THE SOURCE DIRECTORY\n")
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(outside, src / name)

    res, src, out = _build(ripple_trees, tmp_path, link)
    _not_built(res, out)
    assert f"/{name}: cannot open it in its directory" in res["out"], res["out"]
    assert str(outside) not in res["opened"], res["opened"]


def test_build_publishes_the_bytes_it_checked_when_a_sidecar_is_swapped_later(ripple_trees: Path,
                                                                              tmp_path: Path) -> None:
    """The source's LICENSE becomes a link to outside bytes after the build checked it: the export still carries the
    checked bytes, as a regular file (each sidecar is read once and never re-read by path)."""
    outside = _outside(tmp_path)
    src_lic = tmp_path / "source" / "LICENSE"
    res, src, out = _build(ripple_trees, tmp_path,
                           swaps=[{"at": "load_recipe", "path": str(src_lic), "target": str(outside)}])
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    assert src_lic.is_symlink(), "the swap did not happen"
    assert (out / "LICENSE").read_bytes() == (ripple_trees / "source" / "LICENSE").read_bytes()
    assert not (out / "LICENSE").is_symlink() and str(outside) not in res["opened"]


def test_build_replaces_a_link_the_exporter_left_at_a_sidecar_name(ripple_trees: Path, tmp_path: Path) -> None:
    outside = _outside(tmp_path)
    res, src, out = _build(ripple_trees, tmp_path, export_links={"LICENSE": str(outside)})
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    assert (out / "LICENSE").is_file() and not (out / "LICENSE").is_symlink()
    assert (out / "LICENSE").read_bytes() == (src / "LICENSE").read_bytes()
    assert outside.read_text() == "OUTSIDE THE SOURCE DIRECTORY\n" and str(outside) not in res["opened"]


@pytest.mark.parametrize("name", ["config.json", "hf_quant_config.json"])
def test_build_refuses_an_export_config_linked_outside(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    outside = tmp_path / f"outside-{name}"
    outside.write_text(json.dumps({"quantization_config": {"quant_algo": "MIXED_PRECISION"},
                                   "quantization": {"quant_algo": "MIXED_PRECISION"}}))
    res, src, out = _build(ripple_trees, tmp_path, export_links={name: str(outside)})
    _not_built(res, out, before_export=False)
    assert str(outside) not in res["opened"], res["opened"]


@pytest.mark.parametrize("name", ["NOTICE", ".notice", "extra/.notice", ".cache/download/notice"])
def test_build_refuses_a_link_left_anywhere_in_the_export(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    """No link may remain in the export, dot-named entries included, at the top level and in subdirectories."""
    outside = _outside(tmp_path)
    res, src, out = _build(ripple_trees, tmp_path, export_links={name: str(outside)})
    _not_built(res, out, before_export=False)
    assert f"/{name}: cannot open it in its directory" in res["out"], res["out"]


@pytest.mark.parametrize("cfg", [{"dtype": "bfloat16"}, {"quantization_config": {}, "dtype": "bfloat16"},
                                 {"quantization_config": None}])
def test_build_refuses_an_export_config_without_a_quantization_config(ripple_trees: Path, tmp_path: Path,
                                                                     cfg: dict) -> None:
    """The exporter's config.json must carry a quantization_config: without one the build stops before publishing."""
    res, src, out = _build(ripple_trees, tmp_path, export_files={"config.json": json.dumps(cfg)})
    _not_built(res, out, before_export=False)
    assert "the export's config.json has no quantization_config" in res["out"], res["out"]


def test_control_cpu_smoke_writes_the_pinned_sidecars_into_its_tiny_source(ripple_trees: Path,
                                                                           tmp_path: Path) -> None:
    res, src, out = _build(ripple_trees, tmp_path, argv=("--cpu-smoke",))
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    tiny = next((tmp_path / "work").glob("A-4-*/tiny-src"))
    for f in lay.COPIED_FILES:
        assert (tiny / f).read_bytes() == (src / f).read_bytes() and not (tiny / f).is_symlink()


def test_build_cpu_smoke_refuses_a_sidecar_linked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    outside = _outside(tmp_path)
    res, src, out = _build(ripple_trees, tmp_path, lambda s: _link_into(s, "LICENSE", outside),
                           pins={"LICENSE": _sha(outside.read_bytes())}, argv=("--cpu-smoke",))
    _not_built(res, out)
    assert not list((tmp_path / "work").glob("A-4-*/tiny-src/LICENSE"))


def test_build_cpu_smoke_reads_its_tiny_config_confined(ripple_trees: Path, tmp_path: Path) -> None:
    """The CPU rehearsal's tiny checkpoint is written once and read again by later runs: a config.json linked outside
    there is refused (the swap happens before the build reaches the tiny checkpoint)."""
    work = tmp_path / "work"
    calib = _calib(tmp_path)
    run_key = hashlib.sha256(json.dumps(["A", lay.RECIPES["A"][1], _sha(calib.read_bytes()), 4, True]).encode())
    tiny = work / f"A-4-{run_key.hexdigest()[:12]}" / "tiny-src"
    tiny.mkdir(parents=True)
    (tiny / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    (tiny / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    outside = tmp_path / "outside-tiny-config.json"
    outside.write_text(json.dumps({"num_hidden_layers": 2}))
    res, src, out = _build(ripple_trees, tmp_path, argv=("--cpu-smoke",),
                           swaps=[{"at": "load_recipe", "path": str(tiny / "config.json"), "target": str(outside)}])
    _not_built(res, out)
    assert str(outside) not in res["opened"], res["opened"]


def _teacher(trees: Path, tmp_path: Path, mutate=None, pins: dict | None = None) -> tuple[dict, Path]:
    src, _ = _fresh(trees, tmp_path)
    run_pins = {f: _sha((src / f).read_bytes()) for f in PINNED_SOURCE_FILES}
    run_pins.update(pins or {})
    names = _tensor_names(src)
    os.rename(src / "model.safetensors", src / "model-00001-of-00001.safetensors")
    _index(src, {n: "model-00001-of-00001.safetensors" for n in names})
    if mutate is not None:
        mutate(src)
    res = _run({"mode": "teacher", "script": str(TEACHER), "src": str(src), "tensor": names[0], "pins": run_pins},
               tmp_path)
    return res, src


def test_teacher_reads_its_shards_through_bound_descriptors(ripple_trees: Path, tmp_path: Path) -> None:
    res, src = _teacher(ripple_trees, tmp_path)
    assert res["exit"] == 0 and "TEACHER-STREAMER-OK" in res["out"], res["out"]
    assert res["safe_open"] and all(re.fullmatch(r"/(proc/self|dev)/fd/\d+", p) for p in res["safe_open"]), res
    assert not [p for p in res["opened"] if p.endswith((".safetensors", ".json"))], res["opened"]


def test_teacher_refuses_an_index_pointing_outside(ripple_trees: Path, tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"

    def escape(src: Path) -> None:
        shutil.copytree(src, decoy)
        names = list(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"])
        (src / "model-00001-of-00001.safetensors").unlink()
        _index(src, {n: "../decoy/model-00001-of-00001.safetensors" for n in names})

    res, src = _teacher(ripple_trees, tmp_path, escape)
    assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    assert not [p for p in res["safe_open"] + res["opened"] if str(decoy) in p]


def test_teacher_refuses_a_shard_linked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"

    def link(src: Path) -> None:
        os.rename(src / "model-00001-of-00001.safetensors", outside)
        os.symlink(outside, src / "model-00001-of-00001.safetensors")

    res, _ = _teacher(ripple_trees, tmp_path, link)
    assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    assert str(outside) not in res["safe_open"] + res["opened"]


@pytest.mark.parametrize("name", ["tokenizer_config.json", "config.json", "modeling_ricursive.py"])
def test_teacher_refuses_a_source_file_that_misses_its_pin(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    res, _ = _teacher(ripple_trees, tmp_path, pins={name: "0" * 64})
    assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    assert name in res["out"], res["out"]


def test_teacher_refuses_a_tokenizer_file_linked_outside(ripple_trees: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside-tokenizer_config.json"

    def link(src: Path) -> None:
        shutil.copyfile(src / "tokenizer_config.json", outside)
        _link_into(src, "tokenizer_config.json", outside)

    res, _ = _teacher(ripple_trees, tmp_path, link)
    assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    assert str(outside) not in res["opened"]


@pytest.mark.parametrize("name", [".cache-note", "inference/.cache-note"])
def test_teacher_refuses_a_dot_named_link_in_the_source(ripple_trees: Path, tmp_path: Path, name: str) -> None:
    outside = _outside(tmp_path, "outside-note")

    def link(src: Path) -> None:
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(outside, src / name)

    res, _ = _teacher(ripple_trees, tmp_path, link)
    assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    assert f"/{name}: cannot open it in its directory" in res["out"], res["out"]


def _run_script(script: str, trees: Path, tmp_path: Path, mutate=None, pins: dict | None = None) -> dict:
    """The build or the teacher on a fresh synthetic source; either way it must refuse (nothing built, no tensor read)."""
    if script == "build":
        res, src, out = _build(trees, tmp_path, mutate, pins=pins)
        _not_built(res, out)
    else:
        res, _ = _teacher(trees, tmp_path, mutate, pins=pins)
        assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    return res


@pytest.mark.parametrize("position", [0, 63])
@pytest.mark.parametrize("script", ["build", "teacher"])
def test_a_pin_one_digit_off_is_refused(ripple_trees: Path, tmp_path: Path, script: str, position: int) -> None:
    """A pin that differs from the file's sha256 only in its first or its last hex digit: the whole digest is
    compared."""
    good = _sha((ripple_trees / "source" / "LICENSE").read_bytes())
    bad = good[:position] + ("1" if good[position] == "0" else "0") + good[position + 1:]
    assert len(bad) == 64 and sum(x != y for x, y in zip(good, bad)) == 1
    res = _run_script(script, ripple_trees, tmp_path, pins={"LICENSE": bad})
    assert f"/LICENSE: sha256 {good} is not the pin {bad}" in res["out"], res["out"]


@pytest.mark.parametrize("script", ["build", "teacher"])
def test_a_pinned_source_file_that_is_missing_is_refused(ripple_trees: Path, tmp_path: Path, script: str) -> None:
    """A pinned file removed from the source is refused by name, never skipped."""
    def drop(src: Path) -> None:
        (src / "tokenizer_config.json").unlink()

    res = _run_script(script, ripple_trees, tmp_path, drop)
    assert "/tokenizer_config.json: missing" in res["out"], res["out"]


# -- confinement helpers: one test per rule ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", ".", "..", "../x.safetensors", "a/b.safetensors", "/abs.safetensors",
                                 "a\\b.safetensors", "x\x00.safetensors", 7, None])
def test_shard_name_accepts_only_one_plain_file_name(bad: object) -> None:
    cf = _cf()
    with pytest.raises(cf.ShardError):
        cf.shard_name(bad)
    for good in ("model-00001-of-00002.safetensors", "...", "a..b"):
        assert cf.shard_name(good) == good


def test_shard_name_keeps_the_framework_rule() -> None:
    """eval/checkpoint_files.py imports nothing from model_forge (it runs in the GPU images), so its rule is checked
    against model_forge.pipeline.shard_name() name by name."""
    cf = _cf()
    from model_forge.pipeline import CheckpointPathError
    from model_forge.pipeline import shard_name as framework_shard_name

    for name in ["", ".", "..", "...", "a/b", "a\\b", "x\x00", "/x", "ok.safetensors", "a..b", 3, None]:
        try:
            framework_shard_name(name)
            want = True
        except CheckpointPathError:
            want = False
        try:
            cf.shard_name(name)
            got = True
        except cf.ShardError:
            got = False
        assert got == want, name


@pytest.mark.parametrize("body", ['{"metadata": {}}', '{"weight_map": ["model.safetensors"]}', "[]", "{",
                                  '{"weight_map": {"x": "model.bin"}}'])
def test_an_index_without_a_shard_mapping_is_refused(tmp_path: Path, body: str) -> None:
    """An index with no weight_map mapping, or one naming a file that is not a shard of the directory, is refused
    with its own message (never read as an empty index)."""
    cf = _cf()
    d = tmp_path / "ck"
    d.mkdir()
    (d / "model.safetensors.index.json").write_text(body)
    (d / "model.bin").write_bytes(b"x")
    with pytest.raises(cf.ShardError, match="weight_map|unreadable|not exactly the indexed shards"):
        cf.Shards(d)
    assert cf.weight_map(b'{"weight_map": {}}', "x") == {}
    with pytest.raises(cf.ShardError, match="no weight_map mapping"):
        cf.weight_map(b'{"metadata": {}}', "x")


def test_weight_map_checks_every_shard_name() -> None:
    """Every entry's shard name is checked, not only the first."""
    cf = _cf()
    raw = json.dumps({"weight_map": {"a": "model.safetensors", "b": "../decoy/model.safetensors"}}).encode()
    with pytest.raises(cf.ShardError, match=re.escape("b maps to '../decoy/model.safetensors', not a file name")):
        cf.weight_map(raw, "x")


def test_an_indexed_shard_that_is_missing_is_refused_by_name(tmp_path: Path) -> None:
    """A shard the index names but the directory lacks is refused when the shards are opened, by its name."""
    cf = _cf()
    d = tmp_path / "ck"
    d.mkdir()
    header = b'{"__metadata__":{}}     '
    (d / "model-00001-of-00002.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    _index(d, {"t1": "model-00001-of-00002.safetensors", "t2": "model-00002-of-00002.safetensors"})
    with pytest.raises(cf.ShardError, match=re.escape("missing ['model-00002-of-00002.safetensors']")):
        cf.Shards(d)


def test_a_directory_at_a_file_name_is_refused(tmp_path: Path) -> None:
    cf = _cf()
    d = tmp_path / "d"
    (d / "config.json").mkdir(parents=True)
    with cf.Directory(d) as dd, pytest.raises(cf.ShardError, match="config.json: not a regular file"):
        dd.read_optional("config.json")


@pytest.mark.parametrize("header", [b"[]", b'"x"', b"1"])
def test_a_safetensors_header_that_is_not_an_object_is_refused(tmp_path: Path, header: bytes) -> None:
    cf = _cf()
    d = tmp_path / "ck"
    d.mkdir()
    h = header + b" " * ((8 - len(header) % 8) % 8)
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h)
    with cf.Shards(d) as s, pytest.raises(cf.ShardError, match="not a JSON object"):
        s.tensors()


def test_a_refused_file_never_reads_as_missing(tmp_path: Path) -> None:
    """read_optional returns None only for an absent name: a link there is refused, not taken for a missing file (the
    validator skips a sidecar the source lacks, so a linked source sidecar must not look absent)."""
    cf = _cf()
    d = tmp_path / "d"
    d.mkdir()
    assert cf.Directory(d).read_optional("absent") is None
    os.symlink(tmp_path / "elsewhere", d / "dangling")
    (tmp_path / "target").write_text("t")
    os.symlink(tmp_path / "target", d / "linked")
    for name in ("dangling", "linked"):
        with pytest.raises(cf.ShardError, match="links are refused"):
            cf.Directory(d).read_optional(name)


def test_write_files_replaces_a_link_and_never_writes_through_it(tmp_path: Path) -> None:
    cf = _cf()
    d = tmp_path / "d"
    d.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    os.symlink(outside, d / "LICENSE")
    assert cf.write_files(d, {"LICENSE": b"pinned"}) == ["LICENSE"]
    assert (d / "LICENSE").read_bytes() == b"pinned" and not (d / "LICENSE").is_symlink()
    assert outside.read_text() == "keep" and sorted(os.listdir(d)) == ["LICENSE"]


def test_refuse_links_walks_subdirectories(tmp_path: Path) -> None:
    """Every entry is walked, dot-named ones included (real checkpoints carry .gitattributes and a .cache directory):
    dot-named regular files count, and a link is refused at the top level, in a subdirectory and under a dot-named
    directory."""
    cf = _cf()
    d = tmp_path / "tree"
    (d / "inference" / "server").mkdir(parents=True)
    (d / ".cache" / "download").mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (d / "inference" / "server" / "b.py").write_text("b")
    (d / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (d / ".cache" / "download" / "c.lock").write_text("")
    (d / "inference" / ".hidden-note").write_text("n")
    assert cf.refuse_links(d) == 5
    outside = tmp_path / "elsewhere"
    outside.write_text("outside")
    for where in ("inference/server/c.py", ".notice", "inference/.cache-note", ".cache/download/notice"):
        os.symlink(outside, d / where)
        with pytest.raises(cf.ShardError, match=re.escape(f"/{where}: cannot open it in its directory")):
            cf.refuse_links(d)
        (d / where).unlink()
    assert cf.refuse_links(d) == 5


def _in_subprocess(code: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """Run ``code`` with eval/ on sys.path in its own process: an open that blocks fails the test instead of hanging
    the suite."""
    prog = f"import sys; sys.path.insert(0, {str(FAMILY / 'eval')!r}); " + code
    try:
        return subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"blocked: {code}")


FIFO_CALLS = {
    "refuse_links": ("sub/pipe", "cf.refuse_links(D)"),
    "Directory.read_optional": ("pipe", "cf.Directory(D).read_optional('pipe')"),
    "read_file": ("pipe", "cf.read_file(D, 'pipe')"),
}


@pytest.mark.parametrize("helper", sorted(FIFO_CALLS))
def test_a_fifo_is_refused_without_blocking(tmp_path: Path, helper: str) -> None:
    """Opening a FIFO for reading without O_NONBLOCK waits for a writer: every helper that opens a source file refuses
    it at once instead."""
    _cf()
    where, call = FIFO_CALLS[helper]
    d = tmp_path / "tree"
    (d / "sub").mkdir(parents=True)
    os.mkfifo(d / where)
    body = "\n".join([f"import checkpoint_files as cf; D = {str(d)!r}", "try:", f"    {call}",
                      "except cf.ShardError as err:", "    print('REFUSED', err)"])
    r = _in_subprocess(body)
    assert "REFUSED" in r.stdout and "pipe" in r.stdout, r.stdout + r.stderr


# -- small files read whole are capped (review round 2) ----------------------------------------------------------------------
SMALL_FILE_CAP = 64 * 2**20


def test_small_file_reads_are_capped_at_64_mib() -> None:
    """Configs, sidecars and the index are read whole, so their size is capped: 64 MiB, far above the largest file
    the build reads whole (tokenizer.json, 7,137,177 bytes)."""
    cf = _cf()
    assert getattr(cf, "MAX_SMALL_FILE_BYTES", None) == SMALL_FILE_CAP and 8 * 7_137_177 < SMALL_FILE_CAP


SMALL_FILE_READS = {
    "Directory.read_optional": lambda cf, d: cf.Directory(d).read_optional("config.json"),
    "Directory.read": lambda cf, d: cf.Directory(d).read("config.json"),
    "read_file": lambda cf, d: cf.read_file(d, "config.json"),
    "read_source": lambda cf, d: cf.read_source(d, {"config.json": "0" * 64}),
    "Shards-index": lambda cf, d: cf.Shards(d),
}


@pytest.mark.parametrize("call", sorted(SMALL_FILE_READS))
def test_a_small_file_over_the_cap_is_refused_before_it_is_read(tmp_path: Path, call: str) -> None:
    """One byte over the cap (a sparse file: no disk blocks) is refused from its size, before any of it is read."""
    cf = _cf()
    d = tmp_path / "ck"
    d.mkdir()
    name = "model.safetensors.index.json" if call.startswith("Shards") else "config.json"
    with open(d / name, "wb") as f:
        f.truncate(SMALL_FILE_CAP + 1)
    tracemalloc.start()
    try:
        with pytest.raises(cf.ShardError, match=f"{SMALL_FILE_CAP + 1:,} bytes exceeds the {SMALL_FILE_CAP:,}-byte"):
            SMALL_FILE_READS[call](cf, d)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 8 * 2**20, f"allocated {peak:,} bytes before refusing"


def test_a_small_file_at_the_cap_is_read_and_one_byte_more_is_refused(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is inclusive; each read takes it from the module when it runs (lowered here to 100 bytes)."""
    cf = _cf()
    monkeypatch.setattr(cf, "MAX_SMALL_FILE_BYTES", 100)
    d = tmp_path / "d"
    d.mkdir()
    (d / "at").write_bytes(b"x" * 100)
    (d / "over").write_bytes(b"x" * 101)
    with cf.Directory(d) as dd:
        assert dd.read("at") == b"x" * 100
        with pytest.raises(cf.ShardError, match="101 bytes exceeds the 100-byte limit"):
            dd.read("over")


SMALL_FILES_READ = [  # (side, name): every small file the validator and the size report read whole
    ("source", "config.json"), ("export", "config.json"), ("export", "hf_quant_config.json"),
    ("export", "generation_config.json"), ("source", "LICENSE"), ("export", "LICENSE"),
    ("source", "model.safetensors.index.json"), ("export", "model.safetensors.index.json")]


@pytest.mark.parametrize(("side", "name"), SMALL_FILES_READ, ids=[f"{s}-{n}" for s, n in SMALL_FILES_READ])
def test_validator_refuses_each_small_file_over_the_cap(ripple_trees: Path, tmp_path: Path, side: str,
                                                        name: str) -> None:
    """Each read the validator takes whole goes through the cap (a sparse file one byte over it); the size report
    reads the index the same way."""
    src, exp = _fresh(ripple_trees, tmp_path)
    d = src if side == "source" else exp
    if name == "model.safetensors.index.json":
        _index(d, {n: "model.safetensors" for n in _tensor_names(d)})
    (d / name).touch()
    os.truncate(d / name, SMALL_FILE_CAP + 1)
    res = _validate(exp, src)
    _refused(res, f"/{name}: {SMALL_FILE_CAP + 1:,} bytes exceeds the {SMALL_FILE_CAP:,}-byte limit")
    if name == "model.safetensors.index.json":
        assert f"{SMALL_FILE_CAP + 1:,} bytes exceeds the {SMALL_FILE_CAP:,}-byte limit" in _sizes(d)


@pytest.mark.parametrize("where", ["source/LICENSE", "source/config.json", "export/config.json",
                                   "export/hf_quant_config.json"])
def test_build_refuses_a_small_file_over_the_cap(ripple_trees: Path, tmp_path: Path, where: str) -> None:
    """The build's own whole-file reads (the pinned source files, the exporter's config.json and hf_quant_config.json)
    go through the cap; nothing is published."""
    side, name = where.split("/")
    big = SMALL_FILE_CAP + 1
    if side == "source":
        res, src, out = _build(ripple_trees, tmp_path, lambda s: os.truncate(s / name, big))
        _not_built(res, out)
    else:
        res, src, out = _build(ripple_trees, tmp_path, export_sparse={name: big})
        _not_built(res, out, before_export=False)
    assert f"/{name}: {big:,} bytes exceeds the {SMALL_FILE_CAP:,}-byte limit" in res["out"], res["out"]


def test_teacher_refuses_a_pinned_file_over_the_cap(ripple_trees: Path, tmp_path: Path) -> None:
    res, _ = _teacher(ripple_trees, tmp_path, lambda s: os.truncate(s / "tokenizer_config.json", SMALL_FILE_CAP + 1))
    assert res["exit"] != 0 and "TEACHER-STREAMER-OK" not in res["out"], res["out"]
    assert f"/tokenizer_config.json: {SMALL_FILE_CAP + 1:,} bytes exceeds the" in res["out"], res["out"]


def test_build_cpu_smoke_refuses_a_tiny_config_over_the_cap(ripple_trees: Path, tmp_path: Path) -> None:
    """The CPU rehearsal reads its tiny checkpoint's config.json (left by an earlier run) through the cap too."""
    calib = _calib(tmp_path)
    run_key = hashlib.sha256(json.dumps(["A", lay.RECIPES["A"][1], _sha(calib.read_bytes()), 4, True]).encode())
    tiny = tmp_path / "work" / f"A-4-{run_key.hexdigest()[:12]}" / "tiny-src"
    tiny.mkdir(parents=True)
    (tiny / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    (tiny / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    os.truncate(tiny / "config.json", SMALL_FILE_CAP + 1)
    res, src, out = _build(ripple_trees, tmp_path, argv=("--cpu-smoke",))
    _not_built(res, out)
    assert "tiny checkpoint: " in res["out"], res["out"]
    assert f"/config.json: {SMALL_FILE_CAP + 1:,} bytes exceeds the {SMALL_FILE_CAP:,}-byte limit" in res["out"], res["out"]


def test_validator_refuses_a_sparse_small_file_without_allocating_it(ripple_trees: Path, tmp_path: Path) -> None:
    """Review round 2's probe: a sparse 1.5 GB generation_config.json (a JSON prefix, then no disk blocks) made the
    validator allocate about 2.8 GiB before it failed on the JSON. It is refused from its size now."""
    src, exp = _fresh(ripple_trees, tmp_path)
    p = exp / "generation_config.json"
    p.write_bytes(b'{"bos_token_id": 100257}')
    with open(p, "r+b") as f:
        f.truncate(1_500_000_000)
    tracemalloc.start()
    try:
        res = _validate(exp, src)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    _refused(res, f"1,500,000,000 bytes exceeds the {SMALL_FILE_CAP:,}-byte limit")
    assert peak < 8 * 2**20, f"allocated {peak:,} bytes before refusing"


# -- GPQA truncation wording (finding at the card's GPQA row) ---------------------------------------------------------------------
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


# -- provenance (finding at build_ledger.py:32) -----------------------------------------------------------------------------------
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


def test_recipe_edit_manifest_evidence_is_recomputable_from_its_cited_file() -> None:
    ledger = json.loads((FAMILY / "results" / "publication-readiness-ledger.json").read_text())
    gate_ = next(g for p in ledger["products"] for g in p["gates"] if g["id"] == "recipe_edit_manifest")
    cited = (ROOT / gate_["evidence_path"]).read_text()
    hexes = re.findall(r"\b([0-9a-f]{8})\.\.\.", gate_["evidence"])
    assert len(hexes) == 3, gate_["evidence"]
    for h in hexes:
        assert h in cited, h


# ---- review round 1: the gate's surviving mutants (suggestion) ----------------------------------------------------------------
def test_gate_artifact_bar_fails_the_candidate_when_validation_did_not_pass(tmp_path: Path) -> None:
    """The pre-registered fail-closed artifact check: a FAIL verdict, or no validation file, blocks the bars."""
    d = _results(tmp_path, 145, 0.855, 37.0, 0.25)
    assert gate.evaluate(d, ["vendor", "A"], tmp_path / "out")["candidates"]["A"]["passes_bars"]
    (tmp_path / "out" / "A.validation.json").write_text(json.dumps({"verdict": "FAIL"}))
    a = gate.evaluate(d, ["vendor", "A"], tmp_path / "out")["candidates"]["A"]
    assert a["artifact_pass"] is False and not a["passes_bars"] and not a["eligible"]
    (tmp_path / "out" / "A.validation.json").unlink()
    a = gate.evaluate(d, ["vendor", "A"], tmp_path / "out")["candidates"]["A"]
    assert a["artifact_pass"] is False and not a["passes_bars"]


def test_gate_quality_win_needs_ten_percent_lower_kl(tmp_path: Path) -> None:
    """KL_RATIO 0.90: a candidate 5% below the vendor's KL, with the CI below zero, is no quality win."""
    v = {"mean_kl": 1.0, "per_prompt": [(1.0, 1)] * 40}
    c = {"mean_kl": 0.95, "per_prompt": [(0.95, 1)] * 40}
    k = gate.kl_compare(c, v)
    assert k["bootstrap95_diff"][1] < 0 and not k["quality_win"]
    c2 = {"mean_kl": 0.85, "per_prompt": [(0.85, 1)] * 40}
    assert gate.kl_compare(c2, v)["quality_win"]


def test_gate_quality_win_needs_the_ci_below_zero(tmp_path: Path) -> None:
    """A candidate whose mean KL is 20% below the vendor's but whose paired CI crosses zero is no quality win."""
    v = {"mean_kl": 1.0, "per_prompt": [(1.0, 1)] * 40}
    c = {"mean_kl": 0.8, "per_prompt": [(0.0, 1)] * 20 + [(1.8, 1)] * 20}
    k = gate.kl_compare(c, v)
    assert k["ratio"] <= gate.KL_RATIO and k["bootstrap95_diff"][1] >= 0 and not k["quality_win"], k


def test_gate_gpqa_refuses_cells_that_do_not_pair_on_all_198_items(tmp_path: Path) -> None:
    d = _results(tmp_path, 145, 0.855, 37.0, 0.25)
    c, v = gate.gpqa_cell(d, "A"), gate.gpqa_cell(d, "vendor")
    short = dict(c, ids=c["ids"][:-1], per_item={i: c["per_item"][i] for i in c["ids"][:-1]})
    with pytest.raises(ValueError, match="pair on all 198"):
        gate.gpqa_compare(short, v)


def test_gate_a_missing_bar_input_never_passes(tmp_path: Path, monkeypatch) -> None:
    """With the proxy result missing, the bars are incomplete: passes_bars is False even though no bar failed."""
    d = _results(tmp_path, 145, 0.855, 37.0, 0.25)
    (d / "proxy-A.json").unlink()
    a = gate.evaluate(d, ["vendor", "A"], tmp_path / "out")["candidates"]["A"]
    assert a["missing"] and not a["bars_complete"] and a["passes_bars"] is False and not a["eligible"]


# -- no checkpoint file is opened by its path ---------------------------------------------------------------------------
def test_validator_and_sizes_never_open_a_checkpoint_file_by_path(ripple_trees: Path, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file under the two directories is opened through a directory descriptor (os.open with dir_fd), never by a
    path: a by-path open follows whatever link sits at that path when it runs."""
    src, exp = _fresh(ripple_trees, tmp_path)
    (exp / "generation_config.json").write_text(json.dumps({k: vst.source_config()[k] for k in
                                                            ("bos_token_id", "eos_token_id", "pad_token_id")}))
    by_path: list[str] = []
    real_open, real_os_open = open, os.open
    roots = (str(src.resolve()), str(exp.resolve()))

    def watch(path: object) -> None:
        if isinstance(path, (str, os.PathLike)):
            full = os.path.realpath(os.fspath(path))
            if full.startswith(roots) and not os.path.isdir(full):
                by_path.append(full)

    def open_hook(file, *a, **k):  # noqa: ANN001, ANN002, ANN003 - mirrors open()
        watch(file)
        return real_open(file, *a, **k)

    def os_open_hook(path, flags, mode=0o777, *, dir_fd=None):  # noqa: ANN001 - mirrors os.open()
        if dir_fd is None:
            watch(path)
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("builtins.open", open_hook)
    monkeypatch.setattr(os, "open", os_open_hook)
    res = ve.validate(exp, src, "A")
    sizes_mod.sizes(exp)
    monkeypatch.undo()
    assert res["verdict"] == "PASS", res
    assert by_path == [], sorted(set(by_path))


def test_build_never_opens_a_source_file_by_path(ripple_trees: Path, tmp_path: Path) -> None:
    """The build reads its source's small files once, confined; the GPU path never reopens one by name (transformers'
    and the exporter's own reads are stand-ins here, so every recorded open is the build's)."""
    res, src, out = _build(ripple_trees, tmp_path)
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    root = str(src.resolve())
    assert not [p for p in res["opened"] if p.startswith(root)], res["opened"]
    export_root = str(out.resolve()) + ".partial"
    assert not [p for p in res["opened"] if p.startswith(export_root)], res["opened"]


# -- the index and the headers must agree; reads see the file they opened ----------------------------------------------------
def test_an_index_and_its_shards_must_name_the_same_tensors(ripple_trees: Path, tmp_path: Path) -> None:
    src, exp = _fresh(ripple_trees, tmp_path)
    names = _tensor_names(exp)
    _index(exp, {**{n: "model.safetensors" for n in names}, "model.layers.9.mlp.up_proj.weight": "model.safetensors"})
    _refused(_validate(exp, src), "does not match the shard headers")
    _index(exp, {n: "model.safetensors" for n in names[1:]})
    res = _validate(exp, src)
    assert res["verdict"] != "PASS", res


def test_a_shard_truncated_after_it_was_opened_is_a_short_read(ripple_trees: Path, tmp_path: Path) -> None:
    """The same file, cut short after the checks: a read past its new end is refused, never returned short."""
    cf = _cf()
    src, exp = _fresh(ripple_trees, tmp_path)
    ck = ve.Checkpoint(exp)
    try:
        name = max(ck.tensors, key=lambda n: ck.tensors[n][1])  # the last tensor in the file
        os.truncate(exp / "model.safetensors", ck.tensors[name][1] + 1)
        with pytest.raises(cf.ShardError, match="short read"):
            ck.raw(name)
    finally:
        ck.close()


def test_checkpoint_reads_the_directory_it_was_given(ripple_trees: Path, tmp_path: Path) -> None:
    """Checkpoint(Directory) reads the opened directory even after another directory takes its path."""
    cf = _cf()
    src, exp = _fresh(ripple_trees, tmp_path)
    d = cf.Directory(exp)
    try:
        os.rename(exp, tmp_path / "opened")
        shutil.copytree(src, exp)  # a different checkpoint at the old path
        ck = ve.Checkpoint(d)
        try:
            assert sorted(ck.tensors) == _tensor_names(tmp_path / "opened")
        finally:
            ck.close()
    finally:
        d.close()


@pytest.mark.parametrize("side", ["export", "source"])
def test_validator_certifies_the_directories_it_opened(ripple_trees: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch, side: str) -> None:
    """The directory under test has a fault; right after the validator opened it, a clean directory takes its path.
    The verdict must be about the opened directory's bytes (FAIL), not whatever sits at the path later."""
    src, exp = _fresh(ripple_trees, tmp_path)
    target = exp if side == "export" else src
    shutil.copytree(target, tmp_path / "clean")
    if side == "export":
        vst.edit_tensors(exp, lambda t: vst.set_byte(t, "model.layers.1.mlp.up_proj.weight_scale", 5, 0x7F))
        want = "NaN/negative/zero"
    else:
        vst.edit_tensors(src, lambda t: vst.set_byte(t, "model.layers.1.input_layernorm.weight", 0,
                                                     t["model.layers.1.input_layernorm.weight"][2][0] ^ 1))
        want = "BF16 tensors differ"
    real = lay.runtime_plan

    def swap_then_plan(*a, **k):  # runs once the configs are read, before any shard is opened
        if (tmp_path / "clean").exists():
            os.rename(target, tmp_path / "opened")
            os.rename(tmp_path / "clean", target)
        return real(*a, **k)

    monkeypatch.setattr(lay, "runtime_plan", swap_then_plan)
    _refused(_validate(exp, src), want)
    assert not (tmp_path / "clean").exists(), "the swap did not happen"


def test_build_cpu_smoke_writes_the_checked_bytes_when_a_sidecar_is_swapped_later(ripple_trees: Path,
                                                                                  tmp_path: Path) -> None:
    outside = _outside(tmp_path)
    src_lic = tmp_path / "source" / "LICENSE"
    res, src, out = _build(ripple_trees, tmp_path, argv=("--cpu-smoke",),
                           swaps=[{"at": "load_recipe", "path": str(src_lic), "target": str(outside)}])
    assert res["exit"] == 0 and "QUANT-OK" in res["out"], res["out"]
    assert src_lic.is_symlink(), "the swap did not happen"
    tiny = next((tmp_path / "work").glob("A-4-*/tiny-src"))
    assert (tiny / "LICENSE").read_bytes() == (ripple_trees / "source" / "LICENSE").read_bytes()
    assert not (tiny / "LICENSE").is_symlink() and str(outside) not in res["opened"]


# -- the record's builders and checkers, re-run ---------------------------------------------------------------------------
def _record_copy(tmp_path: Path) -> Path:
    """The files the evidence builders and the card checker read, copied into an empty tree."""
    t = tmp_path / "record"
    shutil.copytree(FAMILY / "results", t / "models" / "ripple-65b" / "results")
    shutil.copytree(FAMILY / "model-card", t / "models" / "ripple-65b" / "model-card")
    shutil.copyfile(FAMILY / "README.md", t / "models" / "ripple-65b" / "README.md")
    shutil.copytree(ROOT / "recipes" / "ripple-65b", t / "recipes" / "ripple-65b")
    staged = Path("publication") / "huggingface" / lay.PRODUCT_IDS["A"] / "README.md"
    (t / staged).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / staged, t / staged)
    return t


def _script(name: str, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(FAMILY / "eval" / name), *args], cwd=cwd, capture_output=True,
                          text=True, timeout=120, check=False)


def test_gate_summary_and_ledger_regenerate_byte_identically(tmp_path: Path) -> None:
    """The committed gate-summary.json and ledger are exactly what their builders write from the committed inputs."""
    t = _record_copy(tmp_path)
    res = t / "models" / "ripple-65b" / "results"
    for name, out in (("build_gate_summary.py", "gate-summary.json"),
                      ("build_ledger.py", "publication-readiness-ledger.json")):
        (res / out).unlink()
        r = _script(name, t)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        assert (res / out).read_bytes() == (FAMILY / "results" / out).read_bytes(), out


def test_card_checker_requires_the_logged_serve_flags_and_the_cap_wording(tmp_path: Path) -> None:
    t = _record_copy(tmp_path)
    r = _script("check_card_numbers.py", t)
    assert r.returncode == 0 and re.search(r"^(\d+)/\1 card and README numbers match", r.stdout, re.M), r.stdout
    cards = [t / "models" / "ripple-65b" / "model-card" / "base-nvfp4.md",
             t / "publication" / "huggingface" / lay.PRODUCT_IDS["A"] / "README.md"]
    edits = [("--max-num-seqs 32", "--max-num-seqs 64"), ("--gpu-memory-utilization 0.92", "--gpu-memory-utilization 0.9"),
             ("--max-model-len 131072", "--max-model-len 65536"), ("--tool-call-parser ripple", "--tool-call-parser x"),
             ("--reasoning-parser ripple", "--reasoning-parser x"), ("--enable-auto-tool-choice", "--auto-tool"),
             ('"num_speculative_tokens":7', '"num_speculative_tokens":5'),
             ("| Items without an answer at the 65,536-token cap | 42 and 48 of A's, 42 and 38 of the vendor's (59 and 57 "
              "/ 50 and 49 responses hit the cap) |",
              "| GPQA truncations at 65,536 tokens | 42 and 48 of A's items, 42 and 38 of the vendor's |"),
             ("count as wrong; a\nresponse that hit the cap with an answer is scored.", "count as wrong.")]
    originals = [c.read_text() for c in cards]
    for old, new in edits:
        for c, text in zip(cards, originals):
            assert text.count(old) == 1, (c.name, old)
            c.write_text(text.replace(old, new))
        r = _script("check_card_numbers.py", t)
        assert r.returncode == 1 and "MISSING in" in r.stdout, (old, r.stdout[-800:])
    for c, text in zip(cards, originals):
        c.write_text(text)


def test_serve_args_reads_exactly_one_startup_line(tmp_path: Path) -> None:
    cw = _load(FAMILY / "eval" / "curate_window.py", "curate_window")
    line = ("(APIServer pid=1) INFO 11-13 [startup.py:96] non-default args: {'model': '/m', 'max_model_len': 131072, "
            "'max_num_seqs': 32, 'gpu_memory_utilization': 0.92, 'reasoning_parser': 'ripple', 'tool_call_parser': "
            "'ripple', 'enable_auto_tool_choice': True, 'served_model_name': ['x']}\n")
    log = tmp_path / "serve.log"
    log.write_text("boot\n" + line + "more\n")
    args = cw.serve_args(log)
    assert args["max_num_seqs"] == 32 and args["gpu_memory_utilization"] == 0.92 and args["speculative_config"] is None
    assert args["enable_auto_tool_choice"] is True and args["log_sha256"] == _sha(log.read_bytes())
    for body in ("boot\n", line + line):
        log.write_text(body)
        with pytest.raises(SystemExit, match="non-default args"):
            cw.serve_args(log)
    runs = json.loads((FAMILY / "results" / "serve-runs.json").read_text())
    for kind in ("quality", "speed"):
        for label, e in runs[kind].items():
            assert set(e["serve_args"]) == {*cw.SERVE_ARG_KEYS, "log_sha256"}, (kind, label)


def test_the_receipt_records_the_card_commit_of_the_staged_card() -> None:
    """The staging repository's card commit on record carries the staged card's exact bytes, has the uploaded weights,
    and the boot smoke covers it; every earlier card commit stays listed with its manifest hash."""
    st = json.loads((FAMILY / "results" / "staging-receipt.json").read_text())
    cc = st["card_commit"]
    staged = ROOT / "publication" / "huggingface" / lay.PRODUCT_IDS["A"] / "README.md"
    assert cc["readme_sha256"] == _sha(staged.read_bytes())
    assert cc["weights_identical_to_uploaded_commit"] is True and cc["parent_weights_commit"] == st["uploaded_commit"]
    assert st["boot_smoke"]["covers_card_commit"] == cc["commit"]
    assert [h["commit"] for h in cc["history"]] == ["891ff96ffb402210c20bef38a93ac2632635dcdd"]
    assert cc["parent_card_commit"] == cc["history"][-1]["commit"]
    ledger = (FAMILY / "results" / "publication-readiness-ledger.json").read_text()
    assert f"card @ {cc['commit'][:12]}" in ledger and "891ff96ffb40" not in ledger


# ---- review round 2: each self-check of the record's builders refuses on its own (suggestion) --------------------------
def _edit_record_json(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data, indent=1) + "\n")


def _bump(d: dict, key: str) -> None:
    d[key] += 1


GATE_SUMMARY_CHECKS = {  # check: (results file, edit)
    "length-is-not-hit_max_tokens": ("gpqa-A-r1.json", lambda d: _bump(d["finish_reasons"], "length")),
    "more-without-an-answer-than-capped": (
        "gpqa-A-r2.json", lambda d: d["accuracy"].__setitem__("truncated_no_answer", d["accuracy"]["hit_max_tokens"] + 1)),
    "gate-json-vendor-hit_max_tokens": ("gate.json", lambda d: _bump(d["vendor"]["gpqa_runs"][0], "hit_max_tokens")),
    "gate-json-vendor-truncated_no_answer": (
        "gate.json", lambda d: _bump(d["vendor"]["gpqa_runs"][1], "truncated_no_answer")),
}


@pytest.mark.parametrize("check", sorted(GATE_SUMMARY_CHECKS))
def test_gate_summary_refuses_cap_counts_that_disagree(tmp_path: Path, check: str) -> None:
    """build_gate_summary.py cross-checks each GPQA run's cap counts (finish_reasons.length, and no more items without
    an answer than capped responses) and the vendor's against gate.json: each refuses alone, and nothing is written."""
    t = _record_copy(tmp_path)
    res = t / "models" / "ripple-65b" / "results"
    name, edit = GATE_SUMMARY_CHECKS[check]
    _edit_record_json(res / name, edit)
    (res / "gate-summary.json").unlink()
    r = _script("build_gate_summary.py", t)
    assert r.returncode != 0 and "AssertionError" in r.stderr, r.stdout[-800:] + r.stderr[-800:]
    assert not (res / "gate-summary.json").exists()


@pytest.mark.parametrize("part", ["revision", "repository"])
def test_ledger_refuses_a_gate_summary_naming_another_source(tmp_path: Path, part: str) -> None:
    """build_ledger.py writes the provenance gates only for the pinned source repository and revision."""
    t = _record_copy(tmp_path)
    res = t / "models" / "ripple-65b" / "results"
    old, new = (lay.SOURCE_REV, "0" * 40) if part == "revision" else (lay.SOURCE_REPO, lay.SOURCE_REPO + "-NVFP4")
    _edit_record_json(res / "gate-summary.json", lambda g: g.__setitem__("source", g["source"].replace(old, new)))
    ledger = res / "publication-readiness-ledger.json"
    before = ledger.read_bytes()
    r = _script("build_ledger.py", t)
    assert r.returncode != 0 and "gate-summary.json source " in r.stderr and new in r.stderr, r.stderr[-800:]
    assert ledger.read_bytes() == before


def _card_readback(t: Path) -> tuple[dict, Path, Path]:
    """A read-back of a new card-only commit that passes every check of eval/record_card_commit.py, for the record copy
    ``t``; with the receipt and the staged card."""
    receipt = t / "models" / "ripple-65b" / "results" / "staging-receipt.json"
    card = t / "publication" / "huggingface" / lay.PRODUCT_IDS["A"] / "README.md"
    prev = json.loads(receipt.read_text())["card_commit"]["commit"]
    head, card_sha = "c" * 40, _sha(card.read_bytes())
    rb = {"head": head, "private": True, "files": 19, "commits": [[head[:12], "card"], [prev[:12], "card"]],
          "lfs_unchanged_vs_previous_card_commit": True, "lfs_unchanged_vs_uploaded_commit": True, "shards": 5,
          "manifest_sha256": "a" * 64, "manifest_entries": 18, "entries_not_matching_hub": [],
          "hub_readme_sha256": card_sha, "staged_card_sha256": card_sha, "readme_equals_staged_card": True}
    return rb, receipt, card


def _record_card(t: Path, rb: dict, card: Path) -> subprocess.CompletedProcess:
    p = t / "readback.json"
    p.write_text(json.dumps(rb))
    return _script("record_card_commit.py", t, str(p), str(card))


def test_record_card_commit_records_a_read_back_that_holds(tmp_path: Path) -> None:
    t = _record_copy(tmp_path)
    rb, receipt, card = _card_readback(t)
    prev = json.loads(receipt.read_text())["card_commit"]
    r = _record_card(t, rb, card)
    assert r.returncode == 0 and "RECORD-CARD-OK" in r.stdout, r.stdout + r.stderr
    st = json.loads(receipt.read_text())
    cc = st["card_commit"]
    assert cc["commit"] == rb["head"] and cc["parent_card_commit"] == prev["commit"]
    assert cc["history"][-1] == {"commit": prev["commit"], "manifest_sha256": prev["manifest_sha256"]}
    assert cc["readme_sha256"] == rb["staged_card_sha256"] and st["boot_smoke"]["covers_card_commit"] == rb["head"]


CARD_COMMIT_CHECKS = [  # (the check's name in record_card_commit.py, the read-back field, a value that fails it)
    ("private", "private", False),
    ("weights unchanged against the previous card commit", "lfs_unchanged_vs_previous_card_commit", False),
    ("weights unchanged against the uploaded commit", "lfs_unchanged_vs_uploaded_commit", False),
    ("5 shards", "shards", 4),
    ("every file equal to its manifest line", "entries_not_matching_hub", ["README.md"]),
    ("hub README equals the staged card", "hub_readme_sha256", "0" * 64),
    ("hub README equals the staged card", "staged_card_sha256", "0" * 64),
    ("parent is the previous card commit", "commits", [["c" * 12, "card"], ["0" * 12, "card"]]),
    ("parent is the previous card commit", "head", "d" * 40),
]


@pytest.mark.parametrize(("check", "field", "bad"), CARD_COMMIT_CHECKS, ids=[c[1] for c in CARD_COMMIT_CHECKS])
def test_record_card_commit_refuses_each_failed_check(tmp_path: Path, check: str, field: str, bad: object) -> None:
    """Every check of eval/record_card_commit.py refuses on its own, and the receipt is left as it was."""
    t = _record_copy(tmp_path)
    rb, receipt, card = _card_readback(t)
    before = receipt.read_bytes()
    rb[field] = bad
    r = _record_card(t, rb, card)
    assert r.returncode != 0 and f"RECORD-CARD-FAILED: ['{check}']" in r.stderr, r.stdout + r.stderr
    assert receipt.read_bytes() == before
