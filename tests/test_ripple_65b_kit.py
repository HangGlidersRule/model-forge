"""Issue #39 (Ripple-65B Preview) kit: the pure-Python pieces the GPU window relies on, tested without the model, the GPU
or the network. Covers the candidate layouts and the vLLM modelopt_mixed resolution re-implementation, the calibration
row rule (rank key, smoke take), the export validator (its self-test on synthetic exports with planted faults), the
gate (bars, bootstrap, selection rule) on synthetic result files, the BFCL AST checker re-implementation, the synthetic
tool-call tasks (reference solutions pass and no-ops fail in the local sandbox backend) and the KL estimator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
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
