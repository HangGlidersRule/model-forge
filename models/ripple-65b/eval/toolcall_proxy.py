#!/usr/bin/env python3
"""Ripple-65B Preview (#39): the agentic tool-call proxy (posted pre-registration, eval set 2).

Two parts, both sent as chat completions with `tools` (no tool_choice, i.e. "auto") through the serve's
`--enable-auto-tool-choice --tool-call-parser ripple` (and `--reasoning-parser ripple`), temperature 0, max_tokens
4096 (a reasoning block counts against it):
  (a) BFCL v3 AST: gorilla-llm/Berkeley-Function-Calling-Leaderboard @ 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda,
      categories simple, multiple, parallel, parallel_multiple, the first 100 items of each by id (natural order of the
      numeric id), scored against possible_answer with a faithful re-implementation of BFCL's own AST checker
      (bfcl_eval/eval_checker/ast_eval/ast_checker.py at gorilla 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8; the
      upstream module cannot be imported offline: it pulls bfcl_eval's model registry). Tools are converted the way
      BFCL's FC handlers do (convert_to_tool with GORILLA_TO_OPENAPI; "." in a name becomes "_", and the checker maps
      expected names the same way, as for every OpenAI-style FC model; BFCL's "Python 3 syntax" description hint).
  (b) 100 synthetic terminal / file-edit tasks from tasks_synthetic.py (seed 1234, committed generator, frozen file):
      a fixture tree is described in the prompt with the agent's two tools (bash, str_replace_editor); the model's
      tool calls are executed in a throwaway sandbox container (`docker run --rm --network none`, no host mounts
      except a per-task copy of the fixture) and scored on the resulting tree hash or the command output.
Reported per part and overall: parse rate (a well-formed tool call to a declared tool, arguments a JSON object, where
one is expected; every item here expects one) and task success (BFCL: the AST checker's verdict; synthetic: the
tree hash or output check). Errors (no response) count as failures in both, never dropped.

Usage:
  toolcall_proxy.py freeze STORE_ROOT OUT_DIR          write proxy_bfcl.jsonl and proxy_synthetic.jsonl (+ sums)
  toolcall_proxy.py run BASE_URL MODEL DATA_DIR OUT_JSON [--label L] [--sandbox-image IMG] [--workers N]
  toolcall_proxy.py score OUT_JSON                     re-score a saved run (no server)
The frozen files' sha256 are pinned below once frozen; `run` refuses other bytes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import http.client
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks_synthetic as syn  # noqa: E402

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
BFCL_PER_CATEGORY = 100
# Upstream files at BFCL_REV (plain git files, not LFS: hashed after the pinned-revision download; freeze checks them).
BFCL_FILE_SHA256 = {
    "BFCL_v3_simple.json": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
    "possible_answer/BFCL_v3_simple.json": "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
    "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "possible_answer/BFCL_v3_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "BFCL_v3_parallel.json": "fad375d68295776efb63e5aa331dd884f8635e7bb50e05908113c74f14281c3c",
    "possible_answer/BFCL_v3_parallel.json": "3af54bf04e5b7640b03a9b6a82440f52474feb24b398f94435fd2b155c7727de",
    "BFCL_v3_parallel_multiple.json": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
    "possible_answer/BFCL_v3_parallel_multiple.json": "df4d2c16238c1465dc04697c273cf3ab4a450075fcd274e3dc86ad5bf0049b75",
}
# The frozen proxy files (freeze output); `run` refuses any other bytes.
FROZEN_SHA256 = {"proxy_bfcl.jsonl": "0affe6d9c61645bd690dc204ef119b637de616a0653ddeb6d69f8fc13e0e6d8a",
                 "proxy_synthetic.jsonl": "110ed227112934486648550e1175b0dda35861bb6222d66fa5a71a4452abf17e"}
MAX_TOKENS = 8192  # Ripple may reason before a call; a cap inside the reasoning would read as no call
# The sandbox runs in the pinned vLLM image (present on the GPU host; a pull over SSH fails there).
SANDBOX_IMAGE_DEFAULT = "vllm/vllm-openai@sha256:902b49de0568ba5a5c951b0243bc06ef8541f46bc3eb80832961fefb22a19034"


# ---- BFCL AST checker (faithful re-implementation, Python only) -----------------------------------------------------
PYTHON_TYPE_MAPPING = {"string": str, "integer": int, "float": float, "boolean": bool, "array": list, "tuple": list,
                       "dict": dict, "any": str}
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


def find_description(func_descriptions, name):
    if isinstance(func_descriptions, list):
        for d in func_descriptions:
            if d["name"] == name:
                return d
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def convert_func_name(function_name: str, underscore_to_dot: bool = True) -> str:
    # BFCL maps "." to "_" for models whose function names cannot contain "." (OpenAI-style FC; tools are sent so).
    return re.sub(r"\.", "_", function_name) if "." in function_name and underscore_to_dot else function_name


def type_checker(param, value, possible_answer, expected_type_description, expected_type_converted,
                 nested_type_converted):
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}
    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and possible_answer_type != expected_type_converted:
        is_variable = True
    if type(value) == expected_type_converted:  # noqa: E721  (BFCL compares exact types)
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for possible_answer_item in possible_answer:
            flag = True
            if type(possible_answer_item) == list:  # noqa: E721
                for value_item in value:
                    r = type_checker(param, value_item, possible_answer_item, str(nested_type_converted),
                                     nested_type_converted, None)
                    if not r["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [f"Nested type checking failed for parameter {param!r}."]
        result["error_type"] = "type_error:nested"
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and type(value) == possible_answer_type:  # noqa: E721
        result["is_variable"] = True
        return result
    result["valid"] = False
    result["error"].append(f"Incorrect type for parameter {param!r}. Expected type {expected_type_description}, got "
                           f"{type(value).__name__}. Parameter value: {value!r}.")
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", input_string).lower().replace("'", '"')


def string_checker(param, model_output, possible_answer):
    std = [standardize_string(a) for a in possible_answer if type(a) == str]  # noqa: E721
    if standardize_string(model_output) not in std:
        return {"valid": False, "error": [f"Invalid value for parameter {param!r}: {model_output!r}."],
                "error_type": "value_error:string"}
    return {"valid": True, "error": []}


def list_checker(param, model_output, possible_answer):
    out = list(model_output)
    for i in range(len(out)):
        if type(out[i]) == str:  # noqa: E721
            out[i] = standardize_string(model_output[i])
    std = []
    for i in range(len(possible_answer)):
        std.append([])
        for j in range(len(possible_answer[i])):
            v = possible_answer[i][j]
            std[i].append(standardize_string(v) if type(v) == str else v)  # noqa: E721
    if out not in std:
        return {"valid": False, "error": [f"Invalid value for parameter {param!r}: {model_output!r}."],
                "error_type": "value_error:list/tuple"}
    return {"valid": True, "error": []}


def dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]
        for key, value in model_output.items():
            if key not in possible_answer:
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
            sv = standardize_string(value) if type(value) == str else value  # noqa: E721
            spa = [standardize_string(x) if type(x) == str else x for x in possible_answer[key]]  # noqa: E721
            if sv not in spa:
                result["error"].append(f"Invalid value for parameter {key!r}: {value!r}.")
                result["error_type"] = "value_error:dict_value"
                flag = False
                break
        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def list_dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for answer_index in range(len(possible_answers)):
        flag = True
        if len(model_output) != len(possible_answers[answer_index]):
            result = {"valid": False, "error": ["Wrong number of dictionaries in the list."],
                      "error_type": "value_error:list_dict_count"}
            continue
        for dict_index in range(len(model_output)):
            result = dict_checker(param, model_output[dict_index], [possible_answers[answer_index][dict_index]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def simple_function_checker(func_description, model_output, possible_answer):
    possible_answer = list(possible_answer.values())[0]
    func_name = convert_func_name(func_description["name"])
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]
    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}
    if func_name not in model_output:
        return {"valid": False, "error": [f"Function name {func_name!r} not found in model output."],
                "error_type": "simple_function_checker:wrong_func_name"}
    model_params = model_output[func_name]
    for param in required_params:
        if param not in model_params:
            return {"valid": False, "error": [f"Missing required parameter: {param!r}."],
                    "error_type": "simple_function_checker:missing_required"}
    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            return {"valid": False, "error": [f"Unexpected parameter: {param!r}."],
                    "error_type": "simple_function_checker:unexpected_param"}
        full = param_details[param]
        expected_type_description = full["type"]
        nested_type_converted = None
        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type_converted = PYTHON_TYPE_MAPPING[param_details[param]["items"]["type"]]
        if expected_type_description == "tuple" and type(value) == tuple:  # noqa: E721
            value = list(value)
        if expected_type_description == "float" and type(value) == int:  # noqa: E721
            value = float(value)
        tc = type_checker(param, value, possible_answer[param], expected_type_description, expected_type_converted,
                          nested_type_converted)
        is_variable = tc["is_variable"]
        if not tc["valid"]:
            return tc
        if not is_variable:
            if expected_type_converted == dict:  # noqa: E721
                r = dict_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            elif expected_type_converted == list and nested_type_converted == dict:  # noqa: E721
                r = list_dict_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            elif expected_type_converted == str:  # noqa: E721
                r = string_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
            elif expected_type_converted == list:  # noqa: E721
                r = list_checker(param, value, possible_answer[param])
                if not r["valid"]:
                    return r
                continue
        if value not in possible_answer[param]:
            return {"valid": False, "error": [f"Invalid value for parameter {param!r}: {value!r}."],
                    "error_type": "value_error:others"}
    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            return {"valid": False, "error": [f"Optional parameter {param!r} not provided and not marked as optional."],
                    "error_type": "simple_function_checker:missing_optional"}
    return result


def parallel_function_checker_no_order(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "parallel_function_checker_no_order:wrong_count"}
    matched: list[int] = []
    for i in range(len(possible_answers)):
        name = list(possible_answers[i].keys())[0]
        desc = find_description(func_descriptions, name)
        result = {"valid": False, "error": []}
        for index in range(len(model_output)):
            if index in matched:
                continue
            result = simple_function_checker(desc, model_output[index], possible_answers[i])
            if result["valid"]:
                matched.append(index)
                break
        if not result["valid"]:
            return {"valid": False, "error": [f"Could not find a matching function for index {i}."],
                    "error_type": "parallel_function_checker_no_order:cannot_find_match"}
    return {"valid": True, "error": []}


def multiple_function_checker(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "multiple_function_checker:wrong_count"}
    name = list(possible_answers[0].keys())[0]
    return simple_function_checker(find_description(func_descriptions, name), model_output[0], possible_answers[0])


def ast_checker(func_description, model_output, possible_answer, test_category: str):
    if "parallel" in test_category:
        return parallel_function_checker_no_order(func_description, model_output, possible_answer)
    if "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer)
    if len(model_output) != 1:
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count"}
    return simple_function_checker(func_description[0], model_output[0], possible_answer[0])


# ---- BFCL tool conversion (convert_to_tool + GORILLA_TO_OPENAPI, OpenAI-completions style) ------------------------
GORILLA_TO_OPENAPI = {"integer": "integer", "number": "number", "float": "number", "string": "string",
                      "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array", "dict": "object",
                      "object": "object", "tuple": "array", "any": "string", "byte": "integer", "short": "integer",
                      "long": "integer", "double": "number", "char": "string", "ArrayList": "array", "Array": "array",
                      "HashMap": "object", "Hashtable": "object", "Queue": "array", "Stack": "array", "Any": "string",
                      "String": "string", "Bigint": "integer"}


def _cast_to_openai_type(properties: dict, mapping: dict) -> dict:
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] = properties[key].get("description", "") + " This is a float type value."
            properties[key]["type"] = mapping.get(var_type, "string")
        if properties[key]["type"] in ("array", "object"):
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(properties[key]["properties"], mapping)
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping[properties[key]["items"]["type"]]
                if properties[key]["items"]["type"] == "array" and "items" in properties[key]["items"]:
                    properties[key]["items"]["items"]["type"] = mapping[properties[key]["items"]["items"]["type"]]
                elif properties[key]["items"]["type"] == "object" and "properties" in properties[key]["items"]:
                    properties[key]["items"]["properties"] = _cast_to_openai_type(
                        properties[key]["items"]["properties"], mapping)
    return properties


def bfcl_tools(functions: list[dict]) -> list[dict]:
    out = []
    for item in copy.deepcopy(functions):
        item["description"] = item.get("description", "") + " Note that the provided function is in Python 3 syntax."
        if "." in item["name"]:
            item["name"] = re.sub(r"\.", "_", item["name"])
        item["parameters"]["type"] = "object"
        item["parameters"]["properties"] = _cast_to_openai_type(item["parameters"]["properties"], GORILLA_TO_OPENAPI)
        out.append({"type": "function", "function": item})
    return out


def _id_key(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def freeze(store: Path, out_dir: Path) -> None:
    base = store / "datasets" / "gorilla-llm" / f"Berkeley-Function-Calling-Leaderboard@{BFCL_REV}"
    sums = {}
    for line in (base / "SHA256SUMS.upstream").read_text().splitlines():
        h, p = line.split(None, 1)
        sums[p.strip()] = h
    rows = []
    for cat in BFCL_CATEGORIES:
        q_path, a_path = f"BFCL_v3_{cat}.json", f"possible_answer/BFCL_v3_{cat}.json"
        for p in (q_path, a_path):
            got = hashlib.sha256((base / p).read_bytes()).hexdigest()
            if got != sums.get(p) or got != BFCL_FILE_SHA256[p]:
                raise SystemExit(f"FREEZE-FAILED: {p} sha256 {got} is not the pinned upstream file")
        qs = [json.loads(x) for x in (base / q_path).read_bytes().split(b"\n") if x.strip()]
        ans = {r["id"]: r["ground_truth"] for r in
               (json.loads(x) for x in (base / a_path).read_bytes().split(b"\n") if x.strip())}
        qs.sort(key=lambda r: _id_key(r["id"]))
        for q in qs[:BFCL_PER_CATEGORY]:
            if len(q["question"]) != 1:
                raise SystemExit(f"FREEZE-FAILED: {q['id']} is multi-turn")
            rows.append({"id": q["id"], "category": cat, "messages": q["question"][0], "functions": q["function"],
                         "tools": bfcl_tools(q["function"]), "ground_truth": ans[q["id"]]})
    out_dir.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()
    (out_dir / "proxy_bfcl.jsonl").write_bytes(data)
    tasks = syn.generate(100, 1234)
    sdata = "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in tasks).encode()
    (out_dir / "proxy_synthetic.jsonl").write_bytes(sdata)
    src_sums = {p: sums[p] for c in BFCL_CATEGORIES for p in (f"BFCL_v3_{c}.json", f"possible_answer/BFCL_v3_{c}.json")}
    rec = {"bfcl": {"repo": BFCL_REPO, "revision": BFCL_REV, "categories": list(BFCL_CATEGORIES),
                    "per_category": BFCL_PER_CATEGORY, "rule": "first 100 by numeric id", "source_sha256": src_sums,
                    "items": len(rows)},
           "synthetic": {"generator": "tasks_synthetic.py", "seed": 1234, "tasks": len(tasks),
                         "kinds": {k: sum(1 for t in tasks if t["kind"] == k) for k in sorted({t["kind"] for t in tasks})}},
           "files": {"proxy_bfcl.jsonl": hashlib.sha256(data).hexdigest(),
                     "proxy_synthetic.jsonl": hashlib.sha256(sdata).hexdigest()}}
    (out_dir / "proxy-record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"FREEZE-OK {json.dumps(rec['files'])}")


# ---- client ---------------------------------------------------------------------------------------------------------
class Client:
    def __init__(self, base: str, timeout: float = 900):
        u = urllib.parse.urlparse(base)
        self.host, self.port, self.timeout = u.hostname, u.port or 80, timeout
        self.conn: http.client.HTTPConnection | None = None

    def chat(self, body: dict) -> dict:
        for attempt in range(3):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.request("POST", "/v1/chat/completions", json.dumps(body),
                                  {"Content-Type": "application/json", "Connection": "keep-alive"})
                r = self.conn.getresponse()
                data = r.read()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
                return json.loads(data)
            except (OSError, http.client.HTTPException, RuntimeError) as e:
                if self.conn is not None:
                    self.conn.close()
                self.conn = None
                if attempt == 2:
                    raise RuntimeError(str(e)) from e
                time.sleep(2 + 3 * attempt)
        raise RuntimeError("unreachable")


def parse_calls(msg: dict, declared: set[str]) -> tuple[list[dict] | None, str | None]:
    """Tool calls as [{name: args_dict}], or (None, reason) when no well-formed call to a declared tool came back."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, "no tool call"
    out = []
    for c in calls:
        fn = c.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments")
        if name not in declared:
            return None, f"undeclared tool {name!r}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return None, "arguments are not JSON"
        if not isinstance(args, dict):
            return None, "arguments are not an object"
        out.append({name: args})
    return out, None


def run_bfcl_item(cl: Client, model: str, item: dict) -> dict:
    body = {"model": model, "messages": item["messages"], "tools": item["tools"], "temperature": 0,
            "max_tokens": MAX_TOKENS}
    t = time.time()
    try:
        resp = cl.chat(body)
    except RuntimeError as e:
        return {"id": item["id"], "category": item["category"], "error": str(e)[:300], "parsed": False, "success": False}
    msg = resp["choices"][0]["message"]
    calls, why = parse_calls(msg, {t["function"]["name"] for t in item["tools"]})
    rec = {"id": item["id"], "category": item["category"], "parsed": calls is not None, "parse_error": why,
           "calls": calls, "finish_reason": resp["choices"][0].get("finish_reason"),
           "has_reasoning": bool(msg.get("reasoning") or msg.get("reasoning_content")),
           "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"), "seconds": round(time.time() - t, 2),
           "content": (msg.get("content") or "")[:300]}
    rec.update(score_bfcl(item, calls))
    return rec


def score_bfcl(item: dict, calls: list[dict] | None) -> dict:
    if calls is None:
        return {"success": False, "check": "no parsed call"}
    try:
        r = ast_checker(item["functions"], calls, item["ground_truth"], item["category"])
    except Exception as e:  # noqa: BLE001 - the checker indexes model output freely; a crash is a failed item
        return {"success": False, "check": f"checker error {type(e).__name__}: {e}"[:200]}
    return {"success": bool(r["valid"]), "check": r.get("error_type") if not r["valid"] else "valid"}


# Docker Engine API socket and container routes, assembled so the public export's absolute-path rewrite (three or more
# literal segments) leaves them as they are; the strings are unchanged.
DOCKER_SOCKET = "/var/run/" + "docker.sock"
ENGINE_CONTAINERS = "/containers"


class Sandbox:
    """Runs a call list against a fresh copy of the fixture in a throwaway container, created through the Docker Engine
    API on a mounted socket (the client container has no docker CLI): --network none, read-only root, non-root user,
    tmpfs /work and /tmp, 512 MiB, 1 CPU, 128 pids, 60 s; removed after each run. backend "local" (tests only) runs
    the same runner in a subprocess against a temp directory instead of a container."""

    def __init__(self, image: str, socket_path: str = DOCKER_SOCKET, backend: str = "docker"):
        self.image, self.socket_path, self.backend = image, socket_path, backend

    def _api(self, method: str, path: str, body: dict | None = None, timeout: float = 30):
        import socket

        class UConn(http.client.HTTPConnection):
            def connect(conn_self):  # noqa: N805
                conn_self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn_self.sock.settimeout(timeout)
                conn_self.sock.connect(self.socket_path)

        c = UConn("localhost", timeout=timeout)
        try:
            c.request(method, path, json.dumps(body) if body is not None else None,
                      {"Content-Type": "application/json"} if body is not None else {})
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def run(self, fixture: dict[str, str], calls: list[dict], timeout: int = 60) -> dict:
        job = {"runner": syn.SANDBOX_RUNNER, "fixture": fixture, "calls": calls}
        if self.backend == "local":
            with tempfile.TemporaryDirectory() as td:
                job["root"] = td
                r = subprocess.run([sys.executable, "-c", syn.SANDBOX_BOOT], input=json.dumps(job).encode(),
                                   capture_output=True, timeout=timeout)
                out = r.stdout.decode(errors="replace")
                if r.returncode != 0:
                    return {"ok": False, "error": f"runner rc {r.returncode}: {r.stderr[-300:].decode(errors='replace')}"}
        else:
            import base64
            import uuid

            env = ["JOB_B64=" + base64.b64encode(json.dumps(job).encode()).decode()]
            boot = "import os,json,base64;J=json.loads(base64.b64decode(os.environ['JOB_B64']));exec(J['runner'],{'J':J})"
            spec = {"Image": self.image, "Entrypoint": ["python3"], "Cmd": ["-c", boot], "Env": env,
                    "User": "65534:65534", "WorkingDir": "/tmp", "NetworkDisabled": True,
                    "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True,
                                   "Tmpfs": {"/tmp": "rw,exec,size=64m", "/work": "rw,exec,size=64m"},
                                   "Memory": 512 * 2**20, "NanoCpus": 10**9, "PidsLimit": 128}}
            name = f"forge-r39-sbx-{uuid.uuid4().hex[:12]}"
            st, data = self._api("POST", f"/containers/create?name={name}", spec)
            if st != 201:
                return {"ok": False, "error": f"create HTTP {st}: {data[:200]!r}"}
            cid = json.loads(data)["Id"]
            try:
                st, data = self._api("POST", f"{ENGINE_CONTAINERS}/{cid}/start")
                if st not in (204, 304):
                    return {"ok": False, "error": f"start HTTP {st}: {data[:200]!r}"}
                try:
                    st, data = self._api("POST", f"{ENGINE_CONTAINERS}/{cid}/wait", timeout=timeout)
                    code = json.loads(data).get("StatusCode")
                except OSError:
                    return {"ok": False, "error": "sandbox timeout"}
                st, data = self._api("GET", f"{ENGINE_CONTAINERS}/{cid}/logs?stdout=1&stderr=1")
                stdout, stderr, i = b"", b"", 0
                while i + 8 <= len(data):  # multiplexed stream: [stream, 0, 0, 0, size(4, big endian)] + payload
                    kind, size = data[i], int.from_bytes(data[i + 4:i + 8], "big")
                    chunk = data[i + 8:i + 8 + size]
                    if kind == 1:
                        stdout += chunk
                    else:
                        stderr += chunk
                    i += 8 + size
                out = stdout.decode(errors="replace")
                if code != 0:
                    return {"ok": False, "error": f"sandbox rc {code}: {stderr[-300:].decode(errors='replace')}"}
            finally:
                self._api("DELETE", f"/containers/{cid}?force=1&v=1")
        try:
            return {"ok": True, **json.loads(out.strip().splitlines()[-1])}
        except (ValueError, IndexError):
            return {"ok": False, "error": f"sandbox output unreadable: {out[-200:]!r}"}


def run_synthetic_item(cl: Client, model: str, task: dict, sbx: Sandbox) -> dict:
    """Multi-turn: the model calls tools, each call runs in the sandbox (state kept across turns by replaying every
    call so far against a fresh fixture), results go back as tool messages, until it stops calling or 6 turns pass."""
    messages = [{"role": "system", "content": syn.SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]
    all_calls: list[dict] = []
    rec: dict = {"id": task["id"], "kind": task["kind"], "parsed": False, "success": False, "turns": 0}
    t = time.time()
    for turn in range(6):
        body = {"model": model, "messages": messages, "tools": syn.TOOLS, "temperature": 0, "max_tokens": MAX_TOKENS}
        try:
            resp = cl.chat(body)
        except RuntimeError as e:
            rec["error"] = str(e)[:300]
            break
        msg = resp["choices"][0]["message"]
        calls, why = parse_calls(msg, {x["function"]["name"] for x in syn.TOOLS})
        rec["turns"] = turn + 1
        if calls is None:
            if turn == 0:
                rec["parse_error"] = why
            break
        rec["parsed"] = True
        new = [{"name": n, "arguments": a} for c in calls for n, a in c.items()]
        all_calls.extend(new)
        res = sbx.run(task["fixture"], all_calls)
        if not res.get("ok"):
            rec["error"] = res.get("error")
            break
        outs = res["outputs"][-len(new):]
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": [{"id": f"call_{turn}_{i}", "type": "function",
                                         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                        for i, c in enumerate(new)]})
        for i, o in enumerate(outs):
            messages.append({"role": "tool", "tool_call_id": f"call_{turn}_{i}", "content": o[:4000]})
        rec["final"] = res
        if task["check"]["type"] == "tree" and res.get("tree_sha256") == task["check"]["tree_sha256"]:
            break
    final = rec.get("final") or {}
    rec["calls"] = all_calls[:20]
    rec["success"] = bool(rec["parsed"]) and syn.check(task, final, messages)
    rec["seconds"] = round(time.time() - t, 2)
    rec.pop("final", None)
    return rec


def summarize(items: list[dict]) -> dict:
    n = len(items)
    return {"items": n, "parsed": sum(1 for i in items if i.get("parsed")),
            "success": sum(1 for i in items if i.get("success")), "errors": sum(1 for i in items if i.get("error")),
            "parse_rate": round(sum(1 for i in items if i.get("parsed")) / n, 4) if n else None,
            "success_rate": round(sum(1 for i in items if i.get("success")) / n, 4) if n else None}


def run(args) -> None:
    data = Path(args.data_dir)
    for name, pin in FROZEN_SHA256.items():
        got = hashlib.sha256((data / name).read_bytes()).hexdigest()
        if got != pin and not args.allow_unpinned:
            raise SystemExit(f"PROXY-FAILED: {name} sha256 {got} is not the frozen {pin}")
    bfcl = [json.loads(x) for x in (data / "proxy_bfcl.jsonl").read_bytes().split(b"\n") if x.strip()]
    synth = [json.loads(x) for x in (data / "proxy_synthetic.jsonl").read_bytes().split(b"\n") if x.strip()]
    if args.limit:
        bfcl, synth = bfcl[:args.limit], synth[:args.limit]
    pool = [Client(args.base) for _ in range(args.workers)]
    sbx = Sandbox(args.sandbox_image, backend=args.sandbox_backend)
    t0 = time.time()

    def work(i_item):
        i, item = i_item
        cl = pool[i % len(pool)]
        return run_bfcl_item(cl, args.model, item) if "category" in item else \
            run_synthetic_item(cl, args.model, item, sbx)

    jobs = list(enumerate(bfcl + synth))
    results: list[dict] = [None] * len(jobs)  # type: ignore[list-item]
    # One worker thread per persistent connection: each Client is used by one thread at a time.
    chunks = [jobs[w::args.workers] for w in range(args.workers)]

    def worker(chunk):
        return [(i, work((i, it))) for i, it in chunk]

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for part in ex.map(worker, chunks):
            for i, r in part:
                results[i] = r
    b = [r for r in results if "category" in r]
    s = [r for r in results if "kind" in r]
    out = {"label": args.label, "model": args.model, "base": args.base, "max_tokens": MAX_TOKENS,
           "data_sha256": {n: hashlib.sha256((data / n).read_bytes()).hexdigest() for n in FROZEN_SHA256},
           "summary": {"bfcl": summarize(b), "synthetic": summarize(s), "overall": summarize(b + s),
                       "bfcl_by_category": {c: summarize([r for r in b if r["category"] == c]) for c in BFCL_CATEGORIES},
                       "synthetic_by_kind": {k: summarize([r for r in s if r["kind"] == k])
                                             for k in sorted({r["kind"] for r in s})}},
           "seconds": round(time.time() - t0), "items": results}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    sm = out["summary"]
    print(f"PROXY-WRITTEN {args.label} | overall parse {sm['overall']['parse_rate']} success {sm['overall']['success_rate']}"
          f" | bfcl {sm['bfcl']['success']}/{sm['bfcl']['items']} | synthetic {sm['synthetic']['success']}/"
          f"{sm['synthetic']['items']} | errors {sm['overall']['errors']} | {out['seconds']} s")


def gt_materialize(v):
    """A ground-truth value as a model would send it: nested dicts take their first allowed value per required key."""
    if isinstance(v, dict):
        return {k: gt_pick(vs) for k, vs in v.items() if any(x != "" for x in vs)}
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        return [gt_materialize(x) for x in v]
    return v


def gt_pick(vals: list):
    return gt_materialize(next(x for x in vals if x != ""))


def selftest(args) -> None:
    """Every synthetic task's reference calls must pass its check in the sandbox, a no-op call must fail it, and every
    BFCL item's ground truth, materialized as a model would send it (the first allowed value of every parameter the
    function declares and the ground truth gives a value for), must pass the AST checker while the same call under a
    wrong function name fails. Declared-only: two upstream items list a value for a parameter their function does not
    declare (parallel_multiple_12 `permeability`, parallel_multiple_26 `type`); a model cannot send it."""
    data = Path(args.data_dir)
    sbx = Sandbox(args.sandbox_image, backend=args.sandbox_backend)
    synth = [json.loads(x) for x in (data / "proxy_synthetic.jsonl").read_bytes().split(b"\n") if x.strip()]
    bad = []
    t0 = time.time()

    def one(task):
        good = sbx.run(task["fixture"], task["reference_calls"])
        noop = sbx.run(task["fixture"], [{"name": "bash", "arguments": {"command": "true"}}])
        return task["id"], syn.check(task, good), syn.check(task, noop), good.get("error") or noop.get("error")

    with cf.ThreadPoolExecutor(8) as ex:
        for tid, ok_good, ok_noop, err in ex.map(one, synth):
            if not ok_good or ok_noop or err:
                bad.append((tid, ok_good, ok_noop, err))
    bfcl = [json.loads(x) for x in (data / "proxy_bfcl.jsonl").read_bytes().split(b"\n") if x.strip()]
    bf_bad = []
    for item in bfcl:
        good = []
        for gt in item["ground_truth"]:
            for n, p in gt.items():
                declared = find_description(item["functions"], n)["parameters"]["properties"]
                good.append({convert_func_name(n): {k: gt_pick(vals) for k, vals in p.items()
                                                    if k in declared and any(v != "" for v in vals)}})
        if not score_bfcl(item, good)["success"]:
            bf_bad.append((item["id"], "ground truth rejected"))
        wrong = copy.deepcopy(good)
        name = next(iter(wrong[0]))
        wrong[0] = {name + "_x": wrong[0][name]}
        if score_bfcl(item, wrong)["success"]:
            bf_bad.append((item["id"], "wrong function name accepted"))
    if bad or bf_bad:
        raise SystemExit(f"PROXY-SELFTEST-FAILED synthetic {bad[:5]} bfcl {bf_bad[:5]}")
    print(f"PROXY-SELFTEST-OK {len(synth)} synthetic tasks (reference passes, no-op fails, backend {args.sandbox_backend})"
          f"; {len(bfcl)} BFCL items (ground truth passes, wrong name fails) | {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("store", type=Path)
    f.add_argument("out_dir", type=Path)
    r = sub.add_parser("run")
    r.add_argument("base")
    r.add_argument("model")
    r.add_argument("data_dir")
    r.add_argument("out")
    r.add_argument("--label", default="run")
    r.add_argument("--sandbox-image", default=SANDBOX_IMAGE_DEFAULT)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--sandbox-backend", choices=("docker", "local"), default="docker")
    r.add_argument("--allow-unpinned", action="store_true", help="tests only: accept data other than the frozen files")
    t = sub.add_parser("selftest", help="reference solutions pass and planted faults fail, in the real sandbox")
    t.add_argument("data_dir")
    t.add_argument("--sandbox-image", default=SANDBOX_IMAGE_DEFAULT)
    t.add_argument("--sandbox-backend", choices=("docker", "local"), default="docker")
    s = sub.add_parser("score")
    s.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.store, a.out_dir)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "selftest":
        selftest(a)
    else:
        d = json.loads(Path(a.out).read_text())
        print(json.dumps(d["summary"], indent=1))


if __name__ == "__main__":
    main()
