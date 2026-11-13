#!/usr/bin/env python3
"""Ripple-65B Preview (#39): build the pinned calibration set (512 rows) and the 64 held-out KL prompts.

Sources (posted pre-registration, every file hashed against its pin from the same bytes/stream that is parsed):
  128 abisee/cnn_dailymail 3.0.0 train shard 0          summarize turn (article) + assistant highlights
  128 nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 data/reasoning_on.jsonl (assistant reasoning_content rendered
      by the template as a <think> block; empty system turns dropped)
  192 SWE-bench/SWE-smith-trajectories tool split shard 0 (OpenAI tool-call messages; the rows carry no tool schema, so
      SWE-agent's bash / str_replace_editor / submit definitions below are passed as `tools`; JSON-string arguments are
      parsed to objects, as vLLM does before rendering)
   64 NousResearch/hermes-function-calling-v1 func-calling.json (Hermes <tool_call>/<tool_response> text converted to
      tool_calls and tool messages; the row's own `tools` list is passed so the template writes its tools block; the
      Hermes system preamble, which restates the tools in Hermes' format, is dropped)
Every row is rendered with the checkpoint's own chat template (chat_template.jinja, sha256 pinned below) through its
tokenizer, no generation prompt, and tokenized without added special tokens (the tokenizer adds none).

Selection: within each file, row indices are ranked by sha256("1234:<repo path of the file>:<row index>") ascending
(seed 1234); rows are taken in that order, skipping ineligible ones (reason recorded).
Length: each row is cut at the last message boundary whose render is <= CAP tokens and keeps at least one assistant
turn. CAP = 4,096 unless the 512 rows would exceed 1,048,576 tokens in total; then CAP is the largest multiple of 64
below 4,096 for which the total fits (the same CAP applies to every row; a row with no fitting assistant turn is
skipped and the next row in hash order is taken).
Held-out: the next 16 eligible rows per source in the same hash order (disjoint from the calibration rows by file and
row index), cut by the same CAP rule; the 16 SWE-smith prompts must end at an assistant tool-call turn (largest cut
<= CAP whose last message is an assistant message with tool calls), and their record carries span_start, the index of
the first token of that final assistant turn's body (after "<|im_start|>assistant\\n"), so KL can also be read on the
tool-call span.

Outputs in OUT_DIR:
  ripple39_calib_512.jsonl        {"text", "tokens", "source", "file", "index", "messages_kept", "messages_total"}
  ripple39_heldout_64.jsonl       the same fields plus "token_ids" and "span_start" (SWE-smith only, else null)
  calibration-record.json         pins, rule, CAP, token statistics (p50/p90/max, total), the row lists, skips, the
                                  output sha256s and the row-list digest (sha256 of the canonical JSON row list)
Usage (PTQ image, CPU, --network none): build_ripple_calib.py STORE_ROOT SOURCE_DIR OUT_DIR
  STORE_ROOT: the NAS model store root holding datasets/<org>/<name>@<rev>/ (fetch_dataset_to_store.py layout)
  SOURCE_DIR: the BF16 checkpoint (tokenizer.json, tokenizer_config.json, chat_template.jinja)
Prints CALIB-OK on success."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import sys
from pathlib import Path

SEED = 1234
MAX_ROW_TOKENS = 4096
MAX_TOTAL_TOKENS = 1_048_576
CAP_STEP = 64
N_HELDOUT = 16
TEMPLATE_SHA256 = "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3"
VOCAB_FILE_SHA256 = "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"  # the checkpoint's vocabulary file
CNN_INSTRUCTION = "Summarize the following news article.\n\n"

# (key, rows, repo, revision, file, sha256)
SOURCES = [
    ("cnn_dailymail", 128, "abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
     "3.0.0/train-00000-of-00003.parquet", "a56093836f6283ceac48128515f6c87ade70abb11aec31d23a176c4d90ae1736"),
    ("nemotron_if_chat_reasoning_on", 128, "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
     "1a9454ed054b8544503ab8d8c0a519d141a44c5b", "data/reasoning_on.jsonl",
     "2ab423660dd2bfd6b219f1745821ac5d5fb13643755ef77b56765fca524cbb4e"),
    ("swe_smith_tool", 192, "SWE-bench/SWE-smith-trajectories", "08e109b4a59eaeebf80e4675cd125d42e7ac99a4",
     "data/tool-00000-of-00008.parquet", "ac76e9efe75978f83d4daec98a55604587dd58b0f1509364893b8778cbf5b487"),
    ("hermes_function_calling", 64, "NousResearch/hermes-function-calling-v1",
     "dae3e1d28cfbcf4b915c04ea1e072030529b4bda", "func-calling.json",
     "769478035a886678d525d057ea66e3fd1e247f43e35c8d6d7cc866e080b6742b"),
]
assert sum(s[1] for s in SOURCES) == 512

# SWE-agent's tool definitions (the SWE-smith rows record calls to these three tools but no schema).
SWE_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the terminal.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to execute."}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": "Custom editing tool for viewing, creating and editing files. State is persistent across "
                       "command calls. `view` shows a file with line numbers or lists a directory; `create` writes "
                       "a new file; `str_replace` replaces exactly one occurrence of old_str; `insert` inserts "
                       "new_str after insert_line; `undo_edit` reverts the last edit of the file.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "The command to run."},
            "path": {"type": "string", "description": "Absolute path to the file or directory."},
            "file_text": {"type": "string", "description": "Content of the file to create (`create`)."},
            "old_str": {"type": "string", "description": "The exact text to replace (`str_replace`)."},
            "new_str": {"type": "string", "description": "The replacement (`str_replace`) or inserted text (`insert`)."},
            "insert_line": {"type": "integer", "description": "Insert new_str after this line (`insert`)."},
            "view_range": {"type": "array", "items": {"type": "integer"},
                           "description": "Line range to show (`view`)."}},
            "required": ["command", "path"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the solution when the task is done.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


class Ineligible(Exception):
    pass


def rank_key(file: str, index: int) -> str:
    return hashlib.sha256(f"{SEED}:{file}:{index}".encode()).hexdigest()


def open_regular(path: Path) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SystemExit(f"CALIB-FAILED: {path} is not a regular file")
    return fd


def read_pinned(path: Path, want: str) -> bytes:
    fd = open_regular(path)
    with open(fd, "rb") as f:
        data = f.read()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SystemExit(f"CALIB-FAILED: {path} sha256 {got} is not the pin {want}")
    return data


def text_of(content) -> str:
    """Message content as the template needs it (a string); text parts are joined, anything else is ineligible."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise Ineligible("non-text content part")
            out.append(part["text"])
        return "".join(out)
    raise Ineligible(f"content of type {type(content).__name__}")


# ---- per-source conversion to (messages, tools) ------------------------------------------------------------------------
def conv_cnn(row: dict):
    article, highlights = row.get("article"), row.get("highlights")
    if not isinstance(article, str) or not isinstance(highlights, str) or not article.strip() or not highlights.strip():
        raise Ineligible("empty article or highlights")
    return [{"role": "user", "content": CNN_INSTRUCTION + article},
            {"role": "assistant", "content": highlights}], None


def conv_nemotron(row: dict):
    msgs = []
    for m in row.get("messages") or []:
        role = m.get("role")
        if role == "system" and not text_of(m.get("content")).strip():
            continue  # empty system turn: dropped (a client sends no system message)
        if role not in ("system", "user", "assistant"):
            raise Ineligible(f"role {role!r}")
        out = {"role": role, "content": text_of(m.get("content"))}
        if role == "assistant" and isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip():
            out["reasoning_content"] = m["reasoning_content"]
        msgs.append(out)
    if not any(m["role"] == "assistant" for m in msgs):
        raise Ineligible("no assistant turn")
    return msgs, None


def _args_object(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            raise Ineligible("tool-call arguments are not JSON") from e
        if isinstance(v, dict):
            return v
    raise Ineligible("tool-call arguments are not an object")


def conv_swe(row: dict):
    raw = row.get("messages")
    msgs_in = json.loads(raw) if isinstance(raw, str) else raw
    msgs = []
    for m in msgs_in or []:
        role = m.get("role")
        if role in ("system", "user"):
            msgs.append({"role": role, "content": text_of(m.get("content"))})
        elif role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if not isinstance(fn.get("name"), str):
                    raise Ineligible("tool call without a name")
                calls.append({"type": "function", "function": {"name": fn["name"],
                                                               "arguments": _args_object(fn.get("arguments"))}})
            out = {"role": "assistant", "content": text_of(m.get("content"))}
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
        elif role == "tool":
            msgs.append({"role": "tool", "content": text_of(m.get("content"))})
        else:
            raise Ineligible(f"role {role!r}")
    if not any(m["role"] == "assistant" for m in msgs):
        raise Ineligible("no assistant turn")
    return msgs, SWE_TOOLS


_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TR = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def conv_hermes(row: dict):
    tools = row.get("tools")
    tools = json.loads(tools) if isinstance(tools, str) else tools
    if not isinstance(tools, list) or not tools:
        raise Ineligible("no tools list")
    msgs = []
    for t in row.get("conversations") or []:
        who, val = t.get("from"), t.get("value")
        if not isinstance(val, str):
            raise Ineligible("non-string turn")
        if who == "system":
            continue  # Hermes' preamble restates the tools in its own format; the template writes its own block
        if who == "human":
            msgs.append({"role": "user", "content": val})
        elif who == "gpt":
            calls = []
            for body in _TC.findall(val):
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError as e:
                    raise Ineligible("unparseable <tool_call>") from e
                if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                    raise Ineligible("tool call without a name")
                calls.append({"type": "function", "function": {"name": obj["name"],
                                                               "arguments": _args_object(obj.get("arguments", {}))}})
            out = {"role": "assistant", "content": _TC.sub("", val).strip()}
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
        elif who == "tool":
            if not msgs or msgs[-1]["role"] != "assistant" or not msgs[-1].get("tool_calls"):
                raise Ineligible("tool turn without a preceding tool call")
            bodies = _TR.findall(val) or [val.strip()]
            msgs.extend({"role": "tool", "content": b} for b in bodies)
        else:
            raise Ineligible(f"turn {who!r}")
    if not any(m["role"] == "assistant" for m in msgs):
        raise Ineligible("no assistant turn")
    return msgs, tools


CONVERT = {"cnn_dailymail": conv_cnn, "nemotron_if_chat_reasoning_on": conv_nemotron, "swe_smith_tool": conv_swe,
           "hermes_function_calling": conv_hermes}


# ---- source readers ------------------------------------------------------------------------------------------------------
class ParquetSource:
    def __init__(self, data: bytes, columns: list[str]):
        import pyarrow.parquet as pq
        self.table = pq.read_table(io.BytesIO(data), columns=columns)
        self.n = self.table.num_rows

    def row(self, i: int) -> dict:
        return self.table.slice(i, 1).to_pylist()[0]


class JsonListSource:
    def __init__(self, data: bytes):
        self.rows = json.loads(data)
        self.n = len(self.rows)

    def row(self, i: int) -> dict:
        return self.rows[i]


class JsonlStream:
    """A large JSONL file: one pass hashes the bytes and records line offsets (fd kept open, size/mtime rechecked)."""

    def __init__(self, path: Path, want: str):
        self.fd = open_regular(path)
        st0 = os.fstat(self.fd)
        h = hashlib.sha256()
        offsets, pos, tail = [0], 0, b""
        with open(self.fd, "rb", closefd=False) as f:
            f.seek(0)
            while True:
                chunk = f.read(1 << 24)
                if not chunk:
                    break
                h.update(chunk)
                base = pos
                j = chunk.find(b"\n")
                while j != -1:
                    offsets.append(base + j + 1)
                    j = chunk.find(b"\n", j + 1)
                pos += len(chunk)
                tail = chunk[-1:]
        if h.hexdigest() != want:
            raise SystemExit(f"CALIB-FAILED: {path} sha256 {h.hexdigest()} is not the pin {want}")
        if tail == b"\n":
            offsets.pop()  # no row after the final newline
        self.offsets, self.size, self.st0 = offsets, pos, st0
        self.n = len(offsets)

    def row(self, i: int) -> dict:
        st = os.fstat(self.fd)
        if (st.st_size, st.st_mtime_ns) != (self.st0.st_size, self.st0.st_mtime_ns):
            raise SystemExit("CALIB-FAILED: a pinned source changed while it was being read")
        end = self.offsets[i + 1] - 1 if i + 1 < self.n else self.size
        line = os.pread(self.fd, end - self.offsets[i], self.offsets[i])
        return json.loads(line)


def main() -> None:
    store, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from transformers import AutoTokenizer

    read_pinned(src / "chat_template.jinja", TEMPLATE_SHA256)
    read_pinned(src / "tokenizer.json", VOCAB_FILE_SHA256)
    tok = AutoTokenizer.from_pretrained(src)
    if hashlib.sha256(tok.chat_template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise SystemExit("CALIB-FAILED: the tokenizer's chat template is not the pinned chat_template.jinja")

    def render(msgs, tools) -> str:
        return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)

    def ntok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    record: dict = {"seed": SEED, "rank_key": "sha256('1234:<file>:<row index>') ascending",
                    "max_row_tokens": MAX_ROW_TOKENS, "max_total_tokens": MAX_TOTAL_TOKENS,
                    "template_sha256": TEMPLATE_SHA256, "vocab_file_sha256": VOCAB_FILE_SHA256,
                    "cnn_instruction": CNN_INSTRUCTION, "sources": []}
    # Candidates per source: prefix token counts for every message boundary up to the first one above 4,096.
    cands: dict[str, list[dict]] = {}
    skipped: dict[str, list] = {}
    for key, n_rows, repo, rev, file, sha in SOURCES:
        org, name = repo.split("/")
        path = store / "datasets" / org / f"{name}@{rev}" / file
        if key == "nemotron_if_chat_reasoning_on":
            reader = JsonlStream(path, sha)
        elif file.endswith(".parquet"):
            cols = ["article", "highlights", "id"] if key == "cnn_dailymail" else ["messages", "instance_id"]
            reader = ParquetSource(read_pinned(path, sha), cols)
        else:
            reader = JsonListSource(read_pinned(path, sha))
        order = sorted(range(reader.n), key=lambda i: rank_key(file, i))
        want = n_rows + N_HELDOUT
        got, skips = [], []
        for i in order:
            if len(got) >= 2 * want:  # margin: a smaller CAP can make kept rows ineligible
                break
            try:
                msgs, tools = CONVERT[key](reader.row(i))
                counts, k = [], 0
                for k in range(1, len(msgs) + 1):
                    c = ntok(render(msgs[:k], tools))
                    counts.append(c)
                    if c > MAX_ROW_TOKENS:
                        break
                got.append({"index": i, "msgs": msgs, "tools": tools, "counts": counts})
            except Ineligible as e:
                skips.append([i, str(e)])
        cands[key], skipped[key] = got, skips
        record["sources"].append({"key": key, "rows": n_rows, "repo": repo, "revision": rev, "file": file,
                                  "sha256": sha, "file_rows": reader.n})
        print(f"{key}: {reader.n} rows in file, {len(got)} candidates converted, {len(skips)} ineligible", flush=True)

    def cut(c: dict, cap: int, tool_call_end: bool = False) -> int:
        """Messages kept: the largest k with render(msgs[:k]) <= cap tokens and an assistant turn kept (0: none)."""
        best = 0
        for k, n in enumerate(c["counts"], start=1):
            if n > cap:
                break
            m = c["msgs"][:k]
            if not any(x["role"] == "assistant" for x in m):
                continue
            if tool_call_end and not (m[-1]["role"] == "assistant" and m[-1].get("tool_calls")):
                continue
            best = k
        return best

    def select(cap: int):
        calib, held, skip_cap = [], [], []
        for key, n_rows, _, _, file, _ in SOURCES:
            taken = 0
            for c in cands[key]:
                if taken >= n_rows + N_HELDOUT:
                    break
                is_held = taken >= n_rows
                k = cut(c, cap, tool_call_end=is_held and key == "swe_smith_tool")
                if k == 0:
                    skip_cap.append([key, c["index"], f"no message boundary <= {cap} tokens"
                                     + (" ending at an assistant tool call" if is_held and key == "swe_smith_tool"
                                        else "")])
                    continue
                (held if is_held else calib).append((key, file, c, k))
                taken += 1
            if taken < n_rows + N_HELDOUT:
                raise SystemExit(f"CALIB-FAILED: {key} has only {taken} eligible rows at cap {cap}")
        return calib, held, skip_cap

    cap = MAX_ROW_TOKENS
    while True:
        calib, held, skip_cap = select(cap)
        total = sum(c["counts"][k - 1] for _, _, c, k in calib)
        if total <= MAX_TOTAL_TOKENS:
            break
        cap -= CAP_STEP
        if cap < 1024:
            raise SystemExit("CALIB-FAILED: no row cap >= 1,024 fits the total")
    print(f"row cap {cap} tokens; calibration total {total} tokens", flush=True)

    def finish(key, file, c, k, with_ids=False):
        text = render(c["msgs"][:k], c["tools"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert len(ids) == c["counts"][k - 1]
        row = {"text": text, "tokens": len(ids), "source": key, "file": file, "index": c["index"],
               "messages_kept": k, "messages_total": len(c["msgs"])}
        if with_ids:
            row["token_ids"] = ids
            row["span_start"] = None
            if key == "swe_smith_tool":
                prefix = render(c["msgs"][:k - 1], c["tools"])
                p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
                head = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
                if ids[:len(p_ids)] != p_ids or ids[len(p_ids):len(p_ids) + len(head)] != head:
                    raise SystemExit(f"CALIB-FAILED: tool-call span of {file}:{c['index']} is not token-aligned")
                row["span_start"] = len(p_ids) + len(head)
        return row

    calib_rows = [finish(*x) for x in calib]
    held_rows = [finish(*x, with_ids=True) for x in held]
    keys = {(r["file"], r["index"]) for r in calib_rows}
    assert len(keys) == 512 and not keys & {(r["file"], r["index"]) for r in held_rows}
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> str:
        data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
        (out / name).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    calib_sha = dump("ripple39_calib_512.jsonl", calib_rows)
    held_sha = dump("ripple39_heldout_64.jsonl", held_rows)
    row_list = [{"source": r["source"], "file": r["file"], "index": r["index"], "messages_kept": r["messages_kept"],
                 "tokens": r["tokens"]} for r in calib_rows]
    held_list = [{"source": r["source"], "file": r["file"], "index": r["index"], "messages_kept": r["messages_kept"],
                  "tokens": r["tokens"], "span_start": r["span_start"]} for r in held_rows]

    def digest(x) -> str:
        return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    lens = sorted(r["tokens"] for r in calib_rows)
    hl = sorted(r["tokens"] for r in held_rows)
    record.update({
        "row_cap_tokens": cap, "calibration_total_tokens": total,
        "calibration_tokens": {"p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)], "max": lens[-1],
                               "min": lens[0]},
        "heldout_total_tokens": sum(hl), "heldout_tokens": {"p50": hl[len(hl) // 2], "max": hl[-1], "min": hl[0]},
        "calibration_per_source": {k: sum(1 for r in calib_rows if r["source"] == k) for k, *_ in SOURCES},
        "heldout_per_source": {k: sum(1 for r in held_rows if r["source"] == k) for k, *_ in SOURCES},
        "outputs": {"ripple39_calib_512.jsonl": calib_sha, "ripple39_heldout_64.jsonl": held_sha},
        "row_list_digest": digest(row_list), "heldout_list_digest": digest(held_list),
        "row_list": row_list, "heldout_list": held_list,
        "skipped_ineligible": {k: v for k, v in skipped.items()}, "skipped_at_cap": skip_cap,
    })
    (out / "calibration-record.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"CALIB-OK calib {calib_sha} ({len(calib_rows)} rows, {total} tokens, p50 {record['calibration_tokens']['p50']}"
          f" p90 {record['calibration_tokens']['p90']} max {lens[-1]}, cap {cap}) heldout {held_sha} "
          f"({len(held_rows)} prompts, {sum(hl)} tokens) row-list {record['row_list_digest']}", flush=True)


if __name__ == "__main__":
    main()
