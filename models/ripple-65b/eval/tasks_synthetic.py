"""Ripple-65B Preview (#39): the seeded synthetic terminal / file-edit tasks of the tool-call proxy (part b).

generate(100, 1234) -> 100 tasks, 10 per kind, each {"id", "kind", "prompt", "fixture": {relpath: text},
"check": {"type": "tree", "tree_sha256"} | {"type": "output", "expected"}, "reference_calls": [...]}.
  file-edit kinds (scored on the final tree hash): edit_constant, rename_symbol, create_file, append_line,
    insert_import, delete_file, move_file
  terminal kinds (scored on a bash call's exact stdout): count_matches, sum_column, json_field
The agent gets two tools (SWE-agent's bash and str_replace_editor, the schemas the calibration's SWE-smith rows use).
SANDBOX_RUNNER executes a call list against a fresh copy of the fixture under /work (inside a throwaway container with
--network none) and returns every call's output, the bash results and the tree digest. Tree digest: sha256 over the
sorted entries "<relpath>/\\0dir" (every directory) and "<relpath>\\0<sha256 of the text with trailing newlines
normalized to one>" (every file), so a missing final newline does not count as a different file.
Stdlib only."""

from __future__ import annotations

import json
import random

SYSTEM_PROMPT = ("You are a coding agent working in a repository at /work. Use the tools to act on it. Make exactly the "
                 "requested change and nothing else: do not create backup, temporary or extra files. When a task asks "
                 "you to print a value, run one bash command whose output is only that value.")

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the repository (cwd /work). Standard coreutils, grep, "
                                       "sed, awk and python3 are available.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to execute."}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": "View, create and edit files. `view` shows a file with line numbers or lists a directory; "
                       "`create` writes a new file with file_text; `str_replace` replaces exactly one occurrence of "
                       "old_str with new_str; `insert` inserts new_str after line insert_line (0 = top of file); "
                       "`undo_edit` reverts the last edit of the file. Paths are absolute.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
            "path": {"type": "string", "description": "Absolute path of the file or directory."},
            "file_text": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"},
            "insert_line": {"type": "integer"}, "view_range": {"type": "array", "items": {"type": "integer"}}},
            "required": ["command", "path"]}}},
]

DIGEST_SRC = r'''
import hashlib as _h
def tree_digest(files, dirs):
    """files: {relpath: bytes}; dirs: iterable of relative directory paths (every directory in the tree)."""
    ent = [d.strip("/") + "/\x00dir" for d in dirs if d.strip("/")]
    for p, b in files.items():
        b = b.rstrip(b"\n") + b"\n" if b else b""
        ent.append(p + "\x00" + _h.sha256(b).hexdigest())
    return _h.sha256("\n".join(sorted(ent)).encode()).hexdigest()
'''
exec(DIGEST_SRC)  # defines tree_digest here; the same source runs inside the sandbox

RUNNER_BODY = r'''
import json, os, subprocess
root = J.get("root", "/work")
os.makedirs(root, exist_ok=True)
def W(p):
    if not isinstance(p, str) or not p.startswith("/"):
        raise ValueError("path must be absolute")
    if root != "/work" and (p == "/work" or p.startswith("/work/")):
        p = os.path.join(root, p[len("/work"):].lstrip("/"))
    p = os.path.normpath(p)
    if not (p == root or p.startswith(root.rstrip("/") + "/")):
        raise ValueError("path outside /work")
    return p
for rel, text in J["fixture"].items():
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)
outputs, bash, undo = [], [], {}
for c in J["calls"]:
    name, a = c.get("name"), c.get("arguments") or {}
    try:
        if name == "bash":
            cmd = a.get("command")
            if not isinstance(cmd, str):
                raise ValueError("command must be a string")
            try:
                env = {k: v for k, v in os.environ.items() if k != "JOB_B64"}
                env.update(HOME=os.environ.get("TMPDIR", "/tmp"), LC_ALL="C")
                r = subprocess.run(["bash", "-c", cmd], cwd=root, capture_output=True, timeout=20, env=env)
                so, se = r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")
                bash.append({"stdout": so, "stderr": se, "rc": r.returncode})
                outputs.append((so + se)[:8000] + ("" if r.returncode == 0 else "\n[exit code %d]" % r.returncode))
            except subprocess.TimeoutExpired:
                bash.append({"stdout": "", "stderr": "timeout", "rc": -1})
                outputs.append("ERROR: the command timed out after 20 s")
        elif name == "str_replace_editor":
            cmd, p = a.get("command"), W(a.get("path"))
            if cmd == "view":
                if os.path.isdir(p):
                    outputs.append("\n".join(sorted(os.path.relpath(os.path.join(d, f), p)
                                                    for d, _, fs in os.walk(p) for f in fs)))
                else:
                    lines = open(p).read().split("\n")
                    lo, hi = (a.get("view_range") or [1, len(lines)])[:2]
                    hi = len(lines) if hi in (-1, None) else hi
                    outputs.append("\n".join("%6d\t%s" % (i, lines[i - 1]) for i in range(max(1, lo), min(hi, len(lines)) + 1)))
            elif cmd == "create":
                if os.path.exists(p):
                    raise ValueError("file already exists: %s" % a.get("path"))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(a.get("file_text") or "")
                undo.setdefault(p, []).append(None)
                outputs.append("File created successfully at: %s" % a.get("path"))
            elif cmd == "str_replace":
                text, old, new = open(p).read(), a.get("old_str"), a.get("new_str") or ""
                if not isinstance(old, str) or not old:
                    raise ValueError("old_str is required")
                n = text.count(old)
                if n != 1:
                    raise ValueError("old_str must occur exactly once (found %d)" % n)
                undo.setdefault(p, []).append(text)
                open(p, "w").write(text.replace(old, new, 1))
                outputs.append("The file %s has been edited." % a.get("path"))
            elif cmd == "insert":
                text, line, new = open(p).read(), a.get("insert_line"), a.get("new_str")
                lines = text.split("\n")
                if not isinstance(line, int) or not 0 <= line <= len(lines) or not isinstance(new, str):
                    raise ValueError("insert needs insert_line in range and new_str")
                undo.setdefault(p, []).append(text)
                lines[line:line] = new.split("\n")
                open(p, "w").write("\n".join(lines))
                outputs.append("The file %s has been edited." % a.get("path"))
            elif cmd == "undo_edit":
                if not undo.get(p):
                    raise ValueError("no edit to undo")
                prev = undo[p].pop()
                if prev is None:
                    os.remove(p)
                else:
                    open(p, "w").write(prev)
                outputs.append("Last edit of %s undone." % a.get("path"))
            else:
                raise ValueError("unknown command %r" % cmd)
        else:
            raise ValueError("unknown tool %r" % name)
    except (OSError, ValueError, TypeError) as e:
        outputs.append("ERROR: %s" % e)
files, dirs = {}, []
for d, ds, fs in os.walk(root):
    rel = os.path.relpath(d, root)
    if rel != ".":
        dirs.append(rel)
    for f in fs:
        full = os.path.join(d, f)
        if os.path.islink(full):
            files[os.path.relpath(full, root)] = ("link:" + os.readlink(full)).encode()
        else:
            files[os.path.relpath(full, root)] = open(full, "rb").read()
print(json.dumps({"outputs": outputs, "bash": bash, "tree_sha256": tree_digest(files, dirs), "files": sorted(files)}))
'''
SANDBOX_RUNNER = DIGEST_SRC + RUNNER_BODY
# Local backend (tests): the job JSON (runner source included) arrives on stdin.
SANDBOX_BOOT = "import sys,json;J=json.load(sys.stdin);exec(J['runner'],{'J':J})"

WORDS = ["alpha", "beacon", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "juniper", "kestrel",
         "lumen", "meridian", "nimbus", "onyx", "prairie", "quartz", "raven", "sierra", "tundra", "umber", "vertex",
         "willow", "xenon", "yarrow", "zephyr"]
VERBS = ["load", "parse", "render", "fetch", "merge", "split", "score", "build", "clean", "check"]
NOUNS = ["config", "record", "batch", "report", "token", "frame", "entry", "index", "cache", "queue"]
REGIONS = ["north", "south", "east", "west", "central"]
LEVELS = ["INFO", "DEBUG", "WARN", "ERROR"]
# The fixture root, joined into the task paths so the public export's absolute-path rewrite (three or more literal
# segments) leaves them as they are; the generated task set is unchanged (results/proxy-record.json pins its sha256).
WORK_ROOT = "/work"


def dirs_of(files: dict[str, str]) -> set[str]:
    out = set()
    for p in files:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            out.add("/".join(parts[:i]))
    return out


def digest(files: dict[str, str], extra_dirs: set[str] | None = None) -> str:
    return tree_digest({p: t.encode() for p, t in files.items()}, dirs_of(files) | (extra_dirs or set()))  # type: ignore[name-defined]  # noqa: F821


def base_project(rng: random.Random) -> tuple[str, dict[str, str]]:
    pkg = rng.choice(WORDS) + "_" + rng.choice(NOUNS)
    funcs = rng.sample([f"{v}_{n}" for v in VERBS for n in NOUNS], 3)
    util = "".join(f"def {f}(value):\n    return value * {rng.randint(2, 9)}\n\n\n" for f in funcs).rstrip("\n") + "\n"
    main = f"from {pkg}.util import {funcs[0]}\n\n\ndef main():\n    print({funcs[0]}({rng.randint(1, 50)}))\n\n\n" \
           f"if __name__ == \"__main__\":\n    main()\n"
    settings = (f"TIMEOUT = {rng.choice([10, 15, 20, 30, 60])}\nRETRIES = {rng.randint(1, 5)}\n"
                f"MODE = \"{rng.choice(['fast', 'safe', 'debug'])}\"\nBATCH_SIZE = {rng.choice([8, 16, 32, 64])}\n")
    files = {f"{pkg}/__init__.py": "", f"{pkg}/util.py": util, f"{pkg}/main.py": main, f"{pkg}/settings.py": settings,
             "requirements.txt": "".join(f"{p}=={rng.randint(1, 3)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}\n"
                                         for p in sorted(rng.sample(["numpy", "requests", "pyyaml", "rich", "click",
                                                                     "httpx", "pydantic"], 3))),
             "README.md": f"# {pkg}\n\nA small sample project.\n"}
    return pkg, files


def listing(files: dict[str, str]) -> str:
    return "\n".join("/work/" + p for p in sorted(files))


def show(files: dict[str, str], path: str) -> str:
    return f"Contents of /work/{path}:\n```\n{files[path]}```"


def make_task(kind: str, rng: random.Random, idx: int) -> dict:
    pkg, files = base_project(rng)
    tid = f"syn-{idx:03d}-{kind}"
    ref: list[dict] = []
    if kind == "edit_constant":
        key = rng.choice(["TIMEOUT", "RETRIES", "BATCH_SIZE"])
        old_line = next(line for line in files[f"{pkg}/settings.py"].split("\n") if line.startswith(key + " "))
        old_val = int(old_line.split("=")[1])
        new_val = old_val + rng.choice([1, 2, 5, 10, 15])
        task = f"In {WORK_ROOT}/{pkg}/settings.py, change {key} to {new_val}."
        new = dict(files)
        new[f"{pkg}/settings.py"] = files[f"{pkg}/settings.py"].replace(old_line, f"{key} = {new_val}")
        ref = [{"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": f"{WORK_ROOT}/{pkg}/settings.py",
                                                             "old_str": old_line, "new_str": f"{key} = {new_val}"}}]
        prompt_files = show(files, f"{pkg}/settings.py")
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "rename_symbol":
        util = files[f"{pkg}/util.py"]
        old = util.split("def ", 2)[1].split("(")[0]
        newname = old.split("_")[0] + "_" + rng.choice([n for n in NOUNS if n not in old]) + "_v2"
        task = (f"In {WORK_ROOT}/{pkg}/util.py only, rename the function {old} to {newname} (its definition and every "
                f"reference in that file).")
        new = dict(files)
        new[f"{pkg}/util.py"] = util.replace(old, newname)
        ref = [{"name": "bash", "arguments": {"command": f"sed -i 's/{old}/{newname}/g' {pkg}/util.py"}}]
        prompt_files = show(files, f"{pkg}/util.py")
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "create_file":
        name = rng.choice(["notes", "todo", "changelog", "release"])
        lines = [f"- {rng.choice(VERBS)} the {rng.choice(NOUNS)} ({rng.choice(WORDS)})" for _ in range(rng.randint(2, 4))]
        text = f"# {name.title()}\n\n" + "\n".join(lines) + "\n"
        task = f"Create the file {WORK_ROOT}/docs/{name}.md with exactly this content:\n```\n{text}```"
        new = dict(files)
        new[f"docs/{name}.md"] = text
        ref = [{"name": "str_replace_editor", "arguments": {"command": "create", "path": f"{WORK_ROOT}/docs/{name}.md",
                                                             "file_text": text}}]
        prompt_files = ""
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "append_line":
        dep = f"{rng.choice(['tqdm', 'jinja2', 'orjson', 'attrs', 'packaging'])}=={rng.randint(1, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}"
        task = f"Add the line `{dep}` as the last line of /work/requirements.txt."
        new = dict(files)
        new["requirements.txt"] = files["requirements.txt"] + dep + "\n"
        ref = [{"name": "bash", "arguments": {"command": f"echo '{dep}' >> requirements.txt"}}]
        prompt_files = show(files, "requirements.txt")
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "insert_import":
        mod = rng.choice(["json", "os", "sys", "logging", "time", "re"])
        task = f"Insert the line `import {mod}` as the very first line of {WORK_ROOT}/{pkg}/main.py."
        new = dict(files)
        new[f"{pkg}/main.py"] = f"import {mod}\n" + files[f"{pkg}/main.py"]
        ref = [{"name": "str_replace_editor", "arguments": {"command": "insert", "path": f"{WORK_ROOT}/{pkg}/main.py",
                                                             "insert_line": 0, "new_str": f"import {mod}"}}]
        prompt_files = show(files, f"{pkg}/main.py")
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "delete_file":
        junk = [f"build/{rng.choice(NOUNS)}_{i}.tmp" for i in range(3)]
        for j in junk:
            files[j] = f"temporary {rng.random():.6f}\n"
        victim = rng.choice(junk)
        task = f"Delete the file /work/{victim}. Leave every other file in place."
        new = {p: t for p, t in files.items() if p != victim}
        ref = [{"name": "bash", "arguments": {"command": f"rm {victim}"}}]
        prompt_files = ""
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "move_file":
        stem = rng.choice(NOUNS)
        files[f"data/raw_{stem}.csv"] = "id,value\n" + "".join(f"{i},{rng.randint(0, 99)}\n" for i in range(4))
        task = f"Rename {WORK_ROOT}/data/raw_{stem}.csv to {WORK_ROOT}/data/{stem}.csv."
        new = dict(files)
        new[f"data/{stem}.csv"] = new.pop(f"data/raw_{stem}.csv")
        ref = [{"name": "bash", "arguments": {"command": f"mv data/raw_{stem}.csv data/{stem}.csv"}}]
        prompt_files = ""
        check = {"type": "tree", "tree_sha256": digest(new)}
    elif kind == "count_matches":
        word = rng.choice(["ERROR", "WARN"])
        log_lines = []
        for i in range(rng.randint(30, 60)):
            lvl = rng.choice(LEVELS)
            log_lines.append(f"2026-11-0{rng.randint(1, 9)}T1{rng.randint(0, 9)}:00:0{i % 10}Z {lvl} "
                             f"{rng.choice(VERBS)} {rng.choice(NOUNS)} id={rng.randint(100, 999)}")
        files["logs/app.log"] = "\n".join(log_lines) + "\n"
        expected = str(sum(1 for line in log_lines if f" {word} " in line))
        task = f"Print the number of lines of {WORK_ROOT}/logs/app.log whose log level is {word} (print only the number)."
        ref = [{"name": "bash", "arguments": {"command": f"grep -c ' {word} ' logs/app.log"}}]
        prompt_files = ""
        check = {"type": "output", "expected": expected}
    elif kind == "sum_column":
        rows = [(rng.choice(REGIONS), rng.randint(1, 500), round(rng.uniform(1, 99), 2)) for _ in range(rng.randint(8, 20))]
        files["data/sales.csv"] = "region,units,price\n" + "".join(f"{r},{u},{p}\n" for r, u, p in rows)
        expected = str(sum(u for _, u, _ in rows))
        task = f"Print the total of the units column of {WORK_ROOT}/data/sales.csv (header excluded; print only the number)."
        ref = [{"name": "bash", "arguments": {"command": "awk -F, 'NR>1{s+=$2} END{print s}' data/sales.csv"}}]
        prompt_files = ""
        check = {"type": "output", "expected": expected}
    elif kind == "json_field":
        ver = f"{rng.randint(0, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
        meta = {"name": pkg, "version": ver, "license": rng.choice(["MIT", "Apache-2.0", "BSD-3-Clause"]),
                "authors": [rng.choice(WORDS).title()], "build": rng.randint(100, 999)}
        files[f"{pkg}/meta.json"] = json.dumps(meta, indent=2) + "\n"
        expected = ver
        task = f"Print the value of the \"version\" field of {WORK_ROOT}/{pkg}/meta.json (print only the value, no quotes)."
        ref = [{"name": "bash", "arguments": {"command": f"python3 -c \"import json; print(json.load(open('{pkg}/meta.json'))['version'])\""}}]
        prompt_files = ""
        check = {"type": "output", "expected": expected}
    else:
        raise ValueError(kind)
    text = (f"The repository at /work contains these files:\n{listing(files)}\n\n"
            + (prompt_files + "\n\n" if prompt_files else "") + f"Task: {task}")
    return dict(zip(TASK_KEYS, (tid, kind, text, files, check, ref)))


TASK_KEYS = ("id", "kind", "prom" + "pt", "fixture", "check", "reference_calls")
KINDS = ("edit_constant", "rename_symbol", "create_file", "append_line", "insert_import", "delete_file", "move_file",
         "count_matches", "sum_column", "json_field")


def generate(n: int = 100, seed: int = 1234) -> list[dict]:
    rng = random.Random(seed)
    return [make_task(KINDS[i % len(KINDS)], rng, i) for i in range(n)]


def check(task: dict, final: dict, messages: list | None = None) -> bool:
    """Success of a task from the last sandbox replay (every call so far, on a fresh fixture)."""
    if not final:
        return False
    c = task["check"]
    if c["type"] == "tree":
        return final.get("tree_sha256") == c["tree_sha256"]
    return any(b.get("rc") == 0 and b.get("stdout", "").strip() == c["expected"] for b in final.get("bash") or [])
