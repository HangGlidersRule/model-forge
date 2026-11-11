"""Boundary tables for the export sanitizer's path exemptions and the detector's path rules (PR #96).

Review rounds 5, 6 and 7 each found the next finer mutant of these grammars that the hand-picked cases missed, and the
round-8 review found that the first version of this file still hand-picked its character sets. Here each grammar is
written down as data and the cases are generated from it:

- every alternative, and both ends of every range and every other member of each character class, at the start of an
  entry and after it, is KEPT;
- one look-alike just outside each component (another letter, a case change, no letter, a character the class does not
  allow, anything after the end, a second entry, the grammar matched mid-path) is REWRITTEN or FLAGGED.

Every character set that decides where a path starts is the EXHAUSTIVE one: all of printable ASCII plus tab, newline and
carriage return, split by the class under test into "continues" and "starts", plus the start of a line and the start of
the file. Names run from one character to 64, positionals from one digit to four. Loosening any component then
publishes a generated look-alike, and tightening one rewrites a generated member: either way a case fails. Host-path
fixtures are assembled from parts, so the exporter's own detector does not flag this file.
"""

from __future__ import annotations

import string
from pathlib import Path

import pytest

from model_forge.public_export.detectors import DetectorPolicy, GitleaksStatus, scan_file
from model_forge.public_export.transforms import (
    _PUBLIC_SYSTEM_PATH_PREFIXES,
    TransformContext,
    apply_transform,
)

S = "/"
BS = "\\"
USER = "operator"
UPPER, LOWER, DIGIT = "AZ", "az", "09"  # both ends of each range
PUBLIC_CONTACT = "security@example.com"
# Every character a path can follow: printable ASCII; tab, newline, carriage return, vertical tab and form feed; and
# non-ASCII samples that `\w`, `\d` or `\s` match in a str pattern while the ASCII classes under test do not: no-break,
# em and ideographic spaces, accented Latin, Cyrillic and CJK letters, and an Arabic-Indic digit.
ALL_CHARS = ("".join(chr(i) for i in range(0x20, 0x7F)) + "\t\n\r\x0b\x0c"
             + "\u00a0\u2003\u3000" + "\u00e9\u00df\u044f\u6a21\u0663")
WORD = string.ascii_letters + string.digits
# The generic rewrite's first-segment class and later-segment class, written as data.
FIRST_SEGMENT_CHARS = WORD + "._-"
LATER_SEGMENT_CHARS = FIRST_SEGMENT_CHARS + "${}"
# Segment lengths around every plausible cap, for first and later segments, roots and names.
LENGTHS = (1, 7, 8, 9, 12, 13, 16, 17, 24, 25, 30, 64)

KINDS = {
    "python": ("sanitize_python_script", "models/x/eval/c.py", 'A = "{}"'),
    "doc": ("sanitize_public_markdown", "docs/x.md", "X={}"),
    "shell": ("sanitize_container_script", "scripts/x/run.sh", "#!/bin/bash\nX={}"),
}


def _dedupe(rows: list[str]) -> list[str]:
    return list(dict.fromkeys(rows))


# Names a shell accepts in `${NAME...}`: both ends of each range and "_", as the first character and as a later one,
# names of 2, 13, 14, 30 and 64 characters, and positionals of one to 64 digits with both ends of the digit range.
LONG_NAMES = ["CHECKPOINT_ROO", "CHECKPOINT_ROOT_DIRECTORY_PATH", "model_forge_output_directory_for_the_long_export_" + "x" * 15]
NAMES = ([*UPPER, *LOWER, "_"] + ["x" + c for c in UPPER + LOWER + DIGIT + "_"] + ["ARTIFACT_PATH", *LONG_NAMES]
         + ["0", "9", "10", "99", "100", "999", "1234", "1" * 30, "9" * 64])


# --- _API_ROUTE: "/v" + one or more digits, then one or more segments. A segment starts with any route character but
# "." and continues with any route character; route characters are those the generic rewrite passes in a segment.
ROUTE_SEGMENT_START = UPPER + LOWER + DIGIT + "_${}-"
ROUTE_SEGMENT_REST = ROUTE_SEGMENT_START + "."


def _route_kept() -> list[str]:
    rows = [S + "v1" + S + c + "x" + S + "y" for c in ROUTE_SEGMENT_START]
    rows += [S + "v1" + S + "x" + c + "x" + S + "y" for c in ROUTE_SEGMENT_REST]
    rows += [S + "v" + n + S + "x" + S + "y" for n in ("0", "9", "12", "2024")]
    rows += [S + "v1" + S + "a" + S + "b" + S + "c" + S + "d"]
    # The identity guard matches whole, case-sensitive segments only, at any depth.
    rows += [S + "v1" + S + "users" + S + "{user_id}" + S + "keys", S + "v1" + S + "home2" + S + "x" + S + "y"]
    rows += [S + "v1" + S + "files" + S + "archive" + S + "homework" + S + "x",
             S + "v1" + S + "a" * 30 + S + "b" + S + "Users2" + S + "x"]
    return _dedupe(rows)


def _route_rewritten() -> list[str]:
    tail = S + USER + S + "x"
    tokens = [c + "1" for c in FIRST_SEGMENT_CHARS if c not in "v" + string.digits]  # any other first character
    tokens += ["1", "01", "2024"]  # no letter at all
    tokens += ["v", "v1.0"] + ["v1" + c for c in FIRST_SEGMENT_CHARS if c not in string.digits]  # anything after digits
    tokens += ["v" + c + "1" for c in FIRST_SEGMENT_CHARS + S if c not in string.digits]  # anything between v and digits
    rows = [S + t + tail for t in tokens]
    rows += [S + "v1" + S + "." + USER + S + "x"]  # a segment that starts with "."
    rows += [S + "v1" + S + "x" + S + ".." + S + USER + S + "x"]  # a ".." segment
    # A route mid-path is not a route, with one segment before it or more.
    rows += [S + "srv" + S + USER + S + "v1" + S + "x", S + "data" + S + "v1" + S + USER + S + "runs",
             S + USER + S + "v2" + S + "x" + S + "y"]
    return _dedupe(rows)


# An identity root anywhere in a route makes it a host path: at every depth after the version (0 to 8 segments of
# short and long names), followed by more segments, followed by the user only, and at the very end of the path.
ROUTE_FILLERS = ("x", "files", "chat", "a" * 30)


def _route_identity_at_depth() -> list[str]:
    rows = []
    for depth in range(9):
        for filler in ROUTE_FILLERS[: 1 if depth == 0 else len(ROUTE_FILLERS)]:
            before = S + "v1" + (S + filler) * depth
            for root in IDENTITY_ROOTS:
                rows += [before + S + root + S + USER + S + "x", before + S + root + S + USER]
    return _dedupe(rows)


# --- _KERNEL_PATH: one of the descriptor directories, then at most one entry: a number, `{name}` (dots allowed after
# the first character), `{}`, `$NAME` or `${NAME}`. A name starts with a letter or "_" and continues with letters,
# digits and "_".
KERNEL_DIRS = (("proc", "self", "fd"), ("proc", "thread-self", "fd"), ("dev", "fd"))
NAME_START = UPPER + LOWER + "_"
NAME_REST = UPPER + LOWER + DIGIT + "_"
BRACE_NAME_REST = NAME_REST + "."
KERNEL_BASE = S + "proc" + S + "self" + S + "fd" + S
# Characters that continue the generic path match but are neither digits nor "}".
CONTINUES = UPPER + LOWER + "._$" + "{" + "-"


def _kernel_kept() -> list[str]:
    entries = ["0", "9", "12", "123", "1023", "65535", "{}"]  # descriptor numbers of one to five digits
    entries += ["{" + c + "}" for c in NAME_START] + ["{x" + c + "}" for c in BRACE_NAME_REST]
    entries += ["$" + c for c in NAME_START] + ["$x" + c for c in NAME_REST]
    entries += ["${" + c + "}" for c in NAME_START] + ["${x" + c + "}" for c in NAME_REST]
    entries += ["{fd_in.x2}", "$FD_IN2", "${FD_IN2}"]  # names longer than two characters
    rows = [KERNEL_BASE + e for e in entries]
    for parts in KERNEL_DIRS:  # every directory, bare and with one entry of each kind
        rows += [S + S.join(parts)] + [S + S.join(parts) + S + e for e in ("3", "{fd}", "$FD", "${FD}")]
    return _dedupe(rows)


def _kernel_rewritten() -> list[str]:
    rows: list[str] = []
    for parts in KERNEL_DIRS:
        for i in range(len(parts)):  # each directory component replaced
            rows.append(S + S.join(parts[:i] + (USER,) + parts[i + 1 :]) + S + "3")
        rows.append(S + S.join(p.upper() for p in parts) + S + "3")  # the directories are lowercase
        rows.append(S + S.join(parts) + S + "3" + S + "4")  # a second entry
        rows.append(S + "srv" + S + USER + S + S.join(parts) + S + "3")  # the grammar mid-path (two segments before)
        rows.append(S + USER + S + S.join(parts) + S + "3")  # one segment before
    not_a_name_start = DIGIT + "-.$" + "{" + "}" + S
    not_in_a_name = "-.$" + "{" + "}" + S
    entries = ["3" + c + USER for c in UPPER + LOWER + "_-.$" + "{"]  # a number is digits only
    entries += [c for c in CONTINUES] + ["3" + c for c in CONTINUES] + [c + "3" for c in CONTINUES + "}"]
    entries += ["{" + c + USER + "}" for c in not_a_name_start]
    entries += ["{x" + c + USER + "}" for c in "-$" + "{" + S]
    entries += ["{x}" + c + USER for c in "A_.-$"] + ["{}" + c + USER for c in "A."]
    entries += ["$" + c + USER for c in not_a_name_start]
    entries += ["$x" + c + USER for c in not_in_a_name]
    entries += ["${" + c + USER + "}" for c in not_a_name_start]
    entries += ["${x" + c + USER + "}" for c in not_in_a_name]
    entries += ["${x}" + c + USER for c in "A_.-$"]
    rows += [KERNEL_BASE + e for e in entries]
    return _dedupe(rows)


# --- _PUBLIC_SYSTEM_PATH_PREFIXES: kept only at the start of a path.
PREFIX_KEPT = [p + "tool" + S + "x" + S + "y" for p in _PUBLIC_SYSTEM_PATH_PREFIXES]
PREFIX_REWRITTEN = [S + "data" + S + USER + p + "x" + S + "y" for p in _PUBLIC_SYSTEM_PATH_PREFIXES]
# A kept system path ends at the first character a later segment may not hold: an identity root right after it is its
# own path and is rewritten, after every such character of ALL_CHARS (an empty segment included).
IDENTITY_ROOTS = ("Users", "home", "Volumes")
SEGMENT_BREAKS = "".join(c for c in ALL_CHARS if c not in LATER_SEGMENT_CHARS)


def _prefix_then_identity() -> list[str]:
    first = _PUBLIC_SYSTEM_PATH_PREFIXES[0]
    # Characters that can sit inside a Python double-quoted literal on one line (the table runs in all three KINDS);
    # the rest of SEGMENT_BREAKS (a double quote, a backslash, the line breaks) are covered by the doc-only glue test.
    breaks = [c for c in SEGMENT_BREAKS if c not in "\"\\\r\n\x0b\x0c"]
    rows = [first + "x" + c + S + "Users" + S + USER + S + "x" for c in breaks]
    rows += [p + "x" + ":" + S + "Users" + S + USER + S + "x" for p in _PUBLIC_SYSTEM_PATH_PREFIXES]
    rows += [first + "x" + S + S + root + S + USER + S + "x" for root in IDENTITY_ROOTS]
    return _dedupe(rows)


# --- The generic rewrite starts a path at any "/" that does not continue a word, a relative path or an expansion: the
# complement of its lookbehind class over ALL_CHARS, plus the start of a line and the start of the file.
PATH_CONTINUES = WORD + "._:" + S + "}"
PATH_STARTS = "".join(c for c in ALL_CHARS if c not in PATH_CONTINUES)
# Its first segment may hold any of these, and every later segment these too.
FIRST_SEGMENT = UPPER + LOWER + DIGIT + "._-"
LATER_SEGMENT = FIRST_SEGMENT + "${}"


def _generic_rewritten() -> list[str]:
    rows = [S + c + USER + S + "x" + S + "y" for c in FIRST_SEGMENT]
    rows += [S + USER + c + S + "x" + S + "y" for c in FIRST_SEGMENT]
    rows += [S + "srv" + S + c + USER + S + "y" for c in LATER_SEGMENT]
    rows += [S + "srv" + S + USER + c + S + "y" for c in LATER_SEGMENT.replace("}", "")]
    rows += [S + "a" * n + S + USER + S + "x" + S + "y" for n in LENGTHS]  # a long first segment
    rows += [S + "srv" + S + "b" * n + S + USER + S + "y" for n in LENGTHS]  # a long later segment
    rows += [S + "srv" + S + USER + "c" * n + S + "y" for n in LENGTHS]  # a long segment holding the user
    rows += [S + S.join(["s"] * n) + S + USER for n in range(2, 13)]  # the user as the last of 3 to 13 segments
    return _dedupe(rows)


# A host path glued to a kept system path by any character that ends a later segment and starts a path, and one glued
# to a URL by any character of the token split before the `://` check (whitespace, quotes, a backtick, angle brackets).
SYSTEM_GLUES = "".join(c for c in ALL_CHARS if c not in LATER_SEGMENT_CHARS and c not in PATH_CONTINUES)
TOKEN_SPLIT = "\"'`<>"
URL_GLUES = "".join(c for c in ALL_CHARS if c in TOKEN_SPLIT or c.isspace())


# --- _SHELL_PRIVATE_DEFAULT (scripts): `${NAME<op>value}`, NAME an identifier or a positional, op one of `:-`, `:=`,
# `-`, `=`, the value optionally quoted and absolute (a root plus at least one name, a drive, or a UNC share), then
# anything up to the closing brace.
def _shell_defaults() -> list[str]:
    base = {"name": "OUT", "op": ":-", "quote": "", "value": S + "data" + S + USER}
    roots = [*UPPER, *LOWER, *DIGIT, ".", "_", "-", "data", "Data", "d2", "my_data", "my-data", ".data"]
    roots += ["r" * n for n in LENGTHS]  # roots of every plausible capped length
    values = [S + root + S + USER for root in roots]
    values += [S + USER + S + c for c in "x0._-$@: " + S]  # a single character after the root
    # Every character the value's tail may hold (all but "}" and line breaks), and a command substitution.
    values += [S + "data" + S + USER + c + "x" for c in ALL_CHARS if c not in "}\r\n"]
    values += [S + "data" + S + USER + "$(date +%F)"]
    values += [S + "data" + S + USER + S + "x"]
    values += [d + ":" + sep + USER for d in UPPER + LOWER for sep in (S, BS)]
    values += [BS + BS + "server" + BS + USER]
    axes = {
        "name": NAMES,
        "op": (":-", ":=", "-", "="),
        "quote": ("", '"', "'"),
        "value": values,
    }
    rows = []
    for axis, options in axes.items():
        for option in options:
            r = dict(base, **{axis: option})
            rows.append("${" + r["name"] + r["op"] + r["quote"] + r["value"] + r["quote"] + "}")
    return _dedupe(rows)


# Defaults that are not host paths stay as they are: relative, a bare name, or empty. Other expansions are not
# defaults at all: substitution (`${x//a/b}`, `${x/a/b}`), removal (`${x#..}`, `${x%..}`), alternate value (`${x:+..}`),
# error (`${x:?..}`) and substring (`${x:1}`), even when the text after the operator looks like a path. A root with no
# name after it (`/srv/`) is no host path either, and in a Python source the rewrite leaves it alone like the rest.
SHELL_DEFAULTS_KEPT = ["${OUT:-." + S + "data" + S + "x}", "${OUT:-data" + S + "x}", "${OUT:-x}", "${OUT:-}",
                       "${1:-." + S + "x}", "${OUT=x}", "${OUT:-" + S + "srv" + S + "}"]
SHELL_EXPANSIONS_KEPT = ["${family//_/-}", "${x//" + S + "srv" + S + "team" + S + "y}", "${x/" + S + "a/b}",
                         "${x#" + S + "srv" + S + "team}", "${x%" + S + "srv" + S + "team}",
                         "${x:+" + S + "srv" + S + "team}", "${x:?" + S + "srv" + S + "team}", "${x:1}", "${x: -1}"]


# --- detector default rule: `${NAME<op>` or `:<op>` before a host root (mnt, media, srv, opt, tmp, var; not the Homebrew
# prefix `/opt/homebrew` or anything under it), the value optionally quoted, the root followed by at least one name.
# The segment class `[^/\s\"'<>{}]` written as data: every character of ALL_CHARS outside it ends a segment, and every
# other one starts one.
DETECTOR_FILES = (("docs/run.md", "OUT={}\n"), ("containers/x/docker-compose.yml", "      - OUT={}\n"))
DEFAULT_SEGMENT_EXCLUDED = S + "\"'<>{}"
SEGMENT_STARTS = "".join(c for c in ALL_CHARS if c not in DEFAULT_SEGMENT_EXCLUDED and not c.isspace() and c.isprintable())
SEGMENT_STOPS = "".join(c for c in ALL_CHARS if (c in DEFAULT_SEGMENT_EXCLUDED or c.isspace())
                        and c not in "\r\n\x0b\x0c")


def _detector_defaults() -> list[str]:
    base = {"name": "OUT", "op": "-", "quote": "", "root": "srv", "rest": USER}  # without a colon the name decides
    axes = {
        "name": NAMES,
        "op": ("-", "=", ":-", ":="),
        "quote": ("", '"', "'"),
        "root": ("mnt", "media", "srv", "opt", "tmp", "var"),
        "rest": [c + USER for c in SEGMENT_STARTS] + ["homebrew" + S + USER, "team" + S + USER],
    }
    rows = []
    for axis, options in axes.items():
        for option in options:
            r = dict(base, **{axis: option})
            value = S + r["root"] + S + r["rest"]
            rows.append("${" + r["name"] + r["op"] + r["quote"] + value + r["quote"] + "}")
    rows.append("${PREFIX:-" + S + "opt" + S + "team" + S + USER + "}")  # /opt but not the Homebrew tree
    # A longer word than `homebrew` is no Homebrew prefix.
    rows += ["${PREFIX:-" + S + "opt" + S + "homebrew" + c + USER + "}" for c in SEGMENT_STARTS]
    return _dedupe(rows)


HOMEBREW = S + "opt" + S + "homebrew" + S
DETECTOR_PUBLIC = [
    "${PY:-" + HOMEBREW + "bin" + S + "python3}",
    "${LIB-" + HOMEBREW + "lib" + S + "x}",
    # The Homebrew prefix itself, however the value ends.
    "${HOMEBREW_PREFIX:-" + S + "opt" + S + "homebrew}",
    "${HOMEBREW_PREFIX:-" + HOMEBREW + "}",
    '${HOMEBREW_PREFIX:-"' + S + "opt" + S + 'homebrew"}',
    "${HOMEBREW_PREFIX-" + S + "opt" + S + "homebrew} x",
    "${RUN_ROOT:-" + S + "work" + S + "current-run}",  # container-local defaults stay public
    "${MODEL_DIR:-" + S + "models" + S + "current}",
    # Not a host default: relative, a host root as a later segment, a longer word, a root with no name after it.
    "${OUT:-." + S + "srv" + S + "x}",
    "${OUT:-srv" + S + "x}",
    "${OUT:-" + S + "work" + S + "srv" + S + "x}",
    "${OUT:-" + S + "srvx" + S + "y}",
    "${OUT:-" + S + "srv" + S + "}",
    "p:." + S + "opt" + S + "x",
    # A segment ends at "<" and ">": a placeholder (`/mnt/<drive>`) or a closing tag after the root is no host name.
    "${OUT:-" + S + "mnt" + S + "<drive>}",
    "${OUT:-" + S + "srv" + S + ">x}",
]
# Every character outside the segment class ends the segment, so a root followed by one names nothing.
DETECTOR_PUBLIC += ["${OUT:-" + S + "srv" + S + c + "x}" for c in SEGMENT_STOPS]
# The Homebrew prefix ends at the same characters: `/opt/homebrew` followed by any of them is the prefix itself.
DETECTOR_PUBLIC += ["${HOMEBREW_PREFIX:-" + S + "opt" + S + "homebrew" + c + "x}" for c in SEGMENT_STOPS if c != S]
# Only a default operator (`:-`, `:=`, `-`, `=`) makes a default: removal, alternate value, error and substitution
# expansions whose text after the operator is a host path are no defaults.
DETECTOR_PUBLIC += ["${x" + op + S + "srv" + S + "team}" for op in ("#", "##", "%", "%%", ":+", "+", ":?", "?", "^", ",")]
# --- detector identity roots: a /Users, /home or /Volumes path is flagged after any character that does not continue
# a word or a path: the complement of the lookbehind class over ALL_CHARS, plus the start of a line and of the file.
# (Vertical tab and form feed are left out of the detector tables: a file holding one is refused as binary before any
# text rule runs.)
def _is_text(c: str) -> bool:
    return not (ord(c) < 0x20 and c not in "\t\n\r")


IDENTITY_CONTINUES = WORD + "/:._"
IDENTITY_STARTS = "".join(c for c in ALL_CHARS if c not in IDENTITY_CONTINUES and _is_text(c))
# The segment after the root, `[^/\s"'<>]`, written as data: the rule fires on every character it admits.
IDENTITY_SEGMENT_STARTS = "".join(c for c in ALL_CHARS if c not in S + "\"'<>" and not c.isspace() and _is_text(c))
# --- a default's finding must start at the default, not inside an exempt `re.compile` literal before it, and two
# defaults back to back are two findings (the Python filter drops findings inside an exempt literal). Every host default
# is assembled from parts (`SRV`), so this file's own export scan does not read the fixtures as defaults.
SRV = S + "srv" + S
RECOMPILE_NEIGHBOURS = [
    'P = (re.compile("${A:-' + SRV + 'x"), "${C-' + SRV + USER + '}")\n',
    'P = (re.compile("${A:-' + SRV + 'x"),"${C-' + SRV + USER + '}")\n',
    "P = (re.compile('${A:-" + SRV + "x'),'${C-" + SRV + USER + "}')\n",
    'P = re.compile("${A-"' + SRV + USER + "/x)\n",
    "P = re.compile('${A-'" + SRV + USER + "/x)\n",
    'X = "${A-' + SRV + USER + '}""${B-' + SRV + USER + '}"\n',
    "X = '${A-" + SRV + USER + "}''${B-" + SRV + USER + "}'\n",
    # The key-value branch: its finding starts at the path too, not at the key inside the literal.
    'P = re.compile("root:"' + SRV + USER + "/x)\n",
    "P = re.compile('root:'" + SRV + USER + "/x)\n",
    'P = re.compile("path="' + SRV + USER + "/x)\n",
]


def _context(path: str) -> TransformContext:
    return TransformContext(source_path=path, source_sha="0" * 40, public_contact=PUBLIC_CONTACT)


def _apply(kind: str, value: str) -> tuple[str, str]:
    transform, path, template = KINDS[kind]
    line = template.format(value)
    out = apply_transform(transform, (line + "\n").encode(), _context(path)).data.decode()
    return line.splitlines()[-1], out


def _scan(tmp_path: Path, relative: str, text: str) -> set[str]:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    policy = DetectorPolicy(public_contacts=frozenset({PUBLIC_CONTACT}))
    return {f.rule_id for f in scan_file(tmp_path, relative, policy=policy, gitleaks_status=GitleaksStatus.PASSED)}


@pytest.mark.parametrize("kind", ["python", "doc", "shell"])
@pytest.mark.parametrize("value", _route_kept() + _kernel_kept() + PREFIX_KEPT)
def test_every_member_of_an_exemption_grammar_survives_export(kind: str, value: str) -> None:
    line, out = _apply(kind, value)
    assert line in out.splitlines()


@pytest.mark.parametrize("kind", ["python", "doc", "shell"])
@pytest.mark.parametrize(
    "value",
    _route_rewritten() + _route_identity_at_depth() + _kernel_rewritten() + PREFIX_REWRITTEN + _generic_rewritten()
    + _prefix_then_identity(),
)
def test_every_look_alike_of_an_exemption_grammar_is_rewritten(kind: str, value: str) -> None:
    _, out = _apply(kind, value)
    assert value not in out
    assert USER not in out


@pytest.mark.parametrize("in_default", [False, True])
@pytest.mark.parametrize("route", _route_identity_at_depth())
def test_a_compose_route_with_an_identity_root_at_any_depth_is_rewritten(route: str, in_default: bool) -> None:
    value = "${API:-" + route + "}" if in_default else route
    source = "services:\n  r:\n    environment:\n      - API=" + value + "\n"
    out = apply_transform("sanitize_and_validate_compose", source.encode(),
                          _context("containers/build/docker-compose.yml")).data.decode()
    assert USER not in out


@pytest.mark.parametrize("target", [S + "data" + S + USER + S + "x" + S + "y", S + "Users" + S + USER + S + "x"])
@pytest.mark.parametrize("glue", SYSTEM_GLUES)
def test_a_host_path_glued_to_a_kept_system_path_is_rewritten(glue: str, target: str) -> None:
    """A kept system path ends at the first character a later segment may not hold, so a host path glued after it by
    any such character (a quote, a backtick, a backslash, a line break, non-ASCII text) is still its own path."""
    line = "see " + _PUBLIC_SYSTEM_PATH_PREFIXES[2] + "tool" + glue + target + "\n"
    out = apply_transform("sanitize_public_markdown", line.encode(), _context("docs/x.md")).data.decode()
    assert USER not in out


@pytest.mark.parametrize("url_path", ["~" + S, "~" + S + "a" + S + "b"])
@pytest.mark.parametrize("glue", URL_GLUES)
def test_a_host_path_glued_to_a_url_is_rewritten(glue: str, url_path: str) -> None:
    """The `://` check keeps only the URL's own token: a host path after any token break is still rewritten."""
    line = "see http://example.org/" + url_path + glue + S + "data" + S + USER + S + "x" + S + "y\n"
    out = apply_transform("sanitize_public_markdown", line.encode(), _context("docs/x.md")).data.decode()
    assert USER not in out


@pytest.mark.parametrize("start", PATH_STARTS)
def test_a_path_after_any_non_word_character_is_rewritten(start: str) -> None:
    line = "see " + start + S + "srv" + S + USER + S + "x" + S + "y\n"
    out = apply_transform("sanitize_public_markdown", line.encode(), _context("docs/x.md")).data.decode()
    assert USER not in out


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("doc", S + "srv" + S + USER + S + "x" + S + "y\n"),  # the start of the file
        ("doc", "```\n" + S + "srv" + S + USER + S + "x" + S + "y\n```\n"),  # alone on a line (a code block)
        ("shell", "cat <<EOF\n" + S + "srv" + S + USER + S + "x" + S + "y\nEOF"),  # a heredoc line
        ("python", '"""\n' + S + "srv" + S + USER + S + "x" + S + 'y\n"""'),  # a triple-quoted string
    ],
)
def test_a_path_at_the_start_of_a_line_or_the_file_is_rewritten(kind: str, text: str) -> None:
    transform, path, _ = KINDS[kind]
    prefix = "#!/bin/bash\n" if kind == "shell" else ("A = " if kind == "python" else "")
    out = apply_transform(transform, (prefix + text + "\n").encode(), _context(path)).data.decode()
    assert USER not in out


@pytest.mark.parametrize("before", PATH_CONTINUES)
def test_a_path_that_continues_a_word_or_an_expansion_is_left_alone(before: str) -> None:
    """`models/a/b/c`, `./a/b/c`, `${ROOT}/a/b/c` and `https://host/a/b` are not absolute paths. (Two letters before the
    character, so `x:` is not read as a Windows drive.)"""
    line = "see xy" + before + S + "srv" + S + "team" + S + "x\n"
    out = apply_transform("sanitize_public_markdown", line.encode(), _context("docs/x.md")).data.decode()
    assert line in out


def test_a_url_earlier_on_the_line_does_not_exempt_a_later_path() -> None:
    """The `://` check keeps a URL's own path; a separate host path after it is still rewritten."""
    line = "See http://example.org/?root=" + S + " and the copy in " + S + "srv" + S + USER + S + "runs" + S + "x\n"
    out = apply_transform("sanitize_public_markdown", line.encode(), _context("docs/x.md")).data.decode()
    assert USER not in out


@pytest.mark.parametrize("expansion", _shell_defaults())
def test_every_shell_default_form_requires_an_argument(expansion: str) -> None:
    _, out = _apply("shell", '"' + expansion + '"')
    assert USER not in out
    assert ":?" in out


# Python sources and Dockerfiles take the same rewrite. Two-segment values only, so the generic path rewrite (which
# needs three) cannot hide a missing default rewrite.
SHORT_DEFAULTS = ["${OUT:-" + S + "r" * n + S + USER + "}" for n in LENGTHS] + ["${1:-" + S + "data" + S + USER + "}",
                                                                              "${OUT=" + S + "data" + S + USER + "}"]


@pytest.mark.parametrize("expansion", SHORT_DEFAULTS)
@pytest.mark.parametrize(
    ("transform", "path", "template"),
    [
        ("sanitize_python_script", "scripts/tool.py", 'A = "{}"\n'),
        ("sanitize_container_script", "containers/x/Dockerfile", "FROM example\nENV OUT={}\n"),
        ("sanitize_container_script", "scripts/x/run.sh", "#!/bin/bash\nX={}\n"),
    ],
)
def test_a_two_segment_default_requires_an_argument_in_every_script_kind(
    transform: str, path: str, template: str, expansion: str
) -> None:
    out = apply_transform(transform, template.format(expansion).encode(), _context(path)).data.decode()
    assert USER not in out
    assert ":?" in out


@pytest.mark.parametrize("root", IDENTITY_ROOTS)
@pytest.mark.parametrize(
    ("kind", "prefix"),
    [
        ("doc", "API="),
        ("compose", "      - API="),
    ],
)
def test_a_route_with_an_identity_root_inside_a_default_is_rewritten(kind: str, prefix: str, root: str) -> None:
    line = prefix + "${API:-" + S + "v1" + S + root + S + USER + S + "x}"
    if kind == "compose":
        source = "services:\n  r:\n    environment:\n" + line + "\n"
        out = apply_transform("sanitize_and_validate_compose", source.encode(),
                              _context("containers/build/docker-compose.yml")).data.decode()
    else:
        out = apply_transform("sanitize_public_markdown", (line + "\n").encode(), _context("docs/x.md")).data.decode()
    assert USER not in out


@pytest.mark.parametrize("expansion", SHELL_DEFAULTS_KEPT)
@pytest.mark.parametrize("kind", ["shell", "python"])
def test_shell_defaults_that_are_not_host_paths_are_kept(kind: str, expansion: str) -> None:
    line, out = _apply(kind, '"' + expansion + '"' if kind == "shell" else expansion)
    assert line in out.splitlines()


@pytest.mark.parametrize("kind", ["shell", "python"])
@pytest.mark.parametrize("sep", ["", " ", "/", ":"])
def test_two_defaults_side_by_side_are_two_required_arguments(kind: str, sep: str) -> None:
    """Each default is rewritten on its own: a value never runs past its closing brace into the next expansion."""
    pair = "${A-" + S + "srv" + S + USER + "}" + sep + "${B-" + S + "srv" + S + USER + "}"
    _, out = _apply(kind, '"' + pair + '"' if kind == "shell" else pair)
    assert USER not in out
    assert out.count(":?Set A") == 1 and out.count(":?Set B") == 1


def test_a_line_break_ends_a_default_value() -> None:
    """An unclosed `${NAME:-/x/y` never swallows the next line up to a later closing brace."""
    source = "#!/bin/bash\nX=\"${A:-" + S + "srv" + S + "team\necho done}\"\n"
    out = apply_transform("sanitize_container_script", source.encode(), _context("scripts/x/run.sh")).data.decode()
    assert "echo done}" in out
    assert ":?Set A" not in out


@pytest.mark.parametrize("expansion", SHELL_EXPANSIONS_KEPT)
def test_other_shell_expansions_are_not_turned_into_required_arguments(expansion: str) -> None:
    """Only a default operator (`:-`, `:=`, `-`, `=`) makes a required argument; a live script such as
    `scripts/verify_family_release.sh` (`${family//_/-}`) must keep its other expansions."""
    _, out = _apply("shell", '"' + expansion + '"')
    assert ":?Set" not in out and ":?Pass" not in out


@pytest.mark.parametrize(("relative", "template"), DETECTOR_FILES)
@pytest.mark.parametrize("expansion", _detector_defaults())
def test_detector_flags_every_host_default_form(tmp_path: Path, relative: str, template: str, expansion: str) -> None:
    assert "path.absolute-posix" in _scan(tmp_path, relative, template.format(expansion))


@pytest.mark.parametrize(("relative", "template"), DETECTOR_FILES)
@pytest.mark.parametrize("expansion", DETECTOR_PUBLIC)
def test_detector_keeps_public_defaults_public(tmp_path: Path, relative: str, template: str, expansion: str) -> None:
    assert "path.absolute-posix" not in _scan(tmp_path, relative, template.format(expansion))


@pytest.mark.parametrize("root", IDENTITY_ROOTS)
@pytest.mark.parametrize("start", IDENTITY_STARTS)
def test_detector_flags_an_identity_root_after_any_non_word_character(tmp_path: Path, root: str, start: str) -> None:
    text = "see " + start + S + root + S + USER + S + "x\n"
    assert "path.absolute-posix" in _scan(tmp_path, "docs/run.md", text)


@pytest.mark.parametrize("root", IDENTITY_ROOTS)
@pytest.mark.parametrize("first", IDENTITY_SEGMENT_STARTS)
def test_detector_flags_an_identity_root_whatever_the_next_segment_starts_with(
    tmp_path: Path, root: str, first: str
) -> None:
    text = "see " + S + root + S + first + USER + S + "x\n"
    assert "path.absolute-posix" in _scan(tmp_path, "docs/run.md", text)


@pytest.mark.parametrize("root", IDENTITY_ROOTS)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("tail", ["", S + "x", " and more"])
def test_detector_flags_an_identity_root_whatever_the_user_segment_length(
    tmp_path: Path, root: str, length: int, tail: str
) -> None:
    """A one-character user or volume name (`/home/a`, `/Volumes/D`) is as private as a long one."""
    text = "see " + S + root + S + "u" * length + tail + "\n"
    assert "path.absolute-posix" in _scan(tmp_path, "docs/run.md", text)


@pytest.mark.parametrize("root", IDENTITY_ROOTS)
@pytest.mark.parametrize("before", WORD + "._")
def test_detector_leaves_an_identity_root_that_continues_a_word_alone(tmp_path: Path, root: str, before: str) -> None:
    """`docs/home/x`, `v2.home/x` and `my_Users/x` are relative names, not host paths."""
    text = "see xy" + before + S + root + S + USER + S + "x\n"
    assert "path.absolute-posix" not in _scan(tmp_path, "docs/run.md", text)


@pytest.mark.parametrize("root", IDENTITY_ROOTS)
@pytest.mark.parametrize(
    ("relative", "template"),
    [
        ("docs/run.md", "{p}\n"),  # the start of the file
        ("docs/run.md", "intro\n{p}\n"),  # the start of a line
        ("docs/run.md", "Copy from `{p}` first.\n"),
        ("models/x/eval/defaults.py", "DEFAULT = '{p}'\n"),
        ("configs/x.json", '{{"source": "{p}"}}\n'),
    ],
)
def test_detector_flags_an_identity_root_at_a_line_start_and_in_quotes(
    tmp_path: Path, root: str, relative: str, template: str
) -> None:
    text = template.format(p=S + root + S + USER + S + "models")
    assert "path.absolute-posix" in _scan(tmp_path, relative, text)


@pytest.mark.parametrize("text", RECOMPILE_NEIGHBOURS)
def test_detector_flags_a_default_beside_an_exempt_pattern_literal(tmp_path: Path, text: str) -> None:
    """A default's finding starts at the default itself, so it is never inside the exempt `re.compile` literal before
    it, and a default directly after another one is still found."""
    assert "path.absolute-posix" in _scan(tmp_path, "models/x/eval/patterns.py", text)


@pytest.mark.parametrize(
    ("text", "start"),
    [
        ("OUT=${OUT-" + SRV + USER + "}\n", len("OUT=${OUT-")),
        ("x\n  - OUT=${OUT:-" + S + "var" + S + "a" + S + "b}\n", len("x\n  - OUT=${OUT:-")),
        ("root: " + SRV + USER + "\n", len("root: ")),
    ],
)
def test_a_path_finding_points_at_the_first_character_of_the_path(tmp_path: Path, text: str, start: int) -> None:
    """The finding's offset (from the start of the file) is where the path starts, its first "/", not where the
    match or the path ends."""
    policy = DetectorPolicy(public_contacts=frozenset({PUBLIC_CONTACT}))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "run.md").write_text(text, encoding="utf-8")
    found = scan_file(tmp_path, "docs/run.md", policy=policy, gitleaks_status=GitleaksStatus.PASSED)
    assert [f.offset for f in found if f.rule_id == "path.absolute-posix"] == [start]
