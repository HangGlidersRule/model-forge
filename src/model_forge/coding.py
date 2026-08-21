from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeResult:
    passed: bool
    timed_out: bool
    stdout: str
    stderr: str


def run_python_tests(code: str, tests: str, *, timeout: float = 3.0) -> CodeResult:
    with tempfile.TemporaryDirectory(prefix="model-forge-") as directory:
        path = Path(directory) / "submission.py"
        path.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            completed = subprocess.run([sys.executable, "-I", str(path)], cwd=directory, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            return CodeResult(False, True, stdout, stderr)
        return CodeResult(completed.returncode == 0, False, completed.stdout, completed.stderr)
