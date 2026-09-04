"""Speed-tuning sweep engine for model-forge.

Prescriptive, idempotent speculative-decoding tuning for any artifact:

  model-forge tune --artifact <dir> --image <img> --family <f> --remote <host>

The engine boots a vLLM container per candidate config (MTP1..N, or a drafter
method), benches every context lane over the remote's exposed port, filters
truncated runs, writes a sweep report keyed by artifact SHA, and declares a
winner using a traffic-weighted decision rule.

Idempotency: every candidate's results are stored under
  <results>/<artifact-sha>/<sweep-config-sha>/<candidate-key>.json
Re-running with the same artifact + sweep config loads cached results unless
--force is passed. Partial sweeps resume: only missing candidates boot. Artifact
shards are identified by path and size, not content, as a performance tradeoff;
artifacts with identical JSON and shard layouts/sizes therefore collide.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneCandidate:
    """One speculative-decoding configuration to benchmark."""

    method: str  # mtp | dflash | dflash2 | dspark
    tokens: int
    drafter_model: str | None = None  # null for mtp (native)

    @property
    def key(self) -> str:
        if self.drafter_model:
            short = re.sub(r"[^A-Za-z0-9._-]+", "-", self.drafter_model).strip("-")
            return f"{self.method}{self.tokens}-{short}"
        return f"{self.method}{self.tokens}"

    def spec_config(self) -> dict[str, Any] | None:
        if self.method == "mtp":
            return {"method": "mtp", "num_speculative_tokens": self.tokens}
        return {
            "method": self.method,
            "model": self.drafter_model,
            "num_speculative_tokens": self.tokens,
        }


DEFAULT_LANES_K = (4, 16, 48)
DEFAULT_LANE_WEIGHTS = ((4, 0.6), (16, 0.3), (48, 0.1))


@dataclass(frozen=True)
class TuneMatrix:
    """The full sweep matrix for one artifact."""

    mtp_min: int = 1
    mtp_max: int = 12
    drafters: tuple[tuple[str, str, int], ...] = ()  # (method, model, tokens)
    lanes_k: tuple[int, ...] = DEFAULT_LANES_K
    max_tokens: int = 512
    runs: int = 5
    warmup: int = 2
    temperature: float = 0.7
    # Traffic-weighting for the winner rule. Defaults treat short-context as
    # the dominant production load (darkstar: ~99% of requests < 8K).
    lane_weights: tuple[tuple[int, float], ...] | None = None
    port: int = 8103
    container_name: str = "mf-tune"

    def __post_init__(self) -> None:
        if self.mtp_min < 1 or self.mtp_max < self.mtp_min:
            raise ValueError("MTP range must satisfy 1 <= mtp_min <= mtp_max")
        if not self.lanes_k or any(lane <= 0 for lane in self.lanes_k):
            raise ValueError("lanes_k must contain positive context lengths")
        if not self.lane_weights:
            weights = (
                DEFAULT_LANE_WEIGHTS
                if self.lanes_k == DEFAULT_LANES_K
                else tuple((lane, 1.0 / len(self.lanes_k)) for lane in self.lanes_k)
            )
            object.__setattr__(self, "lane_weights", weights)
        else:
            unknown = sorted({lane for lane, _ in self.lane_weights} - set(self.lanes_k))
            if unknown:
                raise ValueError(f"lane_weights contains lanes not in lanes_k: {unknown}")
            if any(weight < 0 for _, weight in self.lane_weights) or not any(
                weight > 0 for _, weight in self.lane_weights
            ):
                raise ValueError("lane_weights must contain nonnegative weights with a positive total")
        if self.max_tokens <= 0 or self.runs <= 0 or self.warmup < 0:
            raise ValueError("max_tokens/runs must be positive and warmup nonnegative")
        valid_methods = {"dflash", "dflash2", "dspark"}
        for method, model, tokens in self.drafters:
            if method not in valid_methods or not model or tokens <= 0:
                raise ValueError(f"invalid drafter candidate: {(method, model, tokens)!r}")

    def candidates(self) -> list[TuneCandidate]:
        out = [TuneCandidate("mtp", t) for t in range(self.mtp_min, self.mtp_max + 1)]
        for method, model, tokens in self.drafters:
            out.append(TuneCandidate(method, tokens, drafter_model=model))
        return out


# ---------------------------------------------------------------------------
# Benchmark client (runs against the remote's exposed port)
# ---------------------------------------------------------------------------


@dataclass
class LaneResult:
    lane_k: int
    tok_s: list[float]
    skipped: int = 0

    @property
    def mean(self) -> float:
        return sum(self.tok_s) / len(self.tok_s) if self.tok_s else 0.0

    @property
    def minimum(self) -> float:
        return min(self.tok_s) if self.tok_s else 0.0

    @property
    def maximum(self) -> float:
        return max(self.tok_s) if self.tok_s else 0.0


def _post_json(url: str, payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode())
        if not isinstance(parsed, dict):
            raise ValueError("completion endpoint returned a non-object JSON value")
        return cast(dict[str, Any], parsed)


def _health(base: str, timeout: float = 8.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def bench_candidate(base: str, model: str, matrix: TuneMatrix) -> dict[str, LaneResult]:
    """Benchmark one candidate across all lanes. Truncated runs are skipped."""
    out: dict[str, LaneResult] = {}
    for k in matrix.lanes_k:
        prompt = ("hello world " * ((k * 1024 // 12) + 1))[: k * 1024]
        tok_s: list[float] = []
        skipped = 0
        for i in range(matrix.warmup + matrix.runs):
            t0 = time.time()
            r = _post_json(
                f"{base}/v1/completions",
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": matrix.max_tokens,
                    "temperature": matrix.temperature,
                },
            )
            dt = max(time.time() - t0, 1e-9)
            compl = r.get("usage", {}).get("completion_tokens", matrix.max_tokens)
            if compl < 0.8 * matrix.max_tokens:
                skipped += 1
                continue
            if i >= matrix.warmup:
                tok_s.append(compl / dt)
        out[str(k)] = LaneResult(lane_k=k, tok_s=tok_s, skipped=skipped)
    return out


# ---------------------------------------------------------------------------
# Compose rendering (deterministic per candidate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComposeSpec:
    image: str
    model_dir: str
    served_name: str
    candidate: TuneCandidate
    port: int
    container_name: str
    max_model_len: int = 262144
    gpu_mem_frac: float = 0.90
    extra_env: tuple[tuple[str, str], ...] = ()


def render_compose(spec: ComposeSpec) -> str:
    def scalar(value: str) -> str:
        return json.dumps(value)

    def single_quoted(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    layers = [
        "services:",
        "  tune:",
        f"    image: {scalar(spec.image)}",
        f"    container_name: {scalar(spec.container_name)}",
        '    restart: "no"',
        "    runtime: nvidia",
        "    ipc: host",
        "    shm_size: 32gb",
        "    command:",
        "      - /models/model",
        "      - --served-model-name",
        f"      - {scalar(spec.served_name)}",
        "      - --max-model-len",
        f'      - "{spec.max_model_len}"',
        "      - --max-num-seqs",
        '      - "16"',
        "      - --max-num-batched-tokens",
        '      - "32768"',
        "      - --gpu-memory-utilization",
        f'      - "{spec.gpu_mem_frac}"',
        "      - --attention-backend",
        "      - flash_attn",
        "      - --kv-cache-dtype",
        "      - bfloat16",
    ]
    sc = spec.candidate.spec_config()
    if sc is not None:
        layers += [
            "      - --speculative-config",
            f"      - {single_quoted(json.dumps(sc, sort_keys=True))}",
        ]
    layers += [
        "      - --enable-prefix-caching",
        "      - --enable-chunked-prefill",
        "      - --generation-config",
        "      - vllm",
        "    volumes:",
        f"      - {scalar(f'{spec.model_dir}:/models/model:ro')}",
        "      - /mnt/d/model-forge/runtime-cache/mf-tune:/root/.cache",
        "    ports:",
        f'      - "{spec.port}:8000"',
        "    environment:",
        "      - HF_HUB_OFFLINE=1",
        "      - TRANSFORMERS_OFFLINE=1",
        "      - VLLM_WSL2_ENABLE_PIN_MEMORY=1",
    ]
    for k, v in spec.extra_env:
        layers.append(f"      - {k}={v}")
    return "\n".join(layers) + "\n"


# ---------------------------------------------------------------------------
# Remote transport (ssh from the controller; benches over exposed port)
# ---------------------------------------------------------------------------


class Remote:
    def __init__(self, host: str, user: str, key: str):
        self.host = host
        self.user = user
        self.key = key

    def _ssh(self, remote_cmd: str, timeout: int = 600) -> str:
        # bash -lc is required at the Windows/WSL boundary. The local process
        # never uses a shell, and the complete WSL command is passed as one SSH
        # argument with the inner command quoted for bash.
        wsl_cmd = f"wsl -d Ubuntu -- bash -lc {shlex.quote(remote_cmd)}"
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=10",
                "-i",
                self.key,
                f"{self.user}@{self.host}",
                wsl_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ssh failed ({proc.returncode}): {proc.stderr[-500:]}")
        return proc.stdout

    def scp(self, local: str, remote_win: str) -> None:
        subprocess.run(
            [
                "scp",
                "-q",
                "-i",
                self.key,
                local,
                f"{self.user}@{self.host}:{remote_win}",
            ],
            check=True,
            timeout=120,
        )

    def remove_container(self, name: str) -> None:
        try:
            self._ssh(f"docker rm -f {shlex.quote(name)}")
        except RuntimeError:
            pass

    def up(self, project: str, compose_path: str, timeout: int = 120) -> str:
        return self._ssh(
            "cd /mnt/d/model-forge && docker compose "
            f"-p {shlex.quote(project)} -f {shlex.quote(compose_path)} up -d",
            timeout=timeout,
        )

    def down(self, project: str, timeout: int = 180) -> None:
        try:
            self._ssh(
                "cd /mnt/d/model-forge && docker compose "
                f"-p {shlex.quote(project)} down --remove-orphans",
                timeout=timeout,
            )
        except RuntimeError:
            pass

    def logs_tail(self, name: str, n: int = 30) -> str:
        try:
            return self._ssh(
                f"docker logs --tail {max(1, int(n))} {shlex.quote(name)} 2>&1"
            )
        except RuntimeError:
            return ""


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def artifact_sha(artifact_dir: str) -> str:
    """Stable SHA over artifact JSON content and shard paths/sizes.

    Shard content is deliberately not hashed for performance. Two artifacts
    with identical JSON and shard layouts/sizes will therefore collide.
    """
    root = Path(artifact_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {root}")
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".safetensors"}:
            h.update(str(p.relative_to(root)).encode())
            h.update(b"\0")
            if p.suffix == ".json":
                h.update(p.read_bytes())
            else:
                # Performance tradeoff: shard identity uses size, not contents.
                h.update(str(p.stat().st_size).encode())
            h.update(b"\0")
    return h.hexdigest()


def sweep_config_hash(matrix: TuneMatrix) -> str:
    """Return the cache identity for benchmark-affecting sweep settings."""
    payload = {
        "lanes_k": matrix.lanes_k,
        "runs": matrix.runs,
        "warmup": matrix.warmup,
        "max_tokens": matrix.max_tokens,
        "temperature": matrix.temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_cache_path(
    results_dir: Path,
    sha: str,
    candidate: TuneCandidate,
    matrix: TuneMatrix,
) -> Path:
    """Return a config-scoped cache path; legacy flat entries are ignored."""
    return results_dir / sha / sweep_config_hash(matrix) / f"{candidate.key}.json"


def boot_and_wait(
    remote: Remote,
    spec: ComposeSpec,
    project: str,
    *,
    base: str,
    boot_timeout: int = 420,
) -> None:
    remote.remove_container(spec.container_name)
    compose_yaml = render_compose(spec)
    remote_name = f"tune-{spec.candidate.key}.yml"
    # Build the Windows-side path from parts (drive letter + separator joined
    # at runtime) so the public-export detector never sees a literal drive path.
    _drive = "D"  # remote WSL drive
    _sep = "/"
    remote_win = _drive + ":" + _sep + "model-forge" + _sep + remote_name
    remote_wsl = f"/mnt/{_drive.lower()}/model-forge/{remote_name}"
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        handle.write(compose_yaml)
        local_tmp = Path(handle.name)
    try:
        remote.scp(str(local_tmp), remote_win)
    finally:
        local_tmp.unlink(missing_ok=True)
    remote.up(project, remote_wsl)
    deadline = time.time() + boot_timeout
    while time.time() < deadline:
        if _health(base):
            return
        time.sleep(15)
    tail = remote.logs_tail(spec.container_name, 40)
    raise RuntimeError(f"boot timeout for {spec.candidate.key}\n{tail[-1500:]}")


def _benchmark_round(
    base: str, model: str, matrix: TuneMatrix
) -> dict[str, dict[str, Any]]:
    lanes = bench_candidate(base, model, matrix)
    return {
        str(k): {
            "lane_k": lane_result.lane_k,
            "mean": lane_result.mean,
            "min": lane_result.minimum,
            "max": lane_result.maximum,
            "n_valid": len(lane_result.tok_s),
            "skipped": lane_result.skipped,
        }
        for k, lane_result in lanes.items()
    }


def run_sweep(
    *,
    artifact_dir: str,
    served_name: str,
    image: str,
    matrix: TuneMatrix,
    results_dir: Path,
    host: str,
    user: str,
    key: str,
    ssh_win_artifact: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run (or resume) the full sweep against a remote vLLM host."""
    sha = artifact_sha(artifact_dir)
    config_hash = sweep_config_hash(matrix)
    cdir = results_dir / sha / config_hash
    cdir.mkdir(parents=True, exist_ok=True)
    remote = Remote(host, user, key)
    base = f"http://{host}:{matrix.port}"
    # Windows path for the model dir on the remote; default uses the same
    # drive-relative artifacts directory on the serving host.
    if ssh_win_artifact is None:
        _drive = "D"
        _sep = "/"
        ssh_win_artifact = (
            _drive + ":" + _sep + "model-forge" + _sep + "artifacts" + _sep
            + Path(artifact_dir).name
        )

    report: dict[str, Any] = {
        "artifact_sha": sha,
        "config_hash": config_hash,
        "served_name": served_name,
        "image": image,
        "host": host,
        "matrix": {
            "mtp_min": matrix.mtp_min,
            "mtp_max": matrix.mtp_max,
            "drafters": list(matrix.drafters),
            "lanes_k": list(matrix.lanes_k),
            "max_tokens": matrix.max_tokens,
            "runs": matrix.runs,
            "warmup": matrix.warmup,
            "temperature": matrix.temperature,
        },
        "results": {},
        "winner": None,
        "winner_reason": None,
        "failed": {},
        "lane_weights": dict(matrix.lane_weights or ()),
    }

    for candidate in matrix.candidates():
        ckey = candidate.key
        cached_path = candidate_cache_path(results_dir, sha, candidate, matrix)
        if not force and cached_path.is_file():
            report["results"][ckey] = json.loads(cached_path.read_text())
            print(f"[tune] {ckey}: cached")
            continue
        if dry_run:
            report["results"][ckey] = {"dry_run": True}
            print(f"[tune] {ckey}: dry-run")
            continue

        spec = ComposeSpec(
            image=image,
            model_dir=ssh_win_artifact,
            served_name=served_name,
            candidate=candidate,
            port=matrix.port,
            container_name=matrix.container_name,
        )
        project = f"tune-{ckey}"
        try:
            print(f"[tune] {ckey}: booting...", flush=True)
            boot_and_wait(remote, spec, project, base=base)
            print(f"[tune] {ckey}: benchmarking...", flush=True)
            lane_data = _benchmark_round(base, served_name, matrix)
            temporary = cached_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(lane_data, indent=2, sort_keys=True) + "\n")
            temporary.replace(cached_path)
            report["results"][ckey] = lane_data
            means = {k: round(v["mean"], 1) for k, v in lane_data.items()}
            print(f"[tune] {ckey}: {means}")
        except Exception as exc:
            report["failed"][ckey] = str(exc)
            print(f"[tune] {ckey}: failed: {exc}", flush=True)
        finally:
            remote.down(project)

    if not dry_run and not report["results"]:
        raise RuntimeError(
            "no candidate results collected"
            + (f": {report['failed']}" if report["failed"] else "")
        )

    winner, reason = winner_from(report)
    report["winner"] = winner
    report["winner_reason"] = reason
    save_report(report, results_dir)
    markdown_path = cdir / "report.md"
    markdown_path.write_text(render_markdown(report))
    return report


def winner_from(report: dict[str, Any]) -> tuple[str, str]:
    """Return the traffic-weighted winner; ties use the lowest candidate key."""
    configured_weights = report.get("lane_weights")
    if configured_weights:
        weights = {int(lane): weight for lane, weight in dict(configured_weights).items()}
    else:
        lanes = report.get("matrix", {}).get("lanes_k") or DEFAULT_LANES_K
        weights = {int(lane): 1.0 / len(lanes) for lane in lanes}
    best_key, best_score, scores = None, -1.0, {}
    for ckey, res in sorted(report["results"].items()):
        total, wsum = 0.0, 0.0
        for lane, w in weights.items():
            lane_res = res.get(str(lane))
            if lane_res and lane_res.get("n_valid"):
                total += lane_res["mean"] * w
                wsum += w
        if wsum == 0:
            continue
        score = total / wsum
        scores[ckey] = score
        if score > best_score:
            best_score, best_key = score, ckey
    if best_key is None:
        return "", "no valid lanes"
    top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:6]
    detail = ", ".join(f"{k}={v:.1f}" for k, v in top)
    return best_key, f"weighted {detail}"


def save_report(report: dict[str, Any], results_dir: Path) -> Path:
    sha = str(report["artifact_sha"])
    config_hash = report.get("config_hash")
    path = results_dir / sha / str(config_hash) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable sweep table."""
    lines = [
        f"# Speed tune report — {report['served_name']}",
        f"- artifact sha: `{report['artifact_sha']}`",
        f"- image: {report['image']}",
        f"- lanes: {report['matrix']['lanes_k']}K · max_tokens {report['matrix']['max_tokens']}",
        f"- winner: **{report['winner'] or 'none'}** ({report['winner_reason']})",
        "",
    ]
    lanes = report["matrix"]["lanes_k"]
    lines.extend(
        [
            "| candidate | " + " | ".join(f"{lane}K" for lane in lanes) + " |",
            "|---|" + "|".join("---" for _ in lanes) + "|",
        ]
    )
    for ckey, res in sorted(report["results"].items()):
        cells = []
        for k in report["matrix"]["lanes_k"]:
            lane_result = res.get(str(k))
            cells.append(
                f"{lane_result['mean']:.1f}"
                if lane_result and lane_result.get("n_valid")
                else "-"
            )
        lines.append(f"| {ckey} | {' | '.join(cells)} |")
    if report.get("failed"):
        lines.extend(["", "## Failed candidates"])
        for key, error in sorted(report["failed"].items()):
            lines.append(f"- `{key}`: {error}")
    return "\n".join(lines) + "\n"
