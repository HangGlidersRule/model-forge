"""Publication eval sets must score the recorded dataset revisions, not a live tip.

The reranker and embedding builders ask the dataset API for a commit id and accept it
when the id merely starts with a 12-hex-character prefix, then download corpus, queries,
and qrels from that commit and score them. These tests fail while that is true.

A fix has to download the full revisions recorded in
``models/qwen3-embedding-8b/results/retrieval-benchmark.json`` and keep that choice if
the dataset API is consulted again before the download. Refusal is accepted only when
no eval file was written. After the fix, a weak ranker stays under the embedding ship
bar (98% of all-BF16 nDCG@10) and under a perfect reranker nDCG@10.
"""

from __future__ import annotations

import email.message
import importlib.util
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]
_RERANK_EVAL = _REPO / "models" / "qwen3-reranker-4b" / "eval" / "reranker_eval.py"
_BM25_EVAL = _REPO / "models" / "qwen3-reranker-4b" / "eval" / "build_evalset_bm25.py"
_RETRIEVAL_EVAL = _REPO / "models" / "qwen3-embedding-8b" / "eval" / "retrieval_benchmark.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reranker_eval = _load("reranker_eval", _RERANK_EVAL)
build_evalset_bm25 = _load("build_evalset_bm25", _BM25_EVAL)
retrieval_benchmark = _load("retrieval_benchmark", _RETRIEVAL_EVAL)

# Full commit ids already recorded for these six sets. The builders store only
# the first 12 hex characters of each id.
PINNED = {
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140fb9af7c75ad0fdbf6350aaf20e94e",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793cd2d4e61bba0dd08fc58de90f3681",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b6febe1495eae9f2ca3eddad074b8",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f6b85dc0622f3d630ed7a1758e9ec",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb865f46c67710221afa7d9096251b",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65d6883d85c8f5cdc88e679633595e",
}
COLLIDING = {repo: sha[:12] + ("f" * 28) for repo, sha in PINNED.items()}
HONEST_QUERY = "alpha beta HONEST_EVALSET"
POISON_QUERY = "alpha beta POISONED_EVALSET"
HONEST_DOC = "HONEST_PASSAGE zeta"
DECOY_DOC = "alpha beta RANK_ME decoy"
EMBED_SHIP_BAR_PCT = 98.0
PARQUET_FILES = (
    "corpus/train-00000-of-00001.parquet",
    "queries/train-00000-of-00001.parquet",
    "qrels/train-00000-of-00001.parquet",
)


def _parquet(rows: list[dict[str, object]]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def _split(query: str, relevant: str) -> dict[str, bytes]:
    """Thirty documents so both builders keep every doc (one positive, 29 negatives)."""
    corpus = [{"_id": "d00", "title": "", "text": HONEST_DOC}]
    corpus.append({"_id": "d01", "title": "", "text": DECOY_DOC})
    corpus.extend({"_id": f"d{i:02d}", "title": "", "text": f"filler {i}"} for i in range(2, 30))
    queries = [{"_id": "q0", "title": "", "text": query}]
    qrels = [
        {"query-id": "q0", "corpus-id": doc_id, "score": 1 if doc_id == relevant else 0}
        for doc_id in ("d00", "d01")
    ]
    return {
        PARQUET_FILES[0]: _parquet(corpus),
        PARQUET_FILES[1]: _parquet(queries),
        PARQUET_FILES[2]: _parquet(qrels),
    }


HONEST_FILES = _split(HONEST_QUERY, "d00")
POISON_FILES = _split(POISON_QUERY, "d01")


class _Hub:
    """Stand-in for the Hub dataset endpoints these builders call.

    ``GET /api/datasets/{repo}`` with no revision returns the moving main-branch
    commit id (``sha``). ``GET /datasets/{repo}/resolve/{revision}/{file}`` returns
    that commit's file, after one redirect to the CDN host. Only ``PINNED`` serves
    the honest split. Every other revision, including a commit that keeps the
    12-character prefix, serves the poisoned split.
    """

    def __init__(self, mode: str) -> None:
        if mode not in ("collision", "reresolve"):
            raise ValueError(mode)
        self.mode = mode
        self.tip_reads: dict[str, int] = defaultdict(int)

    def tip(self, repo: str) -> str:
        self.tip_reads[repo] += 1
        if self.mode == "collision":
            return COLLIDING[repo]
        if self.tip_reads[repo] == 1:
            return PINNED[repo]
        return COLLIDING[repo]

    def files(self, revision: str, repo: str) -> dict[str, bytes]:
        if revision == PINNED[repo]:
            return HONEST_FILES
        return POISON_FILES


class _HubHandler(urllib.request.HTTPSHandler):
    def __init__(self, hub: _Hub) -> None:
        super().__init__()
        self.hub = hub

    def https_open(self, req: urllib.request.Request) -> Any:
        parsed = urlparse(req.full_url)
        path = unquote(parsed.path)
        api = _match_api(path)
        if api is not None:
            repo, revision = api
            sha = PINNED[repo] if revision == PINNED[repo] else self.hub.tip(repo)
            body = json.dumps({"id": repo, "sha": sha, "private": False}).encode()
            return _http_response(req, body, 200, "OK", "application/json")
        resolved = _match_resolve(path)
        if resolved is None:
            raise urllib.error.URLError(f"unexpected dataset URL {req.full_url}")
        repo, revision, filename = resolved
        if parsed.netloc == "huggingface.co":
            target = f"https://cdn.huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
            return _http_response(req, b"", 302, "Found", "text/plain", location=target)
        payload = self.hub.files(revision, repo)[filename]
        return _http_response(req, payload, 200, "OK", "application/octet-stream")


def _match_api(path: str) -> tuple[str, str | None] | None:
    prefix = "/api/datasets/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    marker = "/revision/"
    if marker in rest:
        repo, revision = rest.split(marker, 1)
        return repo, revision
    return rest, None


def _match_resolve(path: str) -> tuple[str, str, str] | None:
    prefix = "/datasets/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    marker = "/resolve/"
    if marker not in rest:
        return None
    repo, after = rest.split(marker, 1)
    revision, filename = after.split("/", 1)
    return repo, revision, filename


def _http_response(
    req: urllib.request.Request,
    body: bytes,
    code: int,
    msg: str,
    content_type: str,
    location: str | None = None,
) -> Any:
    headers = email.message.Message()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(body))
    if location is not None:
        headers["Location"] = location
    response = urllib.response.addinfourl(io.BytesIO(body), headers, req.full_url, code)
    response.msg = msg
    return response


@contextmanager
def _hub(mode: str) -> Iterator[_Hub]:
    hub = _Hub(mode)
    previous = urllib.request._opener
    urllib.request.install_opener(urllib.request.build_opener(_HubHandler(hub)))
    try:
        yield hub
    finally:
        urllib.request._opener = previous


class _ScoreHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        scores = [1.0 if "RANK_ME" in doc else 0.0 for doc in body["text_2"]]
        payload = json.dumps(
            {"data": [{"index": i, "score": score} for i, score in enumerate(scores)]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _score_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _refused(out: Path, exc: BaseException) -> None:
    assert not out.exists(), f"{out} was written before {type(exc).__name__}: {exc}"


def _run_bm25(out: Path) -> None:
    previous = sys.argv
    sys.argv = ["build_evalset_bm25.py", str(out)]
    try:
        rc = build_evalset_bm25.main()
    finally:
        sys.argv = previous
    if rc != 0:
        raise RuntimeError(f"build_evalset_bm25.main returned {rc}")


def _assert_honest_reranker(blob: dict[str, Any], base: str) -> None:
    scored = reranker_eval.quality(base, "weak-reranker", blob)
    for repo, row in scored.items():
        assert row["ndcg10"] < 1.0, (repo, row["ndcg10"])
    for repo, sha in PINNED.items():
        assert blob["dataset_shas"][repo] == sha
    assert "POISONED_EVALSET" not in json.dumps(blob)


def _assert_honest_embedding(data: Path, out_dir: Path) -> None:
    blob = json.loads(data.read_text())
    _write_embeddings(blob, out_dir)
    retrieval_benchmark.score(data, out_dir)
    bench = json.loads((out_dir / "benchmark.json").read_text())
    pct = bench["dims"]["4096"]["all-weak"]["ndcg_vs_all_bf16_pct"]
    assert pct < EMBED_SHIP_BAR_PCT, pct
    for repo, sha in PINNED.items():
        assert blob["dataset_shas"][repo] == sha
    assert "POISONED_EVALSET" not in json.dumps(blob)


def _write_embeddings(blob: dict[str, Any], out_dir: Path) -> None:
    bf16: dict[str, Any] = {}
    weak: dict[str, Any] = {}
    for repo, split in blob["data"].items():
        docs = np.zeros((len(split["docs"]), 4), dtype=np.float32)
        for index, text in enumerate(split["docs"]):
            if "HONEST_PASSAGE" in text:
                docs[index, 0] = 1.0
            elif "RANK_ME" in text:
                docs[index, 1] = 1.0
            else:
                docs[index, 2] = 0.05
        queries_bf16 = np.zeros((len(split["queries"]), 4), dtype=np.float32)
        queries_weak = np.zeros((len(split["queries"]), 4), dtype=np.float32)
        queries_bf16[:, 0] = 1.0
        queries_weak[:, 1] = 1.0
        bf16[f"{repo}|docs"] = docs
        bf16[f"{repo}|queries"] = queries_bf16
        weak[f"{repo}|docs"] = docs.copy()
        weak[f"{repo}|queries"] = queries_weak
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "bf16.npz", **bf16)
    np.savez(out_dir / "weak.npz", **weak)


def _exercise_reranker(mode: str, build: Callable[[Path], None], out: Path) -> None:
    with _score_server() as base:
        with _hub(mode):
            try:
                build(out)
            except (Exception, SystemExit) as exc:
                _refused(out, exc)
                return
        _assert_honest_reranker(json.loads(out.read_text()), base)


def test_prefix_collision_cannot_flip_reranker_ndcg(tmp_path: Path) -> None:
    """A main-branch commit that only keeps the 12-character prefix must not be scored."""
    out = tmp_path / "random.json"
    _exercise_reranker("collision", reranker_eval.build_eval_set, out)


def test_prefix_collision_cannot_flip_bm25_eval_ndcg(tmp_path: Path) -> None:
    out = tmp_path / "bm25.json"
    _exercise_reranker("collision", _run_bm25, out)


def test_prefix_collision_cannot_clear_embedding_ship_bar(tmp_path: Path) -> None:
    data = tmp_path / "retrieval.json"
    with _hub("collision"):
        try:
            retrieval_benchmark.fetch(data)
        except (Exception, SystemExit) as exc:
            _refused(data, exc)
            return
    _assert_honest_embedding(data, tmp_path / "emb")


def test_reresolved_tip_cannot_replace_reranker_eval(tmp_path: Path) -> None:
    """The first tip may be the pinned commit. A later read of the same API must not win."""
    out = tmp_path / "random.json"
    _exercise_reranker("reresolve", reranker_eval.build_eval_set, out)


def test_reresolved_tip_cannot_replace_bm25_eval(tmp_path: Path) -> None:
    out = tmp_path / "bm25.json"
    _exercise_reranker("reresolve", _run_bm25, out)


def test_reresolved_tip_cannot_replace_embedding_benchmark(tmp_path: Path) -> None:
    data = tmp_path / "retrieval.json"
    with _hub("reresolve"):
        try:
            retrieval_benchmark.fetch(data)
        except (Exception, SystemExit) as exc:
            _refused(data, exc)
            return
    _assert_honest_embedding(data, tmp_path / "emb")
