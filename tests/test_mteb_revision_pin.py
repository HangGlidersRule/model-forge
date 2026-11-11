"""Publication gates must freeze the six MTEB Nano sets from one exact commit each.

The authorized revisions are the commits already recorded for these prefixes in
``models/qwen3-embedding-8b/results/retrieval-benchmark.json``. A live dataset
commit that only shares the first 12 hex characters is a different revision.
``build_sets``, ``fetch``, ``build_eval_set``, and the BM25 rebuild must refuse
it before they download parquet or write an eval file. When the live commit is
the authorized one, a later lookup of the same dataset API must not be able to
swap in the other revision's corpus or qrels.

These tests fail while that prefix check is still what decides the download.
Run them with::

    python -m pytest tests/test_mteb_revision_pin.py -q

``pyarrow`` and ``numpy`` are imported by the eval scripts under test (numpy by
the embed and score steps). Install them with pytest if they are not already
present.

Not covered, because those paths do not resolve a dataset commit through the
12-character prefix: the LMEB download inside ``build_sets`` (it fetches one
full revision constant), ``embed_eval.speed``, ``embed_eval.refcheck``,
``retrieval_benchmark.embed``, ``retrieval_benchmark.score``, and the reranker
quality/throughput runners. Nemotron ``embed`` and ``score`` are covered as the
consumers of the frozen MTEB set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
LEGIT = {
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140fb9af7c75ad0fdbf6350aaf20e94e",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793cd2d4e61bba0dd08fc58de90f3681",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b6febe1495eae9f2ca3eddad074b8",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f6b85dc0622f3d630ed7a1758e9ec",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb865f46c67710221afa7d9096251b",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65d6883d85c8f5cdc88e679633595e",
}
SUFFIX = "deadbeefdeadbeefdeadbeefdead"
PARQUET_KIND = {
    "corpus/train-00000-of-00001.parquet": "corpus",
    "queries/train-00000-of-00001.parquet": "queries",
    "qrels/train-00000-of-00001.parquet": "qrels",
}
BLOB_PATH = re.compile(r"^/datasets/mteb/([^/]+)/resolve/([^/]+)/(.+)$")
ORIGINAL_URLOPEN = urllib.request.urlopen


def attacker(dataset: str) -> str:
    """A different 40-character commit that keeps the stored 12-character prefix."""
    colliding = LEGIT[dataset][:12] + SUFFIX
    if len(colliding) != 40 or colliding == LEGIT[dataset]:
        raise AssertionError(colliding)
    return colliding


def _parquet(rows: list[dict]) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def _corpus(kind: str) -> list[dict]:
    rows = []
    for index in range(30):
        if index == 0:
            text = "PINNED_DOC" if kind == "pinned" else "SUBSTITUTED_DOC"
        else:
            text = f"FILLER_DOC {index}"
        rows.append({"_id": f"d{index:02d}", "title": "", "text": text})
    return rows


def _queries() -> list[dict]:
    return [{"_id": "q0", "text": "which document answers the question"}]


def _qrels(kind: str) -> list[dict]:
    # Same judged pair. The substituted revision changes the grade and the passage text.
    score = 1 if kind == "pinned" else 7
    return [{"query-id": "q0", "corpus-id": "d00", "score": score}]


PINNED_BLOBS = {
    "corpus": _parquet(_corpus("pinned")),
    "queries": _parquet(_queries()),
    "qrels": _parquet(_qrels("pinned")),
}
SUBSTITUTED_BLOBS = {
    "corpus": _parquet(_corpus("substituted")),
    "queries": _parquet(_queries()),
    "qrels": _parquet(_qrels("substituted")),
}


def _roundtrip_markers() -> None:
    pinned = pq.read_table(io.BytesIO(PINNED_BLOBS["corpus"])).to_pylist()
    substituted = pq.read_table(io.BytesIO(SUBSTITUTED_BLOBS["qrels"])).to_pylist()
    if pinned[0]["text"] != "PINNED_DOC" or substituted[0]["score"] != 7:
        raise AssertionError((pinned[0], substituted[0]))


_roundtrip_markers()

# The builders also pin every parquet file by sha256. The real digests cannot match synthetic files, so the
# tables are pointed at the PINNED blobs' digests; the substituted blobs still fail that check.
FIXTURE_SPLIT_SHA256 = {kind: hashlib.sha256(blob).hexdigest() for kind, blob in PINNED_BLOBS.items()}


def _pin_fixture_digests(monkeypatch: pytest.MonkeyPatch, *modules) -> None:
    for holder in (*modules, sys.modules.get("reranker_eval")):
        for name in ("SPLIT_SHA256", "MTEB_SPLIT_SHA256"):
            if holder is not None and hasattr(holder, name):
                assert set(getattr(holder, name)) == set(LEGIT)
                pins = {ds: dict(FIXTURE_SPLIT_SHA256) for ds in LEGIT}
                monkeypatch.setattr(holder, name, pins)


def _load(module_name: str, relative: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _vector(text: str) -> list[float]:
    if "SUBSTITUTED_DOC" in text:
        return [-1.0, 0.0]
    if "PINNED_DOC" in text:
        return [1.0, 0.0]
    if "FILLER_DOC" in text:
        return [0.0, 1.0]
    if "which document answers" in text:
        return [1.0, 0.0]
    raise AssertionError(f"unexpected embed text {text!r}")


class _Hub(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, mode: str, lmeb_files: dict[str, bytes]):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mode = mode
        self.lmeb_files = lmeb_files
        self.api_calls: dict[str, int] = {}
        self.paths: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    server: _Hub

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        path = urllib.parse.urlsplit(self.path).path
        self.server.paths.append(path)
        body = self._body(path)
        if body is None:
            message = f"no fixture for {path}".encode()
            self._send(404, message, "text/plain")
            return
        media = (
            "application/json"
            if path.startswith("/api/") or path.endswith(".json")
            else "application/octet-stream"
        )
        self._send(200, body, media)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        vectors = [_vector(text) for text in request["input"]]
        payload = {
            "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]
        }
        self._send(200, json.dumps(payload).encode(), "application/json")

    def _body(self, path: str) -> bytes | None:
        if path.startswith("/api/datasets/"):
            dataset = path.removeprefix("/api/datasets/")
            if dataset not in LEGIT:
                return None
            seen = self.server.api_calls.get(dataset, 0) + 1
            self.server.api_calls[dataset] = seen
            if self.server.mode == "relookup" and seen == 1:
                sha = LEGIT[dataset]
            else:
                sha = attacker(dataset)
            return json.dumps({"sha": sha}).encode()
        blob = BLOB_PATH.fullmatch(path)
        if blob:
            dataset = f"mteb/{blob.group(1)}"
            sha = blob.group(2)
            kind = PARQUET_KIND.get(blob.group(3))
            if dataset not in LEGIT or kind is None:
                return None
            blobs = PINNED_BLOBS if sha == LEGIT[dataset] else SUBSTITUTED_BLOBS
            return blobs[kind]
        lmeb = "/datasets/KaLM-Embedding/LMEB/resolve/"
        if path.startswith(lmeb):
            rev, _, name = path.removeprefix(lmeb).partition("/")
            if name in self.server.lmeb_files and rev == self.server.lmeb_files["__rev__"].decode():
                return self.server.lmeb_files[name]
        return None

    def _send(self, status: int, body: bytes, media: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _WriteSpy:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self._text = Path.write_text
        self._bytes = Path.write_bytes

    def __enter__(self) -> _WriteSpy:
        def write_text(this: Path, data: str, *args, **kwargs):
            self.paths.append(str(this))
            return self._text(this, data, *args, **kwargs)

        def write_bytes(this: Path, data: bytes, *args, **kwargs):
            self.paths.append(str(this))
            return self._bytes(this, data, *args, **kwargs)

        Path.write_text = write_text  # type: ignore[method-assign]
        Path.write_bytes = write_bytes  # type: ignore[method-assign]
        return self

    def __exit__(self, *_) -> None:
        Path.write_text = self._text  # type: ignore[method-assign]
        Path.write_bytes = self._bytes  # type: ignore[method-assign]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)
    embed = _load("mf_embed_eval_hub", "models/nemotron-3.5-embed-8b/eval/embed_eval.py")
    lmeb_files = {
        "__rev__": embed.LMEB_REV.encode(),
        "task_instructions.json": b'{"memory": {"test": "retrieve the matching turn"}}',
        "Dialogue/memory/corpus.jsonl": b'{"id": "doc1", "title": "T", "text": "alpha"}\n',
        "Dialogue/memory/candidates.jsonl": (
            b'{"scene_id": "scene1", "candidate_doc_ids": ["doc1"]}\n'
        ),
    }
    servers: list[_Hub] = []

    def start(mode: str) -> _Hub:
        server = _Hub(mode, lmeb_files)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        def routed(
            url, data=None, timeout=None, *, context=None, cafile=None, capath=None, cadefault=False
        ):
            request = (
                url
                if isinstance(url, urllib.request.Request)
                else urllib.request.Request(url, data=data)
            )
            parts = urllib.parse.urlsplit(request.full_url)
            if parts.scheme == "https" and parts.hostname == "huggingface.co":
                rewritten = urllib.parse.urlunsplit(
                    ("http", f"127.0.0.1:{port}", parts.path, parts.query, "")
                )
                headers = dict(request.header_items())
                forwarded = urllib.request.Request(
                    rewritten, data=request.data, headers=headers, method=request.get_method()
                )
                return ORIGINAL_URLOPEN(forwarded, timeout=timeout)
            if parts.hostname == "127.0.0.1":
                return ORIGINAL_URLOPEN(url, data=data, timeout=timeout)
            raise AssertionError(f"unexpected URL {request.full_url}")

        monkeypatch.setattr(urllib.request, "urlopen", routed)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _modules():
    embed = _load("mf_embed_eval", "models/nemotron-3.5-embed-8b/eval/embed_eval.py")
    retrieval = _load(
        "mf_retrieval_benchmark", "models/qwen3-embedding-8b/eval/retrieval_benchmark.py"
    )
    reranker = _load("mf_reranker_eval", "models/qwen3-reranker-4b/eval/reranker_eval.py")
    bm25 = _load("mf_build_evalset_bm25", "models/qwen3-reranker-4b/eval/build_evalset_bm25.py")
    eval_dir = str(ROOT / "models/qwen3-reranker-4b/eval")
    while eval_dir in sys.path:
        sys.path.remove(eval_dir)
    return embed, retrieval, reranker, bm25


def _rerank_set(embed) -> dict:
    return {
        "source": f"{embed.LMEB}@{embed.LMEB_REV}",
        "data": {
            "memory/test": [
                {
                    "qid": "scene1_q_0",
                    "query": "what happened",
                    "cands": [{"id": "doc1", "rel": 1}],
                }
            ]
        },
    }


def _run(builder: str, directory: Path, embed, retrieval, reranker, bm25) -> list[Path]:
    if builder == "embed":
        rerank = directory / "rerank.json"
        rerank.write_text(json.dumps(_rerank_set(embed)))
        mteb = directory / "mteb.json"
        lmeb = directory / "lmeb.json"
        embed.build_sets(mteb, lmeb, rerank)
        return [mteb, lmeb]
    if builder == "retrieval":
        out = directory / "retrieval.json"
        retrieval.fetch(out)
        return [out]
    if builder == "reranker":
        out = directory / "reranker.json"
        reranker.build_eval_set(out)
        return [out]
    if builder == "bm25":
        out = directory / "bm25.json"
        previous = sys.argv
        sys.argv = ["build_evalset_bm25.py", str(out)]
        try:
            bm25.main()
        finally:
            sys.argv = previous
        return [out]
    raise AssertionError(builder)


def _mteb_blob_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if BLOB_PATH.fullmatch(path)]


def _publication_ndcg(embed, mteb: Path, directory: Path, base: str) -> float:
    embed.STYLE["name"] = "nemotron"
    vectors = directory / "vectors"
    embed.embed(base, "nemotron-embed", "nemotron", mteb, "nemotron-bf16", vectors)
    score_path = directory / "score.json"
    embed.score(mteb, vectors, score_path)
    return json.loads(score_path.read_text())["runs"]["nemotron-bf16"]["macro_ndcg10"]


def _authorized_problems(path: Path) -> list[str]:
    if not path.exists():
        return [f"{path.name} was not written"]
    text = path.read_text()
    problems = []
    if "SUBSTITUTED_DOC" in text:
        problems.append(f"{path.name} contains the substituted corpus")
    if '"d00": 7' in text:
        problems.append(f"{path.name} contains the substituted qrel grade")
    if "PINNED_DOC" not in text:
        problems.append(f"{path.name} is missing the pinned corpus")
    try:
        blob = json.loads(text)
    except json.JSONDecodeError as exc:
        return problems + [f"{path.name} is not the eval JSON ({exc})"]
    shas = blob.get("dataset_shas")
    if shas != LEGIT:
        problems.append(f"{path.name} dataset_shas={shas}")
    return problems


BUILDERS = ("embed", "retrieval", "reranker", "bm25")


def test_mteb_pins_are_the_full_authorized_commits():
    embed, retrieval, reranker, bm25 = _modules()
    assert embed.MTEB == LEGIT
    assert retrieval.DATASETS == LEGIT
    assert reranker.DATASETS == LEGIT
    assert bm25.DATASETS == LEGIT


@pytest.mark.parametrize("builder", BUILDERS)
def test_prefix_colliding_commit_is_not_downloaded_or_written(
    builder: str, tmp_path: Path, hub, monkeypatch: pytest.MonkeyPatch
):
    """A commit that only matches the 12-character prefix must not become the eval set.

    A builder may either read the dataset API and refuse the colliding commit before any
    download, or never read the API and download only the authorized commit.
    """
    server = hub("collision")
    embed, retrieval, reranker, bm25 = _modules()
    _pin_fixture_digests(monkeypatch, embed, retrieval, reranker, bm25)
    refused = False
    error: BaseException | None = None
    with _WriteSpy() as spy:
        try:
            outputs = _run(builder, tmp_path, embed, retrieval, reranker, bm25)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            refused = True
            error = exc
            outputs = {
                "embed": [tmp_path / "mteb.json", tmp_path / "lmeb.json"],
                "retrieval": [tmp_path / "retrieval.json"],
                "reranker": [tmp_path / "reranker.json"],
                "bm25": [tmp_path / "bm25.json"],
            }[builder]
    problems: list[str] = []
    blob_paths = _mteb_blob_paths(server.paths)
    for path in outputs:
        if path.exists():
            text = path.read_text()
            if "SUBSTITUTED_DOC" in text or any(sha in text for sha in map(attacker, LEGIT)):
                problems.append(f"{path.name} records the substituted revision")
    if server.api_calls:
        if blob_paths:
            problems.append(f"downloaded parquet after a non-matching commit: {blob_paths}")
        written = [path for path in spy.paths if Path(path).name in {item.name for item in outputs}]
        if written:
            problems.append(f"wrote an eval file before refusing: {written}")
        for path in outputs:
            if path.exists():
                problems.append(f"left {path.name} on disk")
        if not refused:
            problems.append("builder accepted the prefix-colliding commit")
    else:
        if refused:
            raise AssertionError("builder refused without reading a dataset revision") from error
        for path in blob_paths:
            match = BLOB_PATH.fullmatch(path)
            assert match is not None
            if match.group(2) != LEGIT[f"mteb/{match.group(1)}"]:
                problems.append(f"fetched {path} instead of the authorized commit")
        problems.extend(_authorized_problems(outputs[0]))
    ndcg = None
    mteb = tmp_path / "mteb.json"
    if builder == "embed" and mteb.exists():
        base = f"http://127.0.0.1:{server.server_address[1]}"
        ndcg = _publication_ndcg(embed, mteb, tmp_path, base)
        if ndcg != 1.0:
            problems.append(f"score() published nDCG@10 {ndcg} from the substituted set")
    if problems:
        pytest.fail(
            "prefix collision reached the frozen eval set\n"
            + "\n".join(problems)
            + (f"\nbuilder error: {error!r}" if error else "")
            + (f"\npublication nDCG@10={ndcg}" if ndcg is not None else "")
        )


@pytest.mark.parametrize("builder", BUILDERS)
def test_second_lookup_cannot_swap_the_authorized_revision(
    builder: str, tmp_path: Path, hub, monkeypatch: pytest.MonkeyPatch
):
    """The blobs written after a passing check must be the revision that was authorized.

    The first dataset-API response is the exact commit. A second response is the
    colliding commit. Downloading that second commit, or any other revision,
    fails this test even if the first response was compared successfully.
    """
    server = hub("relookup")
    embed, retrieval, reranker, bm25 = _modules()
    _pin_fixture_digests(monkeypatch, embed, retrieval, reranker, bm25)
    outputs = _run(builder, tmp_path, embed, retrieval, reranker, bm25)
    gated = outputs[0]
    problems = _authorized_problems(gated)
    for path in _mteb_blob_paths(server.paths):
        match = BLOB_PATH.fullmatch(path)
        assert match is not None
        dataset = f"mteb/{match.group(1)}"
        sha = match.group(2)
        if sha != LEGIT[dataset]:
            problems.append(f"fetched {path} instead of {LEGIT[dataset]}")
    expected = [
        f"/datasets/{dataset}/resolve/{sha}/{kind}"
        for dataset, sha in LEGIT.items()
        for kind in PARQUET_KIND
    ]
    for suffix in expected:
        if not any(path.endswith(suffix) or path == suffix for path in server.paths):
            problems.append(f"missing download of the authorized blob {suffix}")
    if builder == "embed":
        base = f"http://127.0.0.1:{server.server_address[1]}"
        ndcg = _publication_ndcg(embed, gated, tmp_path, base)
        if ndcg != 1.0:
            problems.append(f"score() published nDCG@10 {ndcg} instead of the pinned set's 1.0")
    if problems:
        pytest.fail(
            "authorized revision was not the revision that was frozen\n" + "\n".join(problems)
        )
