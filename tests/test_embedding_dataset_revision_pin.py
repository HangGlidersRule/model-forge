"""The six MTEB Nano loaders must download the commit the embedding gate recorded.

The gate record is ``dataset_shas`` in ``retrieval-benchmark.json``. Each loader
currently stores only the first 12 hex characters and accepts whatever commit the
dataset API returns when that commit starts with those characters, then reads
parquet from that commit. These tests fail while that stands.

A passing loader requests only ``.../resolve/<recorded 40-char sha>/...`` and
writes those rows. It still fails if a later lookup of the dataset API supplies
the revision that is downloaded: the first lookup may return the recorded sha
and every later lookup returns a different commit that shares the 12-character
prefix. ``embed`` and the reranker ``quality`` client are not revision resolvers;
``score`` and ``build_gate_summary`` are covered because they publish the rows
``fetch`` wrote.
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDED_PATH = ROOT / "models/qwen3-embedding-8b/results/retrieval-benchmark.json"
GATE_SUMMARY = ROOT / "models/qwen3-embedding-8b/eval/build_gate_summary.py"
HEX = set("0123456789abcdef")
KINDS = (
    "corpus/train-00000-of-00001.parquet",
    "queries/train-00000-of-00001.parquet",
    "qrels/train-00000-of-00001.parquet",
)
PINNED_RELEVANT = "d00"
COLLIDER_RELEVANT = "d01"
DOC_COUNT = 30


def _recorded() -> dict[str, str]:
    shas = json.loads(RECORDED_PATH.read_text())["dataset_shas"]
    assert set(shas) == {
        "mteb/NanoNFCorpusRetrieval",
        "mteb/NanoFiQA2018Retrieval",
        "mteb/NanoTouche2020Retrieval",
        "mteb/NanoArguAnaRetrieval",
        "mteb/NanoSciFactRetrieval",
        "mteb/NanoMSMARCORetrieval",
    }
    for sha in shas.values():
        assert len(sha) == 40 and set(sha) <= HEX
    return shas


RECORDED = _recorded()


def _collider(full: str) -> str:
    """A different 40-character id that still starts with the stored 12 hex digits."""
    other = full[:12] + ("c" * 28)
    assert other != full and other.startswith(full[:12]) and len(other) == 40
    return other


COLLIDER = {ds: _collider(sha) for ds, sha in RECORDED.items()}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _retrieval():
    return _load(
        "retrieval_benchmark_under_test",
        ROOT / "models/qwen3-embedding-8b/eval/retrieval_benchmark.py",
    )


def _embed():
    return _load(
        "embed_eval_under_test",
        ROOT / "models/nemotron-3.5-embed-8b/eval/embed_eval.py",
    )


def _reranker():
    path = ROOT / "models/qwen3-reranker-4b/eval/reranker_eval.py"
    if "reranker_eval" in sys.modules:
        return sys.modules["reranker_eval"]
    return _load("reranker_eval", path)


def _bm25():
    return _load(
        "build_evalset_bm25_under_test",
        ROOT / "models/qwen3-reranker-4b/eval/build_evalset_bm25.py",
    )


def _parquet(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def _corpus(marker: str) -> list[dict]:
    return [
        {"_id": f"d{i:02d}", "title": "", "text": f"{marker} d{i:02d}"} for i in range(DOC_COUNT)
    ]


def _queries(marker: str) -> list[dict]:
    return [{"_id": "q0", "text": f"{marker}-QUERY"}]


def _qrels(relevant: str) -> list[dict]:
    return [{"query-id": "q0", "corpus-id": relevant, "score": 1}]


PINNED_FILES = {
    KINDS[0]: _parquet(_corpus("PINNED")),
    KINDS[1]: _parquet(_queries("PINNED")),
    KINDS[2]: _parquet(_qrels(PINNED_RELEVANT)),
}
COLLIDER_FILES = {
    KINDS[0]: _parquet(_corpus("COLLIDER")),
    KINDS[1]: _parquet(_queries("COLLIDER")),
    KINDS[2]: _parquet(_qrels(COLLIDER_RELEVANT)),
}


def _materialize(files: dict[str, bytes]) -> dict:
    """Same corpus/query/qrel mapping as ``retrieval_benchmark.fetch``."""
    corpus = pq.read_table(io.BytesIO(files[KINDS[0]])).to_pylist()
    queries = pq.read_table(io.BytesIO(files[KINDS[1]])).to_pylist()
    qrels = pq.read_table(io.BytesIO(files[KINDS[2]])).to_pylist()
    id_key = "_id" if "_id" in corpus[0] else "id"
    docs = {
        row[id_key]: (
            (row.get("title") or "").strip() + "\n" + (row.get("text") or "").strip()
        ).strip()
        for row in corpus
    }
    query_text = {row[id_key]: row["text"] for row in queries}
    relevant: dict[str, dict[str, int]] = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    query_ids = sorted(query_id for query_id in relevant if query_id in query_text)
    doc_ids = sorted(docs)
    return {
        "doc_ids": doc_ids,
        "docs": [docs[doc_id] for doc_id in doc_ids],
        "qids": query_ids,
        "queries": [query_text[query_id] for query_id in query_ids],
        "qrels": {query_id: relevant[query_id] for query_id in query_ids},
    }


PINNED_ROWS = _materialize(PINNED_FILES)
COLLIDER_ROWS = _materialize(COLLIDER_FILES)


class _Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class _Hub:
    """Stand-in for the dataset HTTP API the loaders call through ``urlopen``.

    Parquet bytes are real pyarrow files. The API body is the JSON object whose
    ``sha`` string ``fetch`` reads. Nothing else is rewritten between that body
    and the revision interpolated into the download URL.
    """

    def __init__(self, mode: str, extra: dict[str, bytes] | None = None):
        self.mode = mode
        self.extra = extra or {}
        self.urls: list[str] = []
        self._api_hits: dict[str, int] = {}
        self.files: dict[str, bytes] = dict(self.extra)
        for ds, recorded in RECORDED.items():
            for kind in KINDS:
                self.files[_resolve(ds, recorded, kind)] = PINNED_FILES[kind]
                self.files[_resolve(ds, COLLIDER[ds], kind)] = COLLIDER_FILES[kind]
                self.files[_resolve(ds, recorded[:12], kind)] = COLLIDER_FILES[kind]

    def _api_sha(self, dataset: str) -> str:
        seen = self._api_hits.get(dataset, 0) + 1
        self._api_hits[dataset] = seen
        if self.mode == "colliding_commit":
            return COLLIDER[dataset]
        if self.mode == "reresolve_after_check":
            return RECORDED[dataset] if seen == 1 else COLLIDER[dataset]
        raise AssertionError(self.mode)

    def urlopen(self, req: object, data: object = None, timeout: object = None, **_kwargs: object):
        del data, timeout
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        assert isinstance(url, str)
        self.urls.append(url)
        if url.startswith("https://huggingface.co/api/datasets/"):
            dataset = url.removeprefix("https://huggingface.co/api/datasets/")
            if dataset not in RECORDED:
                raise RuntimeError(url)
            return _Body(json.dumps({"sha": self._api_sha(dataset)}).encode())
        if url not in self.files:
            raise RuntimeError(url)
        return _Body(self.files[url])


def _resolve(dataset: str, revision: str, kind: str) -> str:
    return f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{kind}"


def _allowed_mteb() -> set[str]:
    return {_resolve(ds, sha, kind) for ds, sha in RECORDED.items() for kind in KINDS}


def _pins_of(entry: str, module: object) -> dict[str, str]:
    if entry == "embed_build_sets":
        pins = module.MTEB
    else:
        pins = module.DATASETS
    assert set(pins) == set(RECORDED)
    for dataset, pin in pins.items():
        assert RECORDED[dataset].startswith(pin)
    return pins


def _lmeb_fixture(module: object) -> tuple[dict[str, bytes], dict]:
    base = f"https://huggingface.co/datasets/{module.LMEB}/resolve/{module.LMEB_REV}"
    files = {
        f"{base}/task_instructions.json": json.dumps(
            {"LoCoMo": {"single_hop": "find the turn"}}
        ).encode(),
        f"{base}/Dialogue/LoCoMo/corpus.jsonl": (
            json.dumps({"id": "doc1", "title": "t", "text": "body"}) + "\n"
        ).encode(),
        f"{base}/Dialogue/LoCoMo/candidates.jsonl": (
            json.dumps({"scene_id": "scene1", "candidate_doc_ids": ["doc1"]}) + "\n"
        ).encode(),
    }
    rerank_set = {
        "source": f"{module.LMEB}@{module.LMEB_REV}",
        "data": {
            "LoCoMo/single_hop": [
                {
                    "qid": "scene1_q_0",
                    "query": "hello",
                    "cands": [{"id": "doc1", "rel": 1}],
                }
            ]
        },
    }
    return files, rerank_set


def _run(monkeypatch: pytest.MonkeyPatch, module: object, hub: _Hub, call) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", hub.urlopen)
    if hasattr(module, "time"):
        monkeypatch.setattr(module.time, "sleep", lambda *_a, **_k: None)
    produced: list[Path] = []

    def _call() -> None:
        call(produced)

    try:
        _call()
    except Exception:
        for path in produced:
            assert not path.exists(), f"{path} was created before the loader refused"
        raise


def _assert_urls(hub: _Hub, allowed: set[str]) -> None:
    assert set(hub.urls) == allowed, hub.urls
    refused = [url for url in hub.urls if "/api/datasets/" in url or "/resolve/" not in url]
    assert refused == []
    for url in hub.urls:
        revision = url.split("/resolve/", 1)[1].split("/", 1)[0]
        assert revision in RECORDED.values() or url in allowed


def _assert_pinned_rows(entry: str, blob: dict) -> None:
    assert blob["dataset_shas"] == RECORDED
    raw = json.dumps(blob)
    assert "COLLIDER" not in raw
    if entry == "embed_build_sets":
        sets = blob["sets"]
        assert set(sets) == {ds.split("/", 1)[1] for ds in RECORDED}
        for dataset, rows in ((ds, sets[ds.split("/", 1)[1]]) for ds in RECORDED):
            del dataset
            assert rows["docs"] == PINNED_ROWS["docs"]
            assert rows["queries"] == PINNED_ROWS["queries"]
            assert rows["qrels"] == PINNED_ROWS["qrels"]
            assert rows["doc_ids"] == PINNED_ROWS["doc_ids"]
        return
    if entry == "retrieval_fetch":
        assert set(blob["data"]) == set(RECORDED)
        for dataset, rows in blob["data"].items():
            assert rows == PINNED_ROWS, dataset
        return
    assert set(blob["data"]) == set(RECORDED)
    pinned_text = dict(zip(PINNED_ROWS["doc_ids"], PINNED_ROWS["docs"], strict=True))
    for dataset, items in blob["data"].items():
        assert items, dataset
        for item in items:
            assert item["query"] == "PINNED-QUERY"
            relevant = sorted(cand["id"] for cand in item["cands"] if cand["rel"])
            assert relevant == [PINNED_RELEVANT]
            for cand in item["cands"]:
                assert cand["text"] == pinned_text[cand["id"]]


@pytest.mark.parametrize(
    "entry",
    ["retrieval_fetch", "embed_build_sets", "reranker_build_eval_set", "bm25_eval_set"],
)
@pytest.mark.parametrize("mode", ["colliding_commit", "reresolve_after_check"])
def test_download_uses_recorded_commit_not_branch_resolution(
    entry: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if entry == "retrieval_fetch":
        module = _retrieval()
        out = tmp_path / "data.json"
        _pins_of(entry, module)

        def call(produced: list[Path]) -> None:
            produced.append(out)
            module.fetch(out)

        allowed = _allowed_mteb()
        extra: dict[str, bytes] = {}
    elif entry == "embed_build_sets":
        module = _embed()
        out = tmp_path / "mteb.json"
        lmeb_out = tmp_path / "lmeb.json"
        rerank_path = tmp_path / "rerank.json"
        _pins_of(entry, module)
        extra, rerank_set = _lmeb_fixture(module)
        rerank_path.write_text(json.dumps(rerank_set))

        def call(produced: list[Path]) -> None:
            produced.extend((out, lmeb_out))
            module.build_sets(out, lmeb_out, rerank_path)

        allowed = _allowed_mteb() | set(extra)
    elif entry == "reranker_build_eval_set":
        module = _reranker()
        out = tmp_path / "evalset.json"
        _pins_of(entry, module)

        def call(produced: list[Path]) -> None:
            produced.append(out)
            module.build_eval_set(out)

        allowed = _allowed_mteb()
        extra = {}
    elif entry == "bm25_eval_set":
        module = _bm25()
        out = tmp_path / "bm25.json"
        _pins_of("reranker_build_eval_set", module)
        monkeypatch.setattr(sys, "argv", ["build_evalset_bm25.py", str(out)])

        def call(produced: list[Path]) -> None:
            produced.append(out)
            assert module.main() == 0

        allowed = _allowed_mteb()
        extra = {}
    else:
        raise AssertionError(entry)

    hub = _Hub(mode, extra)
    _run(monkeypatch, module, hub, call)
    _assert_urls(hub, allowed)
    _assert_pinned_rows(entry, json.loads(out.read_text()))


def _write_embeddings(out_dir: Path, datasets: list[str]) -> None:
    import numpy as np

    docs = np.eye(DOC_COUNT, dtype=np.float32)

    def query(primary: int) -> "np.ndarray":
        weights = np.zeros(DOC_COUNT, dtype=np.float32)
        for offset in range(DOC_COUNT):
            weights[(primary + offset) % DOC_COUNT] = np.float32(0.5**offset)
        return weights

    labels = {
        "bf16": 0,
        "bf16-replica": 0,
        "nvfp4-A": 0,
        "nvfp4-B": 1,
        "nvfp4-B-clean": 0,
        "nvfp4-B-stable": 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, primary in labels.items():
        arrays = {f"{ds}|docs": docs for ds in datasets}
        arrays.update({f"{ds}|queries": query(primary)[None, :] for ds in datasets})
        np.savez(out_dir / f"{label}.npz", **arrays)


def _score_ship(retrieval: object, blob: dict, work: Path) -> tuple[dict, bool]:
    work.mkdir(parents=True, exist_ok=True)
    data_path = work / "data.json"
    vec_dir = work / "vec"
    data_path.write_text(json.dumps(blob))
    _write_embeddings(vec_dir, list(blob["data"]))
    retrieval.score(data_path, vec_dir)
    benchmark = json.loads((vec_dir / "benchmark.json").read_text())
    results = work / "results"
    results.mkdir()
    (results / "retrieval-benchmark.json").write_text(json.dumps(benchmark))
    _gate_sidecars(results)
    summary_path = work / "gate.json"
    proc = subprocess.run(
        [sys.executable, str(GATE_SUMMARY), str(results), str(summary_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(summary_path.read_text())
    return benchmark, summary["public_benchmark"]["ship_rule_results"]["nvfp4-B"]["passes"]


def _gate_sidecars(results: Path) -> None:
    cells = {"long_c32": {"input_tok_per_s": 10.0}, "short_c1": {"input_tok_per_s": 20.0}}
    faster = {"long_c32": {"input_tok_per_s": 30.0}, "short_c1": {"input_tok_per_s": 40.0}}
    for name, body in (
        ("throughput-bf16.json", {"cells": cells}),
        ("throughput-nvfp4-A.json", {"cells": cells}),
        ("throughput-nvfp4-B.json", {"cells": faster}),
        ("throughput-nvfp4-B-stable.json", {"cells": cells}),
    ):
        (results / name).write_text(json.dumps(body))
    drift = {
        "drift_4096": {
            "candidate_vs_stored": {"mean": 0.1, "p1": 0.2},
            "bf16_vs_stored": {"mean": 0.3},
        }
    }
    for name in ("store-compat-nvfp4-A.json", "store-compat-nvfp4-B.json"):
        (results / name).write_text(json.dumps({"stores": {"transcripts": drift}}))
    mixed = {
        "self_gap_mean": 0.1,
        "group_n": 4,
        "group_b": 4,
        "stored": {"N": {"recall@1": 0.5}, "B": {"recall@1": 0.5}},
        "mixed": {"N": {"recall@1": 0.5}, "B": {"recall@1": 0.5}},
    }
    for name in ("mixed-corpus-nvfp4-A.json", "mixed-corpus-nvfp4-B.json"):
        (results / name).write_text(json.dumps(mixed))
    (results / "artifact-validation.json").write_text(json.dumps({"result": {"ok": True}}))
    (results / "build-summary.json").write_text(
        json.dumps({"serve": {"bf16": {"weights_gib": 1.0}}, "precision_coverage_shipped": {}})
    )
    (results / "activation-outliers.json").write_text(json.dumps({}))
    (results / "instance-agreement.json").write_text(
        json.dumps({"pairs": {"orig1 vs orig2": {"documents": 1}}})
    )
    (results / "stable-profile-reproducibility.json").write_text(
        json.dumps(
            {
                "docs": {"bitwise_identical": 1, "vectors": 1},
                "queries": {"bitwise_identical": 1, "vectors": 1},
            }
        )
    )


def test_score_and_gate_summary_publish_the_recorded_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = _retrieval()
    instruct = retrieval.INSTRUCT
    pinned_blob = {
        "dataset_shas": dict(RECORDED),
        "instruction": instruct,
        "data": {ds: PINNED_ROWS for ds in RECORDED},
    }
    collider_blob = {
        "dataset_shas": dict(COLLIDER),
        "instruction": instruct,
        "data": {ds: COLLIDER_ROWS for ds in RECORDED},
    }
    _pinned_bench, pinned_passes = _score_ship(retrieval, pinned_blob, tmp_path / "pinned")
    _collider_bench, collider_passes = _score_ship(retrieval, collider_blob, tmp_path / "collider")
    assert pinned_passes is False
    assert collider_passes is True

    fetched = tmp_path / "fetched.json"
    hub = _Hub("colliding_commit")

    def call(produced: list[Path]) -> None:
        produced.append(fetched)
        retrieval.fetch(fetched)

    _run(monkeypatch, retrieval, hub, call)
    fetched_blob = json.loads(fetched.read_text())
    fetched_bench, fetched_passes = _score_ship(retrieval, fetched_blob, tmp_path / "fetched")
    assert fetched_blob["dataset_shas"] == RECORDED
    assert fetched_bench["dataset_shas"] == RECORDED
    assert fetched_passes is False
