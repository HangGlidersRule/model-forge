# Qwen3-Reranker-4B calibration set

`build_reranker_calib.py` builds the 512-sample calibration JSONL for the ModelOpt NVFP4 candidates.
The output file is not committed; rebuild it and check the digest.

- **Source:** `microsoft/ms_marco` v1.1 train at revision `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a` (`v1.1/train-00000-of-00001.parquet`).
- **Samples:** 512 query-passage pairs, 256 relevant (`is_selected=1`) and 256 irrelevant, shuffled with seed 1234.
- **Format:** each pair is rendered exactly as served: the Qwen3-Reranker judge system prompt, the default web-search instruction, and the assistant turn with an empty think block. One `{"text": ...}` object per line.
- **Expected output:** sha256 `cada916c2205153cc32ee6a4b30e5a121f8960e6f68f8578130ffc6a47e95261`. Rendered lengths are 486-1,232 characters (median 825).

```bash
python build_reranker_calib.py --out reranker_calib_msmarco_v1.1_512.jsonl
```

ModelOpt 0.48.0rc1's `hf_ptq.py --dataset <path>.jsonl` reads the `text` column as-is.
