# Hugging Face publication runbook (Darkstar campaign)

Deterministic, idempotent steps for publishing a Darkstar checkpoint family.
Baked-in versions of the ad-hoc scripts used in the Nano-Omni/Lightning
campaigns. Every step ends in verification — a tool succeeding is not the
gate, the read-back is.

## Pre-flight

1. All gates measured on the exact artifact being uploaded (behavior eval,
   GPQA under the llm-inference-bench protocol, speed sweep). Evidence JSONs
   committed in `models/<family>/results/`.
2. Card passes `tests/test_model_card_gold_standard.py` — skeleton, safety
   closer, evaluation table, `--served-model-name`.
3. HF token present with write scope; verify with `whoami` (role is not
   reported by the endpoint — verify by creating the repo).

## Repo creation

```python
api.create_repo(repo_id, private=True, exist_ok=True)  # private FIRST, always
```

Private-until-verified is the convention; flip public only after the
post-upload verification passes. (Note: `upload_folder` on a fresh private
repo can land public if the org default is public — VERIFY visibility after
upload; do not trust the creation call's return value alone.)

## Weight upload (62GB+ artifact)

```python
os.environ["HF_HUB_DISABLE_XET"] = "1"   # REQUIRED for large multi-shard uploads
api.upload_folder(folder_path=..., repo_id=..., repo_type="model",
                  commit_message="<gate summary>")
```

**Known failure:** the xet transfer pipeline throws `TimeoutError: Request
error: error decoding response body, domain: no-url` on large LFS batches
(verified on a 62 GB / 17-shard artifact, 2026-09-01). `HF_HUB_DISABLE_XET=1`
forces the classic HTTP LFS path, which completed in ~5.5 minutes for 18 LFS
files.

Run detached with its own container and a name like `omni-upload`; poll
`docker ps` — `nohup` inside SSH sessions has died silently before.

## Post-upload verification (mandatory)

1. `list_repo_files` count matches the local inventory (e.g. 17 shards +
   index + config/tokenizer/wrapper files + README).
2. `model_info(...).private` matches intent (flip with `hf repo settings
   --public` if needed).
3. Anonymous `curl -s -o /dev/null -w "%{http_code}"` on:
   - the card page (200)
   - `${PUBLIC_WORKSPACE}` (200; follow redirects)
   - `${PUBLIC_ARTIFACT_PATH}` (200; ~30s max-time is
     enough to prove reachability, do not download the shard)
4. Card read-back: fetch the raw README and re-run the gold-standard checks —
   HF metadata validation may reject front-matter (`license_link` must be a
   valid https URI — underscores, not spaces).
5. Record the server-reported revision SHA of the published README in the
   family ledger.

## Card update flow (post-publication corrections)

Re-upload just `README.md` with `upload_file(path_or_fileobj=<bytes>,
path_in_repo="README.md", ...)`; the hub kwarg name is `path_or_fileobj`
(NOT `path_or_file` / `path_or_filetype` — verified error on hub
`huggingface_hub` 2026-08 build). Re-run step 3-5 verification after.

## Ledger sync

After each cell ships, update
`models/<family>/results/publication-readiness-ledger.json`: `publication`
field gets the live URL + verification note; the matching `release_gates`
entries flip PASS-with-evidence. Ledger updates are commits, not chat
messages.
