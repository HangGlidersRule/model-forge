# Containers

Generic Docker Compose assets for building and serving model-forge artifacts. Every
environment-specific value is a variable, and no host paths, credentials, or
private hostnames are baked in. The build runner is a **generic shell with a configurable launcher**:
its `BUILD_SCRIPT` variable defaults to the Qwen3.8 pipeline
(`scripts/qwen3_8/run_qwen38_build_mcprue.sh`), so out of the box it builds Qwen. Another family
points `BUILD_SCRIPT` at its own repo-relative launcher — no Compose edits required.

## Layout

```text
containers/
  build/docker-compose.yml   GPU build runner (transform + quantization pipeline)
  serve/docker-compose.yml   vLLM OpenAI-compatible serving
  serve/darkstar-qwen38-abliterated-nvfp4.yml
                             Frozen Product 4 runtime profile
  modelopt/Dockerfile        Pinned NVIDIA ModelOpt NVFP4 quantization image
  modelopt/build.sh          Reproducible build driven entirely by pin.json
```

## Pinned ModelOpt image

`modelopt/Dockerfile` builds the NVFP4 quantization runtime used by the Qwen3.8
mcprue runner. It is pinned by immutable identifiers (never `latest`): the
`nvidia-modelopt` wheel is downloaded from its exact URL and **verified against the
`pin.json` SHA-256 at build time**, installed as `<wheel>[hf]` so the extra never
re-installs the package from PyPI over the verified file, and
`NVIDIA/Model-Optimizer` is checked out at the pinned commit (asserted with
`git rev-parse`) so `examples/hf_ptq/hf_ptq.py` is byte-for-byte the pinned source.
The base is `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` — noble carries `python3.12`
in its default repositories, and the build creates a `/opt/venv` virtualenv that owns
pip for that interpreter — with a Blackwell PyTorch (`cu128`,
`TORCH_CUDA_ARCH_LIST=12.0`) for the RTX PRO 6000 Blackwell (sm_120).

`modelopt/build.sh` reads every build argument from
[`configs/modelopt/pin.json`](../configs/modelopt/pin.json), so the image can never
drift from the recorded pin:

```bash
containers/modelopt/build.sh              # build the pinned image
DRY_RUN=1 containers/modelopt/build.sh    # print the docker build command only
```

## Naming and extension boundary

- Directories are named by role (`build`, `serve`), not by model family.
- Keep these Compose files generic. Family- or run-specific choices (which recipe, which model path,
  MTP depth) are passed through environment variables, not hardcoded.
- Family-specific launch logic belongs in `scripts/<family_slug>/`, which these containers invoke
  through `BUILD_SCRIPT` (default `scripts/qwen3_8/run_qwen38_build_mcprue.sh`). The path is
  repo-relative to the mounted repo (`/repo`) and expanded by the container shell, not baked into the
  command.
- Published image names follow `ghcr.io/hangglidersrule/model-forge-{serve,build}:<version>`.
- Darkstar API aliases follow `darkstar-<family>-<behavior>-<format>` and are passed with
  `--served-model-name`. Repository/model-card identities remain long and precision-encoded.

`src/model_forge/serve_profile.py` validates this split and renders canonical Compose YAML.
`scripts/render_darkstar_serve_profile.py` is the reusable entry point. It derives the filename from
the alias and writes only when content changes, so rerunning it never creates random files; the fixed
Compose project and `container_name` make repeated `docker compose up -d` converge on one container.
For Product 4:

```bash
python3 scripts/render_darkstar_serve_profile.py \
  --family qwen38 \
  --behavior abliterated \
  --format nvfp4 \
  --repository-id HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8 \
  --model-path ${PUBLIC_ARTIFACT_PATH} \
  --artifact-identity Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8 \
  --artifact-precision-class W4A16-NVFP4-Mixed-FP8 \
  --artifact-success-sha256 3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79 \
  --artifact-manifest models/qwen3.8-27b-r3/results/manifests/abliterated-modelopt-mixed-manifest.json \
  --artifact-validation attestation \
  --container-name vllm-darkstar-qwen38-abliterated-modelopt \
  --mtp-depth 10 \
  --scheduler-tokens 32768
```

`attestation` is the explicit clean-checkout rendering mode: it validates the committed immutable
manifest without claiming the host artifact is present. The default `local` mode is used by the
operator launcher and additionally requires the artifact directory and `_SUCCESS.json`, verifies the
marker's actual SHA-256, and rejects malformed markers.

## Configuration

The Compose files read environment variables. `CONFIG` is deliberately required for builds so the
generic runner cannot silently select a rejected historical recipe. Common variables:

| Variable | Purpose | Default |
|---|---|---|
| `REPO_ROOT` | Repo mounted read-only into the build runner | `.` |
| `CONFIG` | Recipe consumed by the build runner | **Required; set explicitly to a supported current recipe** |
| `BUILD_SCRIPT` | Repo-relative launcher the build runner executes | `scripts/qwen3_8/run_qwen38_build_mcprue.sh` |
| `DRY_RUN` | Guard against accidental heavy runs | `1` |
| `VLLM_IMAGE` | Serving image | `vllm/vllm-openai:v0.27.1` |
| `VLLM_MODEL_PATH` | Model path inside the serve container | `/models/current` |
| `VLLM_CTX` | Max context length | `126144` |
| `VLLM_KV_CACHE_DTYPE` | KV cache dtype | `bf16` |
| `VLLM_EXTRA_ARGS` | Extra vLLM CLI flags appended verbatim (e.g. speculative decoding) | *(empty)* |

Every variable the serve template advertises reaches vLLM through a command-line flag. Settings that
the template fixes are not exposed as variables: chunked prefill is enabled unconditionally by
`serve/docker-compose.yml`, because a Compose string command cannot render a bare flag
conditionally. Override it by editing the template or supplying your own Compose file.

### Frozen Product 4 profile

[`serve/darkstar-qwen38-abliterated-nvfp4.yml`](serve/darkstar-qwen38-abliterated-nvfp4.yml) is the
exact locally verified Product 4 profile:

- API alias `darkstar-qwen38-abliterated-nvfp4`
- container `vllm-darkstar-qwen38-abliterated-modelopt`
- artifact `${PUBLIC_ARTIFACT_PATH}`
- MTP10, FlashAttention (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`), BF16 KV, context 126144,
  scheduler budget 32768, `max_num_seqs=16`, prefix caching, and chunked prefill

The frozen attention backend is baked into the profile as a fixed engine setting, not an operator
knob. Bring Product 4 up through the checked-in canonical launcher, which is the operator entrypoint:

```bash
scripts/serve_darkstar_qwen38_abliterated_nvfp4.sh \
  --artifact-path ${PUBLIC_ARTIFACT_PATH}
```

The launcher renders the deterministic Compose, verifies both the render and the checked-in file
against the frozen digest, validates it (`docker compose config -q`), and converges on one container
with a stable project via `up -d --force-recreate`. It supports `--dry-run` and `--print-config`,
accepts no mutable vLLM arguments, and honors only `VLLM_PORT` as a host override.

The final operator snapshot's winning Compose evidence has SHA-256
`85ba68155418dad7387219f62889def88c62a0e2ca35d15e3f83d62879077088`. The
tracked file above is a deterministic repository rendering of that frozen semantic profile; its bytes
also encode the new stable alias/container standard and therefore are not claimed to be the operator
snapshot file.

### Speculative decoding is off by default

Generic serving does not assume any family supports MTP, so the serve template ships **without**
`--speculative-config`. Optional flags are awkward to toggle conditionally in a Compose string
command, so they flow through one `VLLM_EXTRA_ARGS` fragment (default empty) appended verbatim to the
vLLM command. To enable Qwen3.8 MTP speculative decoding, supply the `--speculative-config` flag
through it, for example:

```bash
VLLM_EXTRA_ARGS='--speculative-config {"method":"mtp","num_speculative_tokens":6}' \
  docker compose -f serve/docker-compose.yml up
```

## Rules

- Images must never contain weights, credentials, private hostnames, or local paths.
- Large artifacts live in external named volumes, not in images or the repository.
- Pin runtime versions (vLLM, CUDA) explicitly.
