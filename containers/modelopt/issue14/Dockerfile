# syntax=docker/dockerfile:1.7
#
# NVIDIA ModelOpt 0.48.0rc1 quantization image for the Qwen3-Reranker-4B and
# whisper-large-v3-turbo campaign (public issue HangGlidersRule/model-forge#14).
#
# Same contract as ../Dockerfile, pinned to configs/modelopt/pins/modelopt-0.48.0rc1.json:
#   - nvidia-modelopt from the SHA-256-verified local wheel only
#   - NVIDIA/Model-Optimizer examples checked out at the immutable tag commit
#   - torch cu128 (Blackwell sm_120) and transformers inside 0.48's supported window
# Extra runtime deps: datasets 3.x + soundfile/librosa (Whisper calibration audio decode),
# the examples/hf_ptq requirements, and pyarrow (calibration JSONL loading).
#
# Build with containers/modelopt/build.sh and PIN_JSON pointing at the pin file above.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG PUBLIC_ARTIFACT_PATH
ARG PUBLIC_WORKSPACE
RUN test -n "${PUBLIC_ARTIFACT_PATH}" && test -n "${PUBLIC_WORKSPACE}"

ARG PYTHON_VERSION=3.12
ARG MODELOPT_VERSION=0.48.0rc1
ARG MODELOPT_COMMIT=ff2450c15eb5f1697ec0471704d5421d54fce4f4
ARG MODELOPT_WHEEL_URL=https://files.pythonhosted.org/packages/b1/b9/5888b69f463f2a5246485d2f85a03d138226d849cdd18297f68e717caaac/nvidia_modelopt-0.48.0rc1-py3-none-any.whl
ARG MODELOPT_WHEEL_SHA256=2b874d94c198d72a1ca299a9d2a45ea5ff61f860ca0bd4ed7656465ce5b5f588
ARG MODELOPT_WHEEL_FILENAME=nvidia_modelopt-0.48.0rc1-py3-none-any.whl
ARG TORCH_VERSION=2.8.0
ARG TORCH_SPEC=torch==${TORCH_VERSION}
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TRANSFORMERS_SPEC=transformers>=5.5,<5.19
ARG EXTRA_PIP_SPECS="datasets>=3.0.0,<4.0.0 soundfile librosa pyarrow psutil zstandard fire nvidia-ml-py requests"
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG MODELOPT_REPO_URL=https://github.com/NVIDIA/Model-Optimizer.git
ARG MODELOPT_ROOT=${PUBLIC_WORKSPACE}/modelopt

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MODELOPT_ROOT=${MODELOPT_ROOT} \
    MODELOPT_COMMIT=${MODELOPT_COMMIT} \
    MODELOPT_VERSION=${MODELOPT_VERSION} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=${PUBLIC_WORKSPACE}/venv

ENV PATH=${PUBLIC_WORKSPACE}/venv/bin:${PATH}

# libsndfile1 lets soundfile decode the FLAC audio in the Whisper calibration set.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev" \
        git curl ca-certificates libsndfile1; \
    rm -rf ${PUBLIC_WORKSPACE}/apt-lists/*; \
    "python${PYTHON_VERSION}" -m venv "${VIRTUAL_ENV}"; \
    python -m pip install --upgrade "pip>=24.2" "setuptools>=80" wheel; \
    test "$(command -v python)" = "${VIRTUAL_ENV}/bin/python"; \
    test "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "${PYTHON_VERSION}"

RUN set -eux; \
    python -m pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"

RUN set -eux; \
    mkdir -p ${PUBLIC_WORKSPACE}/wheel; \
    curl -fSL -o "${PUBLIC_ARTIFACT_PATH}" "${MODELOPT_WHEEL_URL}"; \
    echo "${MODELOPT_WHEEL_SHA256}  ${PUBLIC_ARTIFACT_PATH}" | sha256sum -c -; \
    python -m pip install "${PUBLIC_ARTIFACT_PATH}[hf]" "${TRANSFORMERS_SPEC}"; \
    python -c "import importlib.metadata as md, json; d = md.distribution('nvidia-modelopt'); assert d.version == '${MODELOPT_VERSION}', d.version; url = json.loads(d.read_text('direct_url.json'))['url']; assert url.startswith('file://') and url.endswith('/${MODELOPT_WHEEL_FILENAME}'), url"; \
    python -c "import importlib.metadata as md; v = md.version('torch'); assert v.split('+')[0] == '${TORCH_VERSION}', v"; \
    rm -rf ${PUBLIC_WORKSPACE}/wheel; \
    python -c "import modelopt; assert modelopt.__version__ == '${MODELOPT_VERSION}', modelopt.__version__"

RUN set -eux; \
    git clone --filter=blob:none "${MODELOPT_REPO_URL}" "${MODELOPT_ROOT}"; \
    cd "${MODELOPT_ROOT}"; \
    git checkout --detach "${MODELOPT_COMMIT}"; \
    test "$(git rev-parse HEAD)" = "${MODELOPT_COMMIT}"; \
    git rev-parse HEAD > "${MODELOPT_ROOT}/.modelopt_commit"; \
    test -f "${MODELOPT_ROOT}/examples/hf_ptq/hf_ptq.py"

# Extra runtime deps. torch, modelopt and transformers must not move: re-assert all three.
RUN set -eux; \
    python -m pip install ${EXTRA_PIP_SPECS}; \
    python -c "import importlib.metadata as md; assert md.version('torch').split('+')[0] == '${TORCH_VERSION}'; assert md.version('nvidia-modelopt') == '${MODELOPT_VERSION}'; import transformers, datasets, soundfile, librosa, pyarrow; print('transformers', transformers.__version__, 'datasets', datasets.__version__)"

LABEL org.opencontainers.image.title="model-forge-modelopt" \
      org.opencontainers.image.description="Pinned NVIDIA ModelOpt quantization runtime (issue-14 reranker and Whisper campaign, Blackwell sm_120)" \
      org.opencontainers.image.source="https://github.com/NVIDIA/Model-Optimizer" \
      com.hangglidersrule.modelopt.version="${MODELOPT_VERSION}" \
      com.hangglidersrule.modelopt.commit="${MODELOPT_COMMIT}" \
      com.hangglidersrule.modelopt.wheel_sha256="${MODELOPT_WHEEL_SHA256}"

WORKDIR ${MODELOPT_ROOT}
