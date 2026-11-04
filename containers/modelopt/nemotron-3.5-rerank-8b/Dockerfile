# syntax=docker/dockerfile:1.7
#
# NVIDIA ModelOpt 0.48.0 quantization image for the Nemotron-3.5-Rerank-8B campaign (model-forge issue: see
# configs/modelopt/pins/modelopt-0.48.0.json). Newest stack per campaign, each component re-selected:
#   - nvidia-modelopt 0.48.0 (final, 2026-11-02) from the SHA-256-verified PyPI wheel, examples at the tag commit
#   - torch 2.15.0 cu132 (newest; Blackwell sm_120)
#   - transformers 5.18.0: ModelOpt 0.48.0 requires transformers<5.19 ([hf] extra), and 5.18.0 is the newest release
#     inside that window. The checkpoint's model.py and its bidirectional text tower were proven on 5.18.0 (and 5.22.1)
#     on the GPU host before this image was written.
#
# Build on the GPU host: docker build --build-arg ... -t local/model-forge-modelopt:0.48.0-e2f328a-t2150 .

ARG CUDA_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG PUBLIC_ARTIFACT_PATH
ARG PUBLIC_WORKSPACE
RUN test -n "${PUBLIC_ARTIFACT_PATH}" && test -n "${PUBLIC_WORKSPACE}"

ARG PYTHON_VERSION=3.12
ARG MODELOPT_VERSION=0.48.0
ARG MODELOPT_COMMIT=e2f328a19de789e73b91aca2908042388cf6d496
ARG MODELOPT_WHEEL_URL=https://files.pythonhosted.org/packages/f4/c8/dc9ced0713f8ccc7688ea1497c595e7cefabbc656d0aac125a226b425fd9/nvidia_modelopt-0.48.0-py3-none-any.whl
ARG MODELOPT_WHEEL_SHA256=bfcfd425eab15e05eccf065b58d11cc33f9c27b4859a6efb78aba74bfb9f225c
ARG MODELOPT_WHEEL_FILENAME=nvidia_modelopt-0.48.0-py3-none-any.whl
ARG TORCH_VERSION=2.15.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG TRANSFORMERS_SPEC=transformers==5.18.0
ARG EXTRA_PIP_SPECS="datasets>=3.0.0,<4.0.0 pyarrow psutil zstandard fire nvidia-ml-py requests pillow"
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

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev" \
        git curl ca-certificates; \
    rm -rf ${PUBLIC_WORKSPACE}/apt-lists/*; \
    "python${PYTHON_VERSION}" -m venv "${VIRTUAL_ENV}"; \
    python -m pip install --upgrade "pip>=24.2" "setuptools>=80" wheel

RUN set -eux; \
    python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

RUN set -eux; \
    mkdir -p ${PUBLIC_WORKSPACE}/wheel; \
    curl -fSL -o "${PUBLIC_ARTIFACT_PATH}" "${MODELOPT_WHEEL_URL}"; \
    echo "${MODELOPT_WHEEL_SHA256}  ${PUBLIC_ARTIFACT_PATH}" | sha256sum -c -; \
    python -m pip install "${PUBLIC_ARTIFACT_PATH}[hf]" "${TRANSFORMERS_SPEC}"; \
    python -c "import importlib.metadata as md; assert md.version('nvidia-modelopt') == '${MODELOPT_VERSION}'; assert md.version('torch').split('+')[0] == '${TORCH_VERSION}', md.version('torch')"; \
    rm -rf ${PUBLIC_WORKSPACE}/wheel

RUN set -eux; \
    git clone --filter=blob:none "${MODELOPT_REPO_URL}" "${MODELOPT_ROOT}"; \
    cd "${MODELOPT_ROOT}"; \
    git checkout --detach "${MODELOPT_COMMIT}"; \
    test "$(git rev-parse HEAD)" = "${MODELOPT_COMMIT}"; \
    test -f "${MODELOPT_ROOT}/examples/hf_ptq/hf_ptq.py"

RUN set -eux; \
    python -m pip install ${EXTRA_PIP_SPECS}; \
    python -c "import importlib.metadata as md; assert md.version('torch').split('+')[0] == '${TORCH_VERSION}'; assert md.version('nvidia-modelopt') == '${MODELOPT_VERSION}'; import transformers; assert transformers.__version__ == '5.18.0', transformers.__version__; print('OK torch', md.version('torch'), 'modelopt', md.version('nvidia-modelopt'), 'transformers', transformers.__version__)"

LABEL org.opencontainers.image.title="model-forge-modelopt" \
      com.hangglidersrule.modelopt.version="${MODELOPT_VERSION}" \
      com.hangglidersrule.modelopt.commit="${MODELOPT_COMMIT}" \
      com.hangglidersrule.modelopt.wheel_sha256="${MODELOPT_WHEEL_SHA256}"

WORKDIR ${MODELOPT_ROOT}
