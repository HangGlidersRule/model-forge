# syntax=docker/dockerfile:1.7
#
# Reproducible NVIDIA ModelOpt quantization image for Qwen3.8-27B NVFP4 builds.
#
# Pinned by immutable identifiers (never "latest"):
#   - nvidia-modelopt wheel by URL + SHA-256 (verified at build time)
#   - NVIDIA/Model-Optimizer examples checked out at an immutable commit
#   - torch / transformers version specifiers matching the pin.json contract
#
# Target GPU: NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0 / sm_120),
# which requires CUDA >= 12.8 and a Blackwell-enabled PyTorch build (cu128).
#
# Build with containers/modelopt/build.sh, which sources every value from
# configs/modelopt/pin.json so the image can never drift from the recorded pin.

# Ubuntu 24.04 (noble) is the first NVIDIA CUDA base whose default apt repositories
# carry python3.12 (and python3.12-venv / -dev), so the pinned interpreter needs no
# third-party PPA. All Python work happens in a venv at ${PUBLIC_WORKSPACE}/venv: it owns pip for
# the pinned interpreter and sidesteps the distro's externally-managed environment.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG PUBLIC_ARTIFACT_PATH
ARG PUBLIC_WORKSPACE
RUN test -n "${PUBLIC_ARTIFACT_PATH}" && test -n "${PUBLIC_WORKSPACE}"

# --- Pin contract (all overridable only through build.sh from pin.json) ---
ARG PYTHON_VERSION=3.12
ARG MODELOPT_VERSION=0.46.0rc2
ARG MODELOPT_COMMIT=43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a
ARG MODELOPT_WHEEL_URL=https://files.pythonhosted.org/packages/f8/63/539a8628df685d38b9ca23f24752b1bd71a2320d5ae1676adfdfb7a50736/nvidia_modelopt-0.46.0rc2-py3-none-any.whl
ARG MODELOPT_WHEEL_SHA256=d6f6964b76c9e3f156ed1f3627d406b187c454614ab8e409a3796568cd487bbb
# pip rejects a local wheel whose filename is not PEP 427 shaped, so the download
# target keeps the pinned distribution filename rather than a convenience name.
ARG MODELOPT_WHEEL_FILENAME=nvidia_modelopt-0.46.0rc2-py3-none-any.whl
# torch 2.8.x on cu128 is the first Blackwell (sm_120) enabled stable line.
ARG TORCH_VERSION=2.8.0
ARG TORCH_SPEC=torch==${TORCH_VERSION}
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# transformers window from the pin's dependency_versions contract.
ARG TRANSFORMERS_SPEC=transformers>=4.57,<5.15
# Blackwell RTX PRO 6000 architecture. Kept explicit so any source builds target sm_120.
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

# Separate ENV: PATH must resolve the venv created below, and a value set in the
# same ENV instruction is not visible to its siblings.
ENV PATH=${PUBLIC_WORKSPACE}/venv/bin:${PATH}

# Fail-closed: the interpreter that ends up on PATH must be the pinned version and
# must own a working pip, otherwise the build stops here rather than at import time.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev" \
        git curl ca-certificates; \
    rm -rf ${PUBLIC_WORKSPACE}/apt-lists/*; \
    "python${PYTHON_VERSION}" -m venv "${VIRTUAL_ENV}"; \
    python -m pip install --upgrade "pip>=24.2" "setuptools>=70" wheel; \
    test "$(command -v python)" = "${VIRTUAL_ENV}/bin/python"; \
    test "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "${PYTHON_VERSION}"

# Blackwell-enabled PyTorch (cu128). Pinned exact version; never floats.
RUN set -eux; \
    python -m pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"

# nvidia-modelopt from the SHA-256-verified local wheel *only*. The [hf] extra
# (safetensors / accelerate / datasets) is requested on the local file itself, so
# the extra's dependencies resolve from the index while nvidia-modelopt is never
# re-resolved or replaced by an unverified same-named distribution from PyPI.
# direct_url.json is asserted afterwards: it proves the installed distribution
# came from the verified file, which a version string alone cannot.
RUN set -eux; \
    mkdir -p ${PUBLIC_WORKSPACE}/wheel; \
    curl -fSL -o "${PUBLIC_ARTIFACT_PATH}" "${MODELOPT_WHEEL_URL}"; \
    echo "${MODELOPT_WHEEL_SHA256}  ${PUBLIC_ARTIFACT_PATH}" | sha256sum -c -; \
    python -m pip install "${PUBLIC_ARTIFACT_PATH}[hf]" "${TRANSFORMERS_SPEC}"; \
    python -c "import importlib.metadata as md, json; d = md.distribution('nvidia-modelopt'); assert d.version == '${MODELOPT_VERSION}', d.version; url = json.loads(d.read_text('direct_url.json'))['url']; assert url.startswith('file://') and url.endswith('/${MODELOPT_WHEEL_FILENAME}'), url"; \
    python -c "import importlib.metadata as md; v = md.version('torch'); assert v.split('+')[0] == '${TORCH_VERSION}', v"; \
    rm -rf ${PUBLIC_WORKSPACE}/wheel; \
    python -c "import modelopt; assert modelopt.__version__ == '${MODELOPT_VERSION}', modelopt.__version__"

# examples/hf_ptq/hf_ptq.py from the exact pinned commit (immutable checkout).
RUN set -eux; \
    git clone --filter=blob:none "${MODELOPT_REPO_URL}" "${MODELOPT_ROOT}"; \
    cd "${MODELOPT_ROOT}"; \
    git checkout --detach "${MODELOPT_COMMIT}"; \
    test "$(git rev-parse HEAD)" = "${MODELOPT_COMMIT}"; \
    git rev-parse HEAD > "${MODELOPT_ROOT}/.modelopt_commit"; \
    test -f "${MODELOPT_ROOT}/examples/hf_ptq/hf_ptq.py"

# Extra runtime deps used by examples/hf_ptq (datasets/accelerate come from the
# [hf] extra; pin PyYAML/omegaconf per the pin.json dependency contract).
# Fail-closed: build errors are never swallowed on this path.
RUN set -eux; \
    python -m pip install "PyYAML>=6.0" "omegaconf>=2.3.0"

LABEL org.opencontainers.image.title="model-forge-modelopt" \
      org.opencontainers.image.description="Pinned NVIDIA ModelOpt NVFP4 quantization runtime for Qwen3.8-27B (Blackwell sm_120)" \
      org.opencontainers.image.source="https://github.com/NVIDIA/Model-Optimizer" \
      com.hangglidersrule.modelopt.version="${MODELOPT_VERSION}" \
      com.hangglidersrule.modelopt.commit="${MODELOPT_COMMIT}" \
      com.hangglidersrule.modelopt.wheel_sha256="${MODELOPT_WHEEL_SHA256}"

WORKDIR ${MODELOPT_ROOT}
