"""KV-cache dtype / attention-backend compatibility matrix tests.

Encodes the SM120-measured constraints (2026-09-15/16 campaigns):
- fp8-family KV requires TRITON_ATTN or FLASHINFER (FLASH_ATTN fails engine
  init on SM120 with 'FP8 KV cache requires FA3 on SM90 or FA4 on SM100').
  fp8 + TRITON_ATTN on stock 0.27.1 additionally carries a load-triggered
  IMA race (measured 2026-09-15) — refused at validation.
- nvfp4 KV is VALID on the nvfp4kv nightly runtime (0.27.2rc1.dev77 with
  PR #49891 rebased FA2-nvfp4 routing + sm120 linear-V-scale overlay) with
  FLASHINFER — measured 2026-09-16: GPQA 84.85% (bf16 baseline 85.35%),
  zero IMAs under concurrency-30 thinking load, needles to 210K, KV pool
  1.05M tokens at util 0.50. Refused on any other image.
- turboquant KV remains refused (never validated).
- bf16 (native) KV stays valid on every backend and every runtime — the
  always-available co-option.
"""
import pytest

from model_forge.serve_profile import ServeProfile, ServeProfileError

REPO = "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
NVFP4KV_IMAGE = "vllm-qwen38:nvfp4kv"


def _profile(**overrides):
    base = dict(
        family="qwen38",
        behavior="abliterated",
        format="nvfp4",
        repository_id=REPO,
        model_path=REPO,
        container_name="vllm-darkstar-qwen38-abliterated-modelopt",
        mtp_depth=10,
        scheduler_tokens=32768,
        kv_cache_dtype="bf16",
        attention_backend="FLASH_ATTN",
    )
    base.update(overrides)
    return ServeProfile(**base)


def test_bf16_with_flash_attn_is_valid():
    _profile().validate()


def test_bf16_with_flashinfer_is_valid():
    # native KV stays valid on every backend — always the co-option
    _profile(attention_backend="FLASHINFER").validate()


def test_fp8_kv_with_triton_attn_on_stock_is_refused_race():
    # Measured 2026-09-15: boots clean at low load, IMAs under
    # concurrency-30 thinking load (with/without async-scheduling,
    # prefix caching, or mamba align changes).
    with pytest.raises(ServeProfileError, match=r"load-triggered cudaErrorIllegalAddress race"):
        _profile(kv_cache_dtype="fp8_e4m3", attention_backend="TRITON_ATTN").validate()


def test_fp8_kv_with_flash_attn_is_refused():
    # Measured live on SM120: engine init raises
    # 'FP8 KV cache requires FA3 on SM90 or FA4 on SM100'.
    with pytest.raises(ServeProfileError, match=r"FLASH_ATTN does not support FP8 KV"):
        _profile(kv_cache_dtype="fp8_e4m3", attention_backend="FLASH_ATTN").validate()


def test_fp8_kv_with_flashinfer_is_valid():
    _profile(kv_cache_dtype="fp8", attention_backend="FLASHINFER").validate()


def test_nvfp4_kv_on_nvfp4kv_runtime_with_flashinfer_is_valid():
    # THE 2026-09-16 PRODUCTION CONFIG — measured end-to-end.
    _profile(
        kv_cache_dtype="nvfp4",
        attention_backend="FLASHINFER",
        image=NVFP4KV_IMAGE,
    ).validate()


def test_nvfp4_kv_requires_flashinfer():
    with pytest.raises(ServeProfileError, match=r"nvfp4 requires attention_backend FLASHINFER"):
        _profile(kv_cache_dtype="nvfp4", attention_backend="TRITON_ATTN", image=NVFP4KV_IMAGE).validate()


def test_nvfp4_kv_requires_nvfp4kv_runtime():
    # stock 0.27.1 gates nvfp4 KV to SM100-trtllm-gen only
    with pytest.raises(ServeProfileError, match=r"requires the nvfp4kv-nightly"):
        _profile(kv_cache_dtype="nvfp4", attention_backend="FLASHINFER").validate()


def test_nvfp4_kv_compose_emits_dtype_and_backend():
    rendered = _profile(
        kv_cache_dtype="nvfp4",
        attention_backend="FLASHINFER",
        image=NVFP4KV_IMAGE,
    ).compose()
    command = rendered["services"]["vllm"]["command"]
    assert command[command.index("--kv-cache-dtype") + 1] == "nvfp4"


@pytest.mark.parametrize("dtype", ["turboquant_k8v4", "turboquant_4bit_nc"])
def test_unvalidated_kv_dtypes_are_refused(dtype):
    with pytest.raises(ServeProfileError, match="not validated"):
        _profile(kv_cache_dtype=dtype, attention_backend="TRITON_ATTN").validate()


def test_fp8_kv_compose_emits_flag_pair():
    rendered = _profile(kv_cache_dtype="fp8_e4m3", attention_backend="FLASHINFER").compose()
    command = rendered["services"]["vllm"]["command"]
    assert "--kv-cache-dtype" in command
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    env = rendered["services"]["vllm"]["environment"]
    assert any("VLLM_ATTENTION_BACKEND=FLASHINFER" in e for e in env)
