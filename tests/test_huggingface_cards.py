import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = REPO_ROOT / "publication" / "huggingface"
TARGETS = {
    "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8",
    "Darkstar-Qwen3.8-27B-Abliterated-BF16",
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8",
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4",
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16",
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4",
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4",
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16",
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4",
}
EXPECTED_METADATA = {
    "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8": (
        "Qwen/Qwen3.8-27B",
        "quantized",
    ),
    "Darkstar-Qwen3.8-27B-Abliterated-BF16": (
        "Qwen/Qwen3.8-27B",
        "finetune",
    ),
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8": (
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16",
        "quantized",
    ),
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4": (
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "quantized",
    ),
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16": (
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "finetune",
    ),
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4": (
        "HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16",
        "quantized",
    ),
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4": (
        "HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16",
        "quantized",
    ),
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16": (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "finetune",
    ),
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4": (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "quantized",
    ),
}
SOURCE_CARDS = {
    "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8": (
        REPO_ROOT / "models/qwen3.8-27b-r3/model-card/base-nvfp4.md"
    ),
    "Darkstar-Qwen3.8-27B-Abliterated-BF16": (
        REPO_ROOT / "models/qwen3.8-27b-r3/model-card/bf16.md"
    ),
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8": (
        REPO_ROOT / "models/qwen3.8-27b-r3/model-card/nvfp4.md"
    ),
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16": (
        REPO_ROOT / "models/nemotron-3.5-lightning-r1/model-card/abliterated-bf16.md"
    ),
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4": (
        REPO_ROOT / "models/nemotron-3.5-lightning-r1/model-card/base-nvfp4.md"
    ),
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4": (
        REPO_ROOT / "models/nemotron-3.5-lightning-r1/model-card/abliterated-nvfp4.md"
    ),
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4": (
        REPO_ROOT / "models" / "nemotron-3-nano-omni-r1" / "model-card" / "base-nvfp4.md"
    ),
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16": (
        REPO_ROOT / "models" / "nemotron-3-nano-omni-r1" / "model-card" / "abliterated-bf16.md"
    ),
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4": (
        REPO_ROOT / "models" / "nemotron-3-nano-omni-r1" / "model-card" / "abliterated-nvfp4.md"
    ),
}
EXPECTED_LICENSE = {
    "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8": "apache-2.0",
    "Darkstar-Qwen3.8-27B-Abliterated-BF16": "apache-2.0",
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8": "apache-2.0",
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4": "other",
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16": "other",
    "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4": "other",
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4": "other",
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16": "other",
    "Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4": "other",
}
EXPECTED_PUBLIC_REVISIONS = {
    "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8":
        "c3f03c5bf5a28a636d72cd979323ff2f80668fb0",
    "Darkstar-Qwen3.8-27B-Abliterated-BF16":
        "0181d5d178a15c694b1d6708d3ee3d08d2d9db5e",
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8":
        "2e25bd97fd1b6e6c7989e74c261d93a8702496e8",

    "HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4":
        "7c67d5ca5731d690ab0950773a0a625cf5bb0231",
    "HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16":
        "db2ea4ae7563b78ac29953dae6358982a272a4c8",
    "HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4":
        "06c7cc43f7f640e929c91df8a77f418d4eba5f99",
}
STALE_UPLOAD_WORDING = (
    "card staged",
    "card and weights pending",
    "weights pending upload",
    "currently contains only",
    "currently `.gitattributes` only",
    "does not claim",
    "becomes usable after weights",
)


def test_huggingface_cards_are_exactly_the_three_owned_targets() -> None:
    cards = sorted(HF_ROOT.glob("*/README.md"))
    assert {card.parent.name for card in cards} == TARGETS
    assert len(cards) == 9

    forbidden = (
        "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16",
        "PLACEHOLDER",
        "/d/",
        "/Volumes/",
        "C:\\",
        "](../",
    )
    for card in cards:
        text = card.read_text(encoding="utf-8")
        front_matter = text.split("---", 2)[1]
        metadata = yaml.safe_load(front_matter)
        expected_base, expected_relation = EXPECTED_METADATA[card.parent.name]

        assert metadata["license"] == EXPECTED_LICENSE[card.parent.name]
        assert metadata["base_model"] == expected_base
        assert metadata["base_model_relation"] == expected_relation
        assert f"# {card.parent.name}" in text
        assert "**Private checkpoint repository.**" not in text
        if "Qwen" in card.parent.name:
            assert f"vllm serve HangGlidersRule/{card.parent.name}" in text
            assert "--max-num-seqs 16" in text
        else:
            assert "--max-model-len" in text
        assert not any(value in text.lower() for value in STALE_UPLOAD_WORDING)
        assert re.search(r"\d+\.\d{4,}%", text) is None
        assert re.search(r"\d+\.\d{4,}\s+tok/s", text) is None
        assert not any(value in text for value in forbidden)


def test_source_cards_match_huggingface_identity_and_uploaded_state() -> None:
    for target, source_card in SOURCE_CARDS.items():
        text = source_card.read_text(encoding="utf-8")
        metadata = yaml.safe_load(text.split("---", 2)[1])
        expected_base, expected_relation = EXPECTED_METADATA[target]

        assert f"# {target}" in text
        assert metadata["base_model"] == expected_base
        assert metadata["base_model_relation"] == expected_relation
        assert "**Private checkpoint repository.**" not in text
        # Publication claim: gold-standard Release reference (immutable tag +
        # contract) or the explicit public-on-HF phrase (older variant).
        assert (
            "This immutable tag exists and the release contract is published." in text
            or "public on Hugging Face" in text
        )
        if "Qwen" in target:
            assert "darkstar-qwen3.8-27b-v1.0.0" in text
            assert f"vllm serve HangGlidersRule/{target}" in text
            assert "--max-num-seqs 16" in text
        elif "Lightning" in target:
            assert "darkstar-nemotron-3.5-lightning-v1.0.0" in text
            assert "--max-model-len" in text
        else:
            assert "darkstar-nemotron-3-nano-omni-v1.0.0" in text
            assert f"vllm serve HangGlidersRule/{target}" in text
            assert "--max-num-seqs 16" in text
        assert not any(value in text.lower() for value in STALE_UPLOAD_WORDING)
        assert re.search(r"\d+\.\d{4,}%", text) is None
        assert re.search(r"\d+\.\d{4,}\s+tok/s", text) is None


def test_public_manifest_names_exactly_the_three_huggingface_cards() -> None:
    manifest = yaml.safe_load(
        (REPO_ROOT / "tools/public_export/public-files.yaml").read_text(encoding="utf-8")
    )
    sources = {
        rule["source"]
        for rule in manifest["rules"]
        if rule["source"].startswith("publication/huggingface/")
    }
    assert sources == {
        f"publication/huggingface/{target}/README.md" for target in TARGETS
    }


def test_release_ledger_pins_public_revisions_without_contract_publication_claim() -> None:
    import json

    ledger = json.loads(
        (REPO_ROOT / "models/qwen3.8-27b-r3/results/publication-readiness-ledger.json")
        .read_text(encoding="utf-8")
    )
    owned = [product for product in ledger["products"] if product["role"] != "base-bf16"]
    assert len(owned) == 3
    for product in owned:
        target = product["target_repository"].split("/", 1)[1]
        gates = {gate["id"]: gate for gate in product["gates"]}
        assert EXPECTED_PUBLIC_REVISIONS[target] in gates["publication_targets_hf_ghcr"]["evidence"]
        assert gates["model_card_final"]["status"] == "verified"
        assert gates["clean_download_boot_smoke"]["status"] == "verified"
        assert gates["release_tag"]["status"] == "verified"
        assert product["publication_claim"] is True
