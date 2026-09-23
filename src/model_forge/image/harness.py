"""Image-model quality evaluation harness (the D2/D4 gates).

Runs the frozen case manifest through a plain diffusers pipeline
(dequant-on-load consumption model), captures artifacts, and scores with
full-denominator semantics. Mirrors ``gpqa/harness.py``: every case gets a
scored outcome; the run summary records the manifest hash, model revision, and
config so results are reproducible and hash-frozen.
"""

from __future__ import annotations

import glob
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_forge.image.case_manifest import (
    ImageCase,
    manifest_sha256,
)


@dataclass
class CaseResult:
    case_id: str
    capability: str
    status: str  # "ok" | "blank" | "error"
    mean_px: float | None = None
    std_px: float | None = None
    artifact: str | None = None
    error: str | None = None
    secs: float | None = None


@dataclass
class RunSummary:
    model_source: str
    model_revision: str
    manifest_sha256: str
    started: str
    results: list[CaseResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_source": self.model_source,
            "model_revision": self.model_revision,
            "manifest_sha256": self.manifest_sha256,
            "started": self.started,
            "config": self.config,
            "results": [asdict(r) for r in self.results],
            "capability_scores": self.capability_scores(),
        }

    def capability_scores(self) -> dict[str, dict[str, float]]:
        """Full-denominator pass rates per capability.

        pass = status ok AND pixel stats sane (non-blank). No skip-on-fail:
        errors and blanks count in the denominator as failures.
        """
        out: dict[str, dict[str, float]] = {}
        caps = sorted({r.capability for r in self.results})
        for cap in caps:
            rs = [r for r in self.results if r.capability == cap]
            passed = sum(1 for r in rs if r.status == "ok" and (r.std_px or 0) > 1.0)
            out[cap] = {
                "pass": passed,
                "denominator": len(rs),
                "rate": passed / len(rs) if rs else 0.0,
            }
        return out


def load_pipeline(model_source: str, device: str = "cuda"):
    """Load the Qwen-Image-2.1 pipeline from an HF snapshot or local dir."""
    import torch
    from diffusers import QwenImage21Pipeline

    candidates = sorted(glob.glob(f"{model_source}/snapshots/*")) if "/" not in model_source or "models--" in model_source else [model_source]
    path = candidates[0] if candidates else model_source
    pipe = QwenImage21Pipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    return pipe.to(device)


def render_case(pipe, case: ImageCase, out_dir: Path, input_image=None):
    """Generate one case; returns (CaseResult, image|None)."""
    import numpy as np
    import torch

    t0 = time.time()
    try:
        generator = torch.Generator("cuda").manual_seed(case.seed)
        kwargs = dict(
            prompt=case.prompt or case.edit_instruction or "",
            width=case.width,
            height=case.height,
            num_inference_steps=case.steps,
            true_cfg_scale=case.guidance,
            generator=generator,
        )
        if case.capability == "editing":
            if input_image is None:
                return CaseResult(case.case_id, case.capability, "error", error="missing input image"), None
            kwargs["image"] = input_image
        img = pipe(**kwargs).images[0]
        a = np.array(img)
        result = CaseResult(
            case.case_id,
            case.capability,
            status="ok" if a.max() > 0 else "blank",
            mean_px=round(float(a.mean()), 1),
            std_px=round(float(a.std()), 1),
            artifact=str(out_dir / f"{case.case_id}.png"),
            secs=round(time.time() - t0, 2),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        img.save(out_dir / f"{case.case_id}.png")
        return result, img
    except Exception as e:  # noqa: BLE001 — full-denominator: errors are results
        return CaseResult(case.case_id, case.capability, "error", error=str(e)[:200], secs=round(time.time() - t0, 2)), None


def run_eval(
    *,
    model_source: str,
    model_revision: str,
    out_root: Path,
    capabilities: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> RunSummary:
    """Run the frozen manifest; write summary.json + artifacts."""
    from model_forge.image.case_manifest import FROZEN_CASES

    pipe = load_pipeline(model_source)
    out_dir = out_root / "artifacts"
    summary = RunSummary(
        model_source=model_source,
        model_revision=model_revision,
        manifest_sha256=manifest_sha256(),
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=config or {},
    )
    caps = capabilities or ["t2i_quality", "composition", "typography", "rgba", "editing"]
    rendered: dict[str, object] = {}
    for case in FROZEN_CASES:
        if case.capability not in caps:
            continue
        input_image = None
        if case.capability == "editing" and case.input_case_id:
            p = out_dir / f"{case.input_case_id}.png"
            if p.exists():
                from PIL import Image as PILImage

                input_image = PILImage.open(p).convert("RGB")
        result, img = render_case(pipe, case, out_dir, input_image)
        if img is not None:
            rendered[case.case_id] = img
        summary.results.append(result)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary.to_dict(), indent=1))
    return summary


def run_behavior_suite(
    *,
    model_source: str,
    model_revision: str,
    out_root: Path,
    config: dict[str, Any] | None = None,
) -> RunSummary:
    """Run benign + harm behavior suites (E1 measurement; judge scoring is D-phase)."""
    from model_forge.image.case_manifest import BENIGN_SUITE, HARM_SUITE

    pipe = load_pipeline(model_source)
    out_dir = out_root / "artifacts"
    summary = RunSummary(
        model_source=model_source,
        model_revision=model_revision,
        manifest_sha256=manifest_sha256(),
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=config or {},
    )
    for suite in (BENIGN_SUITE, HARM_SUITE):
        for case in suite:
            result, _ = render_case(pipe, case, out_dir)
            summary.results.append(result)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "behavior-summary.json").write_text(json.dumps(summary.to_dict(), indent=1))
    return summary
