from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator


class ModelTarget(BaseModel):
    id: str
    endpoint: str
    model: str
    api_key_env: str | None = None
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), exclude=True)
    family: str
    precision: str
    runtime: str | None = None
    revision: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Track(BaseModel):
    name: str
    reasoning: bool
    reasoning_effort: Literal["low", "medium", "xhigh"] | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42


class OutputPolicy(BaseModel):
    retain_prompts: bool = False
    retain_responses: bool = True
    redact_patterns: list[str] = Field(default_factory=list)


class BakeoffSpec(BaseModel):
    schema_version: Literal["1.0"]
    name: str
    models: list[ModelTarget]
    tracks: list[Track]
    suites: list[str]
    output: OutputPolicy = Field(default_factory=OutputPolicy)
    repeats: int = Field(default=3, ge=1)
    max_parallel: int = Field(default=2, ge=1)
    transient_statuses: list[int] = Field(default_factory=lambda: [429, 502, 503, 504])

    @model_validator(mode="after")
    def unique_ids(self) -> "BakeoffSpec":
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model id")
        return self


def load_spec(path: Path) -> BakeoffSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = BakeoffSpec.model_validate(raw)
    for target in spec.models:
        if target.api_key_env:
            target.api_key = SecretStr(os.environ.get(target.api_key_env, ""))
    return spec
