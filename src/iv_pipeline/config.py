from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    sampler_model: ModelConfig
    constraint_model: ModelConfig
    verifier_model: ModelConfig
    max_samples: int = 1


def default_config() -> PipelineConfig:
    return PipelineConfig(
        sampler_model=ModelConfig(name="mock_math"),
        constraint_model=ModelConfig(name="mock_constraints"),
        verifier_model=ModelConfig(name="mock_verifier"),
        max_samples=1,
    )


def load_config(path: str | Path | None) -> PipelineConfig:
    if path is None:
        return default_config()
    data = json.loads(Path(path).read_text())
    return PipelineConfig(
        sampler_model=ModelConfig(**data["sampler_model"]),
        constraint_model=ModelConfig(**data["constraint_model"]),
        verifier_model=ModelConfig(**data["verifier_model"]),
        max_samples=int(data.get("max_samples", 1)),
    )
