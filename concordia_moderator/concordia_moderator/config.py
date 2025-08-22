from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModeratorConfig:
    seed: int = 7
    followup_limit: int = 2
    logic_threshold: float = 0.65
    plausibility_threshold: float = 0.65
    minority_prompting: bool = True
    minority_min_share: float = 0.15
    log_path: Optional[str] = "moderation_log.jsonl"


@dataclass
class ModelConfig:
    name: str = "openai-gpt4o"
    temperature: float = 0.7


@dataclass
class EmbeddingConfig:
    provider: str = "openai" # or "sentence_tfm"
    model: str = "text-embedding-3-large"


@dataclass
class AppConfig:
    cfg: ModeratorConfig
    model: ModelConfig
    embedding: EmbeddingConfig


@staticmethod
def load(path: str | Path) -> "AppConfig":
    data = yaml.safe_load(Path(path).read_text())
    mc = ModelConfig(**data.get("model", {}))
    ec = EmbeddingConfig(**data.get("embedding", {}))
    c = ModeratorConfig(**{k: v for k, v in data.items() if k in ModeratorConfig.__annotations__})
    return AppConfig(cfg=c, model=mc, embedding=ec)