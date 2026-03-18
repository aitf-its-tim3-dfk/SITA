"""
Experiment configuration schema.

Configs are defined as dataclasses and loaded from YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for the model loader."""

    name: str  # registry key, e.g. "hf_causal_lm", "hf_vlm"
    pretrained: str  # HF model id or local path
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """Configuration for the PEFT adapter."""

    name: str  # registry key, e.g. "lora", "qlora", "prefix_tuning"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for the dataset loader."""

    name: str  # registry key, e.g. "hf_dataset"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportingConfig:
    """Configuration for experiment reporting/logging."""

    wandb: bool = False
    wandb_project: str = "sita"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    name: str = "loss"  # registry key
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """Configuration for which trainer to use."""

    name: str = "hf_trainer"  # registry key, "hf_trainer" or "custom_loop"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "experiment"
    seed: int = 42
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="", pretrained="")
    )
    adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig(name=""))
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name=""))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _dict_to_dataclass(cls, data: dict) -> Any:
    """Recursively convert a dict to a dataclass, ignoring unknown fields."""
    import typing

    if not isinstance(data, dict):
        return data

    # Resolve type hints properly (handles `from __future__ import annotations`)
    try:
        resolved_hints = typing.get_type_hints(cls)
    except Exception:
        resolved_hints = {}

    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k in field_names:
            field_type = resolved_hints.get(k)
            # resolve nested dataclasses
            if (
                isinstance(v, dict)
                and field_type is not None
                and hasattr(field_type, "__dataclass_fields__")
            ):
                filtered[k] = _dict_to_dataclass(field_type, v)
            else:
                # coerce primitive types so that e.g. "2e-4" to float works
                if field_type in (float, int, bool, str) and not isinstance(
                    v, field_type
                ):
                    try:
                        v = field_type(v)
                    except (ValueError, TypeError):
                        pass
                filtered[k] = v
    return cls(**filtered)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a YAML mapping at top level, got {type(raw).__name__}"
        )

    return _dict_to_dataclass(ExperimentConfig, raw)
