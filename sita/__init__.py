"""SITA: Standardized Infrastructure for the Training of Adapters."""

__version__ = "0.1.0"

from sita.core.registry import (
    ADAPTER_REGISTRY,
    DATASET_REGISTRY,
    EVALUATOR_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
)

__all__ = [
    "MODEL_REGISTRY",
    "ADAPTER_REGISTRY",
    "DATASET_REGISTRY",
    "EVALUATOR_REGISTRY",
    "TRAINER_REGISTRY",
]
