"""Abstract base class for model loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from sita.core.config import ModelConfig


class BaseModelLoader(ABC):
    """Base class for loading models.

    A model loader is responsible for instantiating a model and its
    associated tokenizer/processor from a config.

    To register a new model loader::

        from sita.core.registry import MODEL_REGISTRY

        @MODEL_REGISTRY.register("my_model")
        class MyModelLoader(BaseModelLoader):
            def load(self, config):
                model = ...
                tokenizer = ...
                return model, tokenizer
    """

    @abstractmethod
    def load(self, config: ModelConfig) -> tuple[nn.Module, Any]:
        """Load and return (model, tokenizer_or_processor).

        Args:
            config: Model configuration with pretrained path and kwargs.

        Returns:
            Tuple of (model, tokenizer/processor). The second element is
            intentionally `Any` — it can be a tokenizer for LLMs or a
            processor for VLMs. Downstream components handle both.
        """
        ...
