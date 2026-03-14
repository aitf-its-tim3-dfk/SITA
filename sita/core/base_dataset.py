"""Abstract base class for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sita.core.config import DatasetConfig


class BaseDatasetLoader(ABC):
    """Base class for loading & preprocessing datasets.

    To register a new dataset loader::

        from sita.core.registry import DATASET_REGISTRY

        @DATASET_REGISTRY.register("my_dataset")
        class MyDatasetLoader(BaseDatasetLoader):
            def load(self, config, tokenizer):
                train_ds = ...
                eval_ds = ...
                return train_ds, eval_ds
    """

    @abstractmethod
    def load(self, config: DatasetConfig, tokenizer: Any) -> tuple[Any, Any | None]:
        """Load and preprocess a dataset.

        Args:
            config: Dataset configuration with name and kwargs.
            tokenizer: The tokenizer or processor from the model loader.
                       For VLMs this is a processor; for LLMs a tokenizer.

        Returns:
            Tuple of (train_dataset, eval_dataset).
            eval_dataset may be None if no eval split exists.
        """
        ...
