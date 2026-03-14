"""Abstract base class for trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from sita.core.config import TrainingConfig


class BaseTrainer(ABC):
    """Base class for training loops.

    SITA ships with two built-in trainers:
      - "hf_trainer": wraps HuggingFace Trainer (simple, batteries-included)
      - "custom_loop": bare PyTorch training loop (full control)

    To register your own::

        from sita.core.registry import TRAINER_REGISTRY

        @TRAINER_REGISTRY.register("my_trainer")
        class MyTrainer(BaseTrainer):
            def train(self, model, tokenizer, train_dataset, eval_dataset, config):
                # your custom training logic
                return model
    """

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any | None,
        config: TrainingConfig,
        **kwargs,
    ) -> nn.Module:
        """Run training and return the trained model.

        Args:
            model: The model (already adapted with PEFT layers).
            tokenizer: Tokenizer or processor.
            train_dataset: The training dataset.
            eval_dataset: Optional evaluation dataset.
            config: Training hyperparameters.
            **kwargs: Extra args from TrainerConfig.kwargs.

        Returns:
            The trained model.
        """
        ...
