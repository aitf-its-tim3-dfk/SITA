"""Abstract base class for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class BaseEvaluator(ABC):
    """Base class for evaluation methods.

    To register a new evaluator::

        from sita.core.registry import EVALUATOR_REGISTRY

        @EVALUATOR_REGISTRY.register("my_eval")
        class MyEvaluator(BaseEvaluator):
            def evaluate(self, model, tokenizer, dataset, **kwargs):
                return {"my_metric": 0.95}
    """

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        dataset: Any,
        **kwargs,
    ) -> dict[str, float]:
        """Run evaluation and return metrics.

        Args:
            model: The trained/adapted model.
            tokenizer: Tokenizer or processor.
            dataset: The evaluation dataset.
            **kwargs: Additional evaluator-specific arguments.

        Returns:
            Dict mapping metric names to float values.
        """
        ...
