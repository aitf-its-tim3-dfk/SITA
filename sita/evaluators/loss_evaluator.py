"""Loss-based evaluator."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sita.core.base_evaluator import BaseEvaluator
from sita.core.registry import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("loss")
class LossEvaluator(BaseEvaluator):
    """Evaluate model by computing average loss on a dataset.

    Example YAML::

        evaluation:
          name: loss
          kwargs:
            batch_size: 8
    """

    def evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        dataset: Any,
        **kwargs,
    ) -> dict[str, float]:
        batch_size = kwargs.get("batch_size", 8)

        model.eval()
        device = next(model.parameters()).device

        dataloader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {k: v.to(device) for k, v in batch.items()}
                if "labels" not in inputs and "input_ids" in inputs:
                    inputs["labels"] = inputs["input_ids"].clone()

                outputs = model(**inputs)
                total_loss += outputs.loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }
