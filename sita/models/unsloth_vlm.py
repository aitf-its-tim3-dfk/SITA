"""Unsloth Vision-Language Model loader."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sita.core.base_model import BaseModelLoader
from sita.core.config import ModelConfig
from sita.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("unsloth_vlm")
class UnslothVLMLoader(BaseModelLoader):
    """Load an Unsloth FastVisionModel + Tokenizer.

    Uses ``FastVisionModel.from_pretrained`` which returns (model, tokenizer).
    Config kwargs are forwarded directly, so you can pass ``max_seq_length``,
    ``load_in_4bit``, ``load_in_16bit``, ``dtype``, etc.

    Example YAML::

        model:
          name: unsloth_vlm
          pretrained: unsloth/Qwen3.5-0.8B
          kwargs:
            max_seq_length: 2048
            load_in_4bit: true
    """

    def load(self, config: ModelConfig) -> tuple[nn.Module, Any]:
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it using instructions from "
                "https://github.com/unslothai/unsloth"
            )

        kwargs = dict(config.kwargs)

        # handle string dtype specs like "float16", "bfloat16", "None"
        if "dtype" in kwargs and isinstance(kwargs["dtype"], str):
            if kwargs["dtype"] == "None":
                kwargs["dtype"] = None
            else:
                kwargs["dtype"] = getattr(torch, kwargs["dtype"])

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.pretrained,
            **kwargs,
        )

        return model, tokenizer
