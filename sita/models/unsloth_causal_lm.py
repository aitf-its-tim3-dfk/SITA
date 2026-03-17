"""Unsloth Causal LM model loader."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sita.core.base_model import BaseModelLoader
from sita.core.config import ModelConfig
from sita.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("unsloth_causal_lm")
class UnslothCausalLMLoader(BaseModelLoader):
    """Load an Unsloth FastLanguageModel + Tokenizer.

    Config kwargs are forwarded directly to `from_pretrained`, so you
    can pass things like `load_in_4bit`, `max_seq_length`, `dtype`, etc.

    Example YAML::

        model:
          name: unsloth_causal_lm
          pretrained: unsloth/tinyllama
          kwargs:
            max_seq_length: 2048
            dtype: None
            load_in_4bit: true
    """

    def load(self, config: ModelConfig) -> tuple[nn.Module, Any]:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it using instructions from "
                "https://github.com/unslothai/unsloth"
            )

        kwargs = dict(config.kwargs)

        # handle string dtype specs like "float16", "bfloat16"
        if "dtype" in kwargs and isinstance(kwargs["dtype"], str):
            if kwargs["dtype"] == "None":
                kwargs["dtype"] = None
            else:
                kwargs["dtype"] = getattr(torch, kwargs["dtype"])
                
        # Unsloth uses `model_name` for `pretrained`
        model_name = config.pretrained

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            **kwargs,
        )

        return model, tokenizer
