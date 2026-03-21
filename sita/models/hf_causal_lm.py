"""HuggingFace Causal LM model loader."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sita.core.base_model import BaseModelLoader
from sita.core.config import ModelConfig
from sita.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("hf_causal_lm")
class HFCausalLMLoader(BaseModelLoader):
    """Load any HuggingFace AutoModelForCausalLM + AutoTokenizer.

    Config kwargs are forwarded directly to `from_pretrained`, so you
    can pass things like `torch_dtype`, `device_map`, `attn_implementation`, etc.

    Example YAML::

        model:
          name: hf_causal_lm
          pretrained: TinyLlama/TinyLlama-1.1B-Chat-v1.0
          kwargs:
            torch_dtype: float16
            device_map: auto
    """

    def load(self, config: ModelConfig) -> tuple[nn.Module, Any]:
        kwargs = dict(config.kwargs)

        # handle string dtype specs like "float16", "bfloat16"
        if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])

        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained,
            **kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for attr, value in config.tokenizer_kwargs.items():
            setattr(tokenizer, attr, value)

        return model, tokenizer
