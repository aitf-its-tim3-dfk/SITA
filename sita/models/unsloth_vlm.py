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

    If ``chat_template`` is provided in kwargs, it is resolved via
    :func:`sita.templates.load_chat_template` (built-in name or file path)
    and applied to the tokenizer after loading.

    Example YAML::

        model:
          name: unsloth_vlm
          pretrained: unsloth/Qwen3.5-0.8B
          kwargs:
            max_seq_length: 2048
            load_in_4bit: true
            chat_template: qwen3.5_chatml
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

        chat_template = kwargs.pop("chat_template", None)

        if "dtype" in kwargs and kwargs["dtype"] in ("None", None):
            kwargs.pop("dtype", None)

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.pretrained,
            **kwargs,
        )

        if chat_template is not None:
            from sita.templates import load_chat_template
            template_str = load_chat_template(chat_template)
            if template_str is not None:
                tokenizer.chat_template = template_str
            else:
                raise FileNotFoundError(
                    f"Chat template '{chat_template}' not found. "
                    f"Place a .jinja file in sita/templates/ or provide a full path."
                )

        # Apply any tokenizer overrides (e.g. padding_side, pad_token, etc.)
        # Set on both the processor wrapper AND the inner tokenizer so that
        # model.generate() (which checks the inner one) picks it up too.
        for attr, value in config.tokenizer_kwargs.items():
            setattr(tokenizer, attr, value)
            if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, attr):
                setattr(tokenizer.tokenizer, attr, value)

        return model, tokenizer
