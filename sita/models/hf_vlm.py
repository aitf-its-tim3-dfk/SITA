"""HuggingFace Vision-Language Model loader."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

from sita.core.base_model import BaseModelLoader
from sita.core.config import ModelConfig
from sita.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("hf_vlm")
class HFVLMLoader(BaseModelLoader):
    """Load a HuggingFace Vision-Language Model + Processor.

    This uses AutoModel + AutoProcessor, which works for models like
    LLaVA, Qwen-VL, InternVL, PaliGemma, etc.

    For models that need a specific AutoClass (e.g. LlavaForConditionalGeneration),
    pass the class name in kwargs::

        model:
          name: hf_vlm
          pretrained: llava-hf/llava-1.5-7b-hf
          kwargs:
            auto_class: LlavaForConditionalGeneration
            torch_dtype: float16
            device_map: auto

    The adapter layer works identically on VLMs — LoRA patches nn.Linear
    regardless of whether it's in a vision encoder or language decoder.
    """

    def load(self, config: ModelConfig) -> tuple[nn.Module, Any]:
        kwargs = dict(config.kwargs)

        # handle dtype
        if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])

        # optionally use a specific model class instead of AutoModel
        auto_class_name = kwargs.pop("auto_class", None)
        if auto_class_name:
            import transformers
            auto_class = getattr(transformers, auto_class_name)
        else:
            auto_class = AutoModel

        model = auto_class.from_pretrained(config.pretrained, **kwargs)
        processor = AutoProcessor.from_pretrained(config.pretrained)

        for attr, value in config.tokenizer_kwargs.items():
            setattr(processor, attr, value)

        return model, processor
