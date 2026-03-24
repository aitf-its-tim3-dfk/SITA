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

        # handle dtype synonyms
        dtype_key = "torch_dtype" if "torch_dtype" in kwargs else "dtype"
        if dtype_key in kwargs and isinstance(kwargs[dtype_key], str):
            if kwargs[dtype_key] == "None":
                kwargs[dtype_key] = None
            else:
                kwargs[dtype_key] = getattr(torch, kwargs[dtype_key])
        
        # Ensure we use torch_dtype for AutoModel
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")

        # optionally use a specific model class instead of AutoModel
        auto_class_name = kwargs.pop("auto_class", None)
        
        # Separate processor kwargs if provided, otherwise default to empty
        processor_kwargs = kwargs.pop("processor_kwargs", {})
        if not isinstance(processor_kwargs, dict):
            processor_kwargs = {}

        if auto_class_name:
            import transformers
            auto_class = getattr(transformers, auto_class_name)
        else:
            auto_class = AutoModel

        model = auto_class.from_pretrained(config.pretrained, **kwargs)
        
        # Merge processor_kwargs with common defaults if needed
        processor = AutoProcessor.from_pretrained(config.pretrained, **processor_kwargs)

        # Set on both the processor wrapper AND the inner tokenizer so that
        # model.generate() (which checks the inner one) picks it up too.
        for attr, value in config.tokenizer_kwargs.items():
            setattr(processor, attr, value)
            if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, attr):
                setattr(processor.tokenizer, attr, value)

        return model, processor
