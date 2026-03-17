"""LoRA adapter using Unsloth."""

from __future__ import annotations

from typing import Any

from torch import nn

from sita.core.base_adapter import BaseAdapter
from sita.core.config import AdapterConfig
from sita.core.registry import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("unsloth_lora")
class UnslothLoRAAdapter(BaseAdapter):
    """LoRA adapter optimized by Unsloth.

    All kwargs in the config are forwarded to `FastLanguageModel.get_peft_model`,
    so you can configure `r`, `target_modules`, `lora_alpha`, `lora_dropout`, etc.

    Example YAML::

        adapter:
          name: unsloth_lora
          kwargs:
            r: 16
            target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"]
            lora_alpha: 16
            lora_dropout: 0
            bias: "none"
            use_gradient_checkpointing: "unsloth"
            random_state: 3407
            use_rslora: False
            loftq_config: None
    """

    def apply(self, model: nn.Module, config: AdapterConfig) -> nn.Module:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it using instructions from "
                "https://github.com/unslothai/unsloth"
            )
            
        model = FastLanguageModel.get_peft_model(
            model,
            **config.kwargs
        )
        model.print_trainable_parameters()
        return model

    def save(self, model: nn.Module, path: str) -> None:
        model.save_pretrained(path)

    def load(self, model: nn.Module, path: str) -> nn.Module:
        # For Unsloth, usually loading is done via FastLanguageModel.from_pretrained
        # but if we just want to load the PEFT weights into the base model:
        from peft import PeftModel
        return PeftModel.from_pretrained(model, path)
