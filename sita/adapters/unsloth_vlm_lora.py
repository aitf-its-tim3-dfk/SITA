"""LoRA adapter for Unsloth Vision-Language Models."""

from __future__ import annotations

from typing import Any

from torch import nn

from sita.core.base_adapter import BaseAdapter
from sita.core.config import AdapterConfig
from sita.core.registry import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("unsloth_vlm_lora")
class UnslothVLMLoRAAdapter(BaseAdapter):
    """LoRA adapter for vision-language models via Unsloth.

    Uses ``FastVisionModel.get_peft_model`` which supports vision-specific
    options like ``finetune_vision_layers``, ``finetune_language_layers``,
    ``finetune_attention_modules``, ``finetune_mlp_modules``, plus standard
    LoRA parameters (``r``, ``lora_alpha``, ``target_modules``, etc.).

    Example YAML::

        adapter:
          name: unsloth_vlm_lora
          kwargs:
            finetune_vision_layers: true
            finetune_language_layers: true
            finetune_attention_modules: true
            finetune_mlp_modules: true
            r: 16
            lora_alpha: 16
            lora_dropout: 0
            bias: "none"
            target_modules: "all-linear"
            use_gradient_checkpointing: "unsloth"
            random_state: 3407
    """

    def apply(self, model: nn.Module, config: AdapterConfig) -> nn.Module:
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it using instructions from "
                "https://github.com/unslothai/unsloth"
            )

        model = FastVisionModel.get_peft_model(
            model,
            **config.kwargs,
        )
        model.print_trainable_parameters()
        return model

    def save(self, model: nn.Module, path: str) -> None:
        model.save_pretrained(path)

    def load(self, model: nn.Module, path: str) -> nn.Module:
        """Load adapter weights into an already-adapted model (no double-wrapping)."""
        model.load_adapter(path, adapter_name="default", is_trainable=True)
        model.set_adapter("default")
        return model
