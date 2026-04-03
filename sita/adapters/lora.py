"""LoRA adapter using HuggingFace PEFT."""

from __future__ import annotations

from torch import nn
from peft import LoraConfig, get_peft_model

from sita.core.base_adapter import BaseAdapter
from sita.core.config import AdapterConfig
from sita.core.registry import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("lora")
class LoRAAdapter(BaseAdapter):
    """Standard LoRA via HuggingFace PEFT.

    All kwargs in the config are forwarded to `LoraConfig`, so you
    can configure `r`, `lora_alpha`, `lora_dropout`, `target_modules`,
    `task_type`, etc. directly from YAML.

    Example YAML::

        adapter:
          name: lora
          kwargs:
            r: 16
            lora_alpha: 32
            lora_dropout: 0.05
            target_modules: [q_proj, v_proj]
            task_type: CAUSAL_LM
    """

    def apply(self, model: nn.Module, config: AdapterConfig) -> nn.Module:
        lora_config = LoraConfig(**config.kwargs)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def save(self, model: nn.Module, path: str) -> None:
        model.save_pretrained(path)

    def load(self, model: nn.Module, path: str) -> nn.Module:
        """Load adapter weights into an already-adapted model (no double-wrapping)."""
        model.load_adapter(path, is_trainable=True)
        return model
