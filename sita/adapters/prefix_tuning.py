"""Prefix Tuning adapter using HuggingFace PEFT."""

from __future__ import annotations

from torch import nn
from peft import PrefixTuningConfig, PeftModel, get_peft_model

from sita.core.base_adapter import BaseAdapter
from sita.core.config import AdapterConfig
from sita.core.registry import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("prefix_tuning")
class PrefixTuningAdapter(BaseAdapter):
    """Prefix Tuning via HuggingFace PEFT.

    Example YAML::

        adapter:
          name: prefix_tuning
          kwargs:
            num_virtual_tokens: 20
            task_type: CAUSAL_LM
    """

    def apply(self, model: nn.Module, config: AdapterConfig) -> nn.Module:
        pt_config = PrefixTuningConfig(**config.kwargs)
        model = get_peft_model(model, pt_config)
        model.print_trainable_parameters()
        return model

    def save(self, model: nn.Module, path: str) -> None:
        model.save_pretrained(path)

    def load(self, model: nn.Module, path: str) -> nn.Module:
        return PeftModel.from_pretrained(model, path)
