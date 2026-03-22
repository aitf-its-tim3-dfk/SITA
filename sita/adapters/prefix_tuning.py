"""Prefix Tuning adapter using HuggingFace PEFT."""

from __future__ import annotations

import logging

from torch import nn
from peft import PrefixTuningConfig, PeftModel, get_peft_model

from sita.core.base_adapter import BaseAdapter
from sita.core.config import AdapterConfig
from sita.core.registry import ADAPTER_REGISTRY

logger = logging.getLogger("sita.adapters.prefix_tuning")


# ---------------------------------------------------------------------------
# VLM compatibility patch
# ---------------------------------------------------------------------------


def _patch_rope_for_prefix_tuning(model: nn.Module) -> bool:
    """Monkey-patch ``get_rope_index`` for prefix tuning on VLMs.

    PEFT prefix tuning extends ``attention_mask`` by ``num_virtual_tokens`` to
    cover the prefix past_key_values, but Qwen VLMs' ``get_rope_index``
    expects ``attention_mask`` and ``input_ids`` to share the same seq-len
    dimension.

    This patch trims the prefix portion from the mask before the RoPE
    position computation, those virtual tokens live in ``past_key_values``
    and are attended to correctly: they simply don't need RoPE positions.

    Returns ``True`` if a patch was applied.
    """
    target = None
    for module in model.modules():
        if hasattr(module, "get_rope_index"):
            target = module
            break

    if target is None:
        return False

    _original = target.get_rope_index  # bound method

    def _trimmed_rope_index(input_ids, *args, attention_mask=None, **kwargs):
        if attention_mask is not None and input_ids is not None:
            diff = attention_mask.shape[-1] - input_ids.shape[-1]
            if diff > 0:
                attention_mask = attention_mask[:, diff:]
        return _original(input_ids, *args, attention_mask=attention_mask, **kwargs)

    target.get_rope_index = _trimmed_rope_index
    logger.info(
        "Patched get_rope_index for prefix tuning compatibility "
        "(trimming virtual-token prefix from attention_mask)."
    )
    return True


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


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
        # Prefix tuning is incompatible with gradient checkpointing — disable it
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

        pt_config = PrefixTuningConfig(**config.kwargs)
        model = get_peft_model(model, pt_config)
        model.print_trainable_parameters()

        # Patch VLM multimodal-RoPE if the model uses it (Qwen2-VL, Qwen3.5, …)
        _patch_rope_for_prefix_tuning(model)

        return model

    def save(self, model: nn.Module, path: str) -> None:
        model.save_pretrained(path)

    def load(self, model: nn.Module, path: str) -> nn.Module:
        return PeftModel.from_pretrained(model, path)
