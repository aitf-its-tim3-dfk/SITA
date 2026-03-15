"""QLoRA adapter — LoRA with 4-bit quantized base model."""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from sita.core.base_adapter import BaseAdapter
from sita.core.config import AdapterConfig
from sita.core.registry import ADAPTER_REGISTRY

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register("qlora")
class QLoRAAdapter(BaseAdapter):
    """QLoRA: 4-bit quantized base model + LoRA adapters.

    This adapter is special because it also affects how the *model* is loaded.
    When the model loader doesn't handle quantization itself, QLoRA reloads
    the model with 4-bit quantization in `apply()`.

    Tip: For cleaner separation, you can also load the model with
    quantization in the model loader via kwargs and use plain `lora` adapter.

    Config kwargs:
        - All LoraConfig params (r, lora_alpha, target_modules, ...)
        - `bnb_4bit_compute_dtype` (str): compute dtype, default "float16"
        - `bnb_4bit_quant_type` (str): quantization type, default "nf4"
        - `bnb_4bit_use_double_quant` (bool): double quantization, default True

    Example YAML::

        adapter:
          name: qlora
          kwargs:
            r: 16
            lora_alpha: 32
            target_modules: [q_proj, v_proj, k_proj, o_proj]
            task_type: CAUSAL_LM
            bnb_4bit_compute_dtype: bfloat16
    """

    def apply(self, model: nn.Module, config: AdapterConfig) -> nn.Module:
        kwargs = dict(config.kwargs)

        # extract BnB-specific params
        compute_dtype_str = kwargs.pop("bnb_4bit_compute_dtype", "float16")
        compute_dtype = getattr(torch, compute_dtype_str)
        quant_type = kwargs.pop("bnb_4bit_quant_type", "nf4")
        use_double_quant = kwargs.pop("bnb_4bit_use_double_quant", True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=use_double_quant,
        )

        # Reload model with 4-bit quantization if not already quantized.
        # bitsandbytes quantization must be applied at load time via
        # `from_pretrained(quantization_config=...)`, not post-hoc.
        if not getattr(model, "is_quantized", False):
            model_name_or_path = model.config._name_or_path
            logger.info(
                f"Reloading model '{model_name_or_path}' with 4-bit quantization"
            )
            del model  # free memory before reloading
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
            )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(**kwargs)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def save(self, model: nn.Module, path: str) -> None:
        model.save_pretrained(path)

    def load(self, model: nn.Module, path: str) -> nn.Module:
        return PeftModel.from_pretrained(model, path)
