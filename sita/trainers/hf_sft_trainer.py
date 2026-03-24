"""HuggingFace SFTTrainer wrapper — optimized for Supervised Fine-Tuning."""

from __future__ import annotations

import logging
import inspect
from typing import Any

import torch
from torch import nn

from sita.core.base_trainer import BaseTrainer
from sita.core.config import TrainingConfig
from sita.core.registry import TRAINER_REGISTRY

logger = logging.getLogger(__name__)


class VLMResponseMaskingCollator:
    """Wraps a base collator and masks labels for non-assistant tokens.

    After the base collator tokenizes & pads the batch, this finds
    ``<|im_start|>assistant`` → ``<|im_end|>`` spans in each sequence
    and sets labels to -100 for everything *outside* those spans.
    Works with any TRL version and any chat template that uses im_start/im_end.
    """

    def __init__(self, base_collator, tokenizer):
        self.base_collator = base_collator
        # Get the underlying text tokenizer for VLM processors
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        # Resolve the token IDs we need for span detection
        self.im_start_id = text_tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = text_tok.convert_tokens_to_ids("<|im_end|>")
        # "assistant" as token IDs (may be a single token or multiple)
        self.assistant_ids = text_tok.encode("assistant", add_special_tokens=False)

    def __call__(self, features, **kwargs):
        batch = self.base_collator(features, **kwargs)

        if "labels" not in batch or "input_ids" not in batch:
            return batch

        labels = batch["labels"]
        input_ids = batch["input_ids"]

        for i in range(input_ids.shape[0]):
            ids = input_ids[i]
            seq_len = ids.shape[0]
            # Build a mask: True = keep (assistant response), False = mask out
            keep = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)

            j = 0
            while j < seq_len:
                # Look for <|im_start|> followed by "assistant"
                if ids[j].item() == self.im_start_id:
                    # Check if the next tokens spell "assistant"
                    a_len = len(self.assistant_ids)
                    if j + 1 + a_len <= seq_len:
                        candidate = ids[j + 1 : j + 1 + a_len].tolist()
                        if candidate == self.assistant_ids:
                            # Find the newline after "assistant\n" — content starts there
                            content_start = j + 1 + a_len
                            # Skip the newline token right after "assistant"
                            if content_start < seq_len:
                                content_start += 1  # skip \n

                            # Find the matching <|im_end|>
                            end_pos = seq_len  # fallback: end of sequence
                            for k in range(content_start, seq_len):
                                if ids[k].item() == self.im_end_id:
                                    end_pos = k
                                    break

                            # Mark assistant content (between content_start and im_end) as keep
                            keep[content_start:end_pos] = True
                            j = end_pos + 1
                            continue
                j += 1

            # Mask labels: set non-assistant positions to -100
            labels[i][~keep] = -100

        batch["labels"] = labels
        return batch


@TRAINER_REGISTRY.register("hf_sft_trainer")
class HFSFTTrainer(BaseTrainer):
    """HuggingFace SFTTrainer wrapper from the TRL library.

    This trainer is optimized for Supervised Fine-Tuning (SFT) and handles
    dataset formatting, packing, and completion-only loss masking.

    Trainer-level kwargs (from ``trainer.kwargs`` in YAML):
      - ``dataset_text_field`` (str): The column name in the dataset containing the text.
      - ``max_length`` / ``max_seq_length`` (int): Maximum sequence length. Default 1024.
      - ``packing`` (bool): Whether to use packing (concatenating samples). Default False.
      - ``response_template`` (str): Enables response-only masking via custom VLM collator.
      - ``dataset_num_proc`` (int): Number of processes for dataset preprocessing.

    Example YAML::

        trainer:
          name: hf_sft_trainer
          kwargs:
            max_length: 2048
            packing: false
            response_template: "<|im_start|>assistant"
    """

    def train(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any | None,
        config: TrainingConfig,
        **kwargs,
    ) -> nn.Module:
        try:
            from trl import SFTConfig, SFTTrainer
        except ImportError:
            raise ImportError(
                "trl library not found. Please install with `pip install trl`."
            )

        reporting = kwargs.get("reporting")
        report_to = "wandb" if reporting and reporting.wandb else "none"

        # Extract SFT specific kwargs
        trainer_kwargs = kwargs.copy()
        trainer_kwargs.pop("reporting", None)
        trainer_kwargs.pop("evaluation_config", None)

        # These have moved from SFTTrainer to SFTConfig in recent TRL versions
        dataset_text_field = trainer_kwargs.pop("dataset_text_field", None)
        # Handle both 'max_length' and deprecated 'max_seq_length'
        max_length = trainer_kwargs.pop("max_length", None)
        max_seq_length = trainer_kwargs.pop("max_seq_length", None)
        resolved_max_len = max_length or max_seq_length

        packing = trainer_kwargs.pop("packing", None)
        trainer_kwargs.pop("instruction_template", None)  # consumed but unused now
        response_template = trainer_kwargs.pop("response_template", None)
        dataset_num_proc = trainer_kwargs.pop("dataset_num_proc", None)

        # Build SFTConfig kwargs
        sft_config_kwargs = {
            "output_dir": config.output_dir,
            "num_train_epochs": config.num_epochs,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "fp16": config.fp16,
            "bf16": config.bf16,
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
            "eval_steps": config.eval_steps if eval_dataset else None,
            "eval_strategy": "steps" if eval_dataset else "no",
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "save_total_limit": 2,
            "report_to": report_to,
            "remove_unused_columns": False,
        }

        sft_params = inspect.signature(SFTConfig).parameters

        # Add SFT-specific fields if they were provided
        if resolved_max_len is not None:
            if "max_length" in sft_params:
                sft_config_kwargs["max_length"] = resolved_max_len
            else:
                sft_config_kwargs["max_seq_length"] = resolved_max_len

        if dataset_text_field is not None:
            sft_config_kwargs["dataset_text_field"] = dataset_text_field
        if packing is not None:
            sft_config_kwargs["packing"] = packing
        if dataset_num_proc is not None:
            sft_config_kwargs["dataset_num_proc"] = dataset_num_proc

        # Also merge from config.extra (allows overriding anything)
        sft_config_kwargs.update(config.extra)

        # Final resolved values for logging
        log_max_len = sft_config_kwargs.get("max_length") or sft_config_kwargs.get(
            "max_seq_length", 1024
        )
        resolved_packing = sft_config_kwargs.get("packing", False)

        sft_config = SFTConfig(**sft_config_kwargs)

        # Check SFTTrainer signature for 'tokenizer' vs 'processing_class' (TRL 0.12.0+)
        trainer_params = inspect.signature(SFTTrainer).parameters
        sft_trainer_kwargs = {
            "model": model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            **trainer_kwargs,
        }

        if "processing_class" in trainer_params:
            sft_trainer_kwargs["processing_class"] = tokenizer
        else:
            sft_trainer_kwargs["tokenizer"] = tokenizer

        trainer = SFTTrainer(**sft_trainer_kwargs)

        # Wrap the trainer's data collator with response masking if requested
        if response_template:
            logger.info("Wrapping data collator with VLMResponseMaskingCollator.")
            trainer.data_collator = VLMResponseMaskingCollator(
                base_collator=trainer.data_collator,
                tokenizer=tokenizer,
            )

        logger.info(
            f"Starting SFT training (packing={resolved_packing}, max_length={log_max_len})..."
        )
        trainer.train()
        logger.info("Training complete.")

        return model
