"""Unsloth Vision SFT Trainer — uses TRL SFTTrainer + UnslothVisionDataCollator."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn
import torch

from sita.core.base_trainer import BaseTrainer
from sita.core.config import TrainingConfig
from sita.core.registry import TRAINER_REGISTRY

logger = logging.getLogger("sita.trainers.unsloth_vlm_sft")


@TRAINER_REGISTRY.register("unsloth_vlm_sft")
class UnslothVLMSFTTrainer(BaseTrainer):
    """SFT trainer for Unsloth vision-language models.

    Uses TRL's ``SFTTrainer`` with ``UnslothVisionDataCollator`` for:
      - Vision-aware data collation and padding
      - Response-only loss masking (only backprop on assistant output tokens)
      - Proper handling of vision placeholder tokens

    Trainer-level kwargs (from ``trainer.kwargs`` in YAML):
      - ``instruction_part`` (str): chat template token marking end of user turn
      - ``response_part`` (str): chat template token marking start of assistant turn
      - ``train_on_responses_only`` (bool): default True
      - ``optim`` (str): optimizer name, default ``adamw_8bit``

    Example YAML::

        trainer:
          name: unsloth_vlm_sft
          kwargs:
            train_on_responses_only: true
            instruction_part: "<|im_start|>user\\n"
            response_part: "<|im_start|>assistant\\n"
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
            from trl import SFTTrainer, SFTConfig
            from unsloth.trainer import UnslothVisionDataCollator
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}. " "Please install: pip install unsloth trl"
            )

        # Apply Unsloth's performance patches and memory optimizations for VLM fine-tuning
        FastVisionModel.for_training(model)

        # Extract trainer-specific kwargs
        trainer_kwargs = kwargs.copy()
        # Remove 'reporting' which comes from the runner, not relevant here
        reporting = trainer_kwargs.pop("reporting", None)

        train_on_responses_only = trainer_kwargs.pop("train_on_responses_only", True)
        instruction_part = trainer_kwargs.pop("instruction_part", "<|im_start|>user\n")
        response_part = trainer_kwargs.pop("response_part", "<|im_start|>assistant\n")
        optim = trainer_kwargs.pop("optim", "adamw_8bit")

        report_to = "wandb" if reporting and reporting.wandb else "none"

        # Build warmup config w/ prefer warmup_steps over deprecated warmup_ratio
        warmup_steps = config.extra.pop("warmup_steps", None)
        warmup_kwargs = {}
        if warmup_steps is not None:
            warmup_kwargs["warmup_steps"] = int(warmup_steps)
        else:
            warmup_kwargs["warmup_steps"] = int(
                config.warmup_ratio * config.extra.get("max_steps", 100)
            )

        use_bf16 = config.extra.pop("bf16", None)
        use_fp16 = config.extra.pop("fp16", None)

        # Build SFTConfig from training config
        sft_config = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=(
                use_fp16 if use_fp16 is not None else not torch.cuda.is_bf16_supported()
            ),
            bf16=use_bf16 if use_bf16 is not None else torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            **warmup_kwargs,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            optim=optim,
            save_total_limit=2,
            seed=config.extra.pop("seed", 3407),
            report_to=report_to,
            # Vision SFT specific — let the data collator handle seq length
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            **{k: v for k, v in config.extra.items()},
        )

        # Build data collator with response-only masking
        data_collator = UnslothVisionDataCollator(
            model,
            tokenizer,
            train_on_responses_only=train_on_responses_only,
            instruction_part=instruction_part,
            response_part=response_part,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

        logger.info("Starting VLM SFT training...")
        trainer.train()
        logger.info("Training complete.")

        return model
