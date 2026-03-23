"""HuggingFace Trainer wrapper — the default, batteries-included option."""

from __future__ import annotations

from typing import Any

from torch import nn
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from sita.core.base_trainer import BaseTrainer
from sita.core.config import TrainingConfig
from sita.core.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("hf_trainer")
class HFTrainer(BaseTrainer):
    """Wraps HuggingFace `Trainer` for simple, config-driven training.

    This is the default trainer. It builds `TrainingArguments` from the
    YAML config and handles everything (gradient accumulation, mixed
    precision, checkpointing, logging, etc.) automatically.

    Example YAML::

        trainer:
          name: hf_trainer

        training:
          output_dir: ./output/lora-tinyllama
          num_epochs: 3
          batch_size: 8
          learning_rate: 2e-4
          fp16: true
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
        reporting = kwargs.get("reporting")
        report_to = "wandb" if reporting and reporting.wandb else "none"

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            save_total_limit=2,
            report_to=report_to,
            **config.extra,
        )

        # If tokenizer is a VLM processor, use its inner text tokenizer for padding
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=text_tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        return model
