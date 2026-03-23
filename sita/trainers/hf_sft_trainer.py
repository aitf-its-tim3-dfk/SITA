"""HuggingFace SFTTrainer wrapper — optimized for Supervised Fine-Tuning."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

from sita.core.base_trainer import BaseTrainer
from sita.core.config import TrainingConfig
from sita.core.registry import TRAINER_REGISTRY

logger = logging.getLogger(__name__)


@TRAINER_REGISTRY.register("hf_sft_trainer")
class HFSFTTrainer(BaseTrainer):
    """HuggingFace SFTTrainer wrapper from the TRL library.

    This trainer is optimized for Supervised Fine-Tuning (SFT) and handles
    dataset formatting, packing, and completion-only loss masking.

    Trainer-level kwargs (from ``trainer.kwargs`` in YAML):
      - ``dataset_text_field`` (str): The column name in the dataset containing the text.
      - ``max_seq_length`` (int): Maximum sequence length for the trainer. Default 1024.
      - ``packing`` (bool): Whether to use packing (concatenating samples). Default False.
      - ``instruction_template`` (str): Optional template for completion-only masking.
      - ``response_template`` (str): Optional template for completion-only masking.
      - ``dataset_num_proc`` (int): Number of processes for dataset preprocessing.

    Example YAML::

        trainer:
          name: hf_sft_trainer
          kwargs:
            dataset_text_field: "text"
            max_seq_length: 2048
            packing: false
            response_template: "### Response:"
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

            # Try to import DataCollatorForCompletionOnlyLM from top level, then from submodules
            try:
                from trl import DataCollatorForCompletionOnlyLM
            except ImportError:
                try:
                    # In some recent TRL versions it might be here
                    from trl.trainer import DataCollatorForCompletionOnlyLM
                except ImportError:
                    DataCollatorForCompletionOnlyLM = None
        except ImportError:
            raise ImportError("trl library not found. Please install with `pip install trl`.")

        reporting = kwargs.get("reporting")
        report_to = "wandb" if reporting and reporting.wandb else "none"

        # Extract SFT specific kwargs
        trainer_kwargs = kwargs.copy()
        trainer_kwargs.pop("reporting", None)
        trainer_kwargs.pop("evaluation_config", None)

        dataset_text_field = trainer_kwargs.pop("dataset_text_field", None)
        max_seq_length = trainer_kwargs.pop("max_seq_length", 1024)
        packing = trainer_kwargs.pop("packing", False)
        instruction_template = trainer_kwargs.pop("instruction_template", None)
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
            "max_seq_length": max_seq_length,
            "packing": packing,
            "dataset_text_field": dataset_text_field,
            "dataset_num_proc": dataset_num_proc,
        }
        sft_config_kwargs.update(config.extra)

        data_collator = None
        if response_template:
            # Check if SFTConfig supports the new completion_only_loss (added in TRL 0.12.0+)
            import inspect

            sft_params = inspect.signature(SFTConfig).parameters
            if "completion_only_loss" in sft_params:
                logger.info("Using SFTConfig(completion_only_loss=True) for masking.")
                sft_config_kwargs["completion_only_loss"] = True
                sft_config_kwargs["instruction_template"] = instruction_template
                sft_config_kwargs["response_template"] = response_template
            elif DataCollatorForCompletionOnlyLM is not None:
                logger.info("Using DataCollatorForCompletionOnlyLM for masking.")
                # If tokenizer is a VLM processor, use its inner text tokenizer
                text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template,
                    instruction_template=instruction_template,
                    tokenizer=text_tokenizer,
                )
            else:
                logger.warning(
                    "Completion masking requested but neither SFTConfig(completion_only_loss) "
                    "nor DataCollatorForCompletionOnlyLM are available. Masking will be skipped."
                )

        sft_config = SFTConfig(**sft_config_kwargs)

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **trainer_kwargs,
        )

        logger.info(
            f"Starting SFT training (packing={packing}, max_seq_length={max_seq_length})..."
        )
        trainer.train()
        logger.info("Training complete.")

        return model
