"""Custom PyTorch training loop — full control over the optimization process.

This serves as both a usable trainer and a reference implementation showing
how to write your own training loop within SITA.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sita.core.base_trainer import BaseTrainer
from sita.core.config import TrainingConfig
from sita.core.registry import TRAINER_REGISTRY

logger = logging.getLogger(__name__)


@TRAINER_REGISTRY.register("custom_loop")
class CustomLoopTrainer(BaseTrainer):
    """Bare PyTorch training loop for maximum flexibility.

    Use this when you need full control over:
    - Gradient manipulation (e.g. selective freezing mid-training)
    - Custom schedulers, optimizers, or loss functions
    - Multi-task training with custom batch logic
    - Anything the HF Trainer doesn't expose easily

    Extra kwargs (from trainer.kwargs in config):
        - `optimizer` (str): "adamw" (default) or "sgd"
        - `scheduler` (str): "cosine" (default), "linear", or "none"
        - `collate_fn_name` (str): custom collation strategy (optional)
        - `log_grad_norm` (bool): log gradient norms per step, default False

    Example YAML::

        trainer:
          name: custom_loop
          kwargs:
            optimizer: adamw
            scheduler: cosine
            log_grad_norm: true

        training:
          output_dir: ./output/lora-custom
          num_epochs: 5
          batch_size: 4
          learning_rate: 1e-4
          gradient_accumulation_steps: 4
          bf16: true
          max_grad_norm: 1.0
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
        # Setup
        device = next(model.parameters()).device
        use_amp = config.fp16 or config.bf16
        amp_dtype = torch.bfloat16 if config.bf16 else torch.float16

        # Optimizer
        optimizer_name = kwargs.get("optimizer", "adamw")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        total_steps = len(dataloader) * config.num_epochs
        effective_steps = total_steps // config.gradient_accumulation_steps

        # Scheduler
        scheduler_name = kwargs.get("scheduler", "cosine")
        warmup_steps = int(effective_steps * config.warmup_ratio)

        if scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                optimizer, T_max=effective_steps - warmup_steps
            )
        elif scheduler_name == "linear":
            from torch.optim.lr_scheduler import LinearLR

            scheduler = LinearLR(
                optimizer, start_factor=1.0, total_iters=effective_steps
            )
        else:
            scheduler = None

        scaler = (
            torch.amp.GradScaler("cuda")
            if use_amp and amp_dtype == torch.float16
            else None
        )
        log_grad_norm = kwargs.get("log_grad_norm", False)

        # Training Loop
        global_step = 0
        model.train()

        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

            for step, batch in enumerate(progress):
                inputs = {k: v.to(device) for k, v in batch.items()}
                if "labels" not in inputs and "input_ids" in inputs:
                    inputs["labels"] = inputs["input_ids"].clone()

                # Forward pass
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(**inputs)
                    loss = outputs.loss / config.gradient_accumulation_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * config.gradient_accumulation_steps

                # Optimizer step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if config.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, config.max_grad_norm
                        )

                    if log_grad_norm:
                        total_norm = torch.norm(
                            torch.stack(
                                [
                                    torch.norm(p.grad.detach())
                                    for p in trainable_params
                                    if p.grad is not None
                                ]
                            )
                        ).item()
                        logger.info(f"Step {global_step} | grad_norm: {total_norm:.4f}")

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

                # Checkpointing
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    ckpt_path = f"{config.output_dir}/checkpoint-{global_step}"
                    model.save_pretrained(ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

            avg_loss = epoch_loss / max(len(dataloader), 1)
            logger.info(f"Epoch {epoch + 1} — avg_loss: {avg_loss:.4f}")

        return model
