"""Unsloth Vision Rejection Sampling Fine-Tuning (RFT) Trainer."""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sita.core.base_trainer import BaseTrainer
from sita.core.config import TrainingConfig
from sita.core.registry import TRAINER_REGISTRY

logger = logging.getLogger("sita.trainers.rejection_sampling_vlm")


@TRAINER_REGISTRY.register("unsloth_vlm_rft_trainer")
class UnslothVLMRFTTrainer(BaseTrainer):
    """Rejection Sampling Fine-Tuning (RFT) for Unsloth Vision models.

    Generates reasoning inside the training loop, validates output,
    and performs SFT only on the valid reasoning traces.

    Validator expects standard form: "Label: <label> \n\nAnalisis: <analisis>"
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
            from unsloth.trainer import UnslothVisionDataCollator
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Please install unsloth")

        from sita.core.registry import VALIDATOR_REGISTRY
        import sita.validators.dfk_validator  # ensure registered

        device = next(model.parameters()).device
        use_amp = config.fp16 or config.bf16
        amp_dtype = torch.bfloat16 if config.bf16 else torch.float16

        reporting = kwargs.get("reporting")
        use_wandb = reporting is not None and reporting.wandb

        # Extract config kwargs
        trainer_kwargs = kwargs.copy()

        evaluation_config = trainer_kwargs.pop("evaluation_config", None)

        num_samples = int(trainer_kwargs.pop("num_samples", 2))
        max_new_tokens = int(trainer_kwargs.pop("max_new_tokens", 256))
        temperature = float(trainer_kwargs.pop("temperature", 0.7))

        train_on_responses_only = trainer_kwargs.pop("train_on_responses_only", True)
        instruction_part = trainer_kwargs.pop("instruction_part", "<|im_start|>user\n")
        response_part = trainer_kwargs.pop("response_part", "<|im_start|>assistant\n")
        validator_name = trainer_kwargs.pop("validator", "dfk_vlm_validator")
        validator_kwargs = trainer_kwargs.pop("validator_kwargs", {})

        # Backwards compat for old config styles
        if "semantic_threshold" in trainer_kwargs:
            validator_kwargs["semantic_threshold"] = float(
                trainer_kwargs.pop("semantic_threshold")
            )

        # Set up Data Validator
        logger.info(f"Loading RFT Validator: {validator_name}")
        validator_cls = VALIDATOR_REGISTRY.get(validator_name)
        validator = validator_cls(device=device, **validator_kwargs)

        # Set up Unsloth data collator
        data_collator = UnslothVisionDataCollator(
            model,
            tokenizer,
            train_on_responses_only=train_on_responses_only,
            instruction_part=instruction_part,
            response_part=response_part,
        )

        # Optimizer Setup
        optimizer_name = trainer_kwargs.get("optim", "adamw")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_cls = (
            torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
        )
        optimizer = optimizer_cls(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # Dummy collate_fn to just return list of dictionaries
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

        total_steps = len(dataloader) * config.num_epochs
        effective_steps = max(1, total_steps // config.gradient_accumulation_steps)

        # Scheduler
        warmup_steps = int(effective_steps * config.warmup_ratio)
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(optimizer, T_max=effective_steps - warmup_steps)

        scaler = (
            torch.amp.GradScaler("cuda")
            if use_amp and amp_dtype == torch.float16
            else None
        )

        global_step = 0
        last_loss_val = None

        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            total_valid_samples = 0
            total_candidates = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

            for step, batch in enumerate(progress):
                valid_sft_batch = []

                # --- 1. INFERENCE PHASE ---
                FastVisionModel.for_inference(model)
                model.eval()

                with torch.no_grad():
                    prompts_text = []
                    images_list = []
                    gt_texts = []
                    user_msgs = []

                    for example in batch:
                        messages = example["messages"]
                        user_msg = next(
                            (m for m in messages if m["role"] == "user"), None
                        )
                        assistant_msg = next(
                            (m for m in messages if m["role"] == "assistant"), None
                        )

                        if not user_msg or not assistant_msg:
                            continue

                        # Extract Ground Truth Texts
                        gt_text = "".join(
                            [
                                c["text"]
                                for c in assistant_msg["content"]
                                if c["type"] == "text"
                            ]
                        )
                        if not gt_text:
                            continue

                        # Extract Target image
                        images = example.get("images", [])

                        # Apply chat template
                        prompt_text = tokenizer.apply_chat_template(
                            [user_msg], tokenize=False, add_generation_prompt=True
                        )

                        # Add `<think>\n` to encourage reasoning if requested
                        prompt_text += "<think>\n"

                        prompts_text.append(prompt_text)
                        images_list.append(images)
                        gt_texts.append(gt_text)
                        user_msgs.append(user_msg)

                    if not prompts_text:
                        continue

                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    tokenizer.padding_side = "left"

                    # Batch encode
                    inputs = tokenizer(
                        text=prompts_text,
                        images=images_list,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    input_len = inputs.input_ids.shape[1]

                    # Generate Multiple Samples
                    # Repeat interleave to expand [A, B] -> [A, A, B, B] (for num_samples=2)
                    inputs = {
                        k: v.repeat_interleave(num_samples, dim=0)
                        for k, v in inputs.items()
                    }

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    tokenizer.padding_side = "right"  # Restore for DataCollator

                    # Check Validity
                    for i in range(len(prompts_text)):
                        gt_text = gt_texts[i]
                        user_msg = user_msgs[i]

                        for s in range(num_samples):
                            idx = i * num_samples + s
                            gen_ids = outputs[idx][input_len:]
                            gen_text = tokenizer.decode(
                                gen_ids, skip_special_tokens=True
                            )

                            # Prefix reconstructed `<think>\n`
                            full_gen_text = "<think>\n" + gen_text

                            if validator(full_gen_text, gt_text):
                                # Valid sample! Construct SFT dict
                                new_conv = {
                                    "messages": [
                                        user_msg,
                                        {
                                            "role": "assistant",
                                            "content": [
                                                {"type": "text", "text": full_gen_text}
                                            ],
                                        },
                                    ],
                                    "images": images,
                                }
                                valid_sft_batch.append(new_conv)

                # --- 2. TRAINING PHASE ---
                n_candidates = len(prompts_text) * num_samples
                total_candidates += n_candidates
                total_valid_samples += len(valid_sft_batch)
                accept_rate = (
                    total_valid_samples / total_candidates * 100
                    if total_candidates > 0
                    else 0.0
                )

                # Always update progress bar (even when 0 valid)
                postfix = dict(
                    accept=f"{accept_rate:.1f}%",
                    valids=f"{total_valid_samples}/{total_candidates}",
                )
                if last_loss_val is not None:
                    postfix["loss"] = f"{last_loss_val:.4f}"
                progress.set_postfix(**postfix)

                if len(valid_sft_batch) == 0:
                    continue  # No valid reasoning traces found in this batch

                FastVisionModel.for_training(model)
                model.train()

                # Collate inputs (Unsloth auto-masks prompt if configured)
                # Ensure the collator processes a standard list
                sft_inputs = data_collator(valid_sft_batch)
                sft_inputs = {k: v.to(device) for k, v in sft_inputs.items()}

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    sft_outputs = model(**sft_inputs)
                    loss = sft_outputs.loss / config.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * config.gradient_accumulation_steps

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if config.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, config.max_grad_norm
                        )

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
                    _loss_val = loss.item() * config.gradient_accumulation_steps
                    last_loss_val = _loss_val
                    progress.set_postfix(
                        loss=f"{_loss_val:.4f}",
                        accept=f"{accept_rate:.1f}%",
                        valids=f"{total_valid_samples}/{total_candidates}",
                    )

                    if use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "train/loss": _loss_val,
                                "train/learning_rate": lr,
                                "train/epoch": epoch + (step + 1) / len(dataloader),
                                "train/valid_samples": len(valid_sft_batch),
                                "train/accept_rate": accept_rate,
                            },
                            step=global_step,
                        )

                # Evaluation
                if (
                    eval_dataset is not None
                    and config.eval_steps > 0
                    and global_step % config.eval_steps == 0
                ):
                    if evaluation_config is not None:
                        from sita.core.registry import EVALUATOR_REGISTRY

                        evaluator_cls = EVALUATOR_REGISTRY.get(evaluation_config.name)
                        if evaluator_cls:
                            evaluator = evaluator_cls()
                            logger.info(f"Running evaluation at step {global_step}...")
                            metrics = evaluator.evaluate(
                                model=model,
                                tokenizer=tokenizer,
                                dataset=eval_dataset,
                                **evaluation_config.kwargs,
                            )
                            if use_wandb:
                                import wandb

                                wandb.log(
                                    {f"eval/{k}": v for k, v in metrics.items()},
                                    step=global_step,
                                )

                            # Revert model to training mode
                            FastVisionModel.for_training(model)
                            model.train()

                # Checkpointing
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    ckpt_path = f"{config.output_dir}/checkpoint-{global_step}"
                    model.save_pretrained(ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

            avg_loss = epoch_loss / max(len(dataloader), 1)
            logger.info(
                f"Epoch {epoch + 1} — avg_loss: {avg_loss:.4f} — valid_samples/epoch: {total_valid_samples}"
            )

        return model
