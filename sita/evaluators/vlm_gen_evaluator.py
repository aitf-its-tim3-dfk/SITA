"""VLM generative evaluator — classification metrics + BERTScore.

Runs model.generate() on each eval sample, parses the predicted text for
``Label`` and ``Analisis`` fields, then computes:
  - Classification: accuracy, macro precision / recall / F1
  - BERTScore: average P / R / F1 on the analysis text
"""

from __future__ import annotations

import logging
import re
from typing import Any
import inspect
import transformers

import torch
from torch import nn
from tqdm import tqdm

from sita.core.base_evaluator import BaseEvaluator
from sita.core.registry import EVALUATOR_REGISTRY

logger = logging.getLogger("sita.evaluators.vlm_gen_evaluator")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LABEL_RE = re.compile(r"Label\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ANALISIS_RE = re.compile(r"Analisis\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_response(text: str) -> tuple[str, str]:
    """Extract (label, analisis) from a generated / ground-truth response."""
    # Strip thinking blocks so regexes only match the actual answer
    text = _THINK_RE.sub("", text).strip()

    label_m = _LABEL_RE.search(text)
    label = label_m.group(1).strip() if label_m else ""

    analisis_m = _ANALISIS_RE.search(text)
    analisis = analisis_m.group(1).strip() if analisis_m else ""

    return label, analisis


def _extract_ground_truth(sample: dict) -> tuple[str, str]:
    """Pull the reference answer text from a conversation dict."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    return _parse_response(part["text"])
    return "", ""


def _build_user_messages(sample: dict) -> list[dict]:
    """Return only the user turn(s) for generation (strip assistant)."""
    return [m for m in sample.get("messages", []) if m.get("role") != "assistant"]


def _resize_image(img, max_size: int) -> "PIL.Image.Image":
    """Resize an image so its longest side is at most *max_size* px.

    Uses ``Image.thumbnail`` which preserves aspect ratio and never
    upscales.  This prevents the Pixtral vision encoder from producing
    an unmanageable number of patches on very-high-res inputs.
    """
    from PIL import Image as PILImage

    if isinstance(img, str):
        img = PILImage.open(img).convert("RGB")
    if not hasattr(img, "thumbnail"):
        return img  # not a PIL image, leave as-is
    img = img.copy()  # don't mutate the dataset's cached image
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    return img


def _extract_images(sample: dict, max_image_size: int | None = None) -> list:
    """Pull all PIL images from the conversation dict (standard TRL format)."""
    # Prefer the 'images' field (TRL standard)
    if "images" in sample:
        images = list(sample["images"])
    else:
        # Fallback to searching messages (legacy)
        images = []
        for msg in sample.get("messages", []):
            if msg.get("role") != "user":
                continue
            for part in msg.get("content", []):
                if part.get("type") == "image" and "image" in part:
                    images.append(part["image"])

    if max_image_size is not None:
        images = [_resize_image(im, max_image_size) for im in images]
    return images


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@EVALUATOR_REGISTRY.register("vlm_gen")
class VLMGenEvaluator(BaseEvaluator):
    """Generate predictions and compute classification + BERTScore metrics.

    Example YAML::

        evaluation:
          name: vlm_gen
          kwargs:
            max_new_tokens: 512
            temperature: 0.0
            bert_model: bert-base-multilingual-cased
    """

    def evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        dataset: Any,
        **kwargs,
    ) -> dict[str, float]:
        max_new_tokens = int(kwargs.get("max_new_tokens", 512))
        temperature = float(kwargs.get("temperature", 0.0))
        bert_model_name = kwargs.get("bert_model", "bert-base-multilingual-cased")
        batch_size = int(kwargs.get("batch_size", 1))
        num_workers = int(kwargs.get("num_workers", 0))
        enable_thinking = kwargs.get("enable_thinking", False)
        max_image_size = kwargs.get("max_image_size", 1024)  # cap longest side (px)
        if max_image_size is not None:
            max_image_size = int(max_image_size)
            logger.info(f"Capping eval images to {max_image_size}px max dimension.")

        # Switch model to inference mode (Unsloth optimization)
        try:
            from unsloth import FastVisionModel

            FastVisionModel.for_inference(model)
            logger.info("Switched model to Unsloth inference mode.")
        except ImportError:
            model.eval()
            logger.info("Unsloth not available — using model.eval().")

        device = next(model.parameters()).device

        # Force left-padding for generation.  Training (SFTTrainer / DataCollator)
        # may set padding_side to 'right'
        for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
            if obj is not None and hasattr(obj, "padding_side"):
                obj.padding_side = "left"

        gt_labels: list[str] = []
        pred_labels: list[str] = []
        gt_analyses: list[str] = []
        pred_analyses: list[str] = []

        logger.info(f"Running generation on {len(dataset)} samples…")

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        for batch in tqdm(dataloader, desc="Generating"):
            texts = []
            batch_images: list[list] = []
            valid_gts = []

            for sample in batch:
                gt_label, gt_analisis = _extract_ground_truth(sample)
                if not gt_label:
                    continue

                # Build input — user turn only
                user_msgs = _build_user_messages(sample)
                images = _extract_images(sample, max_image_size=max_image_size)

                input_text = tokenizer.apply_chat_template(
                    user_msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=enable_thinking,
                )

                texts.append(input_text)
                batch_images.append(images if images else [])
                valid_gts.append((gt_label, gt_analisis))

            if not texts:
                continue

            # Tokenize with image processing — pass images to the processor
            # Images must be a list-of-lists so each text entry gets its
            # own image list (required by Gemma4 / multi-sample processors).
            proc_kwargs: dict[str, Any] = {
                "text": texts if len(texts) > 1 else texts[0],
                "return_tensors": "pt",
                "padding": True,
            }
            has_any_images = any(imgs for imgs in batch_images)
            if has_any_images:
                if len(texts) > 1:
                    proc_kwargs["images"] = batch_images
                else:
                    # Single sample, pass flat list directly
                    proc_kwargs["images"] = batch_images[0]

            inputs = tokenizer(**proc_kwargs).to(device)

            # Generate
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            # Dynamically filter inputs using the model's actual forward signature
            # to gracefully handle PEFT wraps, Liger Kernel missing kwargs,
            # or running a pure text model locally for a VLM task.
            unwrapped_model = model
            if hasattr(unwrapped_model, "get_base_model"):
                unwrapped_model = unwrapped_model.get_base_model()
            elif hasattr(unwrapped_model, "base_model"):
                unwrapped_model = unwrapped_model.base_model

            # If unwrapped_model is still a PEFT internal wrapper (e.g. LoraModel)
            if hasattr(unwrapped_model, "model") and not isinstance(
                unwrapped_model, transformers.PreTrainedModel
            ):
                unwrapped_model = unwrapped_model.model

            sig = inspect.signature(unwrapped_model.forward)
            valid_keys = set(sig.parameters.keys())

            # If the model does not accept **kwargs, we strictly filter
            if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                valid_keys = set(sig.parameters.keys())
                # ensure standard generation keys are kept if they might be passed later
                valid_keys.update(["input_ids", "attention_mask"])

                # pop anything not valid
                inputs = {k: v for k, v in inputs.items() if k in valid_keys}

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[-1]
            generated_texts = tokenizer.batch_decode(
                output_ids[:, input_len:],
                skip_special_tokens=True,
            )

            for (gt_label, gt_analisis), generated in zip(valid_gts, generated_texts):
                generated = generated.strip()
                pred_label, pred_analisis = _parse_response(generated)

                gt_labels.append(gt_label)
                pred_labels.append(pred_label if pred_label else "<UNPARSED>")
                gt_analyses.append(gt_analisis)
                pred_analyses.append(pred_analisis if pred_analisis else generated)

        if not gt_labels:
            logger.warning("No valid samples found for evaluation.")
            return {}

        # ---- Classification metrics ----
        metrics = self._compute_classification(gt_labels, pred_labels)

        # ---- BERTScore ----
        metrics.update(
            self._compute_bertscore(gt_analyses, pred_analyses, bert_model_name)
        )

        # Log summary
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_classification(gt: list[str], pred: list[str]) -> dict[str, float]:
        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_recall_fscore_support,
            )
        except ImportError:
            logger.warning(
                "scikit-learn not installed — skipping classification metrics."
            )
            return {}

        acc = accuracy_score(gt, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            gt, pred, average="macro", zero_division=0
        )
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
            gt, pred, average="weighted", zero_division=0
        )

        metrics: dict[str, float] = {
            "cls_accuracy": acc,
            "cls_precision_macro": prec,
            "cls_recall_macro": rec,
            "cls_f1_macro": f1,
            "cls_precision_weighted": w_prec,
            "cls_recall_weighted": w_rec,
            "cls_f1_weighted": w_f1,
        }

        # Per-class breakdown
        all_labels = sorted(set(gt) | set(pred))
        pc_prec, pc_rec, pc_f1, pc_sup = precision_recall_fscore_support(
            gt, pred, labels=all_labels, average=None, zero_division=0
        )
        for label, p, r, f, s in zip(all_labels, pc_prec, pc_rec, pc_f1, pc_sup):
            safe = label.lower().replace(" ", "_")
            metrics[f"cls_{safe}_precision"] = p
            metrics[f"cls_{safe}_recall"] = r
            metrics[f"cls_{safe}_f1"] = f
            metrics[f"cls_{safe}_support"] = int(s)

        return metrics

    @staticmethod
    def _compute_bertscore(
        gt: list[str],
        pred: list[str],
        model_name: str,
    ) -> dict[str, float]:
        # Filter out empty pairs
        pairs = [(g, p) for g, p in zip(gt, pred) if g and p]
        if not pairs:
            logger.warning("No valid analysis pairs for BERTScore.")
            return {}

        gt_filtered, pred_filtered = zip(*pairs)

        try:
            from bert_score import score as bert_score_fn
        except ImportError:
            logger.warning("bert-score not installed — skipping BERTScore metrics.")
            return {}

        logger.info(
            f"Computing BERTScore on {len(gt_filtered)} pairs "
            f"with model={model_name}…"
        )
        P, R, F1 = bert_score_fn(
            list(pred_filtered),
            list(gt_filtered),
            model_type=model_name,
            verbose=True,
        )

        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
