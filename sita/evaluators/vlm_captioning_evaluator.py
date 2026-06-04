"""VLM captioning evaluator — BERTScore + ROUGE-L on generated vs reference text.

Runs model.generate() on each eval sample, takes the raw generated text and
compares it against the reference (ringkasan) using text-similarity metrics.

No label parsing or classification metrics — this is purely for captioning.

Supported metrics (via ``metrics`` kwarg):
  - BERTScore: average P / R / F1
  - ROUGE-L: precision / recall / F1
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

import torch
import transformers
from torch import nn
from tqdm import tqdm

from sita.core.base_evaluator import BaseEvaluator
from sita.core.registry import EVALUATOR_REGISTRY

logger = logging.getLogger("sita.evaluators.vlm_captioning_evaluator")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks from generated text."""
    return _THINK_RE.sub("", text).strip()


def _extract_ground_truth(sample: dict) -> str:
    """Pull the reference caption text from a conversation dict."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    return part["text"].strip()
    return ""


def _build_user_messages(sample: dict) -> list[dict]:
    """Return only the user turn(s) for generation (strip assistant)."""
    return [m for m in sample.get("messages", []) if m.get("role") != "assistant"]


def _resize_image(img, max_size: int) -> "PIL.Image.Image":
    """Resize so longest side is at most *max_size* px."""
    from PIL import Image as PILImage

    if isinstance(img, str):
        img = PILImage.open(img).convert("RGB")
    if not hasattr(img, "thumbnail"):
        return img
    img = img.copy()
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    return img


def _extract_images(sample: dict, max_image_size: int | None = None) -> list:
    """Pull all PIL images from the conversation dict."""
    if "images" in sample:
        images = list(sample["images"])
    else:
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


@EVALUATOR_REGISTRY.register("vlm_captioning")
class VLMCaptioningEvaluator(BaseEvaluator):
    """Generate captions and compute BERTScore / ROUGE-L against references.

    Example YAML::

        evaluation:
          name: vlm_captioning
          kwargs:
            max_new_tokens: 512
            temperature: 0.0
            metrics:
              - bertscore
              - rouge
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
        requested_metrics = kwargs.get("metrics", ["bertscore"])
        if isinstance(requested_metrics, str):
            requested_metrics = [requested_metrics]
        requested_metrics = [m.lower().strip() for m in requested_metrics]
        batch_size = int(kwargs.get("batch_size", 1))
        num_workers = int(kwargs.get("num_workers", 0))
        enable_thinking = kwargs.get("enable_thinking", False)
        max_image_size = kwargs.get("max_image_size", 1024)
        if max_image_size is not None:
            max_image_size = int(max_image_size)
            logger.info(f"Capping eval images to {max_image_size}px max dimension.")

        # Switch model to inference mode
        try:
            from unsloth import FastVisionModel

            FastVisionModel.for_inference(model)
            logger.info("Switched model to Unsloth inference mode.")
        except ImportError:
            model.eval()
            logger.info("Unsloth not available — using model.eval().")

        device = next(model.parameters()).device

        # Force left-padding for generation
        for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
            if obj is not None and hasattr(obj, "padding_side"):
                obj.padding_side = "left"

        gt_captions: list[str] = []
        pred_captions: list[str] = []

        logger.info(f"Running captioning eval on {len(dataset)} samples…")

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        for batch in tqdm(dataloader, desc="Generating captions"):
            texts = []
            batch_images: list[list] = []
            valid_refs: list[str] = []

            for sample in batch:
                ref = _extract_ground_truth(sample)
                if not ref:
                    logger.debug("Empty reference caption, skipping sample.")
                    continue

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
                valid_refs.append(ref)

            if not texts:
                continue

            # Tokenize
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

            # Filter inputs for model forward signature compatibility
            unwrapped_model = model
            if hasattr(unwrapped_model, "get_base_model"):
                unwrapped_model = unwrapped_model.get_base_model()
            elif hasattr(unwrapped_model, "base_model"):
                unwrapped_model = unwrapped_model.base_model

            if hasattr(unwrapped_model, "model") and not isinstance(
                unwrapped_model, transformers.PreTrainedModel
            ):
                unwrapped_model = unwrapped_model.model

            sig = inspect.signature(unwrapped_model.forward)
            if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                valid_keys = set(sig.parameters.keys())
                valid_keys.update(["input_ids", "attention_mask"])
                inputs = {k: v for k, v in inputs.items() if k in valid_keys}

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only new tokens
            input_len = inputs["input_ids"].shape[-1]
            generated_texts = tokenizer.batch_decode(
                output_ids[:, input_len:],
                skip_special_tokens=True,
            )

            for ref, generated in zip(valid_refs, generated_texts):
                pred = _strip_thinking(generated)
                gt_captions.append(ref)
                pred_captions.append(pred if pred else "<EMPTY>")

        if not gt_captions:
            logger.warning("No valid samples found for evaluation.")
            return {}

        # ---- Compute metrics ----
        metrics: dict[str, float] = {}

        if "bertscore" in requested_metrics:
            metrics.update(
                self._compute_bertscore(gt_captions, pred_captions, bert_model_name)
            )
        if "rouge" in requested_metrics:
            metrics.update(
                self._compute_rouge(gt_captions, pred_captions)
            )

        # Log summary
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_bertscore(
        gt: list[str],
        pred: list[str],
        model_name: str,
    ) -> dict[str, float]:
        pairs = [(g, p) for g, p in zip(gt, pred) if g and p]
        if not pairs:
            logger.warning("No valid caption pairs for BERTScore.")
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

    @staticmethod
    def _compute_rouge(
        gt: list[str],
        pred: list[str],
    ) -> dict[str, float]:
        """Compute ROUGE-L on (ground-truth, prediction) caption pairs."""
        pairs = [(g, p) for g, p in zip(gt, pred) if g and p]
        if not pairs:
            logger.warning("No valid caption pairs for ROUGE.")
            return {}

        gt_filtered, pred_filtered = zip(*pairs)

        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning(
                "rouge-score not installed — skipping ROUGE metrics. "
                "Install with: pip install rouge-score"
            )
            return {}

        logger.info(f"Computing ROUGE-L on {len(gt_filtered)} pairs…")
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        precisions, recalls, f1s = [], [], []
        for ref, hyp in zip(gt_filtered, pred_filtered):
            scores = scorer.score(ref, hyp)
            precisions.append(scores["rougeL"].precision)
            recalls.append(scores["rougeL"].recall)
            f1s.append(scores["rougeL"].fmeasure)

        return {
            "rouge_l_precision": sum(precisions) / len(precisions),
            "rouge_l_recall": sum(recalls) / len(recalls),
            "rouge_l_f1": sum(f1s) / len(f1s),
        }
