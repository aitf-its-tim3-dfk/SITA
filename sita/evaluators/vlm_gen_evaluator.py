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

import torch
from torch import nn
from tqdm import tqdm

from sita.core.base_evaluator import BaseEvaluator
from sita.core.registry import EVALUATOR_REGISTRY

logger = logging.getLogger("sita.evaluators.vlm_gen_evaluator")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"Label\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ANALISIS_RE = re.compile(r"Analisis\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_response(text: str) -> tuple[str, str]:
    """Extract (label, analisis) from a generated / ground-truth response."""
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


def _extract_images(sample: dict) -> list:
    """Pull all PIL images from the user turn(s) of a conversation dict."""
    images = []
    for msg in sample.get("messages", []):
        if msg.get("role") != "user":
            continue
        for part in msg.get("content", []):
            if part.get("type") == "image" and "image" in part:
                images.append(part["image"])
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

        # Switch model to inference mode (Unsloth optimization)
        try:
            from unsloth import FastVisionModel

            FastVisionModel.for_inference(model)
            logger.info("Switched model to Unsloth inference mode.")
        except ImportError:
            model.eval()
            logger.info("Unsloth not available — using model.eval().")

        device = next(model.parameters()).device

        gt_labels: list[str] = []
        pred_labels: list[str] = []
        gt_analyses: list[str] = []
        pred_analyses: list[str] = []

        logger.info(f"Running generation on {len(dataset)} samples…")

        for sample in tqdm(dataset, desc="Generating"):
            # Ground truth
            gt_label, gt_analisis = _extract_ground_truth(sample)
            if not gt_label:
                continue

            # Build input — user turn only
            user_msgs = _build_user_messages(sample)
            images = _extract_images(sample)

            input_text = tokenizer.apply_chat_template(
                user_msgs,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Tokenize with image processing — pass images to the processor
            proc_kwargs: dict[str, Any] = {
                "text": input_text,
                "return_tensors": "pt",
                "padding": True,
            }
            if images:
                proc_kwargs["images"] = images

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

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[-1]
            generated = tokenizer.decode(
                output_ids[0][input_len:],
                skip_special_tokens=True,
            ).strip()

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

        return {
            "cls_accuracy": acc,
            "cls_precision_macro": prec,
            "cls_recall_macro": rec,
            "cls_f1_macro": f1,
            "cls_precision_weighted": w_prec,
            "cls_recall_weighted": w_rec,
            "cls_f1_weighted": w_f1,
        }

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
