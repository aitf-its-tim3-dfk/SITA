"""Text-only generative evaluator — classification metrics + BERTScore + ROUGE-L.

For the DFK text dataset which uses the format:
    Label: **Fitnah.** penjelasan: Some explanation text.

No image handling — this is a pure text evaluator.

Supported metrics (via ``metrics`` kwarg):
  - Classification: accuracy, macro/weighted precision / recall / F1
  - BERTScore: average P / R / F1 on the explanation text
  - ROUGE-L: precision / recall / F1 on the explanation text
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

logger = logging.getLogger("sita.evaluators.text_gen_evaluator")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Matches: Label: **Fitnah.** penjelasan: ...
# Also handles: Label: Fitnah penjelasan: ...  (no bold)
_PENJELASAN_SPLIT_RE = re.compile(r"\s+penjelasan:\s*", re.IGNORECASE)
_BOLD_LABEL_RE = re.compile(r"\*{2}(.+?)\.?\*{2,4}")

# Fallback: plain "Label: xxx" on a line (like vlm_gen format)
_LABEL_RE = re.compile(r"Label\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ANALISIS_RE = re.compile(r"Analisis\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

# Label normalization: maps old/variant labels to canonical form.
# Applied AFTER parsing, so it catches both ground truth and predictions.
_LABEL_NORMALIZE: dict[str, str] = {
    "non-dfk": "netral",
    "non_dfk": "netral",
    "nondfk": "netral",
    "fakta": "netral",
    "ujaran_kebencian": "ujaran kebencian",
    "disinformasi_dan_ujaran_kebencian": "ujaran kebencian",
}


def _normalize_label(label: str, normalize: bool = True) -> str:
    """Map old/variant label names to canonical form.

    When *normalize* is False, only lowercases and cleans the label
    without remapping (preserves the original 5-class distinctions).
    """
    clean = label.lower().strip().rstrip(".")
    if not normalize:
        return clean
    return _LABEL_NORMALIZE.get(clean, clean)


def _parse_response(text: str, normalize: bool = True) -> tuple[str, str]:
    """Extract (label, explanation) from generated / ground-truth response.

    Handles both formats:
      - Text dataset: ``Label: **Fitnah.** penjelasan: ...``
      - VLM dataset:  ``Label: fitnah\\n\\nAnalisis: ...``
    """
    text = _THINK_RE.sub("", text).strip()
    first_line = text.split("\n")[0].strip()

    # Try text-dataset format first: split on "penjelasan:"
    parts = _PENJELASAN_SPLIT_RE.split(first_line, maxsplit=1)
    if len(parts) == 2:
        label_part, penjelasan = parts
        # Try bold label: **Fitnah.**
        bold_m = _BOLD_LABEL_RE.search(label_part)
        if bold_m:
            label = bold_m.group(1).strip().rstrip(".")
        else:
            label = label_part.replace("Label:", "").strip().strip("*").strip(".")
        return _normalize_label(label, normalize=normalize), penjelasan.strip()

    # Fallback: vlm_gen format (Label: xxx\nAnalisis: yyy)
    label_m = _LABEL_RE.search(text)
    label = label_m.group(1).strip().strip("*").strip(".") if label_m else ""

    analisis_m = _ANALISIS_RE.search(text)
    analisis = analisis_m.group(1).strip() if analisis_m else ""

    return _normalize_label(label, normalize=normalize), analisis


def _extract_ground_truth(sample: dict, normalize: bool = True) -> tuple[str, str]:
    """Pull the reference answer text from a conversation dict."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    return _parse_response(part["text"], normalize=normalize)
    return "", ""


def _build_input_messages(sample: dict) -> list[dict]:
    """Return system + user turns for generation (strip assistant)."""
    return [m for m in sample.get("messages", []) if m.get("role") != "assistant"]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@EVALUATOR_REGISTRY.register("text_gen")
class TextGenEvaluator(BaseEvaluator):
    """Generate predictions and compute classification + text-similarity metrics.

    For text-only datasets (no images). Handles both the text dataset format
    (``Label: **X.** penjelasan: ...``) and the VLM format as fallback.

    Example YAML::

        evaluation:
          name: text_gen
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
        normalize_labels = kwargs.get("normalize_labels", True)

        # Switch model to inference mode
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)
            logger.info("Switched model to Unsloth inference mode.")
        except ImportError:
            try:
                from unsloth import FastVisionModel

                FastVisionModel.for_inference(model)
                logger.info("Switched model to Unsloth vision inference mode.")
            except ImportError:
                model.eval()
                logger.info("Unsloth not available — using model.eval().")

        device = next(model.parameters()).device

        # Force left-padding for generation
        for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
            if obj is not None and hasattr(obj, "padding_side"):
                obj.padding_side = "left"

        gt_labels: list[str] = []
        pred_labels: list[str] = []
        gt_explanations: list[str] = []
        pred_explanations: list[str] = []

        logger.info(f"Running text generation on {len(dataset)} samples…")

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        for batch in tqdm(dataloader, desc="Generating (text)"):
            texts = []
            valid_gts = []

            for sample in batch:
                gt_label, gt_explanation = _extract_ground_truth(sample, normalize=normalize_labels)
                if not gt_label:
                    continue

                input_msgs = _build_input_messages(sample)

                input_text = tokenizer.apply_chat_template(
                    input_msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=enable_thinking,
                )

                texts.append(input_text)
                valid_gts.append((gt_label, gt_explanation))

            if not texts:
                continue

            # Tokenize — text only, no images
            proc_kwargs: dict[str, Any] = {
                "text": texts if len(texts) > 1 else texts[0],
                "return_tensors": "pt",
                "padding": True,
            }

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

            for (gt_label, gt_explanation), generated in zip(valid_gts, generated_texts):
                generated = generated.strip()
                pred_label, pred_explanation = _parse_response(generated, normalize=normalize_labels)

                gt_labels.append(gt_label)
                pred_labels.append(pred_label if pred_label else "<UNPARSED>")
                gt_explanations.append(gt_explanation)
                pred_explanations.append(
                    pred_explanation if pred_explanation else generated
                )

        if not gt_labels:
            logger.warning("No valid samples found for evaluation.")
            return {}

        # ---- Classification metrics ----
        metrics = self._compute_classification(gt_labels, pred_labels)

        # ---- Text-similarity metrics ----
        if "bertscore" in requested_metrics:
            metrics.update(
                self._compute_bertscore(
                    gt_explanations, pred_explanations, bert_model_name
                )
            )
        if "rouge" in requested_metrics:
            metrics.update(
                self._compute_rouge(gt_explanations, pred_explanations)
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
        pairs = [(g, p) for g, p in zip(gt, pred) if g and p]
        if not pairs:
            logger.warning("No valid explanation pairs for BERTScore.")
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
        """Compute ROUGE-L on (ground-truth, prediction) explanation pairs."""
        pairs = [(g, p) for g, p in zip(gt, pred) if g and p]
        if not pairs:
            logger.warning("No valid explanation pairs for ROUGE.")
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
