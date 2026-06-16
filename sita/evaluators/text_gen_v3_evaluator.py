"""Text-only generative evaluator V3 — classification metrics + BERTScore + ROUGE-L.

For the DFK text dataset V3 which uses the format:
    Penjelasan:
    ...
    Kesimpulan:
    Label: Fitnah.

No image handling — this is a pure text evaluator.
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

logger = logging.getLogger("sita.evaluators.text_gen_v3_evaluator")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LABEL_RE = re.compile(r"Label:\s*(.+?)(?:\n|$|\.)", re.IGNORECASE)

_LABEL_NORMALIZE_MERGED: dict[str, str] = {
    "non-dfk": "netral",
    "non_dfk": "netral",
    "nondfk": "netral",
    "bukan_dfk": "netral",
    "bukan dfk": "netral",
    "fakta": "netral",
    "ujaran_kebencian": "ujaran kebencian",
    "disinformasi_dan_ujaran_kebencian": "ujaran kebencian",
}

_LABEL_NORMALIZE_RAW: dict[str, str] = {
    "non-dfk": "non_dfk",
    "non_dfk": "non_dfk",
    "nondfk": "non_dfk",
    "bukan_dfk": "non_dfk",
    "bukan dfk": "non_dfk",
    "fakta": "fakta",
    "disinformasi": "disinformasi",
    "fitnah": "fitnah",
    "ujaran_kebencian": "ujaran kebencian",
    "disinformasi_dan_ujaran_kebencian": "ujaran kebencian",
}


def _normalize_label(label: str, normalize: bool = True) -> str:
    """Map old/variant label names to canonical form."""
    clean = label.lower().strip().rstrip(".")
    if not normalize:
        return _LABEL_NORMALIZE_RAW.get(clean, clean).replace("-", "_")
    return _LABEL_NORMALIZE_MERGED.get(clean, clean)


def _parse_response_v3(text: str, normalize: bool = True) -> tuple[str, str]:
    """Extract (label, explanation) from generated / ground-truth response in V3 format."""
    text = _THINK_RE.sub("", text).strip()
    
    # Extract Label
    label_m = _LABEL_RE.search(text)
    label = label_m.group(1).strip().strip("*").strip(".") if label_m else ""
    
    # Extract Penjelasan
    penjelasan = text
    # Case-insensitive search for Kesimpulan
    kesimpulan_idx = text.lower().find("kesimpulan:")
    if kesimpulan_idx != -1:
        penjelasan = text[:kesimpulan_idx]
    
    # Case-insensitive search for Penjelasan
    penjelasan_idx = penjelasan.lower().find("penjelasan:")
    if penjelasan_idx != -1:
        penjelasan = penjelasan[penjelasan_idx + len("penjelasan:"):]
        
    penjelasan = penjelasan.strip()
    
    return _normalize_label(label, normalize=normalize), penjelasan


def _extract_ground_truth(sample: dict, normalize: bool = True) -> tuple[str, str]:
    """Pull the reference answer text from a conversation dict."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    return _parse_response_v3(part["text"], normalize=normalize)
            # If content is a string
            if isinstance(msg.get("content"), str):
                return _parse_response_v3(msg["content"], normalize=normalize)
    return "", ""


def _build_input_messages(sample: dict) -> list[dict]:
    """Return system + user turns for generation (strip assistant)."""
    return [m for m in sample.get("messages", []) if m.get("role") != "assistant"]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@EVALUATOR_REGISTRY.register("text_gen_v3")
class TextGenV3Evaluator(BaseEvaluator):
    """Generate predictions and compute classification + text-similarity metrics for V3 format.

    Expects the assistant response format:
        Penjelasan:
        ...
        Kesimpulan:
        Label: ...

    Example YAML::

        evaluation:
          name: text_gen_v3
          kwargs:
            max_new_tokens: 512
            temperature: 0.0
            metrics:
              - bertscore
              - rouge
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

        for batch in tqdm(dataloader, desc="Generating (text V3)"):
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

            # Tokenize — text only
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
                pred_label, pred_explanation = _parse_response_v3(generated, normalize=normalize_labels)

                gt_labels.append(gt_label)
                pred_labels.append(pred_label if pred_label else "<UNPARSED>")
                gt_explanations.append(gt_explanation)
                pred_explanations.append(
                    pred_explanation if pred_explanation else generated
                )

        if not gt_labels:
            logger.warning("No valid samples found for evaluation.")
            return {}

        metrics = self._compute_classification(gt_labels, pred_labels)

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

        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    @staticmethod
    def _compute_classification(gt: list[str], pred: list[str]) -> dict[str, float]:
        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_recall_fscore_support,
            )
        except ImportError:
            logger.warning("scikit-learn not installed — skipping classification metrics.")
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
            return {}

        gt_filtered, pred_filtered = zip(*pairs)
        try:
            from bert_score import score as bert_score_fn
        except ImportError:
            return {}

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
        pairs = [(g, p) for g, p in zip(gt, pred) if g and p]
        if not pairs:
            return {}

        gt_filtered, pred_filtered = zip(*pairs)
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            return {}

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
