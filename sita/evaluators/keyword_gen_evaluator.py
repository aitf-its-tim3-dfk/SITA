"""Evaluator for keyword generation task."""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any

import torch
import transformers
from torch import nn
from tqdm import tqdm

from sita.core.base_evaluator import BaseEvaluator
from sita.core.registry import EVALUATOR_REGISTRY

logger = logging.getLogger("sita.evaluators.keyword_gen_evaluator")

# ---------------------------------------------------------------------------
# Metric Helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()

def tokenize_keywords(kw_list: list[str]) -> set[str]:
    return set(normalize_text(" ".join(kw_list)).split())

def token_precision(pred: list[str], ref: list[str]) -> float:
    pt = tokenize_keywords(pred); rt = tokenize_keywords(ref)
    return len(pt & rt) / len(pt) if pt else 0.0

def token_recall(pred: list[str], ref: list[str]) -> float:
    pt = tokenize_keywords(pred); rt = tokenize_keywords(ref)
    return len(pt & rt) / len(rt) if rt else 0.0

def token_f1(pred: list[str], ref: list[str]) -> float:
    p = token_precision(pred, ref); r = token_recall(pred, ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def phrase_f1(pred: list[str], ref: list[str]) -> float:
    pred_set = {normalize_text(k) for k in pred if k.strip()}
    ref_set  = {normalize_text(k) for k in ref  if k.strip()}
    if not pred_set or not ref_set:
        return 0.0
    tp = len(pred_set & ref_set)
    p  = tp / len(pred_set)
    r  = tp / len(ref_set)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def phrase_precision(pred: list[str], ref: list[str]) -> float:
    pred_set = {normalize_text(k) for k in pred if k.strip()}
    ref_set  = {normalize_text(k) for k in ref  if k.strip()}
    if not pred_set: return 0.0
    return len(pred_set & ref_set) / len(pred_set)

def phrase_recall(pred: list[str], ref: list[str]) -> float:
    pred_set = {normalize_text(k) for k in pred if k.strip()}
    ref_set  = {normalize_text(k) for k in ref  if k.strip()}
    if not ref_set: return 0.0
    return len(pred_set & ref_set) / len(ref_set)

def novelty_score(pred: list[str], ref: list[str]) -> float:
    if not pred: return 0.0
    ref_tokens = tokenize_keywords(ref)
    return sum(1 for kw in pred if not set(normalize_text(kw).split()) & ref_tokens) / len(pred)

def diversity_score(pred: list[str]) -> float:
    if not pred: return 0.0
    all_tok = normalize_text(" ".join(pred)).split()
    return len(set(all_tok)) / len(all_tok) if all_tok else 0.0

def exact_hit_rate(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref: return 0.0
    pred_set = {normalize_text(k) for k in pred}
    ref_set  = {normalize_text(k) for k in ref}
    return 1.0 if pred_set & ref_set else 0.0

def precision_at_k(pred: list[str], ref: list[str], k: int = 5) -> float:
    if not pred or not ref: return 0.0
    rt = tokenize_keywords(ref)
    return sum(1 for kw in pred[:k] if set(normalize_text(kw).split()) & rt) / min(k, len(pred))


def _parse_output(decoded: str) -> list[str]:
    """Parse generated text into a list of keywords using fallback methods."""
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()

    decoded = decoded.replace('\\"', '"')

    # Fallback 1: JSON array pertama
    match = re.search(r"(\[.*?\])", decoded, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(1))
            if isinstance(arr, list) and len(arr) > 0:
                return [str(x).strip() for x in arr]
        except json.JSONDecodeError:
            pass

    # Fallback 2: bracket paling akhir
    start = decoded.rfind("[")
    end = decoded.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(decoded[start : end + 1])
            if isinstance(arr, list):
                return [str(x).strip() for x in arr]
        except Exception:
            pass

    # Fallback 3: regex extract string
    items = re.findall(r'"([^"]{3,100})"', decoded)
    if items:
        return [str(x).strip() for x in items]

    return []

def _extract_ground_truth(sample: dict) -> list[str]:
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            # Handle text content in both dict format and simple string format
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return _parse_output(part["text"])
                    elif isinstance(part, str):
                        return _parse_output(part)
            elif isinstance(content, str):
                return _parse_output(content)
    return []

def _build_input_messages(sample: dict) -> list[dict]:
    return [m for m in sample.get("messages", []) if m.get("role") != "assistant"]

@EVALUATOR_REGISTRY.register("keyword_gen")
class KeywordGenEvaluator(BaseEvaluator):
    """Generate keywords and evaluate with precision, recall, F1.
    
    Example YAML:
        evaluation:
          name: keyword_gen
          kwargs:
            max_new_tokens: 300
            temperature: 0.3
    """
    
    def evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        dataset: Any,
        **kwargs,
    ) -> dict[str, float]:
        max_new_tokens = int(kwargs.get("max_new_tokens", 300))
        temperature = float(kwargs.get("temperature", 0.0))
        batch_size = int(kwargs.get("batch_size", 1))
        num_workers = int(kwargs.get("num_workers", 0))
        enable_thinking = kwargs.get("enable_thinking", False)

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

        gt_keywords_list: list[list[str]] = []
        pred_keywords_list: list[list[str]] = []

        logger.info(f"Running keyword generation on {len(dataset)} samples...")

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )

        for batch in tqdm(dataloader, desc="Generating (keywords)"):
            texts = []
            valid_gts = []

            for sample in batch:
                gt_keywords = _extract_ground_truth(sample)
                if not gt_keywords:
                    continue

                input_msgs = _build_input_messages(sample)

                input_text = tokenizer.apply_chat_template(
                    input_msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=enable_thinking,
                )

                texts.append(input_text)
                valid_gts.append(gt_keywords)

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
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
                gen_kwargs["top_p"] = 0.7
                gen_kwargs["repetition_penalty"] = 1.1
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

            for gt_kws, generated in zip(valid_gts, generated_texts):
                pred_kws = _parse_output(generated)
                gt_keywords_list.append(gt_kws)
                pred_keywords_list.append(pred_kws)

        if not gt_keywords_list:
            logger.warning("No valid samples found for evaluation.")
            return {}

        metrics = self._compute_keyword_metrics(gt_keywords_list, pred_keywords_list)

        # Log summary
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    @staticmethod
    def _compute_keyword_metrics(gt_list: list[list[str]], pred_list: list[list[str]]) -> dict[str, float]:
        metrics_sums = {
            "kw_token_precision": 0.0,
            "kw_token_recall": 0.0,
            "kw_token_f1": 0.0,
            "kw_phrase_f1": 0.0,
            "kw_phrase_precision": 0.0,
            "kw_phrase_recall": 0.0,
            "kw_novelty": 0.0,
            "kw_diversity": 0.0,
            "kw_exact_hit": 0.0,
            "kw_precision@5": 0.0,
            "kw_rouge_l": 0.0,
        }
        
        n = len(gt_list)
        if n == 0:
            return metrics_sums

        try:
            from rouge_score import rouge_scorer as rouge_scorer_lib
            _rouge_scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
            has_rouge = True
        except ImportError:
            logger.warning("rouge-score not installed — skipping ROUGE-L metric.")
            has_rouge = False
            _rouge_scorer = None
            
        def _rouge_l_score(pred_kw, ref_kw):
            if not has_rouge or not pred_kw or not ref_kw:
                return 0.0
            pred_str = " ".join(normalize_text(k) for k in pred_kw)
            ref_str  = " ".join(normalize_text(k) for k in ref_kw)
            if not pred_str.strip() or not ref_str.strip():
                return 0.0
            return _rouge_scorer.score(pred_str, ref_str)["rougeL"].fmeasure

        for ref, pred in zip(gt_list, pred_list):
            metrics_sums["kw_token_precision"] += token_precision(pred, ref)
            metrics_sums["kw_token_recall"] += token_recall(pred, ref)
            metrics_sums["kw_token_f1"] += token_f1(pred, ref)
            metrics_sums["kw_phrase_f1"] += phrase_f1(pred, ref)
            metrics_sums["kw_phrase_precision"] += phrase_precision(pred, ref)
            metrics_sums["kw_phrase_recall"] += phrase_recall(pred, ref)
            metrics_sums["kw_novelty"] += novelty_score(pred, ref)
            metrics_sums["kw_diversity"] += diversity_score(pred)
            metrics_sums["kw_exact_hit"] += exact_hit_rate(pred, ref)
            metrics_sums["kw_precision@5"] += precision_at_k(pred, ref, k=5)
            metrics_sums["kw_rouge_l"] += _rouge_l_score(pred, ref)

        return {k: v / n for k, v in metrics_sums.items()}
