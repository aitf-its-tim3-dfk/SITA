"""Validators for Rejection Sampling Fine-Tuning."""

import logging
import re
from collections import Counter
from typing import Any

from sita.core.registry import VALIDATOR_REGISTRY

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    pass

logger = logging.getLogger("sita.validators.dfk_validator")

@VALIDATOR_REGISTRY.register("dfk_vlm_validator")
class DFKVLMValidator:
    """Validator for DFK VLM ground truth logic."""
    
    def __init__(self, device: Any = "cpu", semantic_threshold: float = 0.8, label_only: bool = False, **kwargs):
        self.semantic_threshold = float(semantic_threshold)
        self.label_only = label_only
        self.device = device
        self.embed_model = None
        self._rejection_counts = Counter()
        self._total_calls = 0
        self._log_interval = 50  # log rejection stats every N calls
        
    def _lazy_load_embed_model(self):
        if self.embed_model is None:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _reject(self, reason: str, gen_text_snippet: str = "") -> bool:
        self._rejection_counts[reason] += 1
        if self._total_calls % self._log_interval == 0 and self._total_calls > 0:
            total_rejected = sum(self._rejection_counts.values())
            logger.info(
                f"Validator stats ({self._total_calls} calls, "
                f"{self._total_calls - total_rejected} accepted): "
                f"{dict(self._rejection_counts)}"
            )
        return False

    def __call__(self, generated_text: str, ground_truth_text: str) -> bool:
        """Returns True if the generated reasoning leads to the correct ground truth."""
        self._total_calls += 1
        
        gt_label_match = re.search(r"Label:\s*(.*?)(?:\n|$)", ground_truth_text)
        gt_analisis_match = re.search(r"Analisis:\s*(.*)", ground_truth_text, re.DOTALL)
        
        gt_label = gt_label_match.group(1).strip().lower() if gt_label_match else ""
        gt_analisis = gt_analisis_match.group(1).strip() if gt_analisis_match else ""
        
        if not gt_label:
            return self._reject("no_label_in_gt")
            
        gen_label_match = re.search(r"Label:\s*(.*?)(?:\n|$)", generated_text)
        gen_label = gen_label_match.group(1).strip().lower() if gen_label_match else ""
        
        if not gen_label:
            return self._reject("no_label_in_gen", generated_text[:200])
        
        if gen_label != gt_label:
            return self._reject(f"label_mismatch({gen_label}!={gt_label})")
            
        if self.label_only:
            return True
            
        # Structural match success. Check Semantic.
        gen_analisis_match = re.search(r"Analisis:\s*(.*)", generated_text, re.DOTALL)
        gen_analisis = gen_analisis_match.group(1).strip() if gen_analisis_match else ""
        
        if not gt_analisis or not gen_analisis:
            return True  # Fallback
            
        self._lazy_load_embed_model()
        
        gt_emb = self.embed_model.encode(gt_analisis, convert_to_tensor=True, show_progress_bar=False)
        gen_emb = self.embed_model.encode(gen_analisis, convert_to_tensor=True, show_progress_bar=False)
        sim = util.pytorch_cos_sim(gt_emb, gen_emb).item()
        
        if sim < self.semantic_threshold:
            return self._reject(f"low_similarity({sim:.2f})")
        
        return True
