"""Validators for Rejection Sampling Fine-Tuning."""

import re
from typing import Any

from sita.core.registry import VALIDATOR_REGISTRY

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    pass

@VALIDATOR_REGISTRY.register("dfk_vlm_validator")
class DFKVLMValidator:
    """Validator for DFK VLM ground truth logic."""
    
    def __init__(self, device: Any = "cpu", semantic_threshold: float = 0.8, label_only: bool = False, **kwargs):
        self.semantic_threshold = float(semantic_threshold)
        self.label_only = label_only
        self.device = device
        self.embed_model = None
        
    def _lazy_load_embed_model(self):
        if self.embed_model is None:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
    def __call__(self, generated_text: str, ground_truth_text: str) -> bool:
        """Returns True if the generated reasoning leads to the correct ground truth."""
        
        gt_label_match = re.search(r"Label:\s*(.*?)(?:\n|$)", ground_truth_text)
        gt_analisis_match = re.search(r"Analisis:\s*(.*)", ground_truth_text, re.DOTALL)
        
        gt_label = gt_label_match.group(1).strip().lower() if gt_label_match else ""
        gt_analisis = gt_analisis_match.group(1).strip() if gt_analisis_match else ""
        
        if not gt_label:
            return False  # Malformed GT, cannot validate
            
        gen_label_match = re.search(r"Label:\s*(.*?)(?:\n|$)", generated_text)
        gen_label = gen_label_match.group(1).strip().lower() if gen_label_match else ""
        
        if gen_label != gt_label:
            return False
            
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
        
        return sim >= self.semantic_threshold
