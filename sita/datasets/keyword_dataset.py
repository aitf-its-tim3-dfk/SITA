"""Keyword extraction dataset loader for conversational TRL format."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.keyword_dataset")

@DATASET_REGISTRY.register("keyword_dataset")
class KeywordDataset(BaseDatasetLoader):
    """Load keyword extraction dataset in conversational TRL JSONL format.
    
    The JSONL file should have a `messages` key per row containing the
    system, user, and assistant turns.
    
    Config kwargs:
        - `data_dir` (str): path to dataset directory (required)
        - `filename` (str): JSONL filename
        - `train_ratio` (float): fraction for training, default 0.8
        - `seed` (int): shuffle seed, default 42
        - `max_samples` (int | None): cap total samples
    """

    def load(
        self, config: DatasetConfig, tokenizer: Any
    ) -> tuple[Any, Any | None]:
        kwargs = dict(config.kwargs)
        
        data_dir = Path(kwargs.pop("data_dir"))
        filename = kwargs.pop("filename", "dataset_conversational_trl_2.jsonl")
        train_ratio = float(kwargs.pop("train_ratio", 0.8))
        seed = int(kwargs.pop("seed", 42))
        max_samples = kwargs.pop("max_samples", None)
        
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        try:
            from datasets import load_dataset
            
            # Load the JSONL dataset
            ds = load_dataset("json", data_files=str(filepath), split="train")
            
            # Optional cap
            if max_samples is not None:
                ds = ds.select(range(min(len(ds), max_samples)))
                
            # Shuffle and split
            ds = ds.shuffle(seed=seed)
            split_ds = ds.train_test_split(train_size=train_ratio, seed=seed)
            
            train_ds = split_ds["train"]
            eval_ds = split_ds["test"]
            
            logger.info(
                "Split: %d train, %d eval",
                len(train_ds),
                len(eval_ds),
            )
            
            return train_ds, eval_ds
            
        except ImportError:
            # Fallback to pure python list
            import json
            logger.warning(
                "datasets library not found. "
                "Falling back to list-based loading (slow)."
            )
            
            with open(filepath, "r", encoding="utf-8") as f:
                all_rows = [json.loads(line) for line in f if line.strip()]
                
            rng = random.Random(seed)
            rng.shuffle(all_rows)
            
            if max_samples is not None:
                all_rows = all_rows[:max_samples]
                
            split_idx = int(len(all_rows) * train_ratio)
            train_ds = all_rows[:split_idx]
            eval_ds = all_rows[split_idx:] if split_idx < len(all_rows) else None
            
            logger.info(
                "Split: %d train, %d eval",
                len(train_ds),
                len(eval_ds) if eval_ds else 0,
            )
            return train_ds, eval_ds
