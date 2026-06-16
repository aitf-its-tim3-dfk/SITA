"""DFK Text Dataset V2 — text-only content analysis (JSONL support).

Supports:
- JSONL file with automatic train/val split
- Single JSON with ratio split
- Fixed splits (train.json / val.json)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.dfk_text_dataset_v2")


def _read_json(json_path: Path) -> list[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON root to be a list, got {type(data)}")
        return data


def _read_jsonl(jsonl_path: Path) -> list[Any]:
    """Read a JSONL file and return list of samples."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _parse_sample(sample: list | dict) -> dict | None:
    """Parse sample JSON into internal dict format."""
    if isinstance(sample, dict) and "messages" in sample:
        messages = sample["messages"]
    elif isinstance(sample, list):
        messages = sample
    else:
        logger.warning("Sample format not recognized, skipping.")
        return None

    clean_msgs = []

    for msg in messages:
        role = msg.get("role", "")
        content_val = msg.get("content", [])
        
        if isinstance(content_val, str):
            clean_msgs.append({"role": role, "content": content_val})
        elif isinstance(content_val, list):
            clean_msg = {"role": role, "content": []}
            for item in content_val:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        clean_msg["content"].append({"type": "text", "text": item.get("text", "")})
                else:
                    clean_msg["content"].append({"type": "text", "text": str(item)})
            clean_msgs.append(clean_msg)
        else:
            logger.warning("Unknown content type: %s", type(content_val))
            return None

    if not clean_msgs:
        return None

    return {"messages": clean_msgs}


@DATASET_REGISTRY.register("dfk_text_dataset_v2")
class DFKTextDatasetV2(BaseDatasetLoader):
    """Load DFK text dataset V2 from JSON/JSONL files (messages-based TRL format).

    Supports three input modes:

    1. **JSONL mode** (default): reads a single JSONL file and splits it into
       train/val using ``train_ratio`` after a deterministic shuffle.

    2. **Fixed splits**: reads ``train.json`` and ``val.json`` from
       ``data_dir``. No shuffling or ratio splitting is performed.

    3. **Single JSON mode**: reads a single JSON file and splits it into
       train/val using ``train_ratio`` after a deterministic shuffle.

    Expected JSON/JSONL schema is standard TRL messages format::

        {
          "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``use_jsonl`` (bool): use JSONL format, default True
        - ``use_fixed_splits`` (bool): use train.json/val.json, default False
        - ``jsonl_file`` (str): JSONL filename when use_jsonl=True, default ``dataset.jsonl``
        - ``json_file`` (str): single JSON filename for ratio split, default ``dataset.json``
        - ``train_file`` (str): training JSON filename, default ``train.json``
        - ``val_file`` (str): validation JSON filename, default ``val.json``
        - ``train_ratio`` (float): fraction of data for training when ratio
          splitting, default 0.8
        - ``seed`` (int): shuffle seed (ratio split only), default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)
    """

    def load(self, config: DatasetConfig, tokenizer: Any) -> tuple[list[dict], list[dict] | None]:
        kwargs = dict(config.kwargs)
        data_dir = Path(kwargs.pop("data_dir"))
        use_jsonl = bool(kwargs.pop("use_jsonl", True))
        use_fixed_splits = bool(kwargs.pop("use_fixed_splits", False))
        train_ratio = float(kwargs.pop("train_ratio", 0.8))
        seed = int(kwargs.pop("seed", 42))
        max_samples = kwargs.pop("max_samples", None)

        # Load raw samples
        if use_fixed_splits:
            train_file = kwargs.pop("train_file", "train.json")
            val_file = kwargs.pop("val_file", "val.json")
            train_path = data_dir / train_file
            val_path = data_dir / val_file
            if not train_path.exists() or not val_path.exists():
                raise FileNotFoundError(f"Train or val file not found: {train_path}, {val_path}")
            raw_train = [_parse_sample(r) for r in _read_json(train_path)]
            raw_val = [_parse_sample(r) for r in _read_json(val_path)]
            raw_train = [r for r in raw_train if r is not None]
            raw_val = [r for r in raw_val if r is not None]
            logger.info(
                "Fixed splits — train: %d samples from %s, val: %d samples from %s",
                len(raw_train),
                train_file,
                len(raw_val),
                val_file,
            )

        elif use_jsonl:
            jsonl_file = kwargs.pop("jsonl_file", "dataset.jsonl")
            jsonl_path = data_dir / jsonl_file
            if not jsonl_path.exists():
                raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
            all_rows = [_parse_sample(r) for r in _read_jsonl(jsonl_path)]
            all_rows = [r for r in all_rows if r is not None]
            logger.info("Loaded %d valid samples from %s", len(all_rows), jsonl_path.name)
            rng = random.Random(seed)
            rng.shuffle(all_rows)
            split_idx = int(len(all_rows) * train_ratio)
            raw_train = all_rows[:split_idx]
            raw_val = all_rows[split_idx:] if split_idx < len(all_rows) else []
            logger.info(
                "JSONL split (ratio=%.2f, seed=%d) — train: %d, val: %d",
                train_ratio,
                seed,
                len(raw_train),
                len(raw_val),
            )
        else:
            json_file = kwargs.pop("json_file", "dataset.json")
            json_path = data_dir / json_file
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            all_rows = [_parse_sample(r) for r in _read_json(json_path)]
            all_rows = [r for r in all_rows if r is not None]
            logger.info("Loaded %d valid samples from %s", len(all_rows), json_path.name)
            rng = random.Random(seed)
            rng.shuffle(all_rows)
            split_idx = int(len(all_rows) * train_ratio)
            raw_train = all_rows[:split_idx]
            raw_val = all_rows[split_idx:] if split_idx < len(all_rows) else []
            logger.info(
                "JSON split (ratio=%.2f, seed=%d) — train: %d, val: %d",
                train_ratio,
                seed,
                len(raw_train),
                len(raw_val),
            )

        if max_samples is not None:
            raw_train = raw_train[:max_samples]
            raw_val = raw_val[:max_samples] if raw_val else []
            logger.info("Capped to max_samples=%d", max_samples)

        # Generator to build messages
        def gen(row_list: list[dict]):
            for row in row_list:
                yield {"messages": row["messages"]}

        # Build HuggingFace Dataset
        try:
            from datasets import Dataset

            train_ds = Dataset.from_generator(gen, gen_kwargs={"row_list": raw_train})
            eval_ds = None
            if raw_val:
                eval_ds = Dataset.from_generator(gen, gen_kwargs={"row_list": raw_val})
        except ImportError:
            logger.warning("datasets library not found, falling back to list loading")
            train_ds = list(gen(raw_train))
            eval_ds = list(gen(raw_val)) if raw_val else None

        logger.info(
            "Split: %d train, %d eval",
            len(train_ds),
            len(eval_ds) if eval_ds else 0,
        )
        return train_ds, eval_ds
