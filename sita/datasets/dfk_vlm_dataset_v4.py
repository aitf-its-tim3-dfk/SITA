"""DFK Vision-Language Dataset V4 — social media content analysis (JSONL support).

Supports:
- Fixed splits (train.jsonl / validation.jsonl)
- Single JSON with ratio split
- JSONL file with automatic train/val split
- Generates messages similar to dfk_vlm_dataset_v3
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from PIL import Image
from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.dfk_vlm_dataset_v4")


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


def _parse_sample(sample: list | dict, data_dir: Path, image_cache: dict[str, Path] | None = None) -> dict | None:
    """Parse sample JSON into internal dict format."""
    if isinstance(sample, dict) and "messages" in sample:
        messages = sample["messages"]
    elif isinstance(sample, list):
        messages = sample
    else:
        logger.warning("Sample format not recognized, skipping.")
        return None

    clean_msgs = []
    image_paths = []

    for msg in messages:
        role = msg.get("role", "")
        content_items = msg.get("content", [])
        clean_msg = {"role": role, "content": []}

        for item in content_items:
            item_type = item.get("type")
            if item_type == "text":
                clean_msg["content"].append(
                    {"type": "text", "text": item.get("text", "")}
                )
            elif item_type == "image":
                clean_msg["content"].append({"type": "image"})
                raw_path = item.get("image")
                if raw_path:
                    filename = Path(raw_path).name
                    if image_cache is not None and filename in image_cache:
                        img_path = image_cache[filename]
                    else:
                        img_path = None
                        if image_cache is not None:
                            import re
                            req_p = Path(raw_path)
                            req_parent_name = req_p.parent.name
                            frame_match = re.search(r'(_frame_\d+)', filename)
                            frame_str = frame_match.group(1) if frame_match else ""
                            base_req_stem = req_p.stem.replace(frame_str, "")
                            
                            best_match = None
                            best_score = 0
                            
                            for p in image_cache.values():
                                if frame_str and frame_str not in p.name:
                                    continue
                                
                                base_disk_stem = p.stem.replace(frame_str, "")
                                score = 0
                                if base_disk_stem and base_req_stem:
                                    if base_disk_stem in base_req_stem or base_req_stem in base_disk_stem:
                                        score += min(len(base_disk_stem), len(base_req_stem))
                                
                                if score > 0:
                                    if p.parent.name and (p.parent.name in req_parent_name or req_parent_name in p.parent.name):
                                        score += 50
                                
                                if score > best_score and score >= 5:
                                    best_score = score
                                    best_match = p
                                    
                            if best_match:
                                img_path = best_match

                        if img_path is None:
                            if raw_path.startswith("images/images/"):
                                raw_path = raw_path[len("images/") :]
                            img_path = data_dir / raw_path

                    if not img_path.exists():
                        logger.warning("Image not found, skipping sample: %s (tried filename: %s)", img_path, filename)
                        return None
                    image_paths.append(str(img_path))

        clean_msgs.append(clean_msg)

    if not clean_msgs:
        return None

    return {"messages": clean_msgs, "image_paths": image_paths}


@DATASET_REGISTRY.register("dfk_vlm_dataset_v4")
class DFKVLMDatasetV4(BaseDatasetLoader):
    """Load DFK vision dataset V4 from JSON/JSONL files (messages-based format).

    Supports three input modes:

    1. **Fixed splits** (default): reads ``train.jsonl`` and ``validation.jsonl`` from
       ``data_dir``.  No shuffling or ratio splitting is performed.

    2. **JSONL mode**: reads a single JSONL file and splits it into
       train/val using ``train_ratio`` after a deterministic shuffle.

    3. **Single JSON mode**: reads a single JSON file and splits it into
       train/val using ``train_ratio`` after a deterministic shuffle.

    Expected JSON/JSONL schema::

        {
          "messages": [
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "...instruction..."},
                {"type": "text", "text": "...context..."},
                {"type": "image", "image": "path/to/image.jpg"}
              ]
            },
            {
              "role": "assistant",
              "content": [
                {"type": "text", "text": "...response..."}
              ]
            }
          ]
        }

    The loader validates image paths and ensures they exist in ``data_dir``.
    Images can be in subdirectories (e.g., ``images/img.jpg``).

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``use_fixed_splits`` (bool): use train.jsonl/validation.jsonl, default True
        - ``use_jsonl`` (bool): use JSONL format, default False
        - ``train_file`` (str): training JSON filename, default ``train.jsonl``
        - ``val_file`` (str): validation JSON filename, default ``validation.jsonl``
        - ``jsonl_file`` (str): JSONL filename when use_jsonl=True, default ``dataset.jsonl``
        - ``json_file`` (str): single JSON filename for ratio split, default ``dataset.json``
        - ``train_ratio`` (float): fraction of data for training when ratio
          splitting, default 0.8
        - ``seed`` (int): shuffle seed (ratio split only), default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)
        - ``instruction`` (str): override instruction text in user messages (optional)

    Example YAML (fixed splits)::

        dataset:
          name: dfk_vlm_dataset_v4
          kwargs:
            data_dir: data/aitf-dfk3-vlm-dataset-jsonl

    Example YAML (JSONL with ratio split)::

        dataset:
          name: dfk_vlm_dataset_v4
          kwargs:
            data_dir: data/aitf-dfk3-vlm-dataset-jsonl
            use_fixed_splits: false
            use_jsonl: true
            jsonl_file: dataset.jsonl
            train_ratio: 0.8

    Example YAML (single JSON with ratio split)::

        dataset:
          name: dfk_vlm_dataset_v4
          kwargs:
            data_dir: data/aitf-dfk3-vlm-dataset-jsonl
            use_fixed_splits: false
            json_file: dataset.json
            train_ratio: 0.8
    """

    def load(
        self, config: DatasetConfig, tokenizer: Any
    ) -> tuple[list[dict], list[dict] | None]:
        kwargs = dict(config.kwargs)
        data_dir = Path(kwargs.pop("data_dir"))
        use_fixed_splits = bool(kwargs.pop("use_fixed_splits", True))
        use_jsonl = bool(kwargs.pop("use_jsonl", False))
        train_ratio = float(kwargs.pop("train_ratio", 0.8))
        seed = int(kwargs.pop("seed", 42))
        max_samples = kwargs.pop("max_samples", None)

        instruction = kwargs.pop(
            "instruction",
            "Anda adalah seorang analis konten media sosial ahli. "
            "Diberikan tangkapan layar dari sebuah konten, "
            "tentukan label kategori pelanggaran dan berikan analisis detail "
            "mengenai pelanggaran yang ditemukan.",
        )

        # Build image cache
        logger.info("Building image cache to resolve paths resiliently...")
        image_cache: dict[str, Path] = {}
        search_dirs = [data_dir, data_dir.parent]
        extra_search = kwargs.pop("image_search_dirs", [])
        if isinstance(extra_search, str):
            extra_search = [extra_search]
        for ed in extra_search:
            search_dirs.append(Path(ed))

        for sdir in search_dirs:
            if sdir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.GIF"]:
                    for p in sdir.rglob(ext):
                        if p.name not in image_cache:
                            image_cache[p.name] = p
        logger.info("Found %d images across data directories.", len(image_cache))

        # Load raw samples
        if use_fixed_splits:
            train_file = kwargs.pop("train_file", "train.jsonl")
            val_file = kwargs.pop("val_file", "validation.jsonl")
            train_path = data_dir / train_file
            val_path = data_dir / val_file
            if not train_path.exists() or not val_path.exists():
                raise FileNotFoundError(
                    f"Train or val file not found: {train_path}, {val_path}"
                )
            read_fn_train = _read_jsonl if train_path.suffix == ".jsonl" else _read_json
            read_fn_val = _read_jsonl if val_path.suffix == ".jsonl" else _read_json
            raw_train = [_parse_sample(r, data_dir, image_cache) for r in read_fn_train(train_path)]
            raw_val = [_parse_sample(r, data_dir, image_cache) for r in read_fn_val(val_path)]
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
            all_rows = [_parse_sample(r, data_dir, image_cache) for r in _read_jsonl(jsonl_path)]
            all_rows = [r for r in all_rows if r is not None]
            logger.info(
                "Loaded %d valid samples from %s", len(all_rows), jsonl_path.name
            )
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
            all_rows = [_parse_sample(r, data_dir, image_cache) for r in _read_json(json_path)]
            all_rows = [r for r in all_rows if r is not None]
            logger.info(
                "Loaded %d valid samples from %s", len(all_rows), json_path.name
            )
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

        # Generator to build messages/images
        def gen(row_list: list[dict], as_paths: bool = False):
            for row in row_list:
                images_data = []
                valid = True
                for img_p in row["image_paths"]:
                    try:
                        if as_paths:
                            with Image.open(img_p) as img:
                                img.verify()
                            images_data.append(img_p)
                        else:
                            images_data.append(Image.open(img_p).convert("RGB"))
                    except Exception as e:
                        logger.warning("Failed to open %s: %s, skipping", img_p, e)
                        valid = False
                        break
                if not valid:
                    continue

                yield {"messages": row["messages"], "images": images_data}

        # Build HuggingFace Dataset
        try:
            from datasets import Dataset, Image as HFImage, Sequence

            train_ds = Dataset.from_generator(
                gen, gen_kwargs={"row_list": raw_train, "as_paths": True}
            )
            train_ds = train_ds.cast_column("images", Sequence(HFImage()))
            eval_ds = None
            if raw_val:
                eval_ds = Dataset.from_generator(
                    gen, gen_kwargs={"row_list": raw_val, "as_paths": True}
                )
                eval_ds = eval_ds.cast_column("images", Sequence(HFImage()))
        except ImportError:
            logger.warning("datasets library not found, falling back to list loading")
            train_ds = list(gen(raw_train, as_paths=False))
            eval_ds = list(gen(raw_val, as_paths=False)) if raw_val else None

        logger.info(
            "Split: %d train, %d eval",
            len(train_ds),
            len(eval_ds) if eval_ds else 0,
        )
        return train_ds, eval_ds
