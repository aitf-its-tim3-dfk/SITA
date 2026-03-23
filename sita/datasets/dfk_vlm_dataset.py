"""DFK Vision-Language Dataset V1 — social media content analysis."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.dfk_vlm_dataset")


@DATASET_REGISTRY.register("dfk_vlm_dataset_v1")
class DFKVLMDatasetV1(BaseDatasetLoader):
    """Load the DFK vision dataset (V1) from a local directory.

    Expects a directory with:
      - A CSV file (``csv_file``) with columns: ``image_filename``,
        ``analisis_pelanggaran``, ``label``
      - An ``images/`` subdirectory containing the referenced image files

    Each row is converted into a multi-turn conversation suitable for
    Unsloth vision fine-tuning::

        [
          {"role": "user",    "content": [{"type":"text", ...}, {"type":"image", ...}]},
          {"role": "assistant","content": [{"type":"text", ...}]},
        ]

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``csv_file`` (str): CSV filename, default ``metadata-updt.csv``
        - ``instruction`` (str): system instruction for the user prompt
        - ``train_ratio`` (float): fraction of data for training, default 0.9
        - ``seed`` (int): shuffle seed, default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)

    Example YAML::

        dataset:
          name: dfk_vlm_dataset_v1
          kwargs:
            data_dir: samples/dataset
            csv_file: metadata-updt.csv
            train_ratio: 0.9
    """

    def load(self, config: DatasetConfig, tokenizer: Any) -> tuple[list[dict], list[dict] | None]:
        kwargs = dict(config.kwargs)

        data_dir = Path(kwargs.pop("data_dir"))
        csv_file = kwargs.pop("csv_file", "metadata-updt.csv")
        instruction = kwargs.pop(
            "instruction",
            (
                "Anda adalah seorang analis konten media sosial ahli. "
                "Diberikan tangkapan layar dari sebuah unggahan media sosial, "
                "tentukan label kategori pelanggaran dan berikan analisis detail "
                "mengenai pelanggaran yang ditemukan."
            ),
        )
        train_ratio = float(kwargs.pop("train_ratio", 0.9))
        seed = int(kwargs.pop("seed", 42))
        max_samples = kwargs.pop("max_samples", None)

        csv_path = data_dir / csv_file
        images_dir = data_dir / "images"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Parse CSV
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("image_filename", "").strip()
                label = row.get("label", "").strip()
                analisis = row.get("analisis_pelanggaran", "").strip()

                if not filename or not label:
                    continue

                img_path = images_dir / filename
                if not img_path.exists():
                    logger.warning(f"Image not found, skipping: {img_path}")
                    continue

                rows.append({
                    "image_path": str(img_path),
                    "label": label,
                    "analisis": analisis,
                })

        logger.info(f"Loaded {len(rows)} valid samples from {csv_path}")

        if max_samples is not None:
            rows = rows[: int(max_samples)]

        # Shuffle deterministically
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)

        # Convert to conversation format
        conversations = []
        for row in rows:
            try:
                # We don't open the image here to keep it serializable for Dataset.from_list
                # SFTTrainer/Processor will handle opening/loading if needed, or we provide it.
                # Actually, many VLM processors expect PIL images.
                image = Image.open(row["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {row['image_path']}: {e}, skipping")
                continue

            answer = f"Label: {row['label']}\n\nAnalisis: {row['analisis']}"

            conv = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": image},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    },
                ]
            }
            conversations.append(conv)

        # Train/eval split
        split_idx = int(len(conversations) * train_ratio)
        train_convs = conversations[:split_idx]
        eval_convs = conversations[split_idx:] if split_idx < len(conversations) else []

        # Convert to HF Dataset (required by recent TRL versions)
        try:
            from datasets import Dataset
            train_ds = Dataset.from_list(train_convs)
            eval_ds = Dataset.from_list(eval_convs) if eval_convs else None
        except ImportError:
            logger.warning("datasets library not found. Returning raw lists (may cause SFTTrainer to fail).")
            train_ds = train_convs
            eval_ds = eval_convs if eval_convs else None

        logger.info(
            f"Split: {len(train_ds)} train, "
            f"{len(eval_ds) if eval_ds else 0} eval"
        )

        return train_ds, eval_ds
