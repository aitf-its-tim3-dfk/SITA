"""DFK Vision-Language Dataset (Eval Only) — social media content analysis.

Supports the new CSV schema with columns:
    link, frame_path, ringkasan, klaim, fakta, label, analisis

Label classes: netral, disinformasi, ujaran kebencian, fitnah

This loader acts exactly like v3 but SKIPS all text elements (ringkasan, klaim, fakta) 
in the prompt. It only provides the image and the system instruction.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

from PIL import Image

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.dfk_vlm_image_only")

# ---------------------------------------------------------------------------
# Label whitelist (for sanity-checking rows)
# ---------------------------------------------------------------------------
VALID_LABELS = frozenset(
    {
        "netral",
        "disinformasi",
        "ujaran kebencian",
        "fitnah",
    }
)

# ---------------------------------------------------------------------------
# Column names in the v3 CSV
# ---------------------------------------------------------------------------
_COL_LINK = "link"
_COL_FRAME_PATH = "frame_path"
_COL_RINGKASAN = "ringkasan"
_COL_KLAIM = "klaim"
_COL_FAKTA = "fakta"
_COL_LABEL = "label"
_COL_ANALISIS = "analisis"

_REQUIRED_COLUMNS = {_COL_FRAME_PATH, _COL_LABEL, _COL_ANALISIS}


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV and return list of row dicts."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"CSV {csv_path.name} is missing required columns: {missing}. "
                f"Found: {fieldnames}"
            )
        return list(reader)


def _parse_row(row: dict[str, str], data_dir: Path) -> dict | None:
    """Parse a single CSV row into an internal sample dict.

    Returns None if the row should be skipped (missing data, bad image, etc.).
    """
    frame_path_raw = (row.get(_COL_FRAME_PATH) or "").strip()
    label = (row.get(_COL_LABEL) or "").strip().lower()
    analisis = (row.get(_COL_ANALISIS) or "").strip()

    if not frame_path_raw or not label:
        return None

    # CSV paths have a redundant leading "images/" prefix — strip it
    if frame_path_raw.startswith("images/images/"):
        frame_path_raw = frame_path_raw[len("images/"):]

    img_path = data_dir / frame_path_raw
    if not img_path.exists():
        logger.warning("Image not found, skipping: %s", img_path)
        return None

    # Optional label validation (warn but don't skip — dataset might evolve)
    if label not in VALID_LABELS:
        logger.warning(
            "Unknown label '%s' (expected one of %s), keeping anyway",
            label,
            VALID_LABELS,
        )

    return {
        "image_path": str(img_path),
        "label": label,
        "analisis": analisis,
        "link": (row.get(_COL_LINK) or "").strip(),
        "ringkasan": (row.get(_COL_RINGKASAN) or "").strip(),
        "klaim": (row.get(_COL_KLAIM) or "").strip(),
        "fakta": (row.get(_COL_FAKTA) or "").strip(),
    }


@DATASET_REGISTRY.register("dfk_vlm_image_only")
class DFKVLMImageOnlyDataset(BaseDatasetLoader):
    """Load the DFK vision dataset (Eval Only) from a local directory.

    This behaves identically to v3 but skips adding any text bits to the prompt.
    """

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(
        self, config: DatasetConfig, tokenizer: Any
    ) -> tuple[list[dict], list[dict] | None]:
        kwargs = dict(config.kwargs)

        data_dir = Path(kwargs.pop("data_dir"))
        use_fixed_splits = bool(kwargs.pop("use_fixed_splits", True))

        instruction = kwargs.pop(
            "instruction",
            (
                "Anda adalah seorang analis konten media sosial ahli. "
                "Diberikan tangkapan layar dari sebuah konten, "
                "tentukan label kategori pelanggaran dan berikan analisis detail "
                "mengenai pelanggaran yang ditemukan."
            ),
        )

        seed = int(kwargs.pop("seed", 42))
        max_samples = kwargs.pop("max_samples", None)

        # ----- load rows -----
        if use_fixed_splits:
            train_csv = kwargs.pop("train_csv", "train_aug.csv")
            val_csv = kwargs.pop("val_csv", "val_aug.csv")

            train_path = data_dir / train_csv
            val_path = data_dir / val_csv

            if not train_path.exists():
                raise FileNotFoundError(f"Train CSV not found: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"Val CSV not found: {val_path}")

            raw_train = _read_csv(train_path)
            raw_val = _read_csv(val_path)

            train_rows = [
                r
                for r in (_parse_row(row, data_dir) for row in raw_train)
                if r is not None
            ]
            eval_rows = [
                r
                for r in (_parse_row(row, data_dir) for row in raw_val)
                if r is not None
            ]

            logger.info(
                "Fixed splits — train: %d rows from %s, val: %d rows from %s",
                len(train_rows),
                train_csv,
                len(eval_rows),
                val_csv,
            )

        else:
            csv_file = kwargs.pop("csv_file", "reformat_result_clean.csv")
            train_ratio = float(kwargs.pop("train_ratio", 0.8))
            csv_path = data_dir / csv_file

            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            raw_rows = _read_csv(csv_path)
            all_rows = [
                r
                for r in (_parse_row(row, data_dir) for row in raw_rows)
                if r is not None
            ]

            logger.info("Loaded %d valid samples from %s", len(all_rows), csv_path.name)

            # deterministic shuffle then split
            rng = random.Random(seed)
            rng.shuffle(all_rows)

            split_idx = int(len(all_rows) * train_ratio)
            train_rows = all_rows[:split_idx]
            eval_rows = all_rows[split_idx:] if split_idx < len(all_rows) else []

        # ----- optional cap -----
        if max_samples is not None:
            cap = int(max_samples)
            train_rows = train_rows[:cap]
            eval_rows = eval_rows[:cap]

        # ----- generator -----
        def gen(row_list: list[dict], as_paths: bool = False):
            for row in row_list:
                try:
                    if as_paths:
                        with Image.open(row["image_path"]) as img:
                            img.verify()
                        image_data = row["image_path"]
                    else:
                        image_data = Image.open(row["image_path"]).convert("RGB")
                except Exception as e:
                    logger.warning(
                        "Failed to open %s: %s, skipping", row["image_path"], e
                    )
                    continue

                # Build user content: instruction + image (NO TEXT BITS)
                user_content = [{"type": "text", "text": instruction}]
                user_content.append({"type": "image"})

                answer = f"Label: {row['label']}\n\nAnalisis: {row['analisis']}"

                yield {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": answer},
                            ],
                        },
                    ],
                    "images": [image_data],
                }

        # ----- build HF Datasets -----
        try:
            from datasets import Dataset, Image as HFImage, Sequence

            train_ds = Dataset.from_generator(
                gen, gen_kwargs={"row_list": train_rows, "as_paths": True}
            )
            train_ds = train_ds.cast_column("images", Sequence(HFImage()))

            if eval_rows:
                eval_ds = Dataset.from_generator(
                    gen, gen_kwargs={"row_list": eval_rows, "as_paths": True}
                )
                eval_ds = eval_ds.cast_column("images", Sequence(HFImage()))
            else:
                eval_ds = None
        except ImportError:
            logger.warning(
                "datasets library not found. "
                "Falling back to list-based loading (slow)."
            )
            train_ds = list(gen(train_rows, as_paths=False))
            eval_ds = list(gen(eval_rows, as_paths=False)) if eval_rows else None

        logger.info(
            "Split: %d train, %d eval",
            len(train_ds),
            len(eval_ds) if eval_ds else 0,
        )

        return train_ds, eval_ds
