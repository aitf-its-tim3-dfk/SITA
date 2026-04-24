"""DFK Vision-Language Dataset V2 — social media content analysis.

Supports the new CSV schema with columns:
    link, img, ringkasan, klaim, fakta, label, analisis

Label classes: netral, disinformasi, ujaran kebencian, fitnah

By default uses pre-split files (train.csv / val.csv).  Optionally
falls back to a single CSV with a configurable train/val ratio.
"""

from __future__ import annotations

import ast
import csv
import logging
import random
from pathlib import Path
from typing import Any

from PIL import Image

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.dfk_vlm_dataset_v2")

# ---------------------------------------------------------------------------
# Label whitelist (for sanity-checking rows)
# ---------------------------------------------------------------------------
VALID_LABELS = frozenset({
    "netral",
    "disinformasi",
    "ujaran kebencian",
    "fitnah",
})

# ---------------------------------------------------------------------------
# Column names in the v2 CSV
# ---------------------------------------------------------------------------
_COL_LINK = "link"
_COL_IMG = "img"
_COL_RINGKASAN = "ringkasan"
_COL_KLAIM = "klaim"
_COL_FAKTA = "fakta"
_COL_LABEL = "label"
_COL_ANALISIS = "analisis"

_REQUIRED_COLUMNS = {_COL_IMG, _COL_LABEL, _COL_ANALISIS}


def _parse_img_field(raw: str) -> list[str]:
    """Parse the ``img`` column which is a stringified Python list of paths.

    Examples::

        "['images/foo.jpg']"
        "['images/a.jpg', 'images/b.png']"
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        stripped = raw.strip("[] '\"")
        parsed = [s.strip().strip("'\"") for s in stripped.split(",") if s.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]
    return parsed


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
    raw_img = (row.get(_COL_IMG) or "").strip()
    label = (row.get(_COL_LABEL) or "").strip()
    analisis = (row.get(_COL_ANALISIS) or "").strip()

    if not raw_img or not label:
        return None

    # Parse image paths — use first image
    paths = _parse_img_field(raw_img)
    if not paths:
        return None

    img_path = data_dir / paths[0]
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


@DATASET_REGISTRY.register("dfk_vlm_dataset_v2")
class DFKVLMDatasetV2(BaseDatasetLoader):
    """Load the DFK vision dataset (V2) from a local directory.

    CSV schema (v2)::

        link, img, ringkasan, klaim, fakta, label, analisis

    Label classes: ``netral``, ``disinformasi``, ``ujaran kebencian``, ``fitnah``

    **Split modes** (controlled by ``use_fixed_splits``):

    1. *Fixed splits* (default): reads ``train.csv`` and ``val.csv`` from
       ``data_dir``.  No shuffling or ratio splitting is performed.

    2. *Ratio split*: reads a single CSV (``csv_file``) and splits it into
       train/val using ``train_ratio`` after a deterministic shuffle.

    Each row is converted into a multi-turn conversation suitable for
    Unsloth vision fine-tuning::

        [
          {"role": "user",      "content": [{"type":"text", ...}, {"type":"image", ...}]},
          {"role": "assistant", "content": [{"type":"text", ...}]},
        ]

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``use_fixed_splits`` (bool): use train.csv/val.csv, default True
        - ``train_csv`` (str): training CSV filename, default ``train.csv``
        - ``val_csv`` (str): validation CSV filename, default ``val.csv``
        - ``csv_file`` (str): single CSV filename when not using fixed splits,
          default ``reformat_result_clean.csv``
        - ``instruction`` (str): system instruction for the user prompt
        - ``train_ratio`` (float): fraction of data for training when ratio
          splitting, default 0.8
        - ``seed`` (int): shuffle seed (ratio split only), default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)

    Example YAML (fixed splits)::

        dataset:
          name: dfk_vlm_dataset_v2
          kwargs:
            data_dir: dataset_v2

    Example YAML (ratio split)::

        dataset:
          name: dfk_vlm_dataset_v2
          kwargs:
            data_dir: dataset_v2
            use_fixed_splits: false
            csv_file: reformat_result_clean.csv
            train_ratio: 0.8
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
            train_csv = kwargs.pop("train_csv", "train.csv")
            val_csv = kwargs.pop("val_csv", "val.csv")

            train_path = data_dir / train_csv
            val_path = data_dir / val_csv

            if not train_path.exists():
                raise FileNotFoundError(f"Train CSV not found: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"Val CSV not found: {val_path}")

            raw_train = _read_csv(train_path)
            raw_val = _read_csv(val_path)

            train_rows = [r for r in (
                _parse_row(row, data_dir) for row in raw_train
            ) if r is not None]
            eval_rows = [r for r in (
                _parse_row(row, data_dir) for row in raw_val
            ) if r is not None]

            logger.info(
                "Fixed splits — train: %d rows from %s, val: %d rows from %s",
                len(train_rows), train_csv, len(eval_rows), val_csv,
            )

        else:
            csv_file = kwargs.pop("csv_file", "reformat_result_clean.csv")
            train_ratio = float(kwargs.pop("train_ratio", 0.8))
            csv_path = data_dir / csv_file

            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            raw_rows = _read_csv(csv_path)
            all_rows = [r for r in (
                _parse_row(row, data_dir) for row in raw_rows
            ) if r is not None]

            logger.info(
                "Loaded %d valid samples from %s", len(all_rows), csv_path.name
            )

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

                # Build user content: instruction + context + image
                user_content = [{"type": "text", "text": instruction}]

                context_parts = []
                if row.get("ringkasan"):
                    context_parts.append(f"Ringkasan: {row['ringkasan']}")
                if row.get("klaim"):
                    context_parts.append(f"Klaim: {row['klaim']}")
                if row.get("fakta"):
                    context_parts.append(f"Fakta: {row['fakta']}")
                if context_parts:
                    user_content.append(
                        {"type": "text", "text": "\n".join(context_parts)}
                    )

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
