"""DFK Vision-Language Captioning Dataset — image-to-ringkasan.

Reuses the V3 CSV schema:
    link, frame_path, ringkasan, klaim, fakta, label, analisis

but only uses `frame_path` (image) and `ringkasan` (caption/summary).
Rows with empty ringkasan are skipped.

Produces a simple captioning conversation::

    [
      {"role": "user",      "content": [{"type":"text", ...}, {"type":"image"}]},
      {"role": "assistant", "content": [{"type":"text", "text": "<ringkasan>"}]},
    ]
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

logger = logging.getLogger("sita.datasets.dfk_vlm_captioning")

# ---------------------------------------------------------------------------
# Column names (same as v3)
# ---------------------------------------------------------------------------
_COL_FRAME_PATH = "frame_path"
_COL_RINGKASAN = "ringkasan"
_COL_LABEL = "label"

_REQUIRED_COLUMNS = {_COL_FRAME_PATH, _COL_RINGKASAN}


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
    """Parse a single CSV row into a captioning sample dict.

    Returns None if the row should be skipped (missing ringkasan, bad image, etc.).
    """
    frame_path_raw = (row.get(_COL_FRAME_PATH) or "").strip()
    ringkasan = (row.get(_COL_RINGKASAN) or "").strip()
    label = (row.get(_COL_LABEL) or "unknown").strip().lower()

    if not frame_path_raw or not ringkasan:
        return None

    # CSV paths have a redundant leading "images/" prefix — strip it
    if frame_path_raw.startswith("images/images/"):
        frame_path_raw = frame_path_raw[len("images/"):]

    img_path = data_dir / frame_path_raw
    if not img_path.exists():
        logger.warning("Image not found, skipping: %s", img_path)
        return None

    return {
        "image_path": str(img_path),
        "ringkasan": ringkasan,
        "label": label,
    }


def _stratified_subsample(
    rows: list[dict], ratio: float, rng: random.Random,
) -> list[dict]:
    """Subsample *rows* while preserving label distribution."""
    from collections import defaultdict

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r.get("label", "unknown")].append(r)

    sampled: list[dict] = []
    for label, group in sorted(by_label.items()):
        k = max(1, int(len(group) * ratio))
        sampled.extend(rng.sample(group, k=min(k, len(group))))

    rng.shuffle(sampled)
    return sampled


@DATASET_REGISTRY.register("dfk_vlm_captioning")
class DFKVLMCaptioning(BaseDatasetLoader):
    """Load the DFK dataset for image captioning (image → ringkasan).

    Uses the same V3 CSV schema but only cares about ``frame_path`` and
    ``ringkasan``.  Rows without a ringkasan are silently skipped.

    **Split modes** (controlled by ``use_fixed_splits``):

    1. *Fixed splits* (default): reads ``train_aug.csv`` and ``val_aug.csv``
       from ``data_dir``.

    2. *Ratio split*: reads a single CSV (``csv_file``) and splits it into
       train/val using ``train_ratio`` after a deterministic shuffle.

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``use_fixed_splits`` (bool): use train_aug.csv/val_aug.csv, default True
        - ``train_csv`` (str): training CSV filename, default ``train_aug.csv``
        - ``val_csv`` (str): validation CSV filename, default ``val_aug.csv``
        - ``csv_file`` (str): single CSV filename when not using fixed splits,
          default ``reformat_result_clean.csv``
        - ``instruction`` (str): user prompt instruction for captioning
        - ``train_ratio`` (float): fraction of data for training when ratio
          splitting, default 0.8
        - ``subsample_ratio`` (float): fraction of data to keep before
          splitting, default 1.0 (keep all).  e.g. 0.1 keeps ~10%%.
          Sampling is **stratified by label** to preserve content distribution.
        - ``seed`` (int): shuffle seed, default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)

    Example YAML::

        dataset:
          name: dfk_vlm_captioning
          kwargs:
            data_dir: dataset_v3

    Example YAML (ratio split)::

        dataset:
          name: dfk_vlm_captioning
          kwargs:
            data_dir: dataset_v3
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
                "Berikan ringkasan dari konten yang ditampilkan "
                "pada tangkapan layar berikut."
            ),
        )

        subsample_ratio = float(kwargs.pop("subsample_ratio", 1.0))
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

            # stratified undersample
            if subsample_ratio < 1.0:
                rng = random.Random(seed)
                train_rows = _stratified_subsample(train_rows, subsample_ratio, rng)
                eval_rows = _stratified_subsample(eval_rows, subsample_ratio, rng)

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

            # deterministic shuffle then undersample + split
            rng = random.Random(seed)
            rng.shuffle(all_rows)

            if subsample_ratio < 1.0:
                all_rows = _stratified_subsample(all_rows, subsample_ratio, rng)

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

                user_content = [
                    {"type": "text", "text": instruction},
                    {"type": "image"},
                ]

                yield {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": row["ringkasan"]},
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
