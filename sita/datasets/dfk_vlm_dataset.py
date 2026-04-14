"""DFK Vision-Language Dataset V1 — social media content analysis."""

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

logger = logging.getLogger("sita.datasets.dfk_vlm_dataset")

# ---------------------------------------------------------------------------
# CSV format identifiers
# ---------------------------------------------------------------------------
_FORMAT_LEGACY = "legacy"  # columns: image_filename, analisis_pelanggaran, label
_FORMAT_IMAGES = "images"  # columns: title, link, text, img_path, label


def _detect_csv_format(fieldnames: list[str]) -> str:
    """Auto-detect which CSV schema we're looking at."""
    if "image_filename" in fieldnames:
        return _FORMAT_LEGACY
    if "img_path" in fieldnames:
        return _FORMAT_IMAGES
    raise ValueError(
        f"Unrecognised CSV columns: {fieldnames}. "
        f"Expected either 'image_filename' (legacy) or 'img_path' (images) column."
    )


def _parse_img_path_field(raw: str) -> list[str]:
    """Parse the ``img_path`` column which is a stringified Python list.

    Examples::

        "['images/fakta/dfakta_0.jpg']"
        "['images/fakta/kfakta_0_0.jpg', 'images/fakta/kfakta_0_1.png']"
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        # fall back: strip brackets and split on comma
        stripped = raw.strip("[] '\"")
        parsed = [s.strip().strip("'\"") for s in stripped.split(",") if s.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]
    return parsed


@DATASET_REGISTRY.register("dfk_vlm_dataset_v1")
class DFKVLMDatasetV1(BaseDatasetLoader):
    """Load the DFK vision dataset (V1) from a local directory.

    Supports **two CSV formats** (auto-detected from column headers):

    **Legacy format** (``metadata-updt.csv``):
      Columns: ``image_filename``, ``analisis_pelanggaran``, ``label``
      Images live in ``<data_dir>/images/<filename>``

    **Images format** (``images.csv``):
      Columns: ``title``, ``link``, ``text``, ``img_path``, ``label``
      ``img_path`` is a stringified Python list of relative paths, e.g.
      ``['images/fakta/dfakta_0.jpg']``.  Paths are resolved relative to
      ``data_dir``.  If a row has multiple images only the first is used.

    Each row is converted into a multi-turn conversation suitable for
    Unsloth vision fine-tuning::

        [
          {"role": "user",    "content": [{"type":"text", ...}, {"type":"image", ...}]},
          {"role": "assistant","content": [{"type":"text", ...}]},
        ]

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``csv_file`` (str): CSV filename, default ``images.csv``
        - ``instruction`` (str): system instruction for the user prompt
        - ``train_ratio`` (float): fraction of data for training, default 0.9
        - ``seed`` (int): shuffle seed, default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)

    Example YAML::

        dataset:
          name: dfk_vlm_dataset_v1
          kwargs:
            data_dir: dataset
            csv_file: images.csv
            train_ratio: 0.9
    """

    # Row parsers per format
    @staticmethod
    def _parse_row_legacy(row: dict, images_dir: Path) -> dict | None:
        """Parse a row from the legacy CSV format."""
        filename = row.get("image_filename", "").strip()
        label = row.get("label", "").strip()
        analisis = row.get("analisis_pelanggaran", "").strip()

        if not filename or not label:
            return None

        img_path = images_dir / filename
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            return None

        return {
            "image_path": str(img_path),
            "label": label,
            "analisis": analisis,
        }

    @staticmethod
    def _parse_row_images(row: dict, data_dir: Path) -> dict | None:
        """Parse a row from the images CSV format."""
        raw_paths = row.get("img_path", "").strip()
        label = row.get("label", "").strip()
        analisis = row.get("analisis", "").strip() or row.get("text", "").strip()
        title = row.get("title", "").strip()
        text = row.get("text", "").strip()

        if not raw_paths or not label:
            return None

        paths = _parse_img_path_field(raw_paths)
        if not paths:
            return None

        # use first image only (multi-image rows exist but VLM expects one)
        rel_path = paths[0]
        img_path = data_dir / rel_path
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            return None

        return {
            "image_path": str(img_path),
            "label": label,
            "analisis": analisis,
            "title": title,
            "text": text,
        }

    # Main load
    def load(
        self, config: DatasetConfig, tokenizer: Any
    ) -> tuple[list[dict], list[dict] | None]:
        kwargs = dict(config.kwargs)

        data_dir = Path(kwargs.pop("data_dir"))
        csv_file = kwargs.pop("csv_file", "images.csv")
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

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Parse CSV, auto-detect format from header row
        rows: list[dict] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fmt = _detect_csv_format(list(reader.fieldnames or []))
            logger.info("Detected CSV format: %s (%s)", fmt, csv_path.name)

            images_dir = data_dir / "images"  # only needed for legacy

            for raw_row in reader:
                if fmt == _FORMAT_LEGACY:
                    if not images_dir.exists():
                        raise FileNotFoundError(
                            f"Images directory not found: {images_dir}"
                        )
                    parsed = self._parse_row_legacy(raw_row, images_dir)
                else:
                    parsed = self._parse_row_images(raw_row, data_dir)

                if parsed is not None:
                    rows.append(parsed)

        logger.info("Loaded %d valid samples from %s", len(rows), csv_path)

        if max_samples is not None:
            rows = rows[: int(max_samples)]

        # Shuffle deterministically
        rng = random.Random(seed)
        rng.shuffle(rows)

        # Split
        split_idx = int(len(rows) * train_ratio)
        train_rows = rows[:split_idx]
        eval_rows = rows[split_idx:] if split_idx < len(rows) else []

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

                # Build user content: instruction + optional context + image
                user_content = [{"type": "text", "text": instruction}]

                context_parts = []
                if row.get("title"):
                    context_parts.append(f"Judul: {row['title']}")
                if row.get("text"):
                    context_parts.append(f"Konteks: {row['text']}")
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

        # Build HF Datasets
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
                "datasets library not found. Falling back to list-based loading (slow)."
            )
            train_ds = list(gen(train_rows, as_paths=False))
            eval_ds = list(gen(eval_rows, as_paths=False)) if eval_rows else None

        logger.info(
            "Split: %d train, %d eval",
            len(train_ds),
            len(eval_ds) if eval_ds else 0,
        )

        return train_ds, eval_ds
