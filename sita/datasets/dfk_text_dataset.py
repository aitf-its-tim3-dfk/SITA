"""DFK Text Dataset — text-only content analysis (v1 CSV format).

Supports the legacy CSV schema with columns:
    instruction, input, output, _label

This loader parses the v1 (Alpaca-style) CSV into the same conversation
format used by V3, so the two datasets can be used interchangeably for
training.

Parsing rules:
  - ``input`` is split on ``Artikel Rujukan:`` into *klaim* and *fakta*.
  - Only the **first line** of ``output`` is kept.  From that line we
    extract the label and the *penjelasan* (→ ``analisis``).
  - Labels ``non_dfk`` and ``fakta`` are merged into ``netral``.
  - ``ujaran_kebencian`` is normalised to ``ujaran kebencian``.

Label classes (after remapping): netral, disinformasi, ujaran kebencian, fitnah
"""

from __future__ import annotations

import csv
import logging
import random
import re
from pathlib import Path
from typing import Any

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY

logger = logging.getLogger("sita.datasets.dfk_text_dataset")

# ---------------------------------------------------------------------------
# Label whitelist (after remapping)
# ---------------------------------------------------------------------------
VALID_LABELS_MERGED = frozenset(
    {
        "netral",
        "disinformasi",
        "ujaran kebencian",
        "fitnah",
    }
)

VALID_LABELS_RAW = frozenset(
    {
        "non_dfk",
        "fakta",
        "disinformasi",
        "fitnah",
        "ujaran kebencian",
    }
)

# Raw label → canonical label (merged)
_LABEL_MAP: dict[str, str] = {
    "non_dfk": "netral",
    "fakta": "netral",
    "disinformasi": "disinformasi",
    "fitnah": "fitnah",
    "ujaran_kebencian": "ujaran kebencian",
}

# Raw label → canonical label (unmerged / raw)
_LABEL_MAP_RAW: dict[str, str] = {
    "non_dfk": "non_dfk",
    "fakta": "fakta",
    "disinformasi": "disinformasi",
    "fitnah": "fitnah",
    "ujaran_kebencian": "ujaran kebencian",
}

# Output text label → canonical label (for replacing inside the output text)
_OUTPUT_LABEL_MAP: dict[str, str] = {
    "Non-DFK": "Netral",
    "Fakta": "Netral",
}

# No-op map: keep original label text as-is
_OUTPUT_LABEL_MAP_RAW: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Column names in the v1 CSV
# ---------------------------------------------------------------------------
_COL_INSTRUCTION = "instruction"
_COL_INPUT = "input"
_COL_OUTPUT = "output"
_COL_LABEL = "_label"

_REQUIRED_COLUMNS = {_COL_INPUT, _COL_OUTPUT, _COL_LABEL}


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


def _remap_label(raw_label: str, merge: bool = True) -> str:
    """Map a raw label to the canonical label set.

    When *merge* is True (default), collapses to 4 classes.
    When False, keeps all 5 original classes.
    """
    lmap = _LABEL_MAP if merge else _LABEL_MAP_RAW
    return lmap.get(raw_label, raw_label)


def _parse_output_first_line(output: str) -> tuple[str, str]:
    """Extract (label_text, analisis) from the first line of output.

    Expected format:
        ``Label: **Fitnah.** penjelasan: Some explanation text.``
        ``Label: **Non-DFK.**** penjelasan: Some explanation text.``

    Returns (label_text, penjelasan_text).  Falls back gracefully if
    the format doesn't match.
    """
    first_line = output.split("\n")[0].strip()

    # Try to split on " penjelasan:" (case-insensitive)
    parts = re.split(r"\s+penjelasan:\s*", first_line, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        label_part, penjelasan = parts
        # Clean up the label part: "Label: **Fitnah.**" → "Fitnah"
        label_match = re.search(r"\*{2}(.+?)\.?\*{2,4}", label_part)
        if label_match:
            label_text = label_match.group(1).strip().rstrip(".")
        else:
            label_text = label_part.replace("Label:", "").strip().strip("*").strip(".")
        return label_text, penjelasan.strip()

    # Fallback: return whole first line as analisis
    return "", first_line


def _remap_output_label(label_text: str, merge: bool = True) -> str:
    """Remap label text that appears in the output (e.g. 'Non-DFK' → 'Netral').

    When *merge* is False, the original label text is kept as-is.
    """
    omap = _OUTPUT_LABEL_MAP if merge else _OUTPUT_LABEL_MAP_RAW
    return omap.get(label_text, label_text)


def _parse_row(row: dict[str, str], merge_labels: bool = True) -> dict | None:
    """Parse a single CSV row into an internal sample dict.

    Returns None if the row should be skipped (missing data, etc.).
    """
    raw_input = (row.get(_COL_INPUT) or "").strip()
    raw_output = (row.get(_COL_OUTPUT) or "").strip()
    raw_label = (row.get(_COL_LABEL) or "").strip().lower()

    if not raw_input or not raw_output or not raw_label:
        return None

    # ---- Parse input into klaim + fakta ----
    if "Artikel Rujukan:" in raw_input:
        parts = raw_input.split("Artikel Rujukan:", 1)
        klaim = parts[0].strip()
        fakta = parts[1].strip()
    else:
        klaim = raw_input
        fakta = ""

    # ---- Parse output: first line only ----
    output_label_text, analisis = _parse_output_first_line(raw_output)

    if not analisis:
        return None

    # ---- Remap labels ----
    label = _remap_label(raw_label, merge=merge_labels)

    # Also remap the label text that will appear in the answer
    remapped_output_label = _remap_output_label(output_label_text, merge=merge_labels)

    # Validate label
    valid = VALID_LABELS_MERGED if merge_labels else VALID_LABELS_RAW
    if label not in valid:
        logger.warning(
            "Unknown label '%s' (raw: '%s'), expected one of %s, keeping anyway",
            label,
            raw_label,
            valid,
        )

    return {
        "label": label,
        "analisis": analisis,
        "klaim": klaim,
        "fakta": fakta,
        "output_label_text": remapped_output_label,  # for building the answer
    }


@DATASET_REGISTRY.register("dfk_text_dataset")
class DFKTextDataset(BaseDatasetLoader):
    """Load the DFK text dataset from a local directory.

    CSV schema (v1 / Alpaca-style)::

        instruction, input, output, _label

    After parsing, labels are remapped to 4 classes:
        ``netral``, ``disinformasi``, ``ujaran kebencian``, ``fitnah``

    The ``input`` column is split on ``Artikel Rujukan:`` into *klaim* and
    *fakta*.  Only the first line of the ``output`` column is used, parsed
    into a label + *penjelasan* (mapped to *analisis*).

    Each row is converted into a multi-turn conversation::

        [
          {"role": "system",    "content": [{"type":"text", ...}]},
          {"role": "user",      "content": [{"type":"text", ...}]},
          {"role": "assistant", "content": [{"type":"text", ...}]},
        ]

    Config kwargs:
        - ``data_dir`` (str): path to dataset directory (required)
        - ``csv_file`` (str): CSV filename, default ``dfk1test.csv``
        - ``instruction`` (str): system prompt text
        - ``train_ratio`` (float): fraction for training, default 0.8
        - ``seed`` (int): shuffle seed, default 42
        - ``max_samples`` (int | None): cap total samples (for debugging)

    Example YAML::

        dataset:
          name: dfk_text_dataset
          kwargs:
            data_dir: dataset_v1
            csv_file: dfk1test.csv
    """

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(
        self, config: DatasetConfig, tokenizer: Any
    ) -> tuple[list[dict], list[dict] | None]:
        kwargs = dict(config.kwargs)

        data_dir = Path(kwargs.pop("data_dir"))

        _default_instruction_merged = (
            "Anda adalah sistem deteksi konten DFK berbasis artikel rujukan. "
            "Tugas Anda adalah membandingkan klaim dengan artikel rujukan, "
            "lalu mengklasifikasikan teks ke dalam salah satu label: "
            "Netral, Disinformasi, Fitnah, atau Ujaran Kebencian. "
            'Jawab dengan format: Label: **NamaLabel.** penjelasan: ...'
        )
        _default_instruction_raw = (
            "Anda adalah sistem deteksi konten DFK berbasis artikel rujukan. "
            "Tugas Anda adalah membandingkan klaim dengan artikel rujukan, "
            "lalu mengklasifikasikan teks ke dalam salah satu label: "
            "Fakta, Disinformasi, Fitnah, Ujaran Kebencian, atau Non-DFK. "
            'Jawab dengan format: Label: **NamaLabel.** penjelasan: ...'
        )
        instruction = kwargs.pop(
            "instruction",
            _default_instruction_merged if merge_labels else _default_instruction_raw,
        )

        csv_file = kwargs.pop("csv_file", "dfk1test.csv")
        train_ratio = float(kwargs.pop("train_ratio", 0.8))
        seed = int(kwargs.pop("seed", 42))
        max_samples = kwargs.pop("max_samples", None)
        merge_labels = bool(kwargs.pop("merge_labels", True))

        csv_path = data_dir / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        raw_rows = _read_csv(csv_path)
        all_rows = [
            r
            for r in (_parse_row(row, merge_labels=merge_labels) for row in raw_rows)
            if r is not None
        ]

        logger.info("Loaded %d valid samples from %s", len(all_rows), csv_path.name)

        # Log label distribution
        label_counts: dict[str, int] = {}
        for r in all_rows:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        logger.info("Label distribution: %s", label_counts)

        # Deterministic shuffle then split
        rng = random.Random(seed)
        rng.shuffle(all_rows)

        split_idx = int(len(all_rows) * train_ratio)
        train_rows = all_rows[:split_idx]
        eval_rows = all_rows[split_idx:] if split_idx < len(all_rows) else []

        # Optional cap
        if max_samples is not None:
            cap = int(max_samples)
            train_rows = train_rows[:cap]
            eval_rows = eval_rows[:cap]

        # ----- generator -----
        def gen(row_list: list[dict]):
            for row in row_list:
                # Build user content: klaim + fakta
                user_parts = []

                if row.get("klaim"):
                    user_parts.append(row["klaim"])
                if row.get("fakta"):
                    user_parts.append(f"Artikel Rujukan: {row['fakta']}")

                user_text = "\n\n".join(user_parts)

                answer = (
                    f"Label: **{row['output_label_text']}.** "
                    f"penjelasan: {row['analisis']}"
                )

                yield {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": instruction}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_text}],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": answer},
                            ],
                        },
                    ],
                }

        # ----- build HF Datasets -----
        try:
            from datasets import Dataset

            train_ds = Dataset.from_generator(
                gen, gen_kwargs={"row_list": train_rows}
            )

            if eval_rows:
                eval_ds = Dataset.from_generator(
                    gen, gen_kwargs={"row_list": eval_rows}
                )
            else:
                eval_ds = None
        except ImportError:
            logger.warning(
                "datasets library not found. "
                "Falling back to list-based loading (slow)."
            )
            train_ds = list(gen(train_rows))
            eval_ds = list(gen(eval_rows)) if eval_rows else None

        logger.info(
            "Split: %d train, %d eval",
            len(train_ds),
            len(eval_ds) if eval_ds else 0,
        )

        return train_ds, eval_ds
