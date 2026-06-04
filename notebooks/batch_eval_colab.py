"""
Batch Adapter Evaluation — Colab Notebook Script
=================================================

Copy-paste cells into Colab, or upload this file and run cells via
the `# %%` markers (Colab recognizes them as cell boundaries).

Workflow:
  1. Install deps & clone SITA
  2. Define adapter list (urls, sources, per-adapter config)
  3. Download & extract adapters (gdown for gdrive, aria2c for direct)
  4. Run eval_only.py per adapter as a subprocess (clean CUDA each time)
  5. Aggregate results into a comparison table
"""

# %% [markdown]
# # Batch Adapter Re-evaluation
# Downloads adapters from various sources, evaluates each with configurable
# metrics (ROUGE-L, BERTScore, or both), and aggregates results.

# %% — Setup
# !pip install -q unsloth rouge-score bert-score gdown scikit-learn pandas

# # Clone SITA (skip if already mounted/cloned)
# !git clone https://github.com/YOUR_ORG/SITA.git /content/sita 2>/dev/null || true

# # Install SITA in editable mode
# !pip install -q -e /content/sita

# # Install aria2 for fast downloads from direct links
# !apt-get install -y -qq aria2

# %% — Configuration

import os
from pathlib import Path

# === EDIT THESE ===

SITA_DIR = "/content/sita"
ADAPTERS_DIR = "/content/adapters"
RESULTS_DIR = "/content/results"
DATA_DIR = "/content/dataset/images/images"   # dataset images root
BASE_MODEL = "unsloth/Qwen3.5-0.8B"

# Default eval settings (can be overridden per adapter)
DEFAULT_METRICS = ["rouge", "bertscore"]
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_NEW_TOKENS = 512

# Adapter list — fill in your actual URLs
ADAPTERS = [
    {
        "name": "qwen35-ws3-final",
        "url": "https://drive.google.com/file/d/XXXXXXX/view?usp=sharing",
        "source": "gdrive",
        "dataset": "dfk_vlm_dataset_v3",
        "chat_template": f"{SITA_DIR}/sita/templates/qwen3.5_chatml.jinja",
    },
    # {
    #     "name": "qwen25-ws2-old",
    #     "url": "https://files.catbox.moe/xxxxxx.zip",
    #     "source": "direct",
    #     "dataset": "dfk_vlm_dataset_v2",
    #     "chat_template": f"{SITA_DIR}/sita/templates/some_other.jinja",
    # },
    # Add more adapters here...
]

os.makedirs(ADAPTERS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% — Download & Extract Adapters

import subprocess
import zipfile
import glob
import shutil


def download_gdrive(url: str, output_path: str) -> None:
    """Download a file from Google Drive using gdown."""
    # gdown accepts both file/d/ID/view and full URLs
    subprocess.run(
        ["gdown", "--fuzzy", url, "-O", output_path],
        check=True,
    )


def download_direct(url: str, output_path: str) -> None:
    """Download from a direct URL using aria2c (with wget fallback)."""
    try:
        subprocess.run(
            [
                "aria2c", "-x", "16", "-s", "16",
                "--max-connection-per-server=16",
                "-d", str(Path(output_path).parent),
                "-o", str(Path(output_path).name),
                url,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("aria2c not found, falling back to wget...")
        subprocess.run(
            ["wget", "-q", "-O", output_path, url],
            check=True,
        )


def find_adapter_dir(extract_dir: str) -> str | None:
    """Auto-discover the 'adapter' subfolder regardless of nesting depth."""
    # Look for a directory named 'adapter' anywhere in the tree
    matches = glob.glob(os.path.join(extract_dir, "**", "adapter"), recursive=True)
    # Filter to only directories
    matches = [m for m in matches if os.path.isdir(m)]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  ⚠ Found multiple 'adapter' dirs, using first: {matches[0]}")
    return matches[0]


print("=" * 60)
print("Downloading & extracting adapters...")
print("=" * 60)

adapter_paths = {}

for adapter in ADAPTERS:
    name = adapter["name"]
    url = adapter["url"]
    source = adapter["source"]

    adapter_dir = os.path.join(ADAPTERS_DIR, name)
    zip_path = os.path.join(ADAPTERS_DIR, f"{name}.zip")

    print(f"\n📦 [{name}]")

    # Skip if already extracted
    if os.path.exists(adapter_dir):
        found = find_adapter_dir(adapter_dir)
        if found:
            print(f"  ✓ Already extracted, adapter at: {found}")
            adapter_paths[name] = found
            continue

    # Download
    print(f"  ↓ Downloading from {source}: {url[:80]}...")
    try:
        if source == "gdrive":
            download_gdrive(url, zip_path)
        else:
            download_direct(url, zip_path)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Download failed: {e}")
        continue

    # Extract
    print(f"  📂 Extracting...")
    os.makedirs(adapter_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(adapter_dir)
    except zipfile.BadZipFile:
        print(f"  ✗ Not a valid zip file: {zip_path}")
        continue

    # Find adapter subfolder
    found = find_adapter_dir(adapter_dir)
    if found:
        print(f"  ✓ Adapter found at: {found}")
        adapter_paths[name] = found
    else:
        print(f"  ✗ Could not find 'adapter' subfolder in {adapter_dir}")
        print(f"    Contents: {os.listdir(adapter_dir)}")

    # Clean up zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

print(f"\n{'=' * 60}")
print(f"Ready to evaluate {len(adapter_paths)}/{len(ADAPTERS)} adapters")
print(f"{'=' * 60}")

# %% — Run Evaluation

import json

print("=" * 60)
print("Running evaluations...")
print("=" * 60)

eval_script = os.path.join(SITA_DIR, "eval_only.py")

for adapter in ADAPTERS:
    name = adapter["name"]
    if name not in adapter_paths:
        print(f"\n⏭ [{name}] skipped (no adapter path)")
        continue

    adapter_path = adapter_paths[name]
    dataset_name = adapter.get("dataset", "dfk_vlm_dataset_v3")
    chat_template = adapter.get("chat_template")
    metrics = adapter.get("metrics", DEFAULT_METRICS)
    batch_size = adapter.get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers = adapter.get("num_workers", DEFAULT_NUM_WORKERS)
    max_new_tokens = adapter.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    output_file = os.path.join(RESULTS_DIR, f"{name}_metrics.json")

    print(f"\n🔬 [{name}]")
    print(f"   adapter:  {adapter_path}")
    print(f"   dataset:  {dataset_name}")
    print(f"   metrics:  {metrics}")

    cmd = [
        "python", eval_script,
        "--base-model", BASE_MODEL,
        "--adapter-path", adapter_path,
        "--dataset-name", dataset_name,
        "--data-dir", DATA_DIR,
        "--metrics", *metrics,
        "--output", output_file,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if chat_template:
        cmd.extend(["--chat-template", chat_template])

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"   ✓ Done! Results saved to {output_file}")
    else:
        print(f"   ✗ Failed with return code {result.returncode}")

# %% — Aggregate Results

import pandas as pd

print("\n" + "=" * 60)
print("Aggregating results...")
print("=" * 60)

rows = []
for adapter in ADAPTERS:
    name = adapter["name"]
    metrics_file = os.path.join(RESULTS_DIR, f"{name}_metrics.json")

    if not os.path.exists(metrics_file):
        print(f"⏭ [{name}] no results file found")
        continue

    with open(metrics_file) as f:
        metrics = json.load(f)

    row = {"adapter": name, "dataset": adapter.get("dataset", "?")}
    row.update(metrics)
    rows.append(row)

if rows:
    df = pd.DataFrame(rows)

    # Reorder columns: adapter info first, then key metrics, then the rest
    priority_cols = [
        "adapter", "dataset",
        "rouge_l_f1", "rouge_l_precision", "rouge_l_recall",
        "bertscore_f1", "bertscore_precision", "bertscore_recall",
        "cls_accuracy", "cls_f1_macro", "cls_f1_weighted",
    ]
    ordered = [c for c in priority_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    # Round floats
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)

    print("\n📊 Comparison Table:\n")
    print(df.to_string(index=False))
    print()

    # Save full results
    summary_path = os.path.join(RESULTS_DIR, "comparison.csv")
    df.to_csv(summary_path, index=False)
    print(f"💾 Full comparison saved to {summary_path}")
else:
    print("No results to aggregate :(")
