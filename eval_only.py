"""
Standalone evaluation-only script for SITA VLM adapters.

Run as a subprocess per adapter to get a clean CUDA context each time
(avoids memory leaks from repeated load/unload in the same process).

Usage:
    python eval_only.py \
        --base-model unsloth/Qwen3.5-0.8B \
        --adapter-path /content/adapters/my_adapter \
        --dataset-name dfk_vlm_dataset_v3 \
        --data-dir /content/dataset/images/images \
        --metrics rouge bertscore \
        --output /content/results/my_adapter_metrics.json \
        --chat-template /content/sita/sita/templates/qwen3.5_chatml.jinja \
        --max-new-tokens 512 \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pkgutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("eval_only")


def _import_builtins() -> None:
    """Import all built-in SITA modules so they register themselves."""
    import sita.models
    import sita.adapters
    import sita.datasets
    import sita.evaluators
    import sita.trainers

    for package in [
        sita.models,
        sita.adapters,
        sita.datasets,
        sita.evaluators,
        sita.trainers,
    ]:
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package.__name__}.{module_name}")


def main():
    parser = argparse.ArgumentParser(
        description="SITA eval-only: load adapter → evaluate → save metrics",
    )
    parser.add_argument("--base-model", required=True,
                        help="HF model ID or local path for the base model")
    parser.add_argument("--adapter-path", required=True,
                        help="Path to the adapter directory")
    parser.add_argument("--dataset-name", default="dfk_vlm_dataset_v3",
                        help="Dataset registry key (dfk_vlm_dataset_v2 or v3)")
    parser.add_argument("--data-dir", required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--metrics", nargs="+", default=["bertscore"],
                        help="Metrics to compute: bertscore, rouge, or both")
    parser.add_argument("--output", default="metrics.json",
                        help="Path to save the output metrics JSON")
    parser.add_argument("--chat-template", default=None,
                        help="Path to a Jinja chat template file")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bert-model", default="bert-base-multilingual-cased",
                        help="BERT model for BERTScore (ignored if bertscore not in metrics)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode for generation")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load base model in 4-bit quantization")

    args = parser.parse_args()

    # ---- Register SITA components ----
    _import_builtins()

    from sita.core.registry import (
        DATASET_REGISTRY,
        EVALUATOR_REGISTRY,
        MODEL_REGISTRY,
    )
    from sita.core.config import DatasetConfig, ModelConfig

    # ---- 1. Load base model ----
    logger.info(f"Loading base model: {args.base_model}")

    model_kwargs = {"load_in_4bit": args.load_in_4bit}
    if args.chat_template:
        model_kwargs["chat_template"] = args.chat_template

    model_config = ModelConfig(
        name="unsloth_vlm",
        pretrained=args.base_model,
        kwargs=model_kwargs,
    )
    model_loader = MODEL_REGISTRY.get("unsloth_vlm")()
    model, tokenizer = model_loader.load(model_config)

    # ---- 2. Load adapter ----
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    logger.info(f"Loading adapter from: {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    logger.info("Adapter loaded successfully.")

    # ---- 3. Load dataset (eval split only) ----
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_config = DatasetConfig(
        name=args.dataset_name,
        kwargs={"data_dir": args.data_dir},
    )
    dataset_loader = DATASET_REGISTRY.get(args.dataset_name)()
    _train_ds, eval_ds = dataset_loader.load(dataset_config, tokenizer)

    if eval_ds is None:
        logger.warning("No eval split found, falling back to train split for eval.")
        eval_ds = _train_ds

    # ---- 4. Evaluate ----
    logger.info(f"Running evaluation with metrics: {args.metrics}")
    evaluator = EVALUATOR_REGISTRY.get("vlm_gen")()
    metrics = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_ds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        metrics=args.metrics,
        bert_model=args.bert_model,
        enable_thinking=args.enable_thinking,
    )

    # ---- 5. Save results ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")
    logger.info("═══ Results ═══")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
