"""
SITA Runner — CLI entrypoint that orchestrates the full experiment pipeline.

Usage:
    sita configs/lora_causal_lm.yaml
    python -m sita.runner configs/lora_causal_lm.yaml
"""

from __future__ import annotations

# Unsloth must be imported before transformers/peft to apply optimizations.
# This is a no-op if unsloth is not installed.
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import argparse
import importlib
import json
import logging
import pkgutil
import random
import sys
from pathlib import Path

import numpy as np
import torch

from sita.core.config import load_config, ExperimentConfig
from sita.core.registry import (
    ADAPTER_REGISTRY,
    DATASET_REGISTRY,
    EVALUATOR_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
)

logger = logging.getLogger("sita")


# Auto-discovery of built-in components


def _import_builtins() -> None:
    """Import all built-in modules so they register themselves."""
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


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Pipeline


def run_experiment(config: ExperimentConfig) -> dict:
    """Run a full experiment: load → adapt → train → evaluate."""

    logger.info(f"═══ Experiment: {config.experiment_name} ═══")
    _set_seed(config.seed)

    if config.reporting.wandb:
        try:
            import wandb
        except ImportError:
            logger.error("wandb is not installed. Please install with `pip install sita[wandb]`.")
            sys.exit(1)
            
        from dataclasses import asdict
        run_name = config.reporting.wandb_run_name or config.experiment_name
        wandb.init(
            project=config.reporting.wandb_project,
            entity=config.reporting.wandb_entity,
            name=run_name,
            tags=config.reporting.wandb_tags,
            config={"experiment": asdict(config)},
            **config.reporting.wandb_kwargs,
        )

    # 1. Load model
    logger.info(f"[1/5] Loading model: {config.model.name} ({config.model.pretrained})")
    model_loader = MODEL_REGISTRY.get(config.model.name)()
    model, tokenizer = model_loader.load(config.model)

    # 2. Apply adapter
    logger.info(f"[2/5] Applying adapter: {config.adapter.name}")
    adapter = ADAPTER_REGISTRY.get(config.adapter.name)()
    model = adapter.apply(model, config.adapter)

    if config.adapter.pretrained_adapter:
        logger.info(f"   Warm-starting from pretrained adapter: {config.adapter.pretrained_adapter}")
        model = adapter.load(model, config.adapter.pretrained_adapter)

    param_info = adapter.get_trainable_params(model)
    logger.info(
        f"   Trainable params: {param_info['trainable_params']:,} / "
        f"{param_info['total_params']:,} ({param_info['trainable_pct']:.2f}%)"
    )

    # 3. Load dataset
    logger.info(f"[3/5] Loading dataset: {config.dataset.name}")
    dataset_loader = DATASET_REGISTRY.get(config.dataset.name)()
    train_ds, eval_ds = dataset_loader.load(config.dataset, tokenizer)
    logger.info(f"   Train samples: {len(train_ds)}")
    if eval_ds:
        logger.info(f"   Eval samples: {len(eval_ds)}")

    # 4. Train
    logger.info(f"[4/5] Training with: {config.trainer.name}")
    trainer = TRAINER_REGISTRY.get(config.trainer.name)()
    model = trainer.train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        config=config.training,
        reporting=config.reporting,
        evaluation_config=config.evaluation,
        **config.trainer.kwargs,
    )

    # 5. Evaluate
    logger.info(f"[5/5] Evaluating with: {config.evaluation.name}")
    evaluator = EVALUATOR_REGISTRY.get(config.evaluation.name)()
    metrics = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_ds if eval_ds is not None else train_ds,
        **config.evaluation.kwargs,
    )

    logger.info("═══ Results ═══")
    for k, v in metrics.items():
        logger.info(f"   {k}: {v:.4f}")

    # Save metrics
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"   Metrics saved to {metrics_path}")

    # Save adapter
    adapter.save(model, str(output_dir / "adapter"))
    logger.info(f"   Adapter saved to {output_dir / 'adapter'}")

    if config.reporting.wandb:
        import wandb
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()

    return metrics


# CLI


def main():
    parser = argparse.ArgumentParser(
        description="SITA: Standardized Infrastructure for the Training of Adapters",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        type=str,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--list-registry",
        action="store_true",
        help="List all registered components and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Import all built-ins
    _import_builtins()

    if args.list_registry:
        print("Registered components:")
        print(f"  Models:     {MODEL_REGISTRY.list()}")
        print(f"  Adapters:   {ADAPTER_REGISTRY.list()}")
        print(f"  Datasets:   {DATASET_REGISTRY.list()}")
        print(f"  Evaluators: {EVALUATOR_REGISTRY.list()}")
        print(f"  Trainers:   {TRAINER_REGISTRY.list()}")
        sys.exit(0)

    if args.config is None:
        parser.error("the following arguments are required: config")

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
