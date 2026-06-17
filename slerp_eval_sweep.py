"""
Multi-Adapter Merge Sweep using local Multislerp.

Loads the base model once to get the tokenizer and optionally evaluate the base model.
Then, for each generated or provided weight configuration, it dynamically merges all PEFT adapters 
using `multislerp` (or `slerp` for 2 adapters) via `slerp_merge.py`, loads the resulting 
PEFT adapter into the base model, evaluates it, and unloads it.

Usage with explicit weights:
    python slerp_eval_sweep.py \
        --base-model aitf-komdigi/KomdigiITS-8B-DFK-CPT \
        --adapters /path/to/adapter1 /path/to/adapter2 /path/to/adapter3 \
        --weights "0.5,0.3,0.2" "0.33,0.33,0.34" \
        --evaluator vlm_gen \
        --dataset-name dfk_vlm_dataset_v3

Usage with grid sweep:
    python slerp_eval_sweep.py \
        --base-model aitf-komdigi/KomdigiITS-8B-DFK-CPT \
        --adapters /path/to/adapter1 /path/to/adapter2 /path/to/adapter3 \
        --grid-step 0.2 \
        --evaluator vlm_gen \
        --dataset-name dfk_vlm_dataset_v3
"""

from __future__ import annotations

import unsloth
import argparse
import gc
import importlib
import json
import logging
import pkgutil
import shutil
import tempfile
from pathlib import Path

import torch

from slerp_merge import merge_adapters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("slerp_eval_sweep")


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


def generate_grid(n_models: int, step: float) -> list[tuple[float, ...]]:
    """Generate all combinations of `n_models` weights that sum to 1.0."""
    steps = int(round(1.0 / step))

    def _generate(n_left, target_sum):
        if n_left == 1:
            yield (target_sum,)
            return
        for i in range(target_sum + 1):
            for tail in _generate(n_left - 1, target_sum - i):
                yield (i,) + tail

    grid = []
    for int_tuple in _generate(n_models, steps):
        grid.append(tuple(round(x / steps, 4) for x in int_tuple))
    return grid


def _run_all_evals(model, tokenizer, eval_datasets, evaluators, eval_kwargs):
    """Run all evaluators and return merged metrics dict with prefixed keys."""
    all_metrics: dict[str, float] = {}
    single = len(evaluators) == 1

    for eval_name, evaluator, eval_ds in zip(
        [e[0] for e in evaluators],
        [e[1] for e in evaluators],
        eval_datasets,
    ):
        logger.info(f"  Running evaluator: {eval_name}")
        metrics = evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_ds,
            **eval_kwargs,
        )
        if single:
            all_metrics.update(metrics)
        else:
            for k, v in metrics.items():
                all_metrics[f"{eval_name}/{k}"] = v
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-adapter merge sweep")
    parser.add_argument(
        "--base-model", type=str, required=True, help="Base model path/name"
    )
    parser.add_argument(
        "--adapters", nargs="+", required=True, help="List of adapter paths"
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        help="List of comma-separated weights, e.g. '0.5,0.5' '0.3,0.7'",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        help="Step size for auto-generating weight grid (e.g. 0.2)",
    )

    parser.add_argument(
        "--evaluator", nargs="+", required=True, help="List of evaluator names"
    )
    parser.add_argument(
        "--dataset-name",
        nargs="+",
        required=True,
        help="List of dataset names (1:1 with evaluators)",
    )
    parser.add_argument("--data-dir", type=str, default="", help="Base data directory")
    parser.add_argument(
        "--data-dir-override", nargs="*", default=[], help="Format: eval_name=/path"
    )
    parser.add_argument(
        "--val-file-override", nargs="*", default=[], help="Format: eval_name=filename"
    )

    parser.add_argument("--metrics", nargs="+", default=["bertscore", "rouge"])
    parser.add_argument(
        "--eval-base",
        action="store_true",
        help="Eval base model before adding adapters",
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="merge_sweep_results.json")
    parser.add_argument(
        "--save-adapters-dir",
        type=str,
        default=None,
        help="Directory to save merged adapters permanently",
    )

    # Model kwargs
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--chat-template", type=str, default=None)

    # Eval kwargs
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--bert-model", type=str, default="bert-base-multilingual-cased"
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument(
        "--no-merge-labels",
        action="store_true",
        help="Keep original 5-class labels instead of merging to 4",
    )

    args = parser.parse_args()
    _import_builtins()

    from sita.core.registry import DATASET_REGISTRY, EVALUATOR_REGISTRY, MODEL_REGISTRY
    from sita.core.config import DatasetConfig, ModelConfig

    eval_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "metrics": args.metrics,
        "bert_model": args.bert_model,
        "enable_thinking": args.enable_thinking,
        "max_image_size": args.max_image_size,
        "normalize_labels": not args.no_merge_labels,
    }

    # ---- Setup Weight Sets ----
    weight_sets = []
    n_adapters = len(args.adapters)

    if args.weights:
        for w_str in args.weights:
            w_tuple = tuple(float(x.strip()) for x in w_str.split(","))
            if len(w_tuple) != n_adapters:
                parser.error(
                    f"Weight vector '{w_str}' length does not match number of adapters ({n_adapters})"
                )
            weight_sets.append(w_tuple)
    elif args.grid_step:
        weight_sets = generate_grid(n_adapters, args.grid_step)
        logger.info(
            f"Generated {len(weight_sets)} weight combinations using step {args.grid_step}"
        )
    else:
        parser.error("Must provide either --weights or --grid-step")

    # ---- Parse overrides ----
    data_dir_overrides: dict[str, str] = {}
    for override in args.data_dir_override:
        if "=" not in override:
            parser.error(f"--data-dir-override must be 'name=/path', got: {override!r}")
        name, path = override.split("=", 1)
        data_dir_overrides[name.strip()] = path.strip()

    val_file_overrides: dict[str, str] = {}
    for override in args.val_file_override:
        if "=" not in override:
            parser.error(
                f"--val-file-override must be 'name=filename', got: {override!r}"
            )
        name, filename = override.split("=", 1)
        val_file_overrides[name.strip()] = filename.strip()

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
    base_model, tokenizer = model_loader.load(model_config)

    # ---- 2. Load Evaluators & Datasets ----
    eval_names = args.evaluator
    ds_names = args.dataset_name

    if len(ds_names) == 1 and len(eval_names) > 1:
        ds_names = ds_names * len(eval_names)
    elif len(ds_names) != len(eval_names):
        parser.error(
            f"--dataset-name must have 1 entry (shared) or match --evaluator count "
            f"({len(eval_names)}), got {len(ds_names)}"
        )

    eval_datasets = []
    evaluators = []
    for eval_name, ds_name in zip(eval_names, ds_names):
        data_dir = data_dir_overrides.get(eval_name, args.data_dir)
        logger.info(
            f"Loading dataset '{ds_name}' for evaluator '{eval_name}' from {data_dir}"
        )
        ds_kwargs = {"data_dir": data_dir}
        if eval_name in val_file_overrides:
            ds_kwargs["val_file"] = val_file_overrides[eval_name]
            logger.info(f"  Overriding val_file -> {val_file_overrides[eval_name]}")
        if args.no_merge_labels:
            ds_kwargs["merge_labels"] = False
        dataset_config = DatasetConfig(
            name=ds_name,
            kwargs=ds_kwargs,
        )
        dataset_loader = DATASET_REGISTRY.get(ds_name)()
        _train_ds, eval_ds = dataset_loader.load(dataset_config, tokenizer)

        if eval_ds is None:
            logger.warning(f"No eval split for {ds_name}, falling back to train.")
            eval_ds = _train_ds

        if args.max_eval_samples is not None and len(eval_ds) > args.max_eval_samples:
            logger.info(
                f"Capping {ds_name} eval from {len(eval_ds)} to {args.max_eval_samples} samples."
            )
            eval_ds = eval_ds.select(range(args.max_eval_samples))

        eval_datasets.append(eval_ds)
        evaluators.append((eval_name, EVALUATOR_REGISTRY.get(eval_name)()))

    results = {}

    # ---- 3. Base Model Eval ----
    if args.eval_base:
        logger.info("Evaluating base model")
        results["base"] = {
            "description": "Base model, no adapters",
            "metrics": _run_all_evals(
                base_model, tokenizer, eval_datasets, evaluators, eval_kwargs
            ),
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # ---- 4. Merge Sweep with local script ----
    import peft

    model = base_model
    is_peft = False

    if args.save_adapters_dir:
        tmp_dir = args.save_adapters_dir
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        is_temp_dir = False
    else:
        tmp_dir = tempfile.mkdtemp(prefix="merge_sweep_")
        is_temp_dir = True

    try:
        for w_tuple in weight_sets:
            w_str = ",".join(f"{w:.2f}" for w in w_tuple)
            config_name = f"w=[{w_str}]"
            logger.info(
                f"========== Testing Merge configuration: {config_name} =========="
            )

            adapter_path = Path(tmp_dir) / f"merged_{w_str.replace(',','_')}"

            logger.info(f"Running multislerp for weights {w_str}...")
            merge_adapters(args.adapters, str(adapter_path), w_tuple)

            logger.info("Loading merged adapter into PEFT...")
            if not is_peft:
                model = peft.PeftModel.from_pretrained(
                    base_model, str(adapter_path), adapter_name="slerp"
                )
                is_peft = True
            else:
                model.load_adapter(str(adapter_path), adapter_name="slerp")

            model.set_adapter("slerp")

            metrics = _run_all_evals(
                model, tokenizer, eval_datasets, evaluators, eval_kwargs
            )
            results[config_name] = {
                "description": f"{'slerp' if n_adapters == 2 else 'multislerp'} weights=[{w_str}]",
                "weights": w_tuple,
                "metrics": metrics,
            }

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

            logger.info("Unloading merged adapter...")
            model.delete_adapter("slerp")

            # Delete merged adapter to save disk space if not saving
            if not args.save_adapters_dir:
                shutil.rmtree(adapter_path, ignore_errors=True)

    finally:
        if is_temp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Sweep complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
