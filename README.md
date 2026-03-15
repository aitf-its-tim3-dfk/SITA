# 👸 **SITA: Standardized Infrastructure for the Training of Adapters**

<img src="./sita.svg">

A modular, registry-based testing harness for PEFT methods (LoRA, QLoRA, Prefix Tuning, MoE, etc.) on LLMs and VLMs.

## **Motivation**

To allow for a more transparent and reproducible comparison of PEFT methods, we need a standardized infrastructure for training and evaluating adapters. This need especially came after the recent meeting in Workshop 1 which seemed to indicate that for future reference, DFK-3 has to be able to more comprehensively and transparently justify the choice of PEFT method. Hence, we created SITA.

## **Architecture**

```
Experiment YAML ──▶ Runner (CLI)
                           │
       ┌───────────────────┼───────────────────┬───────────────────┐
       ▼                   ▼                   ▼                   ▼
MODEL_REGISTRY     ADAPTER_REGISTRY    DATASET_REGISTRY   EVALUATOR_REGISTRY
       │                   │                   │                   │
 hf_causal_lm        lora                hf_dataset          loss
 hf_vlm              qlora                                   (custom)
 (custom)            prefix_tuning
                     (custom)
```

Every component registers itself via a decorator, so adding a new adapter is one file with zero changes to existing code.

## Quick Start

```bash
# Install directly from GitHub
pip install "sita[all] @ git+https://github.com/aitf-its-tim3-dfk/SITA.git"

# OR clone and install from source
git clone https://github.com/aitf-its-tim3-dfk/SITA.git
cd SITA
pip install -e ".[all]"

# Run an experiment
sita configs/lora_causal_lm.yaml

# List all registered components
sita --list-registry
```

## **Adding a New Adapter**

Create a file in `sita/adapters/`:

```python
# sita/adapters/my_adapter.py
from sita.core.registry import ADAPTER_REGISTRY
from sita.core.base_adapter import BaseAdapter

@ADAPTER_REGISTRY.register("my_adapter")
class MyAdapter(BaseAdapter):
    def apply(self, model, config):
        # your PEFT logic here
        return model

    def save(self, model, path):
        model.save_pretrained(path)

    def load(self, model, path):
        # load adapter weights
        return model
```

Then use it in config:

```yaml
adapter:
  name: my_adapter
  kwargs:
    my_param: 42
```

That's it.

## **Adding a New Model**

Same pattern, create `sita/models/my_model.py`:

```python
@MODEL_REGISTRY.register("my_model")
class MyModelLoader(BaseModelLoader):
    def load(self, config):
        model = ...
        tokenizer = ...
        return model, tokenizer
```

This works on **both LLMs and VLMs**, as long as the adapter layer operates on `nn.Module`, you can integrate with any models easily.

## **Training Loops**

Two built-in trainers:

| Trainer     | Config Key    | Use Case                       |
| ----------- | ------------- | ------------------------------ |
| HF Trainer  | `hf_trainer`  | Default, batteries-included    |
| Custom Loop | `custom_loop` | Full control over optimization |

The custom loop supports configurable optimizer, scheduler, gradient norm logging, and AMP. See `configs/lora_custom_loop.yaml` for an example.

You can register your own trainer the same way:

```python
@TRAINER_REGISTRY.register("my_trainer")
class MyTrainer(BaseTrainer):
    def train(self, model, tokenizer, train_dataset, eval_dataset, config, **kwargs):
        # your training loop
        return model
```

## **Example Notebooks**

For an interactive tutorial on setting up and running experiments with SITA, check out our getting started notebook:

- [`examples/getting_started.ipynb`](examples/getting_started.ipynb)

## **Example Configs**

| Config                          | Description                       |
| ------------------------------- | --------------------------------- |
| `configs/lora_causal_lm.yaml`   | LoRA on TinyLlama with HF Trainer |
| `configs/qlora_causal_lm.yaml`  | QLoRA (4-bit) on TinyLlama        |
| `configs/lora_custom_loop.yaml` | LoRA with custom training loop    |
| `configs/lora_vlm.yaml`         | LoRA on a VLM (LLaVA)             |

## **Project Structure**

```
sita/
├── core/                   # Abstract base classes & registries
│   ├── registry.py         # Generic Registry class
│   ├── config.py           # Dataclass config schema + YAML loader
│   ├── base_model.py       # BaseModelLoader
│   ├── base_adapter.py     # BaseAdapter
│   ├── base_dataset.py     # BaseDatasetLoader
│   ├── base_evaluator.py   # BaseEvaluator
│   └── base_trainer.py     # BaseTrainer
├── models/                 # Built-in model loaders
│   ├── hf_causal_lm.py     # AutoModelForCausalLM
│   └── hf_vlm.py           # VLM with AutoProcessor
├── adapters/               # Built-in PEFT adapters
│   ├── lora.py
│   ├── qlora.py
│   └── prefix_tuning.py
├── datasets/               # Built-in dataset loaders
│   └── hf_dataset.py
├── evaluators/             # Built-in evaluators
│   └── loss_evaluator.py
├── trainers/               # Built-in training loops
│   ├── hf_trainer.py       # HuggingFace Trainer wrapper
│   └── custom_loop.py      # Bare PyTorch loop
├── runner.py               # CLI entrypoint
configs/                    # Example experiment configs
```
