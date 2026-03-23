"""Generic HuggingFace datasets loader."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from sita.core.base_dataset import BaseDatasetLoader
from sita.core.config import DatasetConfig
from sita.core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("hf_dataset")
class HFDatasetLoader(BaseDatasetLoader):
    """Load any dataset from HuggingFace Hub or local path.

    Config kwargs:
        - `path` (str): HF dataset name or local path (required)
        - `name` (str): dataset config name (optional)
        - `split_train` (str): train split name, default "train"
        - `split_eval` (str): eval split name, default "test" (set to null to skip)
        - `text_field` (str): field to use as text input, default "text"
        - `max_length` (int): max tokenization length, default 512
        - `streaming` (bool): whether to use streaming mode, default False

    Example YAML::

        dataset:
          name: hf_dataset
          kwargs:
            path: tatsu-lab/alpaca
            text_field: text
            max_length: 1024
    """

    def load(self, config: DatasetConfig, tokenizer: Any) -> tuple[Any, Any | None]:
        kwargs = dict(config.kwargs)

        path = kwargs.pop("path")
        ds_name = kwargs.pop("name", None)
        split_train = kwargs.pop("split_train", "train")
        split_eval = kwargs.pop("split_eval", "test")
        text_field = kwargs.pop("text_field", "text")
        max_length = kwargs.pop("max_length", 512)
        streaming = kwargs.pop("streaming", False)
        skip_tokenization = kwargs.pop("skip_tokenization", False)

        # load raw dataset
        load_kwargs = {"path": path, "streaming": streaming}
        if ds_name:
            load_kwargs["name"] = ds_name
        load_kwargs.update(kwargs)  # pass remaining kwargs through

        train_ds = load_dataset(**load_kwargs, split=split_train)

        eval_ds = None
        if split_eval:
            try:
                eval_ds = load_dataset(**load_kwargs, split=split_eval)
            except ValueError:
                # split doesn't exist, that's fine
                pass

        if skip_tokenization:
            return train_ds, eval_ds

        # tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
        if eval_ds is not None:
            eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

        return train_ds, eval_ds
