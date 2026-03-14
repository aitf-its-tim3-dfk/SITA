"""Abstract base class for PEFT adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from sita.core.config import AdapterConfig


class BaseAdapter(ABC):
    """Base class for PEFT adapter methods.

    An adapter wraps/modifies a base model to add parameter-efficient
    trainable parameters (LoRA, QLoRA, Prefix Tuning, Adapters, MoE, etc.)

    To register a new adapter::

        from sita.core.registry import ADAPTER_REGISTRY

        @ADAPTER_REGISTRY.register("my_adapter")
        class MyAdapter(BaseAdapter):
            def apply(self, model, config):
                # inject your adapter layers
                return modified_model
            ...
    """

    @abstractmethod
    def apply(self, model: nn.Module, config: AdapterConfig) -> nn.Module:
        """Apply the PEFT method to a base model.

        Args:
            model: The base model (could be LLM or VLM — it's just nn.Module).
            config: Adapter-specific config with method kwargs.

        Returns:
            The modified model with adapter layers injected.
        """
        ...

    @abstractmethod
    def save(self, model: nn.Module, path: str) -> None:
        """Save only the adapter weights.

        Args:
            model: The adapted model.
            path: Directory to save adapter weights to.
        """
        ...

    @abstractmethod
    def load(self, model: nn.Module, path: str) -> nn.Module:
        """Load adapter weights into a base model.

        Args:
            model: The base model (without adapter).
            path: Directory containing saved adapter weights.

        Returns:
            Model with adapter weights loaded.
        """
        ...

    def get_trainable_params(self, model: nn.Module) -> dict:
        """Return a summary of trainable vs total parameters.

        This default implementation works for most cases.
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total if total > 0 else 0,
        }
