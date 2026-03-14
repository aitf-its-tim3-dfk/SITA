"""
Generic registry system for SITA components.

Usage:
    from sita.core.registry import MODEL_REGISTRY

    @MODEL_REGISTRY.register("my_model")
    class MyModelLoader(BaseModelLoader):
        ...
"""

from __future__ import annotations

from typing import Any


class Registry:
    """A simple string-keyed registry for component classes.

    Components register themselves via decorator:

        @SOME_REGISTRY.register("key")
        class MyComponent:
            ...

    And are retrieved by name:

        cls = SOME_REGISTRY.get("key")
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Any] = {}

    def register(self, name: str):
        """Decorator to register a class or function under `name`."""

        def decorator(cls):
            if name in self._registry:
                raise ValueError(
                    f"[{self._name}] '{name}' is already registered "
                    f"(existing: {self._registry[name]}, new: {cls})"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Any:
        """Retrieve a registered component by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(f"[{self._name}] '{name}' not found. Available: {available}")
        return self._registry[name]

    def list(self) -> list[str]:
        """Return sorted list of all registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={self.list()})"


# Global registry singletons

MODEL_REGISTRY = Registry("models")
ADAPTER_REGISTRY = Registry("adapters")
DATASET_REGISTRY = Registry("datasets")
EVALUATOR_REGISTRY = Registry("evaluators")
TRAINER_REGISTRY = Registry("trainers")
