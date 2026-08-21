"""Compatibility facade for the integrations-owned runtime factory."""

from trade_rl.integrations.runtime_factory import (
    RuntimeFactory,
    RuntimeFactoryDescriptor,
    describe_runtime_factory,
    load_runtime_factory,
)

__all__ = [
    "RuntimeFactory",
    "RuntimeFactoryDescriptor",
    "describe_runtime_factory",
    "load_runtime_factory",
]
