"""Dynamic runtime-factory loading at the package boundary."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

RuntimeFactory = Callable[..., Any]


def load_runtime_factory(spec: str) -> RuntimeFactory:
    """Load one explicit ``module:function`` runtime factory."""

    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("runtime factory must use module:function syntax")
    module_name, function_name = (part.strip() for part in spec.split(":", 1))
    if not module_name or not function_name:
        raise ValueError("runtime factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError("runtime factory target must be callable")
    return factory
