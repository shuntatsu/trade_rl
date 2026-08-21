"""Dynamic runtime-factory loading with source-bound evidence."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

RuntimeFactory = Callable[..., Any]
_DESCRIPTOR_SCHEMA = "runtime_factory_descriptor_v1"


def _factory_parts(spec: str) -> tuple[str, str]:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("runtime factory must use module:function syntax")
    module_name, function_name = (part.strip() for part in spec.split(":", 1))
    if not module_name or not function_name:
        raise ValueError("runtime factory must use module:function syntax")
    return module_name, function_name


@dataclass(frozen=True, slots=True)
class RuntimeFactoryDescriptor:
    spec: str
    module: str
    callable_name: str
    implementation_digest: str
    schema_version: str = _DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        module_name, callable_name = _factory_parts(self.spec)
        if self.module != module_name or self.callable_name != callable_name:
            raise ValueError("runtime factory descriptor spec mismatch")
        require_sha256(
            self.implementation_digest,
            field="runtime factory implementation_digest",
        )
        if self.schema_version != _DESCRIPTOR_SCHEMA:
            raise ValueError("runtime factory descriptor schema mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "callable_name": self.callable_name,
            "implementation_digest": self.implementation_digest,
            "module": self.module,
            "schema_version": self.schema_version,
            "spec": self.spec,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self.digest_payload(), "descriptor_digest": self.digest}


def load_runtime_factory(spec: str) -> RuntimeFactory:
    """Load one explicit ``module:function`` runtime factory."""

    module_name, function_name = _factory_parts(spec)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError("runtime factory target must be callable")
    return factory


def describe_runtime_factory(
    spec: str,
    *,
    factory: RuntimeFactory | None = None,
) -> RuntimeFactoryDescriptor:
    """Bind one runtime factory to the exact source file that implements it."""

    module_name, function_name = _factory_parts(spec)
    resolved = factory or load_runtime_factory(spec)
    source_name = inspect.getsourcefile(resolved) or inspect.getfile(resolved)
    source_path = Path(source_name)
    if not source_path.is_file():
        raise ValueError("runtime factory implementation source is unavailable")
    implementation_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return RuntimeFactoryDescriptor(
        spec=f"{module_name}:{function_name}",
        module=module_name,
        callable_name=function_name,
        implementation_digest=implementation_digest,
    )


__all__ = [
    "RuntimeFactory",
    "RuntimeFactoryDescriptor",
    "describe_runtime_factory",
    "load_runtime_factory",
]
