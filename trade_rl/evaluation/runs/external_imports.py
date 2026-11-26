"""Opt-in observations of loaded regular-package import origins.

This is for a trusted fresh Python process, not an execution attestation or
a sandbox. Pair it with the independently pinned wheel/source verifier.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec, SourceFileLoader
from types import ModuleType

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.external_provenance import (
    ExternalPackageInput,
    _check_digest,
    _verify_installed,
)

_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


@dataclass(frozen=True, slots=True)
class LoadedPackageImports:
    """Path-independent observation of current module origins and reviewed bytes."""

    package_pin_digest: str
    loaded_modules: tuple[tuple[str, str], ...]
    schema_version: str = field(default="loaded_package_imports_v1", init=False)

    @property
    def digest(self) -> str:
        return content_digest(self)


def _observe(source: ExternalPackageInput) -> tuple[tuple[str, str], ...]:
    root = source.installed_package
    package = source.pin.package
    paths = {entry.path for entry in source.pin.files}
    result: list[tuple[str, str]] = []
    for name, module in sys.modules.copy().items():
        if type(name) is not str or not (
            name == package or name.startswith(package + ".")
        ):
            continue
        if not _MODULE.fullmatch(name) or type(module) is not ModuleType:
            raise ValueError("invalid loaded package module")
        tail = name[len(package) :].lstrip(".").replace(".", "/")
        if not tail:
            relative = "__init__.py"
        else:
            candidates = [
                item for item in (tail + ".py", tail + "/__init__.py") if item in paths
            ]
            if len(candidates) != 1:
                raise ValueError("loaded module does not have one pinned source")
            relative = candidates[0]
        if relative not in paths:
            raise ValueError("loaded module is outside the pinned source roster")
        expected = str(root / relative)
        is_package = relative.endswith("__init__.py") and (
            relative == "__init__.py" or relative.endswith("/__init__.py")
        )
        attrs = vars(module)
        spec = attrs.get("__spec__")
        if type(spec) is not ModuleSpec or type(spec.loader) is not SourceFileLoader:
            raise ValueError("loaded module is not a regular source import")
        loader = spec.loader
        if (
            spec.name != name
            or spec.origin != expected
            or loader.name != name
            or loader.path != expected
            or attrs.get("__name__") != name
            or attrs.get("__file__") != expected
            or attrs.get("__loader__") is not loader
            or attrs.get("__package__")
            != (name if is_package else name.rpartition(".")[0])
        ):
            raise ValueError("loaded module origin differs from the pinned package")
        if is_package:
            expected_path = [str((root / relative).parent)]
            if (
                type(spec.submodule_search_locations) is not list
                or spec.submodule_search_locations != expected_path
                or type(attrs.get("__path__")) is not list
                or attrs["__path__"] != expected_path
            ):
                raise ValueError("loaded package search path differs from its pin")
        elif spec.submodule_search_locations is not None or "__path__" in attrs:
            raise ValueError("loaded non-package module has package search paths")
        result.append((name, relative))
    return tuple(sorted(result))


def verify_loaded_package_imports(
    source: ExternalPackageInput,
    *,
    expected_pin_digest: str,
    required_modules: tuple[str, ...],
    expected_observation_digest: str | None = None,
) -> LoadedPackageImports:
    """Check present module origin/loader/search paths plus all pinned source bytes.

    Call in a trusted fresh noneditable installation with Python -I -B, after
    importing explicitly required entry points and after independently verifying
    the wheel via verify_external_source_provenance. All loaded modules below
    this package must use their pinned regular source paths. The caller may
    compare an observation at the same import boundary on subsequent checks.

    This does not import modules, prove which code objects executed, attest
    dependency files, detect transient mutation or authenticate a malicious
    Python interpreter or in-process code. No paths or authorization are emitted.
    """
    if type(source) is not ExternalPackageInput:
        raise ValueError("expected an external package input")
    _check_digest(expected_pin_digest, name="package pin digest")
    if expected_pin_digest != content_digest(source.pin):
        raise ValueError("package pin digest mismatch")
    if expected_observation_digest is not None:
        _check_digest(expected_observation_digest, name="import observation digest")
    if not sys.dont_write_bytecode:
        raise ValueError("import observation requires no bytecode writes (Python -B)")
    package = source.pin.package
    if (
        type(required_modules) is not tuple
        or not required_modules
        or any(
            type(name) is not str
            or not _MODULE.fullmatch(name)
            or not (name == package or name.startswith(package + "."))
            for name in required_modules
        )
        or tuple(sorted(set(required_modules))) != required_modules
        or package not in required_modules
    ):
        raise ValueError(
            "expected a sorted required module roster including the package"
        )
    before = _observe(source)
    if not set(required_modules).issubset(name for name, _ in before):
        raise ValueError("required package modules have not been loaded")
    _verify_installed(source)
    if _observe(source) != before:
        raise ValueError("loaded module roster changed during source verification")
    result = LoadedPackageImports(expected_pin_digest, before)
    if (
        expected_observation_digest is not None
        and result.digest != expected_observation_digest
    ):
        raise ValueError("import observation digest mismatch")
    return result


__all__ = ["LoadedPackageImports", "verify_loaded_package_imports"]
