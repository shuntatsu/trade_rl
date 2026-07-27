"""Side-effect-free Python import reference extraction for architecture tests."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path
from typing import Literal

ImportKind = Literal["import", "from", "dynamic"]
_DynamicImportKind = Literal["importlib", "builtin"]


@dataclass(frozen=True, slots=True)
class ImportReference:
    """One statically observable Python import dependency."""

    path: Path
    line: int
    kind: ImportKind
    target: str | None
    unresolved: bool = False


def module_name_from_path(
    path: Path,
    *,
    package_root: Path,
    root_package: str,
) -> str:
    """Resolve a Python module name beneath a declared package root."""

    relative = path.relative_to(package_root)
    if relative.suffix != ".py":
        raise ValueError(f"Python source path must end with .py: {path}")
    if relative.name == "__init__.py":
        parts = relative.parent.parts
    else:
        parts = relative.with_suffix("").parts
    return ".".join((root_package, *parts))


class _ImportReferenceVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path, module_name: str) -> None:
        self.path = path
        self.module_name = module_name
        self.package_name = (
            module_name
            if path.name == "__init__.py"
            else module_name.rpartition(".")[0]
        )
        self.references: list[ImportReference] = []
        self.importlib_modules: set[str] = {"importlib"}
        self.builtins_modules: set[str] = {"builtins"}
        self.importlib_functions: set[str] = set()
        self.builtin_functions: set[str] = {"__import__"}

    def _append(
        self,
        node: ast.AST,
        *,
        kind: ImportKind,
        target: str | None,
        unresolved: bool = False,
    ) -> None:
        self.references.append(
            ImportReference(
                path=self.path,
                line=int(getattr(node, "lineno", 0)),
                kind=kind,
                target=target,
                unresolved=unresolved,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._append(node, kind="import", target=alias.name)
            local_name = alias.asname or alias.name.partition(".")[0]
            if alias.name == "importlib":
                self.importlib_modules.add(local_name)
            elif alias.name == "builtins":
                self.builtins_modules.add(local_name)
        self.generic_visit(node)

    def _resolve_from_base(self, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        if not self.package_name:
            return None
        relative_name = f"{'.' * node.level}{node.module or ''}"
        try:
            return resolve_name(relative_name, self.package_name)
        except (ImportError, ValueError):
            return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = self._resolve_from_base(node)
        if base is None:
            self._append(node, kind="from", target=None, unresolved=True)
            self.generic_visit(node)
            return
        for alias in node.names:
            target = base if alias.name == "*" else f"{base}.{alias.name}"
            self._append(node, kind="from", target=target)
            local_name = alias.asname or alias.name
            if base == "importlib" and alias.name == "import_module":
                self.importlib_functions.add(local_name)
            elif base == "builtins" and alias.name == "__import__":
                self.builtin_functions.add(local_name)
        self.generic_visit(node)

    def _dynamic_import_kind(self, function: ast.expr) -> _DynamicImportKind | None:
        if isinstance(function, ast.Name):
            if function.id in self.importlib_functions:
                return "importlib"
            if function.id in self.builtin_functions:
                return "builtin"
            return None
        if not isinstance(function, ast.Attribute) or not isinstance(
            function.value, ast.Name
        ):
            return None
        if (
            function.attr == "import_module"
            and function.value.id in self.importlib_modules
        ):
            return "importlib"
        if function.attr == "__import__" and function.value.id in self.builtins_modules:
            return "builtin"
        return None

    @staticmethod
    def _literal_string(node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    @staticmethod
    def _package_argument(node: ast.Call) -> ast.expr | None:
        if len(node.args) >= 2:
            return node.args[1]
        for keyword in node.keywords:
            if keyword.arg == "package":
                return keyword.value
        return None

    def visit_Call(self, node: ast.Call) -> None:
        dynamic_kind = self._dynamic_import_kind(node.func)
        if dynamic_kind is None:
            self.generic_visit(node)
            return
        target = self._literal_string(node.args[0] if node.args else None)
        if target is None:
            self._append(node, kind="dynamic", target=None, unresolved=True)
            self.generic_visit(node)
            return
        if target.startswith("."):
            if dynamic_kind != "importlib":
                self._append(node, kind="dynamic", target=None, unresolved=True)
                self.generic_visit(node)
                return
            package = self._literal_string(self._package_argument(node))
            if package is None:
                self._append(node, kind="dynamic", target=None, unresolved=True)
                self.generic_visit(node)
                return
            try:
                target = resolve_name(target, package)
            except (ImportError, ValueError):
                self._append(node, kind="dynamic", target=None, unresolved=True)
                self.generic_visit(node)
                return
        self._append(node, kind="dynamic", target=target)
        self.generic_visit(node)


def scan_import_references(
    path: Path,
    *,
    module_name: str,
) -> tuple[ImportReference, ...]:
    """Parse one Python source file and return imports in source order."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _ImportReferenceVisitor(path=path, module_name=module_name)
    visitor.visit(tree)
    return tuple(visitor.references)


def _is_prohibited_target(target: str, *, prohibited_prefix: str) -> bool:
    return target == prohibited_prefix or target.startswith(f"{prohibited_prefix}.")


def causal_scenario_dependency_violations(
    *,
    protected_roots: tuple[Path, ...],
    excluded_root: Path,
    package_root: Path,
    root_package: str,
    prohibited_prefix: str,
) -> tuple[str, ...]:
    """Return stable violations from protected Python dependency roots."""

    violations: list[str] = []
    for root in protected_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.is_relative_to(excluded_root):
                continue
            module_name = module_name_from_path(
                path,
                package_root=package_root,
                root_package=root_package,
            )
            for reference in scan_import_references(path, module_name=module_name):
                if reference.kind == "dynamic" and reference.unresolved:
                    violations.append(f"{path}:{reference.line}:dynamic:<unresolved>")
                    continue
                if reference.target is not None and _is_prohibited_target(
                    reference.target,
                    prohibited_prefix=prohibited_prefix,
                ):
                    violations.append(
                        f"{path}:{reference.line}:{reference.kind}:{reference.target}"
                    )
    return tuple(sorted(violations))


def forbidden_json_key_paths(payload: object, *, key: str) -> tuple[str, ...]:
    """Return JSON paths containing an exact forbidden mapping key."""

    if not key:
        raise ValueError("key must be non-empty")
    violations: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            if key in value:
                violations.append(f"{path}.{key}")
            for child_key in sorted(value, key=str):
                visit(value[child_key], f"{path}.{child_key}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "$")
    return tuple(violations)


__all__ = [
    "ImportReference",
    "causal_scenario_dependency_violations",
    "forbidden_json_key_paths",
    "module_name_from_path",
    "scan_import_references",
]
