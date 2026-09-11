"""Source-only import ownership inspection; never import production modules.

This models static import/re-export spelling, not arbitrary Python execution or
transitive runtime reachability. Caches live for one scan only.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path


def within_module(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _module_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Include conditional imports, but do not confuse local names with exports."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _module_scope(child)


class ImportCollector:
    """Resolve imports and selected static re-exports against a physical tree."""

    def __init__(self, package: Path) -> None:
        self.package = package
        self.modules = {
            self._module(path): path for path in sorted(package.rglob("*.py"))
        }
        self._trees: dict[Path, ast.Module] = {}

    def _module(self, path: Path) -> str:
        parts = path.relative_to(self.package.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _tree(self, path: Path) -> ast.Module:
        if path not in self._trees:
            self._trees[path] = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
        return self._trees[path]

    def _base(self, path: Path, node: ast.ImportFrom) -> str:
        module = self._module(path)
        context = module if path.name == "__init__.py" else module.rpartition(".")[0]
        return resolve_name("." * node.level + (node.module or ""), context)

    def _star_names(
        self, path: Path, visited: frozenset[Path] = frozenset()
    ) -> set[str]:
        if path in visited:
            return set()
        visited = visited | {path}
        nodes = list(_module_scope(self._tree(path)))
        declarations = [
            node
            for node in nodes
            if isinstance(node, ast.Name) and node.id == "__all__"
        ]
        if declarations:
            # A literal declaration is deliberately the only supported star
            # contract. Mutation/conditional assembly must not silently hide an edge.
            assignments = [
                node
                for node in self._tree(path).body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                )
            ]
            if len(assignments) == 1 and len(declarations) == 1:
                value = assignments[0].value
                if isinstance(value, (ast.List, ast.Tuple)):
                    exported_names: set[str] = set()
                    for item in value.elts:
                        if not isinstance(item, ast.Constant) or not isinstance(
                            item.value, str
                        ):
                            break
                        exported_names.add(item.value)
                    else:
                        return exported_names
            raise ValueError(f"{path}: star import requires a literal __all__")
        names: set[str] = set()
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if isinstance(node, ast.ImportFrom) and alias.name == "*":
                        source = self.modules.get(self._base(path, node))
                        if source is not None:
                            names.update(self._star_names(source, visited))
                        continue
                    name = alias.asname or (
                        alias.name.split(".")[0]
                        if isinstance(node, ast.Import)
                        else alias.name
                    )
                    if not name.startswith("_"):
                        names.add(name)
        return names

    def _export(
        self, module: str, name: str, visited: frozenset[tuple[str, str]]
    ) -> set[str]:
        key = (module, name)
        if key in visited:
            return set()
        visited = visited | {key}
        result = {module}
        child = f"{module}.{name}"
        if child in self.modules:
            result.add(child)
        path = self.modules.get(module)
        if path is None:
            return result
        if name == "*":
            for exported in self._star_names(path):
                result.update(self._export(module, exported, visited))
            return result
        for node in _module_scope(self._tree(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[0]) == name:
                        result.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = self._base(path, node)
                for alias in node.names:
                    if alias.name == "*":
                        source = self.modules.get(base)
                        if source is not None and name in self._star_names(source):
                            result.update(self._export(base, name, visited))
                    elif (alias.asname or alias.name) == name:
                        result.update(self._export(base, alias.name, visited))
        return result

    def collect_direct(self, path: Path) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                result.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = self._base(path, node)
                result.add(base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    child = f"{base}.{alias.name}"
                    if child in self.modules:
                        result.add(child)
        return result

    def collect(self, path: Path) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                result.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = self._base(path, node)
                result.add(base)
                for alias in node.names:
                    result.update(self._export(base, alias.name, frozenset()))
        return result
