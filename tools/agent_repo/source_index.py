"""Source-only repository inspection; never import production modules.

This models static import/re-export spelling and local source signals, not
arbitrary Python execution or complete runtime reachability. Caches live for
one scan only and no generated index is persisted.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

FILESYSTEM_EFFECTS = frozenset(
    {"write_text", "write_bytes", "replace", "rename", "unlink", "mkdir", "rmdir"}
)
NETWORK_EFFECTS = frozenset({"urlopen"})


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


@dataclass(frozen=True, slots=True)
class PathContext:
    """Deterministic source-derived context for one repository path."""

    path: str
    module: str | None
    domain: str | None
    capability: str | None
    facade_module: str | None
    direct_imports: tuple[str, ...]
    semantic_imports: tuple[str, ...]
    direct_consumers: tuple[str, ...]
    semantic_consumers: tuple[str, ...]
    public_exports: tuple[str, ...]
    schema_constants: tuple[str, ...]
    test_candidates: tuple[str, ...]
    doc_references: tuple[str, ...]
    effect_signals: tuple[str, ...]


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_public_exports(path: Path) -> tuple[str, ...] | None:
    tree = _parse_module(path)
    assignments: list[ast.Assign | ast.AnnAssign] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            assignments.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            assignments.append(node)
    if not assignments:
        return None
    if len(assignments) != 1:
        return None
    value = assignments[0].value
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None
    names: list[str] = []
    for item in value.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        names.append(item.value)
    return tuple(sorted(names))


def _schema_constants(path: Path) -> tuple[str, ...]:
    result: set[str] = set()
    for node in _parse_module(path).body:
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            if node.value is None:
                continue
            targets = [node.target]
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and "SCHEMA" in target.id:
                result.add(target.id)
    return tuple(sorted(result))


def _effect_signals(path: Path) -> tuple[str, ...]:
    result: set[str] = set()
    for node in ast.walk(_parse_module(path)):
        if not isinstance(node, ast.Call):
            continue
        name: str | None = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FILESYSTEM_EFFECTS:
            result.add(f"filesystem:{name}")
        if name in NETWORK_EFFECTS:
            result.add(f"network:{name}")
    return tuple(sorted(result))


class SourceIndex:
    """One-scan in-memory index over current production source and nearby evidence."""

    def __init__(self, repository: Path, collector: ImportCollector) -> None:
        self.repository = repository
        self.package = repository / "trade_rl"
        self.collector = collector
        self._module_by_path = {
            path: module for module, path in collector.modules.items()
        }
        self._production_paths = tuple(sorted(collector.modules.values()))
        self._direct = {
            path: frozenset(collector.collect_direct(path))
            for path in self._production_paths
        }
        self._semantic = {
            path: frozenset(collector.collect(path)) for path in self._production_paths
        }

    @classmethod
    def build(cls, repository: Path) -> SourceIndex:
        root = repository.resolve()
        package = root / "trade_rl"
        if not package.is_dir():
            raise ValueError("repository must contain trade_rl package")
        return cls(root, ImportCollector(package))

    def _logical_source(self, relative_path: str) -> tuple[str, Path]:
        logical = PurePosixPath(relative_path)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise ValueError(f"unsafe repository path: {relative_path}")
        normalized = logical.as_posix().rstrip("/")
        candidate = self.repository.joinpath(*logical.parts)
        if candidate.is_dir():
            source = candidate / "__init__.py"
            if not source.is_file():
                raise FileNotFoundError(relative_path)
            return normalized, source
        if not candidate.is_file():
            raise FileNotFoundError(relative_path)
        return normalized, candidate

    def _domain_capability(self, relative_path: str) -> tuple[str | None, str | None]:
        parts = PurePosixPath(relative_path).parts
        if len(parts) < 2 or parts[0] != "trade_rl":
            return None, None
        domain = parts[1]
        if len(parts) < 3 or parts[2].endswith(".py"):
            return domain, None
        return domain, parts[2]

    def _facade(self, source: Path) -> tuple[str | None, tuple[str, ...]]:
        directory = source.parent
        while directory == self.package or directory.is_relative_to(self.package):
            init = directory / "__init__.py"
            if init.is_file():
                exports = _literal_public_exports(init)
                module = self._module_by_path.get(init)
                if exports and module is not None:
                    return module, exports
            if directory == self.package:
                break
            directory = directory.parent
        return None, ()

    def _consumers(
        self,
        module: str | None,
        *,
        semantic: bool,
        exclude_module: str | None,
    ) -> tuple[str, ...]:
        if module is None:
            return ()
        imports_by_path = self._semantic if semantic else self._direct
        result = [
            path.relative_to(self.repository).as_posix()
            for path, imports in imports_by_path.items()
            if self._module_by_path.get(path) not in {module, exclude_module}
            and module in imports
        ]
        return tuple(sorted(result))

    def _test_candidates(
        self,
        *,
        module: str | None,
        facade_module: str | None,
        domain: str | None,
        capability: str | None,
    ) -> tuple[str, ...]:
        tests_root = self.repository / "tests"
        if not tests_root.is_dir():
            return ()
        result: set[str] = set()
        for path in sorted(tests_root.rglob("test_*.py")):
            relative = path.relative_to(self.repository).as_posix()
            imports = self.collector.collect(path)
            if module is not None and module in imports:
                result.add(relative)
                continue
            if facade_module is not None and facade_module in imports:
                result.add(relative)
                continue
            parts = PurePosixPath(relative).parts
            if domain is not None and len(parts) >= 2 and parts[1] == domain:
                if (
                    capability is not None
                    and len(parts) >= 3
                    and parts[2] == capability
                ):
                    result.add(relative)
        return tuple(sorted(result))

    def _doc_references(
        self, relative_path: str, module: str | None
    ) -> tuple[str, ...]:
        docs_root = self.repository / "docs"
        if not docs_root.is_dir():
            return ()
        terms = {relative_path}
        if module is not None:
            terms.add(module)
        result: list[str] = []
        for path in sorted(docs_root.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            if any(term in text for term in terms):
                result.append(path.relative_to(self.repository).as_posix())
        return tuple(result)

    def context(self, relative_path: str) -> PathContext:
        logical_path, source = self._logical_source(relative_path)
        module = self._module_by_path.get(source)
        domain, capability = self._domain_capability(logical_path)
        facade_module, public_exports = self._facade(source)
        direct_imports = tuple(sorted(self.collector.collect_direct(source)))
        semantic_imports = tuple(sorted(self.collector.collect(source)))
        return PathContext(
            path=logical_path,
            module=module,
            domain=domain,
            capability=capability,
            facade_module=facade_module,
            direct_imports=direct_imports,
            semantic_imports=semantic_imports,
            direct_consumers=self._consumers(
                module,
                semantic=False,
                exclude_module=facade_module,
            ),
            semantic_consumers=self._consumers(
                module,
                semantic=True,
                exclude_module=facade_module,
            ),
            public_exports=public_exports,
            schema_constants=_schema_constants(source),
            test_candidates=self._test_candidates(
                module=module,
                facade_module=facade_module,
                domain=domain,
                capability=capability,
            ),
            doc_references=self._doc_references(logical_path, module),
            effect_signals=_effect_signals(source),
        )

    def impact(self, relative_paths: tuple[str, ...]) -> tuple[PathContext, ...]:
        return tuple(self.context(path) for path in sorted(set(relative_paths)))


__all__ = ["ImportCollector", "PathContext", "SourceIndex", "within_module"]
