"""Review-oriented semantic surface diff derived from Git and Python source."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

from tools.agent_repo.source_index import (
    FILESYSTEM_EFFECTS,
    NETWORK_EFFECTS,
    literal_public_exports,
)


@dataclass(frozen=True, slots=True, order=True)
class SemanticSignal:
    """One deterministic review signal; not an automatic architecture verdict."""

    kind: str
    change: str
    path: str
    name: str
    detail: str


@dataclass(frozen=True, slots=True)
class _Surface:
    data_shapes: dict[str, str]
    schemas: dict[str, str]
    public_exports: frozenset[str]
    dependencies: frozenset[str]
    filesystem_effects: frozenset[str]
    network_effects: frozenset[str]


def _git(
    repository: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )


def _module_from_path(path: str) -> str:
    parts = PurePosixPath(path).with_suffix("").parts
    if not parts or parts[0] != "trade_rl":
        raise ValueError(f"not a production Python path: {path}")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _git_modules(repository: Path, ref: str) -> frozenset[str]:
    result = _git(
        repository,
        "ls-tree",
        "-r",
        "--name-only",
        ref,
        "--",
        "trade_rl",
    ).stdout
    return frozenset(
        _module_from_path(path) for path in result.splitlines() if path.endswith(".py")
    )


def _worktree_modules(repository: Path) -> frozenset[str]:
    return frozenset(
        _module_from_path(path.relative_to(repository).as_posix())
        for path in sorted((repository / "trade_rl").rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    )


def _changed_paths(repository: Path, merge_base: str) -> tuple[str, ...]:
    changed = set(
        line
        for line in _git(
            repository, "diff", "--name-only", merge_base, "--"
        ).stdout.splitlines()
        if line
    )
    changed.update(
        line
        for line in _git(
            repository,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).stdout.splitlines()
        if line
    )
    return tuple(sorted(changed))


def _base_text(repository: Path, ref: str, path: str) -> str | None:
    result = _git(repository, "show", f"{ref}:{path}", check=False)
    if result.returncode != 0:
        return None
    return result.stdout


def _current_text(repository: Path, path: str) -> str | None:
    candidate = repository / path
    if not candidate.exists() and not candidate.is_symlink():
        return None
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"semantic diff source must be a regular file: {path}")
    return candidate.read_text(encoding="utf-8")


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _data_shapes(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    boundary_bases = {"Protocol", "TypedDict", "NamedTuple", "Enum", "StrEnum"}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(_decorator_name(item) == "dataclass" for item in node.decorator_list):
            result[node.name] = "dataclass"
            continue
        bases = {_base_name(base) for base in node.bases}
        matched = sorted(name for name in bases if name in boundary_bases)
        if matched:
            result[node.name] = matched[0]
    return result


def _schemas(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in tree.body:
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
                result[target.id] = value.value
    return result


def _public_exports(tree: ast.Module) -> frozenset[str]:
    exports = literal_public_exports(tree)
    return frozenset(() if exports is None else exports)


def _direct_dependencies(
    tree: ast.Module,
    *,
    module: str,
    is_package: bool,
    known_modules: frozenset[str],
) -> frozenset[str]:
    result: set[str] = set()
    context = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name(
                "." * node.level + (node.module or ""),
                context,
            )
            result.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}"
                if child in known_modules:
                    result.add(child)
    return frozenset(
        name for name in result if name.split(".", 1)[0] not in sys.stdlib_module_names
    )


def _effects(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    filesystem: set[str] = set()
    network: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name: str | None = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FILESYSTEM_EFFECTS:
            filesystem.add(name)
        if name in NETWORK_EFFECTS:
            network.add(name)
    return frozenset(filesystem), frozenset(network)


def _surface(
    text: str | None,
    *,
    path: str,
    known_modules: frozenset[str],
) -> _Surface:
    if text is None:
        return _Surface({}, {}, frozenset(), frozenset(), frozenset(), frozenset())
    tree = ast.parse(text, filename=path)
    module = _module_from_path(path)
    filesystem, network = _effects(tree)
    return _Surface(
        data_shapes=_data_shapes(tree),
        schemas=_schemas(tree),
        public_exports=_public_exports(tree),
        dependencies=_direct_dependencies(
            tree,
            module=module,
            is_package=PurePosixPath(path).name == "__init__.py",
            known_modules=known_modules,
        ),
        filesystem_effects=filesystem,
        network_effects=network,
    )


def _project_surfaces(text: str | None) -> tuple[dict[str, str], dict[str, str]]:
    if text is None:
        return {}, {}
    payload = tomllib.loads(text)
    project = payload.get("project")
    if project is None:
        return {}, {}
    if not isinstance(project, dict):
        raise ValueError("pyproject project table must be a table")

    extras: dict[str, str] = {}
    optional = project.get("optional-dependencies")
    if optional is not None:
        if not isinstance(optional, dict):
            raise ValueError("project.optional-dependencies must be a table")
        for name, requirements in optional.items():
            if not isinstance(name, str) or not isinstance(requirements, list):
                raise ValueError("optional dependency entries must be string lists")
            if not all(isinstance(item, str) for item in requirements):
                raise ValueError("optional dependency entries must be string lists")
            extras[name] = repr(tuple(sorted(requirements)))

    scripts: dict[str, str] = {}
    script_table = project.get("scripts")
    if script_table is not None:
        if not isinstance(script_table, dict):
            raise ValueError("project.scripts must be a table")
        for name, target in script_table.items():
            if not isinstance(name, str) or not isinstance(target, str):
                raise ValueError("project script entries must be strings")
            scripts[name] = target
    return extras, scripts


def _mapping_signals(
    *,
    kind: str,
    path: str,
    before: dict[str, str],
    after: dict[str, str],
) -> list[SemanticSignal]:
    result: list[SemanticSignal] = []
    for name in sorted(before.keys() - after.keys()):
        result.append(SemanticSignal(kind, "removed", path, name, before[name]))
    for name in sorted(after.keys() - before.keys()):
        result.append(SemanticSignal(kind, "added", path, name, after[name]))
    for name in sorted(before.keys() & after.keys()):
        if before[name] != after[name]:
            result.append(
                SemanticSignal(
                    kind,
                    "changed",
                    path,
                    name,
                    f"{before[name]} -> {after[name]}",
                )
            )
    return result


def _set_signals(
    *,
    kind: str,
    path: str,
    before: frozenset[str],
    after: frozenset[str],
    detail: str,
) -> list[SemanticSignal]:
    return [
        *(
            SemanticSignal(kind, "removed", path, name, detail)
            for name in sorted(before - after)
        ),
        *(
            SemanticSignal(kind, "added", path, name, detail)
            for name in sorted(after - before)
        ),
    ]


def _ci_surface_signal(
    path: str,
    *,
    before: str | None,
    after: str | None,
) -> SemanticSignal | None:
    if before == after:
        return None
    if before is None:
        change = "added"
    elif after is None:
        change = "removed"
    else:
        change = "changed"
    return SemanticSignal(
        kind="CI_SURFACE",
        change=change,
        path=path,
        name=PurePosixPath(path).name,
        detail="workflow file",
    )


def semantic_diff(
    repository: Path,
    *,
    base_ref: str,
) -> tuple[SemanticSignal, ...]:
    """Return narrow semantic review signals without mutating the checkout."""

    root = repository.resolve()
    merge_base = _git(root, "merge-base", "HEAD", base_ref).stdout.strip()
    before_modules = _git_modules(root, merge_base)
    after_modules = _worktree_modules(root)
    changed_paths = _changed_paths(root, merge_base)
    signals: list[SemanticSignal] = []

    production_paths = (
        path
        for path in changed_paths
        if path.startswith("trade_rl/") and path.endswith(".py")
    )
    for path in production_paths:
        before = _surface(
            _base_text(root, merge_base, path),
            path=path,
            known_modules=before_modules,
        )
        after = _surface(
            _current_text(root, path),
            path=path,
            known_modules=after_modules,
        )
        signals.extend(
            _mapping_signals(
                kind="DATA_SHAPE",
                path=path,
                before=before.data_shapes,
                after=after.data_shapes,
            )
        )
        signals.extend(
            _mapping_signals(
                kind="SCHEMA",
                path=path,
                before=before.schemas,
                after=after.schemas,
            )
        )
        signals.extend(
            _set_signals(
                kind="PUBLIC_EXPORT",
                path=path,
                before=before.public_exports,
                after=after.public_exports,
                detail="literal __all__",
            )
        )
        signals.extend(
            _set_signals(
                kind="DEPENDENCY",
                path=path,
                before=before.dependencies,
                after=after.dependencies,
                detail="direct import",
            )
        )
        signals.extend(
            _set_signals(
                kind="FILESYSTEM_EFFECT",
                path=path,
                before=before.filesystem_effects,
                after=after.filesystem_effects,
                detail="effect candidate",
            )
        )
        signals.extend(
            _set_signals(
                kind="NETWORK_EFFECT",
                path=path,
                before=before.network_effects,
                after=after.network_effects,
                detail="effect candidate",
            )
        )

    if "pyproject.toml" in changed_paths:
        before_extras, before_scripts = _project_surfaces(
            _base_text(root, merge_base, "pyproject.toml")
        )
        after_extras, after_scripts = _project_surfaces(
            _current_text(root, "pyproject.toml")
        )
        signals.extend(
            _mapping_signals(
                kind="PROJECT_EXTRA",
                path="pyproject.toml",
                before=before_extras,
                after=after_extras,
            )
        )
        signals.extend(
            _mapping_signals(
                kind="PROJECT_SCRIPT",
                path="pyproject.toml",
                before=before_scripts,
                after=after_scripts,
            )
        )

    for path in changed_paths:
        if not path.startswith(".github/workflows/") or not path.endswith(
            (".yml", ".yaml")
        ):
            continue
        signal = _ci_surface_signal(
            path,
            before=_base_text(root, merge_base, path),
            after=_current_text(root, path),
        )
        if signal is not None:
            signals.append(signal)

    return tuple(sorted(signals))


__all__ = ["SemanticSignal", "semantic_diff"]
