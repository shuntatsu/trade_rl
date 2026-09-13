from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "trade_rl"
SCHEMA_VERSION = "guide-code-symbols-v1"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class GuideCodeSymbolError(ValueError):
    """Raised when the Guide code-symbol index cannot be built safely."""


def _validated_revision(revision: str) -> str:
    if not _REVISION_RE.fullmatch(revision):
        raise GuideCodeSymbolError(
            "revision must be a 40-character hexadecimal commit SHA"
        )
    return revision


def _module_name(source_root: Path, path: Path) -> str:
    relative = path.relative_to(source_root)
    parts = [source_root.name, *relative.parts]
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts)


def _repository_path(source_root: Path, path: Path) -> str:
    return path.relative_to(source_root.parent).as_posix()


def _source_digest(lines: list[str], start: int, end: int) -> str:
    text = "".join(lines[start - 1 : end]).replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_start_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self, arguments: ast.arguments) -> None:
        self.names = _argument_names(arguments)
        self.excluded: set[str] = set()

    def collect(self, body: list[ast.stmt]) -> list[str]:
        for statement in body:
            self.visit(statement)
        return sorted(self.names - self.excluded)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if isinstance(node.name, str):
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_Global(self, node: ast.Global) -> None:
        self.excluded.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.excluded.update(node.names)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.names.add(node.name)
        if node.pattern is not None:
            self.visit(node.pattern)

    def _visit_comprehension_generators(
        self, generators: list[ast.comprehension]
    ) -> None:
        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_generators(node.generators)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_generators(node.generators)
        self.visit(node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_generators(node.generators)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_generators(node.generators)
        self.visit(node.key)
        self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    arguments = [ast.unparse(base) for base in node.bases]
    arguments.extend(
        f"{keyword.arg}={ast.unparse(keyword.value)}"
        for keyword in node.keywords
        if keyword.arg is not None
    )
    suffix = f"({', '.join(arguments)})" if arguments else ""
    return f"class {node.name}{suffix}"


def _decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    return None


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(_decorator_name(decorator) == "overload" for decorator in node.decorator_list)


def _resolved_source_file(source_root: Path, path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GuideCodeSymbolError(f"cannot resolve Python source: {path}") from exc
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise GuideCodeSymbolError(f"Python source escapes source root: {path}") from exc
    if not resolved.is_file():
        raise GuideCodeSymbolError(f"Python source is not a file: {path}")
    return resolved


def _symbol_entry(
    *,
    qualified_name: str,
    kind: str,
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
) -> dict[str, object]:
    end_line = node.end_lineno
    if end_line is None:
        raise GuideCodeSymbolError(f"missing end line for code symbol: {qualified_name}")
    start_line = _source_start_line(node)
    local_names: list[str] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        signature = _function_signature(node)
        local_names = _LocalNameCollector(node.args).collect(node.body)
    else:
        signature = _class_signature(node)
    return {
        "qualified_name": qualified_name,
        "kind": kind,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "signature": signature,
        "source_sha256": _source_digest(lines, start_line, end_line),
        "local_names": local_names,
    }


def _class_symbols(
    node: ast.ClassDef,
    *,
    module_name: str,
    path: str,
    lines: list[str],
    parents: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    class_parts = (*parents, node.name)
    qualified_class = ".".join((module_name, *class_parts))
    symbols = [
        _symbol_entry(
            qualified_name=qualified_class,
            kind="class",
            path=path,
            node=node,
            lines=lines,
        )
    ]
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_overload(child):
                continue
            symbols.append(
                _symbol_entry(
                    qualified_name=".".join((qualified_class, child.name)),
                    kind="method",
                    path=path,
                    node=child,
                    lines=lines,
                )
            )
        elif isinstance(child, ast.ClassDef):
            symbols.extend(
                _class_symbols(
                    child,
                    module_name=module_name,
                    path=path,
                    lines=lines,
                    parents=class_parts,
                )
            )
    return symbols


def build_symbol_index(source_root: Path, *, revision: str) -> dict[str, object]:
    source_root = source_root.resolve()
    if source_root.name != "trade_rl" or not source_root.is_dir():
        raise GuideCodeSymbolError("source root must be an existing trade_rl directory")
    validated_revision = _validated_revision(revision)

    symbols: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(source_root.rglob("*.py")):
        resolved_path = _resolved_source_file(source_root, path)
        text = resolved_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise GuideCodeSymbolError(f"cannot parse Python source: {path}") from exc
        lines = text.splitlines(keepends=True)
        module_name = _module_name(source_root, path)
        repository_path = _repository_path(source_root, path)
        file_symbols: list[dict[str, object]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_overload(node):
                    continue
                file_symbols.append(
                    _symbol_entry(
                        qualified_name=f"{module_name}.{node.name}",
                        kind="function",
                        path=repository_path,
                        node=node,
                        lines=lines,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                file_symbols.extend(
                    _class_symbols(
                        node,
                        module_name=module_name,
                        path=repository_path,
                        lines=lines,
                    )
                )
        for symbol in file_symbols:
            qualified_name = str(symbol["qualified_name"])
            if qualified_name in seen:
                raise GuideCodeSymbolError(f"duplicate code symbol: {qualified_name}")
            seen.add(qualified_name)
            symbols.append(symbol)

    symbols.sort(key=lambda item: str(item["qualified_name"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "source_revision": validated_revision,
        "symbols": symbols,
    }


def _runtime_symbol_names(topics_dir: Path) -> list[str]:
    topics_dir = topics_dir.resolve()
    if not topics_dir.is_dir():
        raise GuideCodeSymbolError(f"topics directory does not exist: {topics_dir}")

    topic_paths = sorted(topics_dir.glob("*.json"))
    if not topic_paths:
        raise GuideCodeSymbolError(f"no Guide topic JSON files found: {topics_dir}")

    names: set[str] = set()
    for path in topic_paths:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(topics_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise GuideCodeSymbolError(f"Guide topic escapes topics directory: {path}") from exc
        try:
            topic = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GuideCodeSymbolError(f"cannot read Guide topic JSON: {path}") from exc
        if not isinstance(topic, dict):
            raise GuideCodeSymbolError(f"Guide topic JSON root must be an object: {path}")
        references = topic.get("code_references", [])
        if not isinstance(references, list):
            raise GuideCodeSymbolError(f"Guide topic code_references must be a list: {path}")
        for index, reference in enumerate(references):
            if not isinstance(reference, dict):
                raise GuideCodeSymbolError(
                    f"Guide topic code reference must be an object: {path}#{index}"
                )
            symbol = reference.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                raise GuideCodeSymbolError(
                    f"Guide topic code reference requires symbol: {path}#{index}"
                )
            names.add(symbol)
    return sorted(names)


def build_runtime_symbol_index(
    source_root: Path,
    topics_dir: Path,
    *,
    revision: str,
) -> dict[str, object]:
    full_index = build_symbol_index(source_root, revision=revision)
    raw_symbols = full_index["symbols"]
    if not isinstance(raw_symbols, list):
        raise GuideCodeSymbolError("code symbol index has malformed symbols")
    symbols_by_name = {
        str(symbol["qualified_name"]): symbol
        for symbol in raw_symbols
        if isinstance(symbol, dict) and "qualified_name" in symbol
    }

    selected: list[dict[str, object]] = []
    for name in _runtime_symbol_names(topics_dir):
        symbol = symbols_by_name.get(name)
        if symbol is None:
            raise GuideCodeSymbolError(f"missing referenced code symbol: {name}")
        selected.append(symbol)

    return {
        "schema_version": full_index["schema_version"],
        "source_revision": full_index["source_revision"],
        "symbols": selected,
    }


def write_symbol_index(
    source_root: Path,
    output: Path,
    *,
    revision: str,
    topics_dir: Path | None = None,
) -> None:
    if topics_dir is None:
        index = build_symbol_index(source_root, revision=revision)
    else:
        index = build_runtime_symbol_index(
            source_root,
            topics_dir,
            revision=revision,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_revision(explicit: str | None = None) -> str:
    candidates = [explicit, os.environ.get("GUIDE_SOURCE_REV"), os.environ.get("GITHUB_SHA")]
    for candidate in candidates:
        if candidate:
            return _validated_revision(candidate)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GuideCodeSymbolError("cannot resolve current Git revision") from exc
    return _validated_revision(completed.stdout.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Human Guide code-symbol index.")
    parser.add_argument("--write", type=Path, required=True, metavar="OUTPUT")
    parser.add_argument("--revision")
    parser.add_argument(
        "--topics",
        type=Path,
        metavar="TOPICS_DIR",
        help="Write only symbols referenced by Guide topic JSON files.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        write_symbol_index(
            DEFAULT_SOURCE_ROOT,
            args.write,
            revision=resolve_revision(args.revision),
            topics_dir=args.topics,
        )
    except GuideCodeSymbolError as exc:
        print(f"guide code symbols: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
