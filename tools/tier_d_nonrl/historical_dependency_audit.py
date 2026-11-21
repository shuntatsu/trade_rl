"""Read-only, result-blind historical Tier-D source dependency audit.

Compare exact Git trees; static imports do not establish valid runtime, native
receipt producer, Windows containment, PIT data or R3 authorization. Example:
python -m tools.tier_d_nonrl.historical_dependency_audit --repo . \
  --historical-sha SHA40 --maintained-sha SHA40
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SEEDS = (
    "tools.tier_d_nonrl.tier_d_runner",
    "tools.tier_d_nonrl.resume",
    "tools.tier_d_nonrl.tier_d_supervisor",
)


def _path(name: str, sources: Mapping[str, str]) -> str | None:
    base = name.replace(".", "/")
    return next(
        (p for p in (base + ".py", base + "/__init__.py") if p in sources), None
    )


def _exports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(x.asname or x.name.split(".")[0] for x in node.names)
        elif isinstance(node, ast.Assign):
            names.update(x.id for x in node.targets if isinstance(x, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _target(module: str, path: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = (
        module.split(".") if path.endswith("/__init__.py") else module.split(".")[:-1]
    )
    depth = node.level - 1
    if depth > len(package):
        return ""
    prefix = package[:-depth] if depth else package
    return ".".join([*prefix, *(node.module.split(".") if node.module else [])])


def analyze_sources(
    historical: Mapping[str, str],
    maintained: Mapping[str, str],
    *,
    seeds: Sequence[str] = SEEDS,
) -> dict[str, Any]:
    """Enumerate only static migration gaps; never grant runtime authority."""
    pending = list(seeds)
    seen: set[str] = set()
    missing: set[str] = set()
    missing_seeds = sorted(s for s in seeds if _path(s, historical) is None)
    errors: set[str] = set()
    drift: set[tuple[str, str, str]] = set()
    while pending:
        module = pending.pop(0)
        if module in seen:
            continue
        path = _path(module, historical)
        if path is None:
            continue
        seen.add(module)
        if path not in maintained:
            missing.add(path)
        try:
            tree = ast.parse(historical[path], filename=path)
        except (SyntaxError, ValueError):
            errors.add(path)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _path(alias.name, historical) is not None:
                        pending.append(alias.name)
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _target(module, path, node)
            target_path = _path(target, historical)
            if target_path is not None:
                pending.append(target)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = target + "." + alias.name
                if _path(child, historical) is not None:
                    pending.append(child)
                    continue
                if target_path is None or target_path not in maintained:
                    continue
                try:
                    old_names = _exports(ast.parse(historical[target_path]))
                    new_names = _exports(ast.parse(maintained[target_path]))
                except (SyntaxError, ValueError):
                    errors.add(target_path)
                    continue
                if alias.name in old_names and alias.name not in new_names:
                    drift.add((module, target, alias.name))
    return {
        "schema": "task18_historical_static_dependency_audit_v1",
        "analysis_complete": not missing_seeds and not errors,
        "recovery_ready": False,
        "reachable_module_count": len(seen),
        "missing_module_count": len(missing),
        "missing_seed_modules": missing_seeds,
        "missing_files": sorted(missing),
        "missing_static_exports": [
            {"consumer": consumer, "module": target, "symbol": symbol}
            for consumer, target, symbol in sorted(drift)
        ],
        "parse_errors": sorted(errors),
        "limits": "Static literal imports and top-level symbols only; no dynamic imports, package initializers, Windows, native execution, ownership, economic evidence or authorization",
    }


def _tree(repo: Path, sha: str) -> dict[str, str]:
    """Read only tracked Python source from a literal pinned Git commit."""
    import io
    import tarfile

    if re.fullmatch(r"[a-f0-9]{40}", sha) is None:
        raise ValueError("expected a literal 40-digit Git SHA")
    kind = (
        subprocess.check_output(
            ["git", "-C", str(repo), "cat-file", "-t", sha], timeout=10
        )
        .decode("utf-8")
        .strip()
    )
    if kind != "commit":
        raise ValueError("the pinned Git object is not a commit")
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "archive", "--format=tar", sha, "tools", "trade_rl"],
        timeout=35,
    )
    result: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("missing source member")
            result[member.name] = handle.read().decode("utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--historical-sha", required=True)
    parser.add_argument("--maintained-sha", required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze_sources(
            _tree(args.repo, args.historical_sha),
            _tree(args.repo, args.maintained_sha),
        )
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        parser.exit(2, "static dependency audit refused; pinned source unreadable\n")
    report.update(
        historical_sha=args.historical_sha, maintained_sha=args.maintained_sha
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report["analysis_complete"]:
        return 2
    return 1 if report["missing_files"] or report["missing_static_exports"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
