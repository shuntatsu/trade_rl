#!/usr/bin/env python3
"""Enforce branch-coverage ratchets for financially critical modules."""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CoverageResult:
    label: str
    observed: float
    minimum: float

    @property
    def passed(self) -> bool:
        return self.observed + 1e-9 >= self.minimum


def _branch_percent(summary: dict[str, Any]) -> float:
    total = int(summary.get("num_branches", 0))
    if total <= 0:
        return float(summary.get("percent_covered", 0.0))
    covered = int(summary.get("covered_branches", 0))
    return 100.0 * covered / total


def _summary(files: dict[str, Any], path: str) -> dict[str, Any]:
    try:
        value = files[path]["summary"]
    except KeyError as error:
        raise ValueError(f"coverage report does not contain {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"coverage summary for {path} is invalid")
    return value


def _file_results(
    files: dict[str, Any], thresholds: dict[str, Any]
) -> list[CoverageResult]:
    results: list[CoverageResult] = []
    for path, minimum in sorted(thresholds.items()):
        results.append(
            CoverageResult(
                label=path,
                observed=_branch_percent(_summary(files, path)),
                minimum=float(minimum),
            )
        )
    return results


def _group_results(
    files: dict[str, Any], groups: dict[str, Any]
) -> list[CoverageResult]:
    results: list[CoverageResult] = []
    for name, raw in sorted(groups.items()):
        if not isinstance(raw, dict):
            raise ValueError(f"critical coverage group {name} must be a table")
        paths = raw.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and path for path in paths)
        ):
            raise ValueError(f"critical coverage group {name} requires paths")
        covered = 0
        total = 0
        fallback_percentages: list[float] = []
        for path in paths:
            summary = _summary(files, path)
            branches = int(summary.get("num_branches", 0))
            if branches > 0:
                covered += int(summary.get("covered_branches", 0))
                total += branches
            else:
                fallback_percentages.append(_branch_percent(summary))
        if total > 0:
            observed = 100.0 * covered / total
        else:
            observed = sum(fallback_percentages) / len(fallback_percentages)
        results.append(
            CoverageResult(
                label=f"group:{name}",
                observed=observed,
                minimum=float(raw.get("minimum", 0.0)),
            )
        )
    return results


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: check_critical_coverage.py COVERAGE_JSON PYPROJECT_TOML",
            file=sys.stderr,
        )
        return 2
    coverage_path = Path(argv[1])
    config_path = Path(argv[2])
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    files = coverage.get("files")
    if not isinstance(files, dict):
        raise ValueError("coverage report has no files table")
    critical = config.get("tool", {}).get("trade_rl", {}).get("critical_coverage", {})
    if not isinstance(critical, dict):
        raise ValueError("critical coverage configuration is invalid")
    if critical.get("metric", "branch") != "branch":
        raise ValueError("only branch critical coverage is supported")
    thresholds = critical.get("files", {})
    groups = critical.get("groups", {})
    if not isinstance(thresholds, dict) or not isinstance(groups, dict):
        raise ValueError("critical coverage files and groups must be tables")
    results = _file_results(files, thresholds) + _group_results(files, groups)
    if not results:
        raise ValueError("critical coverage configuration is empty")
    print("Critical branch coverage ratchet")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status:4} {result.label}: "
            f"{result.observed:.2f}% >= {result.minimum:.2f}%"
        )
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
