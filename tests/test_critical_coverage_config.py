from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_coverage(path: Path, covered: int, total: int) -> None:
    path.write_text(
        json.dumps(
            {
                "files": {
                    "trade_rl/example.py": {
                        "summary": {
                            "covered_branches": covered,
                            "num_branches": total,
                            "percent_covered": 100.0,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _write_config(path: Path, minimum: float) -> None:
    path.write_text(
        "\n".join(
            (
                "[tool.trade_rl.critical_coverage]",
                'metric = "branch"',
                "",
                "[tool.trade_rl.critical_coverage.files]",
                f'"trade_rl/example.py" = {minimum}',
            )
        ),
        encoding="utf-8",
    )


def test_critical_coverage_gate_accepts_exact_threshold(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.json"
    config = tmp_path / "pyproject.toml"
    _write_coverage(coverage, covered=9, total=10)
    _write_config(config, minimum=90.0)

    result = subprocess.run(
        [
            sys.executable,
            ".github/check_critical_coverage.py",
            str(coverage),
            str(config),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "trade_rl/example.py" in result.stdout
    assert "90.00%" in result.stdout


def test_critical_coverage_gate_rejects_regression(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.json"
    config = tmp_path / "pyproject.toml"
    _write_coverage(coverage, covered=8, total=10)
    _write_config(config, minimum=90.0)

    result = subprocess.run(
        [
            sys.executable,
            ".github/check_critical_coverage.py",
            str(coverage),
            str(config),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "FAIL" in result.stdout
    assert "80.00%" in result.stdout


def test_critical_coverage_gate_aggregates_group_branches(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.json"
    config = tmp_path / "pyproject.toml"
    coverage.write_text(
        json.dumps(
            {
                "files": {
                    "trade_rl/a.py": {
                        "summary": {
                            "covered_branches": 3,
                            "num_branches": 4,
                            "percent_covered": 100.0,
                        }
                    },
                    "trade_rl/b.py": {
                        "summary": {
                            "covered_branches": 6,
                            "num_branches": 6,
                            "percent_covered": 100.0,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config.write_text(
        "\n".join(
            (
                "[tool.trade_rl.critical_coverage]",
                'metric = "branch"',
                "",
                "[tool.trade_rl.critical_coverage.groups.core]",
                "minimum = 90.0",
                'paths = ["trade_rl/a.py", "trade_rl/b.py"]',
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".github/check_critical_coverage.py",
            str(coverage),
            str(config),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "group:core" in result.stdout
    assert "90.00%" in result.stdout
