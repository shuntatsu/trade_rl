"""The full suite may be split, but no collected item may disappear."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_test_shard import collection_digest, shard_index, verify_reports

ROOT = Path(__file__).resolve().parents[2]
HEAD = "a" * 40


def _reports(root: Path, items: list[str], count: int = 4) -> list[Path]:
    paths = []
    for index in range(count):
        directory = root / f"shard-{index}"
        directory.mkdir()
        path = directory / "core-shard.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "ci_test_shard_v1",
                    "head": HEAD,
                    "index": index,
                    "count": count,
                    "collection_count": len(items),
                    "collection_digest": collection_digest(items),
                    "completed": sorted(
                        item for item in items if shard_index(item, count) == index
                    ),
                }
            )
        )
        paths.append(path)
    return paths


def test_partition_is_exhaustive_order_independent_and_stable(tmp_path: Path) -> None:
    items = [f"tests/test_example.py::test_cases[{i}]" for i in range(120)]
    assert collection_digest(items) == collection_digest(list(reversed(items)))
    assert [shard_index(item, 4) for item in items[:5]] == [1, 2, 0, 0, 2]
    _reports(tmp_path, items)
    assert verify_reports(tmp_path, count=4, head=HEAD, expected_items=items) == len(
        items
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate_index",
        "wrong_head",
        "wrong_count",
        "omission",
        "duplicate_item",
        "wrong_shard",
        "collection_drift",
        "empty",
        "malformed",
    ],
)
def test_verifier_refuses_incomplete_or_inconsistent_reports(
    tmp_path: Path, case: str
) -> None:
    items = [f"tests/test_example.py::test_cases[{i}]" for i in range(120)]
    paths = _reports(tmp_path, items)
    path = paths[0]
    data = json.loads(path.read_text())
    if case == "missing":
        path.unlink()
    elif case == "malformed":
        path.write_text("not JSON")
    else:
        if case == "duplicate_index":
            data["index"] = 1
        elif case == "wrong_head":
            data["head"] = "b" * 40
        elif case == "wrong_count":
            data["count"] = 3
        elif case == "omission":
            data["completed"].pop()
        elif case == "duplicate_item":
            data["completed"].append(data["completed"][0])
        elif case == "wrong_shard":
            data["completed"].append(json.loads(paths[1].read_text())["completed"][0])
        elif case == "collection_drift":
            data["collection_digest"] = "0" * 64
        elif case == "empty":
            data["completed"] = []
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_reports(tmp_path, count=4, head=HEAD, expected_items=items)


def _run_shard(
    root: Path, index: int, count: int, report: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tools.ci_test_shard",
            "--ci-shard-index",
            str(index),
            "--ci-shard-count",
            str(count),
            "--ci-shard-head",
            HEAD,
            "--ci-shard-report",
            str(report),
            *extra,
            "test_sample.py",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_real_pytest_invocations_cover_skipped_and_passed_items(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n@pytest.mark.parametrize('i', range(24))\ndef test_items(i):\n    if i % 5 == 0: pytest.skip('control')\n    assert i >= 0\n"
    )
    root = tmp_path / "reports"
    root.mkdir()
    for index in range(4):
        directory = root / f"shard-{index}"
        directory.mkdir()
        run = _run_shard(tmp_path, index, 4, directory / "core-shard.json")
        assert run.returncode == 0, run.stdout + run.stderr
    assert (
        verify_reports(
            root,
            count=4,
            head=HEAD,
            expected_items=[f"test_sample.py::test_items[{i}]" for i in range(24)],
        )
        == 24
    )


@pytest.mark.parametrize("extra", [(), ("--collect-only",)])
def test_failed_or_collection_only_run_never_publishes_success(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    (tmp_path / "test_sample.py").write_text("def test_bad():\n    assert False\n")
    report = tmp_path / "core-shard.json"
    index = shard_index("test_sample.py::test_bad", 4)
    run = _run_shard(tmp_path, index, 4, report, *extra)
    assert run.returncode != 0
    assert not report.exists()


def test_invalid_shard_options_refused(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text("def test_ok(): pass\n")
    report = tmp_path / "core-shard.json"
    run = _run_shard(tmp_path, 4, 4, report)
    assert run.returncode != 0
    assert not report.exists()


def test_premature_success_exit_cannot_issue_receipt(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import pytest\ndef test_early_exit(): pytest.exit('early', returncode=0)\ndef test_later(): pass\n"
    )
    report = tmp_path / "core-shard.json"
    run = _run_shard(tmp_path, 0, 1, report)
    assert run.returncode != 0
    assert not report.exists()


def test_filtered_or_stale_receipt_cannot_claim_completion(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text("def test_ok(): pass\n")
    report = tmp_path / "core-shard.json"
    run = _run_shard(tmp_path, 0, 1, report, "-k", "test_ok")
    assert run.returncode != 0
    assert not report.exists()
    report.write_text("prior attempt")
    run = _run_shard(tmp_path, 0, 1, report)
    assert run.returncode != 0
    assert report.read_text() == "prior attempt"


def test_all_shards_agreeing_on_incomplete_collection_is_rejected(
    tmp_path: Path,
) -> None:
    items = [f"tests/test_example.py::test_cases[{i}]" for i in range(120)]
    _reports(tmp_path, items[:-1])
    with pytest.raises(ValueError, match="independent full collection"):
        verify_reports(tmp_path, count=4, head=HEAD, expected_items=items)
