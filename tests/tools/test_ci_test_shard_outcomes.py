"""Prospective skip evidence, not a certificate that skipped tests passed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_test_shard import collection_digest, verify_reports

_ROOT = Path(__file__).resolve().parents[2]
_HEAD = "a" * 40


def test_real_shard_retains_phase_outcomes_and_collection_skip(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "@pytest.mark.skip(reason='setup skip')\n"
        "def test_setup_skip(): assert False\n"
        "def test_call_skip(): pytest.skip('call skip')\n"
        "@pytest.mark.xfail(run=False, reason='no call')\n"
        "def test_no_run_xfail(): assert False\n"
        "@pytest.mark.xfail(reason='call failure')\n"
        "def test_call_xfail(): assert False\n"
        "@pytest.mark.xfail(reason='unexpected pass', strict=False)\n"
        "def test_xpass(): pass\n"
        "@pytest.fixture\n"
        "def cleanup():\n"
        "    yield\n"
        "    pytest.skip('teardown skip')\n"
        "def test_teardown_skip(cleanup): pass\n"
    )
    (tests / "test_module_skip.py").write_text(
        "import pytest\npytest.skip('missing runtime', allow_module_level=True)\n"
    )
    reports = tmp_path / "reports"
    folder = reports / "shard-0"
    folder.mkdir(parents=True)
    report = folder / "core-shard.json"
    env = dict(
        os.environ,
        PYTHONPATH=str(_ROOT),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tools.ci_test_shard",
            "--ci-shard-index",
            "0",
            "--ci-shard-count",
            "1",
            "--ci-shard-head",
            _HEAD,
            "--ci-shard-report",
            str(report),
            "tests",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    record = json.loads(report.read_text())
    assert record["schema"] == "ci_test_shard_v2"
    phases = record["phase_reports"]
    assert set(phases) == set(record["completed"])
    assert len(phases) == 7
    expected = {
        "test_pass": None,
        "test_setup_skip": ("setup", "skipped", "setup skip"),
        "test_call_skip": ("call", "skipped", "call skip"),
        "test_no_run_xfail": ("setup", "xfailed", "no call"),
        "test_call_xfail": ("call", "xfailed", "call failure"),
        "test_xpass": ("call", "xpassed", "unexpected pass"),
        "test_teardown_skip": ("teardown", "skipped", "teardown skip"),
    }
    for name, outcome in expected.items():
        other = [
            row for row in phases[f"tests/test_sample.py::{name}"] if row[1] != "passed"
        ]
        if outcome is None:
            assert other == []
        else:
            assert len(other) == 1
            phase, status, reason = other[0]
            assert (phase, status) == outcome[:2]
            assert outcome[2] in reason
    skipped = record["collection_skips"]
    assert len(skipped) == 1
    assert skipped[0][0] == "tests/test_module_skip.py"
    assert "missing runtime" in skipped[0][1]
    assert (
        verify_reports(
            reports,
            count=1,
            head=_HEAD,
            expected_items=list(phases),
            expected_collection_skips=[tuple(skipped[0])],
        )
        == 7
    )
    with pytest.raises(ValueError, match="collection skip"):
        verify_reports(reports, count=1, head=_HEAD, expected_items=list(phases))

    # Exercise the actual aggregator, including its independent pytest collection.
    aggregate = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci_test_shard",
            "--directory",
            str(reports),
            "--count",
            "1",
            "--head",
            _HEAD,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert aggregate.returncode == 0, aggregate.stdout + aggregate.stderr
    record["collection_skips"] = []
    report.write_text(json.dumps(record))
    rejected = subprocess.run(
        aggregate.args,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert rejected.returncode != 0
    assert "collection skip roster differs" in rejected.stderr


@pytest.mark.parametrize(
    "case",
    [
        "legacy",
        "missing_node",
        "foreign_node",
        "missing_call",
        "bad_reason",
        "bad_skip",
        "duplicate_collection_skip",
    ],
)
def test_verifier_requires_exhaustive_valid_phase_evidence(
    tmp_path: Path, case: str
) -> None:
    folder = tmp_path / "shard-0"
    folder.mkdir()
    node = "test_sample.py::test_pass"
    record = {
        "schema": "ci_test_shard_v2",
        "head": _HEAD,
        "index": 0,
        "count": 1,
        "collection_count": 1,
        "collection_digest": collection_digest([node]),
        "completed": [node],
        "phase_reports": {
            node: [
                ["setup", "passed", None],
                ["call", "passed", None],
                ["teardown", "passed", None],
            ]
        },
        "collection_skips": [],
    }
    if case == "legacy":
        record["schema"] = "ci_test_shard_v1"
        del record["phase_reports"]
        del record["collection_skips"]
    elif case == "missing_node":
        record["phase_reports"] = {}
    elif case == "foreign_node":
        record["phase_reports"]["foreign::test"] = record["phase_reports"][node]
    elif case == "missing_call":
        record["phase_reports"][node].pop(1)
    elif case == "bad_reason":
        record["phase_reports"][node][1][2] = "invented reason"
    elif case == "bad_skip":
        record["phase_reports"][node][1][1] = "skipped"
    elif case == "duplicate_collection_skip":
        record["collection_skips"] = [["other.py", "reason"], ["other.py", "reason"]]
    (folder / "core-shard.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verify_reports(tmp_path, count=1, head=_HEAD, expected_items=[node])
