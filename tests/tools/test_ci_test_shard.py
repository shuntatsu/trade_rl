"""The full suite may be split, but no collected item may disappear."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
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


@pytest.mark.parametrize("key", ["head", "index", "completed", "schema"])
def test_verifier_rejects_duplicate_receipt_keys(tmp_path: Path, key: str) -> None:
    items = [f"tests/test_example.py::test_cases[{i}]" for i in range(120)]
    path = _reports(tmp_path, items)[0]
    original = path.read_text()
    path.write_text("{" + json.dumps(key) + ": null, " + original[1:])
    with pytest.raises(ValueError, match="unreadable shard receipt"):
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


@pytest.mark.parametrize("mode", ["--setup-only", "--setup-plan"])
def test_nonexecuting_setup_modes_cannot_issue_receipt(
    tmp_path: Path, mode: str
) -> None:
    (tmp_path / "test_sample.py").write_text("def test_must_execute(): assert False\n")
    report = tmp_path / "core-shard.json"
    run = _run_shard(tmp_path, 0, 1, report, mode)
    assert run.returncode != 0
    assert not report.exists()


def test_logfinish_without_execution_phases_is_not_completion(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text("def test_must_execute(): assert False\n")
    (tmp_path / "conftest.py").write_text(
        "import pytest\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_runtest_protocol(item, nextitem):\n"
        "    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)\n"
        "    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)\n"
        "    return True\n"
    )
    report = tmp_path / "core-shard.json"
    run = _run_shard(tmp_path, 0, 1, report)
    assert run.returncode != 0
    assert not report.exists()


def test_real_setup_skip_call_skip_and_xfail_keep_valid_completion(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "@pytest.mark.skip(reason='setup skip')\n"
        "def test_setup_skip(): assert False\n"
        "def test_call_skip(): pytest.skip('call skip')\n"
        "@pytest.mark.xfail(run=False, reason='expected no call')\n"
        "def test_no_run_xfail(): assert False\n"
        "@pytest.mark.xfail(reason='expected call failure')\n"
        "def test_call_xfail(): assert False\n"
    )
    report = tmp_path / "core-shard.json"
    run = _run_shard(tmp_path, 0, 1, report)
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(json.loads(report.read_text())["completed"]) == 5


# These are the five module placements derived from the ordinary CI trace;
# include parameterized and class-based node IDs, and avoid prefix matching.
_AFFINITY_CASES = (
    ("tests/evaluation/test_dev_realdata_phase3.py", 1),
    ("tests/evaluation/test_dev_realdata_study.py", 0),
    ("tests/evaluation/test_forward_study_fs1.py", 2),
    ("tests/evaluation/test_signal_diagnosis_run_scenarios.py", 0),
    ("tests/tools/tier_d_nonrl/test_task18_native_artifact.py", 3),
)


@pytest.mark.parametrize("module,expected", _AFFINITY_CASES)
def test_measured_module_fixture_affinity(module: str, expected: int) -> None:
    nodes = [f"{module}::TestShared::test_work[{i}]" for i in range(16)]
    assert {shard_index(node, 4) for node in nodes} == {expected}
    # Other shard counts keep the prior deterministic hash contract.
    assert [shard_index(node, 3) for node in nodes] == [
        int.from_bytes(sha256(node.encode()).digest()[:8], "big") % 3 for node in nodes
    ]
    similar = [f"{module}.copy.py::test_work[{i}]" for i in range(16)]
    assert [shard_index(node, 4) for node in similar] == [
        int.from_bytes(sha256(node.encode()).digest()[:8], "big") % 4
        for node in similar
    ]


def test_affinity_receipts_cover_whole_roster_and_reject_old_assignment(
    tmp_path: Path,
) -> None:
    items = [f"tests/test_example.py::test_cases[{i}]" for i in range(120)]
    items += [
        f"{module}::test_group[{i}]" for module, _ in _AFFINITY_CASES for i in range(8)
    ]
    paths = _reports(tmp_path, items)
    assert verify_reports(tmp_path, count=4, head=HEAD, expected_items=items) == len(
        items
    )
    # Even a complete receipt union cannot be reused under an old placement.
    for index, path in enumerate(paths):
        record = json.loads(path.read_text())
        record["completed"] = sorted(
            node
            for node in items
            if int.from_bytes(sha256(node.encode()).digest()[:8], "big") % 4 == index
        )
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="wrong shard"):
        verify_reports(tmp_path, count=4, head=HEAD, expected_items=items)


def test_measured_module_fixture_runs_once_in_real_shards(tmp_path: Path) -> None:
    module, expected = _AFFINITY_CASES[0]
    path = tmp_path / module
    path.parent.mkdir(parents=True)
    path.write_text(
        "from pathlib import Path\nimport pytest\n"
        "@pytest.fixture(scope='module')\n"
        "def shared():\n"
        "    with Path('fixture-count.txt').open('a') as log: log.write('setup\\n')\n"
        "    return 7\n"
        "@pytest.mark.parametrize('i', range(16))\n"
        "def test_work(shared, i): assert shared == 7 and i >= 0\n"
    )
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n@pytest.mark.parametrize('i', range(24))\n"
        "def test_items(i): assert i >= 0\n"
    )
    root = tmp_path / "reports"
    root.mkdir()
    for index in range(4):
        directory = root / f"shard-{index}"
        directory.mkdir()
        report = directory / "core-shard.json"
        run = _run_shard(tmp_path, index, 4, report, module)
        assert run.returncode == 0, run.stdout + run.stderr
        nodes = json.loads(report.read_text())["completed"]
        assert sum(node.startswith(module + "::") for node in nodes) == (
            16 if index == expected else 0
        )
    assert (tmp_path / "fixture-count.txt").read_text().splitlines() == ["setup"]
    items = [f"{module}::test_work[{i}]" for i in range(16)]
    items += [f"test_sample.py::test_items[{i}]" for i in range(24)]
    assert verify_reports(root, count=4, head=HEAD, expected_items=items) == 40
