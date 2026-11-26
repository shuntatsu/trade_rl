"""Closed capacity preregistration lives in frozen history, not core runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
MODULE = "trade_rl.evaluation.experiments.bootstrap.capacity_robustness_prereg"
SOURCE = "trade_rl/evaluation/experiments/bootstrap/capacity_robustness_prereg.py"
TEST = "tests/evaluation/experiments/bootstrap/test_capacity_robustness_prereg.py"
ARCHIVE_COMMIT = "5a8a7a8c149c1017f2e52ac32392c1810ddc5672"
# These two exact inventory blobs are retained by research PR #22/#25.
ARCHIVED = {
    SOURCE: "c39e0cb19bc01a6279300916cb122b2c1a90a08e9040b39a6a4f72a6979bf529",
    TEST: "91978233d983f0f2defab9769657a10c72e9ee7abb4cb1cbc5e0aaa96a36bcec",
}


def test_closed_capacity_source_and_duplicate_test_are_absent() -> None:
    for name in ARCHIVED:
        assert not (ROOT / name).exists(), name
    assert importlib.util.find_spec(MODULE) is None


def test_frozen_capacity_source_and_test_are_byte_preserved() -> None:
    for name, expected in ARCHIVED.items():
        raw = subprocess.check_output(
            ["git", "show", f"{ARCHIVE_COMMIT}:{name}"], cwd=ROOT
        )
        assert hashlib.sha256(raw).hexdigest() == expected, name


def test_core_has_no_static_dependency_on_closed_capacity_study() -> None:
    collector = ImportCollector(PACKAGE)
    for path in PACKAGE.rglob("*.py"):
        if path != ROOT / SOURCE:
            assert not any(
                within_module(name, MODULE) for name in collector.collect(path)
            ), path


def test_capacity_archive_navigation_does_not_replace_shared_bootstrap() -> None:
    from trade_rl.evaluation.experiments import bootstrap

    assert "bootstrap_canonical_m2_study" in bootstrap.__all__
    assert "CanonicalM2BootstrapConfig" in bootstrap.__all__
    assert "CapacityRobustnessProtocol" not in bootstrap.__all__
    docs = (ROOT / "docs/plans/repository-split.md").read_text(encoding="utf-8")
    assert (
        "https://github.com/shuntatsu/trade_rl_research/blob/main/"
        "migration/capacity_robustness_archive.json"
    ) in docs
