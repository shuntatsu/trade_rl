"""Closed development-study entrypoints are historical, not core commands."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_COMMIT = "97c80a2120b82b7f9bc93c789482be15ea18f5e2"
PACKAGE = "trade_rl.evaluation.dev_realdata_study"
ARCHIVED = {
    "trade_rl/evaluation/dev_realdata_study/__main__.py": "a2ebef0ce9402990da7b853aa4593befeada39dd8b13342682b6776e59e70120",
    "trade_rl/evaluation/dev_realdata_study/cli.py": "58bb5d5ecd1e86f5767c86e7a05c39472019ee2be629ac7e12f1015e6f188b30",
    "trade_rl/evaluation/dev_realdata_study/phase3_cli.py": "992b6398305759cc49c1ea4c80f32814142298e3e3e91398ee0509f9949c5482",
    "trade_rl/evaluation/dev_realdata_study/trial_cli.py": "bb2ce9e8300a82551be707e21a6121d065e1d17b8fcca9e93d52bfaf067c806b",
    "trade_rl/evaluation/dev_realdata_study/trial_study.py": "01eafac8d5feb98ae039c3f2b781e73400363e670f400ff0baf3114a8f8120ba",
    "trade_rl/evaluation/dev_realdata_study/phase3_study.py": "420aa9133ca37b68880c445c82b950d1e2ca5b920c13a8c1a418d28289b855d7",
    "trade_rl/evaluation/dev_realdata_study/dataset_build.py": "7d75dc8f68ec080f1c3a09a4684af8d9a51da4eef85836588542e18a1aacd927",
    "trade_rl/evaluation/dev_realdata_study/trial_labels.py": "4e18fd10daa13980ef1743ee820db17d1c0b2178be59fb2420ed83bc22c5117d",
    "tests/evaluation/test_dev_realdata_dataset_build.py": "c87ffe8fca0c069eb186399bf127572e3072d436fb0a78d019b44a90736150fc",
    "tests/evaluation/test_dev_realdata_study.py": "69d983b86a76399c1b7f992bfe95b15356cd811d4561688705ad714a8c3032bf",
    "tests/evaluation/test_dev_realdata_phase2_trials.py": "315807d57982d684a8054359e2837b124e224904a0fc9fc9c1ced67c22212f78",
    "tests/evaluation/test_dev_realdata_phase2_hardening.py": "5be8f62b2d62b23a6470d324fbf5fba1b85d4f2cd67851c7efea94cb2860a81c",
    "tests/evaluation/test_dev_realdata_phase3.py": "4c903a1cc53aab4956ab8387b938325d9b0347c58dfb5c726f88793bc09202cc",
}
RETIRED = tuple(
    f"{PACKAGE}.{name}"
    for name in (
        "__main__",
        "cli",
        "phase3_cli",
        "trial_cli",
        "trial_study",
        "phase3_study",
        "dataset_build",
        "trial_labels",
    )
)


def test_closed_study_entrypoints_and_stage_runners_are_absent() -> None:
    for module in RETIRED:
        assert not (ROOT / (module.replace(".", "/") + ".py")).exists(), module
        assert importlib.util.find_spec(module) is None


def test_original_entrypoints_and_stage_runner_tests_remain_in_archive_history() -> (
    None
):
    for path, expected in ARCHIVED.items():
        raw = subprocess.check_output(
            ["git", "show", f"{ARCHIVE_COMMIT}:{path}"], cwd=ROOT
        )
        assert hashlib.sha256(raw).hexdigest() == expected, path


def test_current_core_has_no_static_dependency_on_retired_entrypoints() -> None:
    package = ROOT / "trade_rl"
    collector = ImportCollector(package)
    for path in package.rglob("*.py"):
        assert not any(
            within_module(name, module)
            for name in collector.collect(path)
            for module in RETIRED
        ), path


def test_shared_consumers_keep_existing_development_contracts() -> None:
    for module in (
        "protocol",
        "runner",
        "metrics",
        "trial_sources",
        "phase3_sources",
    ):
        assert importlib.util.find_spec(f"{PACKAGE}.{module}") is not None


def test_frozen_development_receipts_are_unchanged() -> None:
    digests = {
        "dev-realdata-study-p2-20261029-provenance.json": "bb51892dc7be20d2c52f8d171df0f011651b2c06bd1322d705d89cdf386a8e5c",
        "issue810-phase2-trial-results-20261029.json": "840faffb1e6b047e45ee8d4eb899dca61ae77c50f921bb62ae2efe59d052c86b",
        "issue810-phase2-trial-execution1-summary-20261029.json": "51b55ce911231d0b771fb70d702acc7fd87d87fbeee930eeaa6aac39831482d0",
        "issue810-phase2-trial-provenance-20261029.json": "e3d9666b21ba4d78377dfa38d6357402eedb530271d11b12f880a09d913de53c",
    }
    for name, expected in digests.items():
        assert (
            hashlib.sha256((ROOT / "report" / name).read_bytes()).hexdigest()
            == expected
        ), name
