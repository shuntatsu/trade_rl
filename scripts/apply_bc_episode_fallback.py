#!/usr/bin/env python3
"""Verify and apply episode-aware BC split fallback, then remove temporary files."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "trade_rl/integrations/behavior_cloning.py"
WORKFLOW = ROOT / ".github/workflows/apply-bc-episode-fallback.yml"
SCRIPT = Path(__file__).resolve()
RED_TEST = (
    "tests/learning/test_behavior_cloning_episode_training.py::"
    "test_episode_behavior_cloning_excludes_purged_rows_from_training_and_metrics"
)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def verify_red() -> None:
    print(f"+ uv run pytest -q {RED_TEST}", flush=True)
    completed = subprocess.run(
        ["uv", "run", "pytest", "-q", RED_TEST],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode == 0:
        raise RuntimeError("episode-aware fallback regression unexpectedly passed")
    if "training_sample_ids" not in completed.stdout:
        raise RuntimeError("episode-aware fallback regression failed unexpectedly")


def patch() -> None:
    text = TARGET.read_text()
    text = replace_once(
        text,
        "from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit\n",
        "from trade_rl.learning.episode_behavior_cloning import (\n"
        "    BehaviorCloningSplit,\n"
        "    behavior_cloning_split,\n"
        ")\n",
        label="episode split import",
    )
    signature = (
        "def _behavior_cloning_indices(\n"
        "    *,\n"
        "    sample_count: int,\n"
        "    config: BehaviorCloningConfig,\n"
        "    split: BehaviorCloningSplit | None,\n"
        ") -> tuple[np.ndarray, np.ndarray]:\n"
        "    if split is None:\n"
    )
    replacement = (
        "def _behavior_cloning_indices(\n"
        "    *,\n"
        "    dataset: SupervisedPolicyDataset,\n"
        "    config: BehaviorCloningConfig,\n"
        "    split: BehaviorCloningSplit | None,\n"
        ") -> tuple[np.ndarray, np.ndarray]:\n"
        "    sample_count = dataset.sample_count\n"
        "    if split is None and hasattr(dataset, \"episode_ids\"):\n"
        "        episode_split = behavior_cloning_split(\n"
        "            dataset,\n"
        "            validation_fraction=config.validation_fraction,\n"
        "        )\n"
        "        return episode_split.train_indices, episode_split.validation_indices\n"
        "    if split is None:\n"
    )
    text = replace_once(text, signature, replacement, label="split helper signature")
    text = replace_once(
        text,
        "    train_indices, validation_indices = _behavior_cloning_indices(\n"
        "        sample_count=sample_count,\n",
        "    train_indices, validation_indices = _behavior_cloning_indices(\n"
        "        dataset=dataset,\n",
        label="split helper call",
    )
    TARGET.write_text(text)


def verify_green() -> None:
    modified = [
        "trade_rl/integrations/behavior_cloning.py",
        "tests/learning/test_behavior_cloning_episode_training.py",
        "tests/learning/test_behavior_cloning.py",
        "tests/learning/test_behavior_cloning_temporal_split.py",
    ]
    run("uv", "run", "ruff", "format", *modified)
    run(
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/learning/test_behavior_cloning_episode_training.py",
        "tests/learning/test_behavior_cloning.py",
        "tests/learning/test_behavior_cloning_temporal_split.py",
        "tests/integrations/test_sb3_training.py",
    )
    run("uv", "run", "ruff", "check", *modified)
    run("uv", "run", "ruff", "format", "--check", *modified)
    run("uv", "run", "mypy", "trade_rl/integrations/behavior_cloning.py")


def main() -> None:
    verify_red()
    patch()
    verify_green()
    WORKFLOW.unlink()
    SCRIPT.unlink()


if __name__ == "__main__":
    main()
