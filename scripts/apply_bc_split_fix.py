#!/usr/bin/env python3
"""Apply and verify the temporary BC split TDD patch, then remove itself."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests/learning/test_behavior_cloning.py"
TRAINER_PATH = ROOT / "trade_rl/integrations/behavior_cloning.py"
ADAPTER_PATH = ROOT / "trade_rl/integrations/sb3_training.py"
WORKFLOW_PATH = ROOT / ".github/workflows/apply-bc-split-fix.yml"
SCRIPT_PATH = Path(__file__).resolve()
TEST_NAME = (
    "tests/learning/test_behavior_cloning.py::"
    "test_explicit_behavior_cloning_split_excludes_purged_samples"
)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def add_regression_test() -> None:
    text = TEST_PATH.read_text()
    import_marker = "from trade_rl.learning.behavior_cloning import BehaviorCloningConfig\n"
    text = replace_once(
        text,
        import_marker,
        import_marker
        + "from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit\n",
        label="behavior-cloning test import",
    )
    insertion_marker = "\n\nclass _SquashedDistribution:"
    regression = textwrap.dedent(
        '''

        class _IndexTrackingProvider:
            def __init__(self, observations: np.ndarray) -> None:
                self.observations = observations
                self.sample_count = len(observations)
                self.requested_indices: set[int] = set()

            def get(self, indices: np.ndarray) -> np.ndarray:
                self.requested_indices.update(int(index) for index in indices)
                return self.observations[indices]


        def test_explicit_behavior_cloning_split_excludes_purged_samples() -> None:
            observations = np.array(
                [
                    [-1.0, 0.0],
                    [-0.5, 0.0],
                    [10.0, 0.0],
                    [11.0, 0.0],
                    [0.5, 0.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            )
            dataset = SupervisedPolicyDataset(
                observations=observations,
                actions=np.clip(observations[:, :1], -1.0, 1.0),
                dataset_id="a" * 64,
                train_start=0,
                train_stop=7,
                environment_digest="b" * 64,
                action_spec_digest="c" * 64,
                teacher_config_digest="d" * 64,
            )
            split = BehaviorCloningSplit(
                train_indices=np.asarray([0, 1], dtype=np.int64),
                validation_indices=np.asarray([4, 5], dtype=np.int64),
                train_episode_ids=np.asarray([0], dtype=np.int64),
                validation_episode_ids=np.asarray([2], dtype=np.int64),
                purged_indices=np.asarray([2, 3], dtype=np.int64),
                purged_episode_ids=np.asarray([1], dtype=np.int64),
            )
            provider = _IndexTrackingProvider(observations)

            result = pretrain_policy(
                _LinearPolicy(),
                dataset,
                config=BehaviorCloningConfig(
                    epochs=2,
                    learning_rate=0.01,
                    batch_size=2,
                    validation_fraction=1 / 3,
                ),
                seed=19,
                observation_provider=provider,
                split=split,
            )

            assert provider.requested_indices == {0, 1, 4, 5}
            assert result.sample_count == 6
            assert result.validation_sample_count == 2
        '''
    )
    text = replace_once(
        text,
        insertion_marker,
        regression + insertion_marker,
        label="behavior-cloning regression insertion",
    )
    TEST_PATH.write_text(text)


def verify_red() -> None:
    print(f"+ uv run pytest -q {TEST_NAME}", flush=True)
    completed = subprocess.run(
        ["uv", "run", "pytest", "-q", TEST_NAME],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode == 0:
        raise RuntimeError("RED regression unexpectedly passed before implementation")
    if "unexpected keyword argument 'split'" not in completed.stdout:
        raise RuntimeError("RED regression failed for an unexpected reason")


def patch_trainer() -> None:
    text = TRAINER_PATH.read_text()
    import_marker = (
        "from trade_rl.learning.behavior_cloning import (\n"
        "    BehaviorCloningConfig,\n"
        "    BehaviorCloningResult,\n"
        "    ObservationBatchProvider,\n"
        ")\n"
    )
    text = replace_once(
        text,
        import_marker,
        import_marker
        + "from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit\n",
        label="behavior-cloning trainer import",
    )

    helper_marker = "\n\ndef pretrain_policy(\n"
    helper = textwrap.dedent(
        '''

        def _behavior_cloning_indices(
            *,
            sample_count: int,
            config: BehaviorCloningConfig,
            split: BehaviorCloningSplit | None,
        ) -> tuple[np.ndarray, np.ndarray]:
            if split is None:
                validation_count = (
                    0
                    if config.validation_fraction == 0.0
                    else max(
                        1,
                        int(math.floor(sample_count * config.validation_fraction)),
                    )
                )
                train_count = sample_count - validation_count
                if train_count <= 0:
                    raise ValueError(
                        "behavior-cloning validation leaves no training samples"
                    )
                indices = np.arange(sample_count, dtype=np.int64)
                return indices[:train_count], indices[train_count:]

            train_indices = np.asarray(split.train_indices, dtype=np.int64)
            validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
            purged_indices = np.asarray(split.purged_indices, dtype=np.int64)
            for name, indices in (
                ("training", train_indices),
                ("validation", validation_indices),
                ("purged", purged_indices),
            ):
                if np.any(indices < 0) or np.any(indices >= sample_count):
                    raise ValueError(
                        f"behavior-cloning {name} index is outside the dataset"
                    )
            partition = np.concatenate(
                (train_indices, validation_indices, purged_indices)
            )
            expected_partition = np.arange(sample_count, dtype=np.int64)
            if partition.size != sample_count or not np.array_equal(
                np.sort(partition), expected_partition
            ):
                raise ValueError(
                    "explicit behavior-cloning split must partition the dataset"
                )
            expected_validation_count = (
                0
                if config.validation_fraction == 0.0
                else max(
                    1,
                    int(math.floor(sample_count * config.validation_fraction)),
                )
            )
            if validation_indices.size != expected_validation_count:
                raise ValueError(
                    "explicit behavior-cloning split disagrees with "
                    "validation_fraction"
                )
            return train_indices, validation_indices
        '''
    )
    text = replace_once(
        text,
        helper_marker,
        helper + helper_marker,
        label="behavior-cloning split helper",
    )
    signature_marker = "    seed: int,\n    observation_provider:"
    text = replace_once(
        text,
        signature_marker,
        "    seed: int,\n    split: BehaviorCloningSplit | None = None,\n"
        "    observation_provider:",
        label="behavior-cloning trainer signature",
    )
    split_block = (
        "    sample_count = dataset.sample_count\n"
        "    validation_count = (\n"
        "        0\n"
        "        if config.validation_fraction == 0.0\n"
        "        else max(1, int(math.floor(sample_count * config.validation_fraction)))\n"
        "    )\n"
        "    train_count = sample_count - validation_count\n"
        "    if train_count <= 0:\n"
        "        raise ValueError(\"behavior-cloning validation leaves no training samples\")\n"
        "    all_indices = np.arange(sample_count, dtype=np.int64)\n"
        "    train_indices = all_indices[:train_count]\n"
        "    validation_indices = all_indices[train_count:]\n"
    )
    replacement = (
        "    sample_count = dataset.sample_count\n"
        "    train_indices, validation_indices = _behavior_cloning_indices(\n"
        "        sample_count=sample_count,\n"
        "        config=config,\n"
        "        split=split,\n"
        "    )\n"
        "    train_count = int(train_indices.size)\n"
        "    validation_count = int(validation_indices.size)\n"
        "    all_indices = np.concatenate((train_indices, validation_indices))\n"
    )
    text = replace_once(
        text,
        split_block,
        replacement,
        label="behavior-cloning trainer split block",
    )
    text = replace_once(
        text,
        '"""Fit legacy MSE or hierarchical BC with a chronological validation tail."""',
        '"""Fit BC using a validated split while excluding purged samples."""',
        label="behavior-cloning trainer docstring",
    )
    TRAINER_PATH.write_text(text)


def patch_sb3_adapter() -> None:
    text = ADAPTER_PATH.read_text()
    marker = (
        "                    cloning = pretrain_policy(\n"
        "                        model.policy,\n"
        "                        teacher_dataset,\n"
        "                        config=cloning_config,\n"
    )
    replacement = marker + "                        split=episode_split,\n"
    text = replace_once(
        text,
        marker,
        replacement,
        label="SB3 behavior-cloning call",
    )
    ADAPTER_PATH.write_text(text)


def verify_green() -> None:
    modified = [
        "trade_rl/integrations/behavior_cloning.py",
        "trade_rl/integrations/sb3_training.py",
        "tests/learning/test_behavior_cloning.py",
        "tests/learning/test_behavior_cloning_temporal_split.py",
    ]
    run("uv", "run", "ruff", "format", *modified)
    run(
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/learning/test_behavior_cloning.py",
        "tests/learning/test_behavior_cloning_temporal_split.py",
        "tests/integrations/test_sb3_training.py",
    )
    run("uv", "run", "ruff", "check", *modified)
    run("uv", "run", "ruff", "format", "--check", *modified)
    run(
        "uv",
        "run",
        "mypy",
        "trade_rl/integrations/behavior_cloning.py",
        "trade_rl/integrations/sb3_training.py",
    )


def main() -> None:
    add_regression_test()
    verify_red()
    patch_trainer()
    patch_sb3_adapter()
    verify_green()
    WORKFLOW_PATH.unlink()
    SCRIPT_PATH.unlink()


if __name__ == "__main__":
    main()
