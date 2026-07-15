from __future__ import annotations

import pytest

from trade_rl.workflows.fold_runner import (
    CheckpointPolicyEvaluation,
    select_seed_checkpoint_finalists,
)


def checkpoint(*, seed: int, digit: int, score: float) -> CheckpointPolicyEvaluation:
    return CheckpointPolicyEvaluation(
        seed=seed,
        policy_digest=f"{digit:064x}",
        score=score,
        evaluation_digest=f"{100 + digit:064x}",
    )


def test_selects_one_checkpoint_finalist_within_each_seed() -> None:
    result = select_seed_checkpoint_finalists(
        checkpoint_evaluations=(
            checkpoint(seed=0, digit=1, score=0.10),
            checkpoint(seed=0, digit=2, score=0.20),
            checkpoint(seed=1, digit=3, score=0.40),
            checkpoint(seed=1, digit=4, score=0.30),
        ),
    )

    assert tuple(item.seed for item in result) == (0, 1)
    assert tuple(item.policy_digest for item in result) == (
        f"{2:064x}",
        f"{3:064x}",
    )
    assert tuple(item.checkpoint_score for item in result) == (
        0.20,
        0.40,
    )


def test_rejects_duplicate_checkpoint_policy_identity() -> None:
    with pytest.raises(ValueError, match="checkpoint policy digests must be unique"):
        select_seed_checkpoint_finalists(
            checkpoint_evaluations=(
                checkpoint(seed=0, digit=1, score=0.10),
                checkpoint(seed=1, digit=1, score=0.20),
            )
        )


def test_breaks_score_ties_by_canonical_policy_digest() -> None:
    result = select_seed_checkpoint_finalists(
        checkpoint_evaluations=(
            checkpoint(seed=0, digit=2, score=0.10),
            checkpoint(seed=0, digit=1, score=0.10),
        ),
    )

    assert result[0].policy_digest == f"{1:064x}"


def test_selects_predeclared_top_k_within_every_seed() -> None:
    result = select_seed_checkpoint_finalists(
        checkpoint_evaluations=(
            checkpoint(seed=0, digit=1, score=0.10),
            checkpoint(seed=0, digit=2, score=0.30),
            checkpoint(seed=0, digit=3, score=0.20),
            checkpoint(seed=1, digit=4, score=0.40),
            checkpoint(seed=1, digit=5, score=0.50),
        ),
        finalists_per_seed=2,
    )

    assert tuple((item.seed, item.policy_digest) for item in result) == (
        (0, f"{2:064x}"),
        (0, f"{3:064x}"),
        (1, f"{5:064x}"),
        (1, f"{4:064x}"),
    )


@pytest.mark.parametrize("value", [0, -1, True])
def test_rejects_invalid_top_k(value: int) -> None:
    with pytest.raises(ValueError, match="finalists_per_seed"):
        select_seed_checkpoint_finalists(
            checkpoint_evaluations=(checkpoint(seed=0, digit=1, score=0.1),),
            finalists_per_seed=value,
        )
