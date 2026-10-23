from pathlib import Path

import pytest

from tests.evaluation.test_allocation_continuation import env_for
from tests.evaluation.test_allocation_continuous_walk_forward import folds_for

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")


def test_real_saved_allocation_bundle_is_admitted_before_continuous_oos(tmp_path):
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
        run_artifact_bound_continuous_allocation_walk_forward,
    )
    from trade_rl.evaluation.rl_allocation.training import fit_allocation_ppo
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    training = env_for(6, 8)
    policy = fit_allocation_ppo(training, total_timesteps=2, seed=0)
    root = Path(tmp_path) / "policy"
    digest = save_allocation_policy(root, policy)

    first, second = env_for(6, 8), env_for(8, 10)
    assert first.recipe_digest == second.recipe_digest == training.recipe_digest
    artifacts = tuple(
        AllocationFoldPolicyArtifact(
            fold_index=index,
            bundle_root=root,
            expected_digest=digest,
            expected_recipe_digest=env.recipe_digest,
        )
        for index, env in enumerate((first, second))
    )
    result = run_artifact_bound_continuous_allocation_walk_forward(
        folds_for(first, second),
        (first, second),
        artifacts,
        reset_seed=7,
    )

    assert result.policy_digests == (digest, digest)
    assert result.folds[0].closing_state_digest == result.folds[1].opening_state_digest
    assert second.index == second.stop_index
