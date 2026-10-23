from pathlib import Path

import pytest

from tests.evaluation.test_allocation_continuation import env_for
from tests.evaluation.test_allocation_continuous_walk_forward import folds_for
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.robustness.walk_forward.sealed_test import SealedTestLedger

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")


def test_real_saved_bundle_runs_only_under_matching_sealed_fold_authorization(tmp_path):
    from trade_rl.evaluation.rl_allocation.policy_admission import (
        AllocationFoldPolicyArtifact,
    )
    from trade_rl.evaluation.rl_allocation.sealed_policy_admission import (
        SealedAllocationFoldPolicyArtifact,
        run_sealed_artifact_bound_continuous_allocation_walk_forward,
    )
    from trade_rl.evaluation.rl_allocation.training import fit_allocation_ppo
    from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy

    training = env_for(6, 8)
    policy = fit_allocation_ppo(training, total_timesteps=2, seed=0)
    bundle = Path(tmp_path) / "policy"
    digest = save_allocation_policy(bundle, policy)

    first, second = env_for(6, 8), env_for(8, 10)
    folds = folds_for(first, second)
    plan = content_digest({"plan": "sealed-allocation-oos-runtime"})
    ledger = SealedTestLedger()
    bindings = []
    for fold, env in zip(folds, (first, second), strict=True):
        artifact = AllocationFoldPolicyArtifact(
            fold_index=fold.fold_index,
            bundle_root=bundle,
            expected_digest=digest,
            expected_recipe_digest=env.recipe_digest,
        )
        access = ledger.authorize_once(
            experiment_plan_digest=plan,
            dataset_id=env.dataset.dataset_id,
            fold_index=fold.fold_index,
            test_range=fold.test,
            selected_configuration="frozen-allocation-ppo",
            selected_policy_digest=digest,
        )
        bindings.append(SealedAllocationFoldPolicyArtifact(access, artifact))

    result = run_sealed_artifact_bound_continuous_allocation_walk_forward(
        folds,
        (first, second),
        tuple(bindings),
        experiment_plan_digest=plan,
        access_ledger=ledger,
        reset_seed=7,
    )

    assert len(ledger.records) == 2
    assert ledger.consumed_access_digests == tuple(
        record.access_digest for record in ledger.records
    )
    assert result.policy_digests == (digest, digest)
    assert result.folds[0].closing_state_digest == result.folds[1].opening_state_digest
    assert second.index == second.stop_index
