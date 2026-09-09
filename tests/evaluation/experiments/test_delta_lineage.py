from __future__ import annotations

from dataclasses import replace

from tests.evaluation.experiments.test_delta import (
    _definition,
    _loaded_evidence,
    _plan,
    _resolved,
)
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.experiments.delta import (
    ControlledVerificationStatus,
    verify_controlled_delta,
)


def test_controlled_delta_allows_definition_bound_accepted_lineage_baseline() -> None:
    initial = _resolved()
    plan = _plan(initial)

    # A prior ACCEPT_CANDIDATE may legitimately move the Study lineage away from
    # the initial StudyPlan baseline. Task 6 owns reachability; delta verification
    # must verify the baseline bound by the frozen ExperimentDefinition.
    accepted_config = replace(initial, ppo_total_timesteps=512)
    baseline = _loaded_evidence(accepted_config, plan)

    candidate_config = replace(accepted_config, ppo_total_timesteps=768)
    candidate = _loaded_evidence(candidate_config, plan)
    definition = _definition(
        plan=plan,
        baseline=baseline,
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate=candidate_config,
    )

    verification = verify_controlled_delta(
        plan=plan,
        definition=definition,
        baseline=baseline,
        candidate=candidate,
    )

    assert verification.status is ControlledVerificationStatus.CONTROLLED
    assert verification.changed_paths == (("ppo_total_timesteps",),)
    assert verification.violations == ()
