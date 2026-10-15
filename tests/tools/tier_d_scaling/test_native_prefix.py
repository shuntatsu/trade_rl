"""Public native partial evidence stays byte-identical with cProfile enabled."""

from pathlib import Path

from tools.tier_d_nonrl.progress import ArmProgressSpec
from tools.tier_d_nonrl.tier_d_fixture import write_artificial_dataset
from tools.tier_d_nonrl.tier_d_native import (
    carrier,
    execution_folds,
    synthetic_protocol,
)
from tools.tier_d_nonrl.tier_d_protocol import market_protocol
from tools.tier_d_nonrl.tier_d_runner import declare_matrix, prepare_inputs
from tools.tier_d_scaling.probe import observe_prefix
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    run_terminal_allocation_execution,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance


def test_profile_keeps_real_failed_prefix_receipt_bytes(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset = write_artificial_dataset(dataset_root)
    protocol = synthetic_protocol(market_protocol(), dataset)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    prepared, costs = prepare_inputs(prepared_root, dataset_root, dataset, protocol)
    pair = declare_matrix(prepared, costs, protocol, build_candidate_run_provenance())[
        0
    ]
    inputs = prepared[pair.configuration]
    plan = pair.controls.nonrl_plan
    spec = ArmProgressSpec(
        pair.configuration,
        pair.account,
        pair.scenario,
        "nonrl",
        plan.digest,
        128,
        160,
        168,
    )
    folds = execution_folds((128,), 160)
    receipts = []
    for profile in (None, tmp_path / "native.prof"):
        env = carrier(
            inputs,
            inputs.costs_by_symbol[0],
            protocol,
            horizon=72,
            symbol_index=0,
            account=pair.account,
            scenario=pair.scenario,
            role="nonrl",
        )
        observation = observe_prefix(
            lambda callback: run_terminal_allocation_execution(
                folds, env, plan, on_advance=callback
            ),
            spec,
            target=3,
            profile_path=profile,
            checkpoint_every=1 if profile is not None else None,
        )
        assert observation.expected_stop and observation.count == 3
        assert observation.profile_publication_failure is None
        assert len(observation.profile_checkpoints) == (0 if profile is None else 3)
        raw = getattr(observation.error, "terminal_execution_receipt")
        assert raw["status"] == "integrity_failure"
        assert raw["settlement_complete"] is False
        reader = TerminalAllocationExecutionReceipt.from_payload(
            raw, expected_plan=plan
        )
        receipts.append(canonical_json_bytes(reader.payload))
    assert receipts[0] == receipts[1]
    assert (tmp_path / "native.prof").stat().st_size > 0
