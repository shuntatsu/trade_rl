from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_training import (
    SharedPortfolioPPOConfig,
    fit_shared_portfolio_ppo,
)
from trade_rl.strategies.rl.portfolio_transform import (
    fit_portfolio_observation_transform,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _dataset() -> MarketDataset:
    close = np.full((8, 2), 100.0, dtype=np.float64)
    zeros = np.zeros_like(close)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        len(close)
    ) * np.timedelta64(1, "h")
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("AAA", "BBB"),
        timestamps=timestamps,
        features=np.arange(16, dtype=np.float32).reshape(8, 2, 1),
        global_features=np.arange(8, dtype=np.float32).reshape(8, 1),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=zeros.copy(),
        funding_due=None,
        borrow_rate=zeros.copy(),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((8, 2, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zeros.copy(),
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(2, dtype=np.float64),
    ).with_content_identity({"fixture": "portfolio-inference-bundle"})


def _environment(
    dataset: MarketDataset,
    start_index: int,
    stop_index: int,
    *,
    target_abs_weights: tuple[float, ...] = (0.5, 0.5),
    execution_cost: ExecutionCostConfig | None = None,
) -> SharedPortfolioGymEnv:
    stop = _START + timedelta(hours=stop_index)
    bound = BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=_START + timedelta(hours=start_index),
            evaluation_stop_exclusive=stop,
            terminal_valuation="settled",
            economics_digest="b" * 64,
            risk_digest="c" * 64,
            deployment_recipe_digest="d" * 64,
            maximum_drawdown=0.20,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=(stop_index - start_index) * 3_600,
            rollout_steps=4,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=bound,
        target_abs_weights=target_abs_weights,
        execution_cost=(
            ExecutionCostConfig.zero() if execution_cost is None else execution_cost
        ),
        risk_config=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.20,
            drawdown_stop=0.20,
        ),
    )


def _train_two_rollouts(root: Path) -> tuple[Any, MarketDataset, SharedPortfolioGymEnv]:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _dataset()
    training_env = _environment(dataset, 0, 5)
    transform = fit_portfolio_observation_transform(training_env)
    result = fit_shared_portfolio_ppo(
        training_env,
        transform=transform,
        output_root=root / "attempt",
        config=SharedPortfolioPPOConfig(
            total_timesteps=8,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            reset_seed=29,
            ppo_seed=31,
        ),
    )
    return result, dataset, training_env


def _publish(result: Any, destination: Path) -> Any:
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    return publish_portfolio_inference_bundle(
        result.output_root,
        destination,
        expected_manifest_sha256=result.manifest_sha256,
        expected_completion_sha256=result.completion_sha256,
        expected_training_context_sha256=result.training_context_sha256,
        expected_checkpoint_sha256=result.final_checkpoint_sha256,
    )


def _load(publication: Any, runtime_env: SharedPortfolioGymEnv) -> Any:
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        load_portfolio_inference_bundle,
    )

    return load_portfolio_inference_bundle(
        publication.root,
        expected_bundle_sha256=publication.bundle_sha256,
        expected_manifest_sha256=publication.training_manifest_sha256,
        expected_completion_sha256=publication.training_completion_sha256,
        expected_training_context_sha256=publication.training_context_sha256,
        expected_checkpoint_sha256=publication.checkpoint_sha256,
        runtime_env=runtime_env,
    )


@pytest.fixture(scope="module")
def completed_attempt(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, MarketDataset, SharedPortfolioGymEnv]:
    return _train_two_rollouts(tmp_path_factory.mktemp("bundle-attempt"))


@pytest.fixture(scope="module")
def published_bundle(
    completed_attempt: tuple[Any, MarketDataset, SharedPortfolioGymEnv],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any]:
    result, dataset, training_env = completed_attempt
    publication = _publish(
        result, tmp_path_factory.mktemp("bundle-publication") / "bundle"
    )
    return result, dataset, training_env, publication


def test_loaded_policy_decides_from_matching_portfolio_snapshot(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.simulation.accounting import BookState
    from trade_rl.strategies.portfolio import PortfolioTargetProposal
    from trade_rl.strategies.portfolio_observation import (
        build_portfolio_decision_snapshot,
    )

    _, dataset, _, publication = published_bundle
    runtime_env = _environment(dataset, 5, 7)
    policy = _load(publication, runtime_env)
    captured: list[dict[str, np.ndarray]] = []

    class FixedPrediction:
        def predict(
            self, observation: object, *, deterministic: bool
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            assert isinstance(observation, dict)
            captured.append(observation)
            return np.asarray([0, 3], dtype=np.int64), None

    policy = replace(policy, _policy=FixedPrediction())
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital=1_000.0,
        initial_prices=dataset.resolved_array("mark_price")[5],
        contract_multipliers=dataset.contract_multipliers,
    )
    snapshot = build_portfolio_decision_snapshot(
        dataset,
        index=5,
        agent_stop_index=6,
        book=book,
        active_orders=(),
        risk_config=runtime_env.risk_config,
    )

    proposal = policy.decide_portfolio(snapshot)

    assert isinstance(proposal, PortfolioTargetProposal)
    assert proposal.target_weights == (0.0, 0.5)
    assert proposal.hold_current_quantity_symbols == (0,)
    assert len(captured) == 1
    assert tuple(captured[0]) == runtime_env.observation_keys


@pytest.mark.parametrize("field", ["symbols", "timestamp_ns", "agent_stop_index"])
def test_loaded_policy_rejects_snapshot_clock_or_roster_before_prediction(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    field: str,
) -> None:
    from trade_rl.simulation.accounting import BookState
    from trade_rl.strategies.portfolio_observation import (
        build_portfolio_decision_snapshot,
    )

    _, dataset, _, publication = published_bundle
    runtime_env = _environment(dataset, 5, 7)
    policy = _load(publication, runtime_env)
    calls = 0

    class MustNotPredict:
        def predict(self, *args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("an invalid snapshot reached the policy")

    policy = replace(policy, _policy=MustNotPredict())
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital=1_000.0,
        initial_prices=dataset.resolved_array("mark_price")[5],
        contract_multipliers=dataset.contract_multipliers,
    )
    snapshot = build_portfolio_decision_snapshot(
        dataset,
        index=5,
        agent_stop_index=6,
        book=book,
        active_orders=(),
        risk_config=runtime_env.risk_config,
    )
    if field == "symbols":
        invalid = replace(snapshot, symbols=("BBB", "AAA"))
    elif field == "timestamp_ns":
        invalid = replace(snapshot, timestamp_ns=snapshot.timestamp_ns + 1)
    else:
        invalid = replace(snapshot, agent_stop_index=7, remaining_decisions=2)

    with pytest.raises(ValueError, match="snapshot"):
        policy.decide_portfolio(invalid)
    assert calls == 0


def test_published_policy_replay_pairs_one_loaded_dataset_and_native_session(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from trade_rl.evaluation.portfolio_inference import (
            run_published_portfolio_inference_replay,
        )
    except ModuleNotFoundError:
        pytest.fail("the paired published-policy replay adapter is not implemented")

    from trade_rl.evaluation import portfolio_inference as inference_module
    from trade_rl.simulation.stateful import portfolio_session as session_module
    from trade_rl.strategies.rl import portfolio_env as env_module
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        PortfolioInferencePolicy,
    )

    _, dataset, _, bundle = published_bundle
    data_artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    runtime_env = _environment(dataset, 5, 7)
    loaded_pairs: list[tuple[Any, MarketDataset]] = []
    validation_envs: list[SharedPortfolioGymEnv] = []
    native_sessions: list[tuple[MarketDataset, dict[str, object]]] = []
    policy_calls = 0

    load_dataset = inference_module.load_published_market_dataset_artifact

    def load_dataset_spy(root: str | Path) -> tuple[Any, MarketDataset]:
        pair = load_dataset(root)
        loaded_pairs.append(pair)
        return pair

    env_init = env_module.SharedPortfolioGymEnv.__init__

    def env_init_spy(
        self: SharedPortfolioGymEnv,
        dataset_value: MarketDataset,
        *args: object,
        **kwargs: object,
    ) -> None:
        env_init(self, dataset_value, *args, **kwargs)
        validation_envs.append(self)

    def reset_must_not_be_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("validation environment must remain unreset")

    session_init = session_module.SharedPortfolioSession.__init__

    def session_init_spy(
        self: object, dataset_value: MarketDataset, **kwargs: object
    ) -> None:
        native_sessions.append((dataset_value, kwargs))
        session_init(self, dataset_value, **kwargs)

    decide = PortfolioInferencePolicy.decide_portfolio

    def decide_spy(self: PortfolioInferencePolicy, snapshot: object) -> object:
        nonlocal policy_calls
        policy_calls += 1
        return decide(self, snapshot)

    monkeypatch.setattr(
        inference_module,
        "load_published_market_dataset_artifact",
        load_dataset_spy,
    )
    monkeypatch.setattr(env_module.SharedPortfolioGymEnv, "__init__", env_init_spy)
    monkeypatch.setattr(
        env_module.SharedPortfolioGymEnv, "reset", reset_must_not_be_called
    )
    monkeypatch.setattr(
        session_module.SharedPortfolioSession, "__init__", session_init_spy
    )
    monkeypatch.setattr(PortfolioInferencePolicy, "decide_portfolio", decide_spy)

    returned_artifact, replay = run_published_portfolio_inference_replay(
        dataset_artifact_root=data_artifact.root,
        expected_dataset_artifact_sha256=data_artifact.artifact_digest,
        bundle_root=bundle.root,
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_manifest_sha256=bundle.training_manifest_sha256,
        expected_completion_sha256=bundle.training_completion_sha256,
        expected_training_context_sha256=bundle.training_context_sha256,
        expected_checkpoint_sha256=bundle.checkpoint_sha256,
        bound=runtime_env.bound,
        start_index=5,
        stop_index=7,
        initial_capital=1_000.0,
        target_abs_weights=(0.5, 0.5),
        execution_cost=runtime_env.execution_cost,
        risk_config=runtime_env.risk_config,
        market_order_profile=None,
        capture_ledger_evidence=True,
    )

    assert returned_artifact.artifact_digest == data_artifact.artifact_digest
    assert len(loaded_pairs) == 1
    assert len(validation_envs) == 1
    assert validation_envs[0]._session is None
    assert len(native_sessions) == 1
    loaded_dataset = loaded_pairs[0][1]
    assert validation_envs[0].dataset is loaded_dataset
    assert native_sessions[0][0] is loaded_dataset
    assert native_sessions[0][1]["execution_cost"] is runtime_env.execution_cost
    assert native_sessions[0][1]["risk"].config is runtime_env.risk_config
    assert policy_calls == len(replay.decisions) == 1
    assert replay.ledger_evidence is not None
    assert all(quantity == 0 for quantity in replay.book.exact_quantities)
    assert replay.ledger_evidence.intervals.active_order_remainders == ()


@pytest.mark.parametrize("mismatch", ["dataset", "bundle"])
def test_paired_replay_rejects_wrong_dataset_or_bundle_identity_before_session(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    from trade_rl.evaluation import portfolio_inference as inference_module
    from trade_rl.simulation.stateful import portfolio_session as session_module
    from trade_rl.strategies.rl import portfolio_env as env_module
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    _, dataset, _, bundle = published_bundle
    artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    runtime_env = _environment(dataset, 5, 7)
    loaded_pairs: list[tuple[Any, MarketDataset]] = []
    validation_envs: list[SharedPortfolioGymEnv] = []
    policy_loads = 0
    session_starts = 0

    original_loader = inference_module.load_published_market_dataset_artifact

    def loader(root: str | Path) -> tuple[Any, MarketDataset]:
        pair = original_loader(root)
        loaded_pairs.append(pair)
        return pair

    original_env_init = env_module.SharedPortfolioGymEnv.__init__

    def env_init_spy(
        self: SharedPortfolioGymEnv,
        dataset_value: MarketDataset,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_env_init(self, dataset_value, *args, **kwargs)
        validation_envs.append(self)

    def reset_must_not_be_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("identity validation must not reset the Gym shell")

    def deserialize_must_not_be_called(*args: object, **kwargs: object) -> object:
        nonlocal policy_loads
        policy_loads += 1
        raise AssertionError("identity mismatch must precede PPO deserialization")

    def session_must_not_start(*args: object, **kwargs: object) -> None:
        nonlocal session_starts
        session_starts += 1
        raise AssertionError("identity mismatch must precede native session creation")

    monkeypatch.setattr(
        inference_module, "load_published_market_dataset_artifact", loader
    )
    monkeypatch.setattr(env_module.SharedPortfolioGymEnv, "__init__", env_init_spy)
    monkeypatch.setattr(
        env_module.SharedPortfolioGymEnv, "reset", reset_must_not_be_called
    )
    monkeypatch.setattr(
        bundle_module, "_load_policy_from_verified_path", deserialize_must_not_be_called
    )
    monkeypatch.setattr(
        session_module.SharedPortfolioSession, "__init__", session_must_not_start
    )

    expected_dataset_digest = artifact.artifact_digest
    expected_bundle_digest = bundle.bundle_sha256
    if mismatch == "dataset":
        expected_dataset_digest = "0" * 64
        if expected_dataset_digest == artifact.artifact_digest:
            expected_dataset_digest = "1" * 64
    else:
        expected_bundle_digest = "0" * 64
        if expected_bundle_digest == bundle.bundle_sha256:
            expected_bundle_digest = "1" * 64

    with pytest.raises(ValueError):
        inference_module.run_published_portfolio_inference_replay(
            dataset_artifact_root=artifact.root,
            expected_dataset_artifact_sha256=expected_dataset_digest,
            bundle_root=bundle.root,
            expected_bundle_sha256=expected_bundle_digest,
            expected_manifest_sha256=bundle.training_manifest_sha256,
            expected_completion_sha256=bundle.training_completion_sha256,
            expected_training_context_sha256=bundle.training_context_sha256,
            expected_checkpoint_sha256=bundle.checkpoint_sha256,
            bound=runtime_env.bound,
            start_index=5,
            stop_index=7,
            initial_capital=1_000.0,
            target_abs_weights=(0.5, 0.5),
            execution_cost=runtime_env.execution_cost,
            risk_config=runtime_env.risk_config,
            market_order_profile=None,
            capture_ledger_evidence=True,
        )

    assert len(loaded_pairs) == 1
    assert policy_loads == 0
    assert session_starts == 0
    if mismatch == "dataset":
        assert validation_envs == []
    else:
        assert len(validation_envs) == 1
        assert validation_envs[0]._session is None


def test_paired_fixed_action_replay_routes_positive_costs_through_native_ledger(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed synthetic policy checks cost routing, not learned behavior."""
    from trade_rl.evaluation import portfolio_inference as inference_module
    from trade_rl.simulation.stateful import portfolio_session as session_module
    from trade_rl.strategies.portfolio_observation import PortfolioDecisionSnapshot
    from trade_rl.strategies.portfolio_policy_codec import (
        ACTION_LONG,
        ACTION_SHORT,
        portfolio_action_to_proposal,
    )
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    _, source_dataset, _, bundle = published_bundle
    fee_rate = np.full_like(source_dataset.close, 0.001, dtype=np.float64)
    funding_rate = np.full_like(source_dataset.close, 0.01, dtype=np.float64)
    funding_due = np.zeros_like(source_dataset.close, dtype=np.bool_)
    funding_due[6, :] = True
    borrow_rate = np.zeros_like(source_dataset.close, dtype=np.float64)
    borrow_rate[6, 1] = 0.0876
    cost_dataset = replace(
        source_dataset,
        fee_rate=fee_rate,
        maker_fee_rate=np.zeros_like(fee_rate),
        taker_fee_rate=np.zeros_like(fee_rate),
        spread_rate=np.zeros_like(fee_rate),
        funding_rate=funding_rate,
        funding_due=funding_due,
        borrow_rate=borrow_rate,
        borrow_available=np.ones_like(funding_due),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "portfolio-inference-cost-routing"})
    data_artifact = publish_market_dataset_artifact(
        tmp_path / "cost-dataset", cost_dataset
    )
    execution_cost = replace(
        ExecutionCostConfig.zero(),
        borrow_rate_multiplier=1.0,
    )
    runtime_env = _environment(
        cost_dataset,
        5,
        7,
        target_abs_weights=(0.4, 0.2),
        execution_cost=execution_cost,
    )
    actions_seen: list[int] = []
    validation_envs: list[SharedPortfolioGymEnv] = []
    session_datasets: list[MarketDataset] = []
    loaded_pairs: list[tuple[Any, MarketDataset]] = []

    class FixedActionPolicy:
        def decide_portfolio(self, snapshot: PortfolioDecisionSnapshot) -> object:
            actions_seen.append(snapshot.index)
            return portfolio_action_to_proposal(
                np.asarray([ACTION_LONG, ACTION_SHORT], dtype=np.int64),
                snapshot,
                (0.4, 0.2),
            )

    fixed_policy = FixedActionPolicy()

    def load_fixed_policy(root: str | Path, **kwargs: object) -> FixedActionPolicy:
        del root
        validation_env = kwargs["runtime_env"]
        assert isinstance(validation_env, SharedPortfolioGymEnv)
        validation_envs.append(validation_env)
        assert validation_env.execution_cost is execution_cost
        assert validation_env.dataset.fee_rate is not None
        assert validation_env.dataset.fee_rate[6, 0] == 0.001
        return fixed_policy

    original_loader = inference_module.load_published_market_dataset_artifact

    def dataset_loader(root: str | Path) -> tuple[Any, MarketDataset]:
        pair = original_loader(root)
        loaded_pairs.append(pair)
        return pair

    original_session_init = session_module.SharedPortfolioSession.__init__

    def session_init_spy(
        self: object, dataset_value: MarketDataset, **kwargs: object
    ) -> None:
        session_datasets.append(dataset_value)
        original_session_init(self, dataset_value, **kwargs)

    monkeypatch.setattr(
        bundle_module, "load_portfolio_inference_bundle", load_fixed_policy
    )
    monkeypatch.setattr(
        inference_module, "load_published_market_dataset_artifact", dataset_loader
    )
    monkeypatch.setattr(
        session_module.SharedPortfolioSession, "__init__", session_init_spy
    )

    returned_artifact, replay = (
        inference_module.run_published_portfolio_inference_replay(
            dataset_artifact_root=data_artifact.root,
            expected_dataset_artifact_sha256=data_artifact.artifact_digest,
            bundle_root=bundle.root,
            expected_bundle_sha256=bundle.bundle_sha256,
            expected_manifest_sha256=bundle.training_manifest_sha256,
            expected_completion_sha256=bundle.training_completion_sha256,
            expected_training_context_sha256=bundle.training_context_sha256,
            expected_checkpoint_sha256=bundle.checkpoint_sha256,
            bound=runtime_env.bound,
            start_index=5,
            stop_index=7,
            initial_capital=1_000.0,
            target_abs_weights=(0.4, 0.2),
            execution_cost=execution_cost,
            risk_config=runtime_env.risk_config,
            market_order_profile=None,
            capture_ledger_evidence=True,
        )
    )

    assert returned_artifact.artifact_digest == data_artifact.artifact_digest
    assert len(loaded_pairs) == 1
    assert len(validation_envs) == len(session_datasets) == 1
    assert validation_envs[0]._session is None
    assert loaded_pairs[0][1] is validation_envs[0].dataset
    assert session_datasets[0] is loaded_pairs[0][1]
    assert actions_seen == [5]
    assert replay.ledger_evidence is not None
    ledger = replay.ledger_evidence.intervals
    assert ledger.active_order_remainders == ()
    assert ledger.terminal_exact_quantities == ("0", "0")

    fills = tuple(
        (
            event.processing_index,
            event.symbol_index,
            event.filled_quantity,
            event.execution_price,
        )
        for interval in ledger.intervals
        for event in interval.order_events
        if event.filled_quantity != 0.0
    )
    assert fills == (
        (6, 0, 4.0, 100.0),
        (6, 1, -2.0, 100.0),
        (7, 0, -4.0, 100.0),
        (7, 1, 2.0, 100.0),
    )

    entry_and_exit_fee = Fraction(2 * (400 + 200), 1_000)
    funding_cash_delta = -Fraction(400 - 200, 100)
    borrow_cash_delta = Fraction(200) * Fraction(876, 10_000) / 8_760
    expected_final_cash = (
        Fraction(1_000) - entry_and_exit_fee + funding_cash_delta - borrow_cash_delta
    )
    assert expected_final_cash == Fraction(498_399, 500)
    assert ledger.final_cash == pytest.approx(float(expected_final_cash), abs=1e-12)
    assert ledger.final_total_cost == pytest.approx(
        float(entry_and_exit_fee), abs=1e-12
    )
    assert ledger.final_funding_pnl == pytest.approx(
        float(funding_cash_delta), abs=1e-12
    )
    assert ledger.final_borrow_cost == pytest.approx(
        float(borrow_cash_delta), abs=1e-12
    )


def _write_canonical(path: Path, value: object) -> bytes:
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    return payload


def _copy_and_repin_attempt(
    result: Any,
    destination: Path,
    *,
    mutate_manifest: Any | None = None,
    mutate_context: Any | None = None,
) -> tuple[Path, str, str]:
    shutil.copytree(result.output_root, destination)
    manifest_path = destination / "manifest.json"
    context_path = destination / "training_context.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    if mutate_context is not None:
        mutate_context(context)
    manifest_bytes = _write_canonical(manifest_path, manifest)
    context_bytes = _write_canonical(context_path, context)
    return (
        destination,
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(context_bytes).hexdigest(),
    )


def _copy_and_repin_bundle(
    publication: Any,
    destination: Path,
    *,
    mutate_training_manifest: Any | None = None,
    mutate_context: Any | None = None,
    mutate_bundle_manifest: Any | None = None,
) -> Any:
    shutil.copytree(publication.root, destination)
    manifest_path = destination / "manifest.json"
    training_manifest_path = destination / "training_manifest.json"
    context_path = destination / "training_context.json"
    bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if mutate_training_manifest is not None:
        mutate_training_manifest(training_manifest)
    if mutate_context is not None:
        mutate_context(context)
    training_manifest_bytes = _write_canonical(
        training_manifest_path, training_manifest
    )
    context_bytes = _write_canonical(context_path, context)
    manifest_sha256 = hashlib.sha256(training_manifest_bytes).hexdigest()
    context_sha256 = hashlib.sha256(context_bytes).hexdigest()
    bundle_manifest["training_attempt"]["manifest_sha256"] = manifest_sha256
    bundle_manifest["training_attempt"]["training_context_sha256"] = context_sha256
    bundle_manifest["training_attempt"]["training_context_digest"] = context[
        "training_context_digest"
    ]
    bundle_manifest["files"]["training_manifest.json"] = {
        "sha256": manifest_sha256,
        "size_bytes": len(training_manifest_bytes),
    }
    bundle_manifest["files"]["training_context.json"] = {
        "sha256": context_sha256,
        "size_bytes": len(context_bytes),
    }
    if mutate_bundle_manifest is not None:
        mutate_bundle_manifest(bundle_manifest)
    _write_canonical(manifest_path, bundle_manifest)
    return publication.__class__(
        root=destination,
        bundle_sha256=content_digest(bundle_manifest),
        training_manifest_sha256=manifest_sha256,
        training_completion_sha256=publication.training_completion_sha256,
        training_context_sha256=context_sha256,
        checkpoint_sha256=publication.checkpoint_sha256,
    )


def _publish_rebound_attempt(result: Any, attempt_root: Path, destination: Path) -> Any:
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    manifest_sha256 = hashlib.sha256(
        (attempt_root / "manifest.json").read_bytes()
    ).hexdigest()
    context_sha256 = hashlib.sha256(
        (attempt_root / "training_context.json").read_bytes()
    ).hexdigest()
    return publish_portfolio_inference_bundle(
        attempt_root,
        destination,
        expected_manifest_sha256=manifest_sha256,
        expected_completion_sha256=result.completion_sha256,
        expected_training_context_sha256=context_sha256,
        expected_checkpoint_sha256=result.final_checkpoint_sha256,
    )


def test_bundle_round_trip_is_strict_and_deterministic(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
) -> None:
    result, dataset, training_env, publication = published_bundle
    assert result.rollout_count == 2
    assert result.final_checkpoint_sha256
    assert result.manifest_sha256
    assert result.training_context_sha256
    assert result.completion_sha256

    altered_attempt = tmp_path / "altered-attempt"
    shutil.copytree(result.output_root, altered_attempt)
    completion = json.loads(
        (altered_attempt / "completion.json").read_text(encoding="utf-8")
    )
    checkpoint_path = altered_attempt / completion["final_checkpoint_path"]
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
    altered_output = tmp_path / "tampered-checkpoint-bundle"
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        publish_portfolio_inference_bundle(
            altered_attempt,
            altered_output,
            expected_manifest_sha256=result.manifest_sha256,
            expected_completion_sha256=result.completion_sha256,
            expected_training_context_sha256=result.training_context_sha256,
            expected_checkpoint_sha256=result.final_checkpoint_sha256,
        )
    assert not altered_output.exists()

    runtime_env = _environment(dataset, 5, 7)
    policy = _load(publication, runtime_env)
    raw_observation, _ = runtime_env.reset(seed=41)
    original_features = raw_observation["features"].copy()
    action_one = policy.predict(raw_observation)
    action_two = policy.predict(raw_observation)

    assert publication.training_context_sha256 == result.training_context_sha256
    assert policy.runtime_binding.runtime_start_index == 5
    assert np.array_equal(action_one, action_two)
    assert action_one.shape == (2,)
    assert np.all((0 <= action_one) & (action_one < 4))
    assert np.array_equal(raw_observation["features"], original_features)
    assert not hasattr(policy, "learn")
    assert not hasattr(policy, "model")
    assert runtime_env._session is not None
    assert runtime_env._session.current_index == 5
    assert training_env._session is not None

    mismatched_env = _environment(dataset, 5, 7, target_abs_weights=(0.4, 0.6))
    with pytest.MonkeyPatch.context() as monkeypatch:
        from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

        def must_not_deserialize(*args: object, **kwargs: object) -> object:
            raise AssertionError("runtime binding rejection must precede PPO load")

        monkeypatch.setattr(
            bundle_module, "_load_policy_from_verified_path", must_not_deserialize
        )
        with pytest.raises(ValueError, match="financial or target-weight context"):
            _load(publication, mismatched_env)

    tampered_root = tmp_path / "tampered"
    shutil.copytree(publication.root, tampered_root)
    context_path = tampered_root / "training_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["transform_digest"] = "0" * 64
    context_path.write_text(json.dumps(context), encoding="utf-8")

    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        raise AssertionError("metadata rejection must precede PPO deserialization")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            bundle_module,
            "_load_policy_from_verified_path",
            must_not_deserialize,
        )
        with pytest.raises(
            ValueError, match="training context.*digest|digest.*context"
        ):
            _load(
                publication.__class__(
                    root=tampered_root,
                    bundle_sha256=publication.bundle_sha256,
                    training_manifest_sha256=publication.training_manifest_sha256,
                    training_completion_sha256=publication.training_completion_sha256,
                    training_context_sha256=publication.training_context_sha256,
                    checkpoint_sha256=publication.checkpoint_sha256,
                ),
                runtime_env,
            )


def _mutate_action_code_boolean(context: dict[str, object]) -> None:
    layout = context["policy_layout"]
    assert isinstance(layout, dict)
    layout["action_codes"] = [False, True, 2, 3]


def _mutate_space_dimension_boolean(context: dict[str, object]) -> None:
    layout = context["policy_layout"]
    assert isinstance(layout, dict)
    observation_space = layout["observation_space"]
    changed = False

    def visit(value: object) -> None:
        nonlocal changed
        if changed:
            return
        if isinstance(value, dict):
            shape = value.get("shape")
            if isinstance(shape, list):
                for index, item in enumerate(shape):
                    if type(item) is int and item == 1:
                        shape[index] = True
                        changed = True
                        return
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(observation_space)
    assert changed, "synthetic observation layout must contain a unit dimension"


def _mutate_boolean_gamma(manifest: dict[str, object]) -> None:
    effective = manifest["effective_config"]
    assert isinstance(effective, dict)
    effective["gamma"] = True


def _mutate_context_boolean_gamma(context: dict[str, object]) -> None:
    config = context["training_config"]
    assert isinstance(config, dict)
    effective = config["effective_config"]
    assert isinstance(effective, dict)
    effective["gamma"] = True


def _mutate_boolean_n_envs(manifest: dict[str, object]) -> None:
    effective = manifest["effective_config"]
    assert isinstance(effective, dict)
    effective["n_envs"] = True


def _mutate_context_boolean_n_envs(context: dict[str, object]) -> None:
    config = context["training_config"]
    assert isinstance(config, dict)
    effective = config["effective_config"]
    assert isinstance(effective, dict)
    effective["n_envs"] = True


def _malformed_coercion_cases() -> list[tuple[str, Any | None, Any | None]]:
    return [
        ("gamma-bool", _mutate_boolean_gamma, _mutate_context_boolean_gamma),
        ("n-envs-bool", _mutate_boolean_n_envs, _mutate_context_boolean_n_envs),
        ("action-code-bools", None, _mutate_action_code_boolean),
        ("space-dimension-bool", None, _mutate_space_dimension_boolean),
    ]


def test_re_pinned_python_bool_numeric_metadata_is_rejected_before_publish(
    completed_attempt: tuple[Any, MarketDataset, SharedPortfolioGymEnv],
    tmp_path: Path,
) -> None:
    result, _, _ = completed_attempt
    for name, mutate_manifest, mutate_context in _malformed_coercion_cases():
        attempt, _, _ = _copy_and_repin_attempt(
            result,
            tmp_path / f"malformed-{name}",
            mutate_manifest=mutate_manifest,
            mutate_context=mutate_context,
        )
        output = tmp_path / f"must-not-publish-{name}"
        with pytest.raises(ValueError):
            _publish_rebound_attempt(result, attempt, output)
        assert not output.exists()


def test_re_pinned_python_bool_numeric_metadata_is_rejected_before_load(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset, _, publication = published_bundle
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        raise AssertionError("malformed metadata must be rejected before PPO load")

    monkeypatch.setattr(
        bundle_module, "_load_policy_from_verified_path", must_not_deserialize
    )
    for name, mutate_manifest, mutate_context in _malformed_coercion_cases():
        malformed = _copy_and_repin_bundle(
            publication,
            tmp_path / f"malformed-bundle-{name}",
            mutate_training_manifest=mutate_manifest,
            mutate_context=mutate_context,
        )
        runtime_env = _environment(dataset, 5, 7)
        with pytest.raises(ValueError):
            _load(malformed, runtime_env)
        assert runtime_env._session is None


def test_malformed_source_runtime_is_rejected_before_publish_reserves_root(
    completed_attempt: tuple[Any, MarketDataset, SharedPortfolioGymEnv],
    tmp_path: Path,
) -> None:
    result, _, _ = completed_attempt

    def malformed_source_runtime(context: dict[str, object]) -> None:
        context["observed_source_runtime"] = {"scope": "not-the-source-schema"}

    attempt, _, _ = _copy_and_repin_attempt(
        result,
        tmp_path / "malformed-source-runtime-attempt",
        mutate_context=malformed_source_runtime,
    )
    output = tmp_path / "must-not-reserve-for-malformed-source"
    with pytest.raises(ValueError, match="observed_source_runtime|source runtime"):
        _publish_rebound_attempt(result, attempt, output)
    assert not output.exists()


def test_source_runtime_mismatch_fails_before_load_or_environment_reset(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset, _, publication = published_bundle
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    observed = deepcopy(bundle_module.capture_portfolio_source_runtime())
    runtime = observed["runtime"]
    assert isinstance(runtime, dict)
    runtime["python_version"] = f"{runtime['python_version']}.mismatch"
    monkeypatch.setattr(
        bundle_module, "capture_portfolio_source_runtime", lambda: observed
    )

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        raise AssertionError("source/runtime mismatch must precede PPO load")

    monkeypatch.setattr(
        bundle_module, "_load_policy_from_verified_path", must_not_deserialize
    )
    runtime_env = _environment(dataset, 5, 7)
    with pytest.raises(ValueError, match="source/runtime differs"):
        _load(publication, runtime_env)
    assert runtime_env._session is None


def test_selected_source_runtime_binds_timestamp_owner_content_digest() -> None:
    import importlib

    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    observed = bundle_module.capture_portfolio_source_runtime()
    modules = observed["source_modules"]
    assert isinstance(modules, list)
    names = [entry["module"] for entry in modules if isinstance(entry, dict)]
    market_index = names.index("trade_rl.data.market")
    assert names[market_index + 1] == "trade_rl.data._timestamps"

    timestamp_module = importlib.import_module("trade_rl.data._timestamps")
    origin = getattr(timestamp_module, "__file__", None)
    assert isinstance(origin, str)
    source_path = Path(origin).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    entry = modules[market_index + 1]
    assert entry == {
        "module": "trade_rl.data._timestamps",
        "origin": source_path.as_posix(),
        "bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def test_re_pinned_bundle_with_previous_source_roster_fails_before_policy_load(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset, _, publication = published_bundle
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    def restore_previous_source_roster(context: dict[str, object]) -> None:
        observed = context["observed_source_runtime"]
        assert isinstance(observed, dict)
        modules = observed["source_modules"]
        assert isinstance(modules, list)
        observed["source_modules"] = [
            entry
            for entry in modules
            if not isinstance(entry, dict)
            or entry.get("module") != "trade_rl.data._timestamps"
        ]

    old_roster = _copy_and_repin_bundle(
        publication,
        tmp_path / "previous-selected-source-roster",
        mutate_context=restore_previous_source_roster,
    )
    policy_loads = 0

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        nonlocal policy_loads
        policy_loads += 1
        raise AssertionError("old selected-source roster must fail before PPO load")

    monkeypatch.setattr(
        bundle_module, "_load_policy_from_verified_path", must_not_deserialize
    )
    runtime_env = _environment(dataset, 5, 7)
    with pytest.raises(ValueError, match="source runtime module roster differs"):
        _load(old_roster, runtime_env)
    assert policy_loads == 0
    assert runtime_env._session is None


@pytest.mark.parametrize("malformation", ["extra-key", "missing-file-record"])
def test_re_pinned_bundle_manifest_rejects_extra_or_missing_metadata(
    malformation: str,
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset, _, publication = published_bundle

    def mutate_bundle_manifest(manifest: dict[str, object]) -> None:
        if malformation == "extra-key":
            manifest["unexpected"] = "synthetic"
        else:
            files = manifest["files"]
            assert isinstance(files, dict)
            del files["training_context.json"]

    malformed = _copy_and_repin_bundle(
        publication,
        tmp_path / f"bad-bundle-{malformation}",
        mutate_bundle_manifest=mutate_bundle_manifest,
    )
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        raise AssertionError("bundle manifest must be rejected before PPO load")

    monkeypatch.setattr(
        bundle_module, "_load_policy_from_verified_path", must_not_deserialize
    )
    runtime_env = _environment(dataset, 5, 7)
    with pytest.raises(ValueError, match="missing or extra fields|roster differs"):
        _load(malformed, runtime_env)
    assert runtime_env._session is None


def test_re_pinned_bundle_rejects_unsupported_features_extractor_declaration(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset, _, publication = published_bundle

    def declare_unsupported_extractor(manifest: dict[str, object]) -> None:
        effective = manifest["effective_config"]
        assert isinstance(effective, dict)
        effective["policy_features_extractor"] = (
            "stable_baselines3.common.torch_layers.FlattenExtractor"
        )

    malformed = _copy_and_repin_bundle(
        publication,
        tmp_path / "unsupported-features-extractor",
        mutate_training_manifest=declare_unsupported_extractor,
    )
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported extractor must be rejected before PPO load")

    monkeypatch.setattr(
        bundle_module, "_load_policy_from_verified_path", must_not_deserialize
    )
    runtime_env = _environment(dataset, 5, 7)
    with pytest.raises(ValueError, match="effective PPO policy configuration"):
        _load(malformed, runtime_env)
    assert runtime_env._session is None


def test_loaded_policy_with_wrong_actual_spaces_is_rejected(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset, _, publication = published_bundle
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module
    from trade_rl.strategies.rl.portfolio_transform import PortfolioObservationTransform

    context = json.loads(
        (publication.root / "training_context.json").read_text(encoding="utf-8")
    )
    transform = PortfolioObservationTransform.from_payload(
        context["transform"], expected_digest=context["transform_digest"]
    )
    fake_loaded_policy = SimpleNamespace(
        observation_space=transform.observation_space,
        action_space=gym.spaces.MultiDiscrete(np.array([3, 4], dtype=np.int64)),
    )
    monkeypatch.setattr(
        bundle_module,
        "_load_policy_from_verified_path",
        lambda path: fake_loaded_policy,
    )
    runtime_env = _environment(dataset, 5, 7)
    with pytest.raises(ValueError, match="loaded PPO policy spaces differ"):
        _load(publication, runtime_env)
    assert runtime_env._session is None


@pytest.mark.parametrize(
    ("architecture_change", "expected_error"),
    [
        ("network", "loaded PPO policy architecture"),
        ("activation", "loaded PPO policy activation"),
        ("features", "loaded PPO policy features extractor"),
        ("policy_type", "loaded PPO policy architecture"),
    ],
)
def test_loaded_policy_with_wrong_actual_architecture_is_rejected(
    published_bundle: tuple[Any, MarketDataset, SharedPortfolioGymEnv, Any],
    monkeypatch: pytest.MonkeyPatch,
    architecture_change: str,
    expected_error: str,
) -> None:
    _, dataset, _, publication = published_bundle
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    load_verified = bundle_module._load_policy_from_verified_path

    def load_same_spaces_wrong_architecture(path: Path) -> Any:
        model = load_verified(path)
        import torch

        if architecture_change == "network":
            actor_layers = model.policy.mlp_extractor.policy_net
            input_width = actor_layers[0].in_features
            model.policy.mlp_extractor.policy_net = torch.nn.Sequential(
                torch.nn.Linear(input_width, 8),
                torch.nn.Tanh(),
                torch.nn.Linear(8, 64),
                torch.nn.Tanh(),
            )
        elif architecture_change == "activation":
            model.policy.activation_fn = torch.nn.ReLU
        elif architecture_change == "features":
            model.policy.features_extractor = torch.nn.Flatten(start_dim=1)
        else:
            model.policy = SimpleNamespace(
                observation_space=model.observation_space,
                action_space=model.action_space,
            )
        return model

    monkeypatch.setattr(
        bundle_module,
        "_load_policy_from_verified_path",
        load_same_spaces_wrong_architecture,
    )
    runtime_env = _environment(dataset, 5, 7)
    with pytest.raises(ValueError, match=expected_error):
        _load(publication, runtime_env)
    assert runtime_env._session is None


@pytest.mark.parametrize("failure_stage", ["copy", "manifest"])
def test_bundle_export_failure_is_preserved_and_root_cannot_be_reused(
    failure_stage: str,
    completed_attempt: tuple[Any, MarketDataset, SharedPortfolioGymEnv],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = completed_attempt
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    destination = tmp_path / f"failed-export-{failure_stage}"
    if failure_stage == "copy":

        def fail_copy(*args: object, **kwargs: object) -> None:
            raise OSError("literal synthetic checkpoint-copy failure")

        monkeypatch.setattr(bundle_module, "_write_copy_verified", fail_copy)
        expected_error = "checkpoint-copy failure"
    else:
        atomic_write = bundle_module.atomic_write_bytes

        def fail_manifest(path: Path, payload: bytes) -> None:
            if path.name == "manifest.json" and path.parent == destination:
                raise OSError("literal synthetic bundle-manifest failure")
            atomic_write(path, payload)

        monkeypatch.setattr(bundle_module, "atomic_write_bytes", fail_manifest)
        expected_error = "bundle-manifest failure"

    with pytest.raises(OSError, match=expected_error):
        _publish(result, destination)

    failure = json.loads((destination / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert not (destination / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        _publish(result, destination)


def test_failed_training_attempt_cannot_be_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _dataset()
    env = _environment(dataset, 0, 5)
    transform = fit_portfolio_observation_transform(env)
    attempt = tmp_path / "failed-attempt"

    def fail_reset(
        self: SharedPortfolioGymEnv,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        raise RuntimeError("literal synthetic reset failure")

    monkeypatch.setattr(SharedPortfolioGymEnv, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="literal synthetic reset failure"):
        fit_shared_portfolio_ppo(
            env,
            transform=transform,
            output_root=attempt,
            config=SharedPortfolioPPOConfig(
                total_timesteps=4,
                n_steps=4,
                batch_size=2,
                n_epochs=1,
                reset_seed=29,
                ppo_seed=31,
            ),
        )

    failure = json.loads((attempt / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert not (attempt / "completion.json").exists()
    output = tmp_path / "must-not-reserve"
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    with pytest.raises((FileNotFoundError, ValueError), match="completion|completed"):
        publish_portfolio_inference_bundle(
            attempt,
            output,
            expected_manifest_sha256=hashlib.sha256(
                (attempt / "manifest.json").read_bytes()
            ).hexdigest(),
            expected_completion_sha256="1" * 64,
            expected_training_context_sha256=hashlib.sha256(
                (attempt / "training_context.json").read_bytes()
            ).hexdigest(),
            expected_checkpoint_sha256="2" * 64,
        )
    assert not output.exists()
