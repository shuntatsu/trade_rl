from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import (
    PPOIntentStrategy,
    PPOTradingEnv,
    fit_ppo_strategy,
)
from trade_rl.strategies.rl.ppo_artifact import (
    load_normalized_ppo,
    load_ppo_inference_bundle,
    save_normalized_ppo,
    save_ppo_inference_bundle,
)


def test_real_checkpoint_reuses_fit_and_replays_identically_after_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("stable_baselines3")
    import trade_rl.evaluation.directional_candidates as candidates
    import trade_rl.evaluation.ppo_feature_checkpoint as checkpoint
    from tests.evaluation.test_ppo_feature_study import _PROTOCOL
    from trade_rl.artifacts import canonical_json_bytes, content_digest
    from trade_rl.data.artifacts import publish_market_dataset_artifact
    from trade_rl.evaluation.directional import evaluate_directional_arm
    from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
    from trade_rl.evaluation.experiments import ResolvedRunConfig
    from trade_rl.evaluation.runs import build_candidate_run_provenance

    dataset = pooled_market().with_content_identity()
    source, root = tmp_path / "synthetic-source", tmp_path / "checkpoints"
    source.mkdir()
    artifact = publish_market_dataset_artifact(source / "dataset", dataset)
    plan_payload = {"synthetic_checkpoint_test": True}
    plan_raw = canonical_json_bytes(plan_payload)
    (source / "study").mkdir()
    (source / "study" / "plan.json").write_bytes(plan_raw)
    plan_digest = content_digest(plan_payload)
    protocol = deepcopy(_PROTOCOL)
    protocol.update(
        source_dataset_id=dataset.dataset_id,
        source_artifact_digest=artifact.artifact_digest,
        source_study_digest=plan_digest,
        source_plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        evaluation_dataset_id=dataset.dataset_id,
        start_index=0,
        stop_index=3,
        interval_count=3,
        symbols=list(dataset.symbols),
        provenance=build_candidate_run_provenance(),
    )
    for factor in protocol["factors"].values():
        factor.update(feature_names=list(dataset.feature_names), feature_indices=[0])
    protocol["training"]["requested_timesteps"] = 2048
    protocol["evaluation"].update(
        start_index=0,
        stop_index=3,
        interval_count=3,
        interval_year_slices={"2026": [0, 3]},
    )
    for scenario in protocol["evaluation"]["scenarios"].values():
        execution = replace(
            DIRECTIONAL_BASE_EXECUTION_COST,
            multiplier=scenario["cost_multiplier"],
            order_latency_bars=scenario["latency_bars"],
        )
        scenario["execution_policy_digest"] = execution.execution_policy_digest
    snapshot = checkpoint.legacy._source_snapshot_bytes(protocol["provenance"])
    protocol["source_snapshot_sha256"] = hashlib.sha256(snapshot).hexdigest()
    config = ResolvedRunConfig(
        signal_name="signal",
        signal_index=0,
        feature_names=dataset.feature_names,
        feature_indices=(0,),
        fit_symbol_names=dataset.symbols,
        fit_symbol_indices=(0, 1),
        fit_cutoff="2026-01-01T04:00:00",
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=2048,
        ppo_seed=0,
        evaluation_start="2026-01-01T00:00:00",
        evaluation_stop_exclusive="2026-01-01T04:00:00",
        gross_budget=0.1,
        initial_capital=10_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )
    monkeypatch.setattr(
        checkpoint.legacy, "expected_protocol", lambda _source: deepcopy(protocol)
    )
    monkeypatch.setattr(
        checkpoint, "_load_context", lambda _source, _protocol: (dataset, config)
    )
    monkeypatch.setattr(
        checkpoint,
        "inspect_study",
        lambda _root: SimpleNamespace(
            plan=SimpleNamespace(digest=plan_digest, dataset_id=dataset.dataset_id)
        ),
    )
    monkeypatch.setattr(candidates, "PPO_TIMESTEPS", 2048)
    monkeypatch.setattr(checkpoint, "PPO_TIMESTEPS", 2048)
    fitted = []

    def fit(*args, **kwargs):
        factory = candidates.fit_directional_candidate(*args, **kwargs)
        fitted.append(factory())
        return factory

    monkeypatch.setattr(checkpoint, "fit_directional_candidate", fit)
    checkpoint.prepare_checkpoint_study(source, root)
    fit_dir = checkpoint.fit_checkpoint(source, root, "baseline", 0)
    assert fitted[0].policy.num_timesteps == 2048
    fitted_bytes = {
        p.relative_to(fit_dir): p.read_bytes()
        for p in fit_dir.rglob("*")
        if p.is_file()
    }

    def unexpected_refit(*_args, **_kwargs):
        pytest.fail("a completed fit must be reused without training")

    monkeypatch.setattr(checkpoint, "fit_directional_candidate", unexpected_refit)
    assert checkpoint.fit_checkpoint(source, root, "baseline", 0) == fit_dir
    loaded_policies = []
    load = checkpoint.load_ppo_inference_bundle

    def load_bundle(*args, **kwargs):
        strategy = load(*args, **kwargs)
        assert strategy.policy.get_env() is None
        assert strategy.policy.device.type == "cpu"
        assert strategy.policy.num_timesteps == 2048
        loaded_policies.append(strategy.policy)
        return strategy

    monkeypatch.setattr(checkpoint, "load_ppo_inference_bundle", load_bundle)

    def evaluate_with_bound_schema(dataset, factory, **kwargs):
        strategy = factory()
        assert strategy.feature_names == fitted[0].feature_names
        assert strategy.feature_indices == fitted[0].feature_indices
        return evaluate_directional_arm(dataset, factory, **kwargs)

    monkeypatch.setattr(
        checkpoint, "evaluate_directional_arm", evaluate_with_bound_schema
    )
    for index in (0, 1):
        expected = evaluate_directional_arm(
            dataset,
            lambda: PPOIntentStrategy(fitted[0].policy, feature_indices=(0,)),
            start_index=0,
            stop_index=3,
            symbol_index=index,
            capture_ledger_evidence=True,
        )
        direct_dir = tmp_path / f"direct-{index}"
        direct_dir.mkdir()
        expected = checkpoint.legacy._persist_replay_ledger(
            direct_dir, expected, scenario_name="base", symbol_index=index
        )
        cell_dir = checkpoint.replay_cell(source, root, "baseline", 0, "base", index)
        # Compare persisted return/accounting bytes against the original policy,
        # independently of the checkpoint reader's own comparison logic.
        actual = json.loads((cell_dir / "cell.json").read_bytes())["replay"]
        assert actual == expected
        ledger_name = expected["ledger_evidence_file"]
        assert (cell_dir / ledger_name).read_bytes() == (
            direct_dir / ledger_name
        ).read_bytes()
    assert len(loaded_policies) == 2
    assert loaded_policies[0] is not loaded_policies[1]

    def unexpected_evaluation(*_args, **_kwargs):
        pytest.fail("a completed cell must be reused without replay")

    monkeypatch.setattr(checkpoint, "evaluate_directional_arm", unexpected_evaluation)
    checkpoint.replay_cell(source, root, "baseline", 0, "base", 0)
    assert len(loaded_policies) == 2
    assert {
        p.relative_to(fit_dir): p.read_bytes()
        for p in fit_dir.rglob("*")
        if p.is_file()
    } == fitted_bytes


def _env() -> PPOTradingEnv:
    return PPOTradingEnv(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
    )


def _content_verified_market(kind: FeatureKind) -> MarketDataset:
    dataset = pooled_market()
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="signal", kind=kind),),
        cross_asset_reference_symbol="BTCUSDT",
    )
    return dataset.with_content_identity({"config": config.canonical_payload()})


def _observation(symbol_index: int = 0) -> StrategyObservation:
    dataset = pooled_market()
    global_available = dataset.resolved_array("global_feature_available")[0]
    return StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[symbol_index],
        features=dataset.features[0, symbol_index],
        feature_available=dataset.feature_available[0, symbol_index],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, symbol_index],
        global_features=dataset.global_features[0],
        global_feature_available=global_available,
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def test_real_sb3_accepts_environment_and_runs_sequential_rollout() -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    torch = pytest.importorskip("torch")

    assert stable_baselines3.__version__ == "2.3.2"
    assert torch.__version__.split("+", 1)[0] == "2.4.1"
    torch.set_num_threads(1)

    env_checker.check_env(_env(), warn=True)

    model = stable_baselines3.PPO(
        "MlpPolicy",
        _env(),
        n_steps=8,
        batch_size=8,
        seed=17,
        ent_coef=0.0,
        verbose=0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )
    model.learn(total_timesteps=16)

    strategy = PPOIntentStrategy(model, feature_indices=(0,))
    decision = strategy.decide(_observation())

    assert decision in {
        PositionIntent.SHORT,
        PositionIntent.FLAT,
        PositionIntent.LONG,
    }
    assert model.device.type == "cpu"
    assert model.num_timesteps == 16


def test_real_sequential_normalized_fit_keeps_verified_scope() -> None:
    pytest.importorskip("stable_baselines3")
    dataset = _content_verified_market(FeatureKind.RELATIVE_RETURN_TO_BTC)

    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(1, 0),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=1,
        seed=19,
        normalize_features=True,
    )

    vector = strategy.policy.get_env()
    assert vector is not None
    assert strategy.feature_normalizer is not None
    assert strategy.feature_normalizer.fit_symbol_indices == (1, 0)
    assert len(vector.envs) == 1

    env = vector.envs[0].unwrapped
    assert isinstance(env, PPOTradingEnv)
    assert env.symbol_indices == (1, 0)
    assert env.information_symbol_indices == (1, 0)
    assert env.feature_normalizer is strategy.feature_normalizer

    _first_observation, first_info = env.reset(seed=19)
    _second_observation, second_info = env.reset()
    assert first_info["symbol"] == "ETHUSDT"
    assert second_info["symbol"] == "BTCUSDT"


def test_real_sb3_interleaved_fit_preserves_verified_information_scope() -> None:
    pytest.importorskip("stable_baselines3")
    dataset = _content_verified_market(FeatureKind.RELATIVE_RETURN_TO_BTC)

    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=21,
        training_layout="interleaved",
        rollout_steps_per_env=32,
        normalize_features=True,
    )

    vector = strategy.policy.get_env()
    assert vector is not None
    assert strategy.feature_normalizer is not None
    assert strategy.feature_normalizer.fit_symbol_indices == (0, 1)
    envs = vector.envs
    assert len(envs) == 2
    for expected_symbol, raw_env in zip(
        ("BTCUSDT", "ETHUSDT"),
        envs,
        strict=True,
    ):
        env = raw_env.unwrapped
        assert isinstance(env, PPOTradingEnv)
        assert env.information_symbol_indices == (0, 1)
        assert env.feature_normalizer is strategy.feature_normalizer
        _observation_value, info = env.reset(seed=21)
        assert info["symbol"] == expected_symbol


def test_real_sb3_interleaved_normalized_fit_roundtrips_bundle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    dataset = pooled_market()

    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=23,
        training_layout="interleaved",
        rollout_steps_per_env=32,
        normalize_features=True,
    )
    assert strategy.feature_normalizer is not None
    assert torch.get_num_threads() == 1

    root = tmp_path / "normalized-ppo"
    digest = save_normalized_ppo(root, strategy)
    torch.set_num_threads(2)
    loaded = load_normalized_ppo(
        root,
        expected_digest=digest,
        feature_names=dataset.feature_names,
    )

    before = strategy.decide(_observation())
    after = loaded.decide(_observation())

    assert after is before
    assert loaded.feature_normalizer == strategy.feature_normalizer
    assert loaded.policy.device.type == "cpu"
    assert torch.get_num_threads() == 1


def test_real_interleaved_fit_is_parameter_deterministic_for_same_seed() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    dataset = pooled_market()

    def fit():
        return fit_ppo_strategy(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0, 1),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            total_timesteps=64,
            seed=31,
            training_layout="interleaved",
            rollout_steps_per_env=32,
        )

    first = fit()
    second = fit()
    first_state = first.policy.policy.state_dict()
    second_state = second.policy.policy.state_dict()

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_real_sequential_fit_is_parameter_deterministic_for_same_seed() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    dataset = pooled_market()

    def fit():
        return fit_ppo_strategy(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0, 1),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            total_timesteps=1,
            seed=35,
        )

    first = fit()
    second = fit()
    first_state = first.policy.policy.state_dict()
    second_state = second.policy.policy.state_dict()

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_real_sequential_fit_records_default_rollout_rounded_timesteps() -> None:
    pytest.importorskip("stable_baselines3")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=1,
        seed=37,
    )

    assert strategy.policy.num_timesteps == 2048


def test_real_interleaved_fit_records_vector_rollout_rounded_timesteps() -> None:
    pytest.importorskip("stable_baselines3")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=65,
        seed=41,
        training_layout="interleaved",
        rollout_steps_per_env=32,
    )

    assert strategy.policy.num_timesteps == 128


def _strong_uptrend_market(n_bars: int = 64) -> MarketDataset:
    open_price = np.empty((n_bars, 1), dtype=np.float64)
    close = np.empty((n_bars, 1), dtype=np.float64)
    open_price[0, 0] = 100.0
    close[0, 0] = 100.0
    for index in range(1, n_bars):
        open_price[index, 0] = close[index - 1, 0]
        close[index, 0] = open_price[index, 0] * 1.05
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.ones((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_real_ppo_learns_trivial_causal_long_signal() -> None:
    pytest.importorskip("stable_baselines3")
    dataset = _strong_uptrend_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        start_index=0,
        stop_index=dataset.n_bars - 1,
        gross_budget=0.5,
        total_timesteps=4096,
        seed=53,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    observation = StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[0],
        features=dataset.features[0, 0],
        feature_available=dataset.feature_available[0, 0],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, 0],
        global_features=dataset.global_features[0],
        global_feature_available=dataset.resolved_array("global_feature_available")[0],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )

    assert strategy.decide(observation) is PositionIntent.LONG
    torch = pytest.importorskip("torch")
    for name, parameter in strategy.policy.policy.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_real_raw_ppo_model_roundtrips_deterministic_intent(tmp_path: Path) -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=59,
        training_layout="interleaved",
        rollout_steps_per_env=32,
    )
    model_path = tmp_path / "model.zip"
    strategy.policy.save(str(model_path))
    loaded = stable_baselines3.PPO.load(str(model_path), device="cpu")

    before = strategy.decide(_observation())
    after = PPOIntentStrategy(loaded, feature_indices=(0,)).decide(_observation())

    assert after is before
    assert loaded.device.type == "cpu"


def _strong_downtrend_market(n_bars: int = 64) -> MarketDataset:
    dataset = _strong_uptrend_market(n_bars)
    open_price = np.empty_like(dataset.open)
    close = np.empty_like(dataset.close)
    open_price[0, 0] = 100.0
    close[0, 0] = 100.0
    for index in range(1, n_bars):
        open_price[index, 0] = close[index - 1, 0]
        close[index, 0] = open_price[index, 0] * 0.95
    return MarketDataset(
        dataset_id="8" * 64,
        symbols=dataset.symbols,
        timestamps=dataset.timestamps,
        features=dataset.features,
        global_features=dataset.global_features,
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=dataset.volume,
        funding_rate=dataset.funding_rate,
        tradable=dataset.tradable,
        feature_available=dataset.feature_available,
        feature_names=dataset.feature_names,
        global_feature_names=dataset.global_feature_names,
        periods_per_year=dataset.periods_per_year,
    )


def test_real_ppo_learns_trivial_causal_short_signal() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    dataset = _strong_downtrend_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        start_index=0,
        stop_index=dataset.n_bars - 1,
        gross_budget=0.5,
        total_timesteps=4096,
        seed=61,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    observation = StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[0],
        features=dataset.features[0, 0],
        feature_available=dataset.feature_available[0, 0],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, 0],
        global_features=dataset.global_features[0],
        global_feature_available=dataset.resolved_array("global_feature_available")[0],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )

    assert strategy.decide(observation) is PositionIntent.SHORT
    for name, parameter in strategy.policy.policy.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_real_ppo_terminal_settlement_fit_uses_agent_only_normalizer_scope() -> None:
    pytest.importorskip("stable_baselines3")
    dataset = pooled_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=71,
        normalize_features=True,
        settle_terminal_position=True,
    )

    assert strategy.feature_normalizer is not None
    assert strategy.feature_normalizer.start_index == 0
    assert strategy.feature_normalizer.stop_index == 2
    assert strategy.policy.device.type == "cpu"


def test_real_interleaved_ppo_runs_with_terminal_settlement() -> None:
    pytest.importorskip("stable_baselines3")
    dataset = pooled_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=73,
        training_layout="interleaved",
        rollout_steps_per_env=32,
        normalize_features=True,
        settle_terminal_position=True,
    )

    assert strategy.feature_normalizer is not None
    assert strategy.feature_normalizer.stop_index == 2
    assert strategy.policy.num_timesteps == 64
    assert strategy.policy.device.type == "cpu"


def test_real_ppo_constructor_contract_is_explicit() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=1,
        seed=89,
    )
    model = strategy.policy

    assert model.learning_rate == pytest.approx(3e-4)
    assert model.n_steps == 2048
    assert model.batch_size == 64
    assert model.n_epochs == 10
    assert model.gamma == pytest.approx(0.99)
    assert model.gae_lambda == pytest.approx(0.95)
    assert model.clip_range(1.0) == pytest.approx(0.2)
    assert model.clip_range_vf is None
    assert model.normalize_advantage is True
    assert model.ent_coef == pytest.approx(0.0)
    assert model.vf_coef == pytest.approx(0.5)
    assert model.max_grad_norm == pytest.approx(0.5)
    assert model.use_sde is False
    assert model.sde_sample_freq == -1
    assert model.target_kl is None
    assert model.policy.activation_fn is torch.nn.Tanh
    assert model.policy.ortho_init is True
    assert model.policy.features_extractor.__class__.__name__ == "FlattenExtractor"
    assert model.policy.share_features_extractor is True
    assert isinstance(model.policy.optimizer, torch.optim.Adam)
    assert model.policy.optimizer.defaults["eps"] == pytest.approx(1e-5)


def test_explicit_ppo_constructor_matches_pinned_implicit_defaults() -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch_layers = pytest.importorskip("stable_baselines3.common.torch_layers")

    assert stable_baselines3.__version__ == "2.3.2"
    implicit = stable_baselines3.PPO(
        "MlpPolicy",
        _env(),
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=97,
        ent_coef=0.0,
        device="cpu",
        verbose=0,
    )
    explicit = stable_baselines3.PPO(
        "MlpPolicy",
        _env(),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.Tanh,
            "ortho_init": True,
            "features_extractor_class": torch_layers.FlattenExtractor,
            "share_features_extractor": True,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"eps": 1e-5},
        },
        seed=97,
        device="cpu",
        verbose=0,
    )

    implicit_state = implicit.policy.state_dict()
    explicit_state = explicit.policy.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    for name in implicit_state:
        assert torch.equal(implicit_state[name], explicit_state[name]), name


def test_explicit_ppo_constructor_matches_pinned_implicit_training_update() -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch_layers = pytest.importorskip("stable_baselines3.common.torch_layers")
    torch.set_num_threads(1)

    def train_implicit() -> dict[str, object]:
        model = stable_baselines3.PPO(
            "MlpPolicy",
            _env(),
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            seed=101,
            ent_coef=0.0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=2048)
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.policy.state_dict().items()
        }

    def train_explicit() -> dict[str, object]:
        model = stable_baselines3.PPO(
            "MlpPolicy",
            _env(),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            policy_kwargs={
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "activation_fn": torch.nn.Tanh,
                "ortho_init": True,
                "features_extractor_class": torch_layers.FlattenExtractor,
                "share_features_extractor": True,
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {"eps": 1e-5},
            },
            seed=101,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=2048)
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.policy.state_dict().items()
        }

    implicit_state = train_implicit()
    explicit_state = train_explicit()

    assert implicit_state.keys() == explicit_state.keys()
    for name in implicit_state:
        assert torch.equal(implicit_state[name], explicit_state[name]), name


def test_real_raw_ppo_roundtrips_schema_bound_inference_bundle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    dataset = pooled_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=1,
        seed=29,
    )

    root = tmp_path / "raw-ppo"
    digest = save_ppo_inference_bundle(
        root,
        strategy,
        feature_names=dataset.feature_names,
    )
    loaded = load_ppo_inference_bundle(
        root,
        expected_digest=digest,
        feature_names=dataset.feature_names,
    )

    assert loaded.decide(_observation()) is strategy.decide(_observation())
    assert loaded.feature_normalizer is None
    assert loaded.policy.device.type == "cpu"


def test_real_normalized_ppo_roundtrips_schema_bound_inference_bundle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    dataset = pooled_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=37,
        training_layout="interleaved",
        rollout_steps_per_env=32,
        normalize_features=True,
    )

    root = tmp_path / "normalized-inference-ppo"
    digest = save_ppo_inference_bundle(
        root,
        strategy,
        feature_names=dataset.feature_names,
    )
    loaded = load_ppo_inference_bundle(
        root,
        expected_digest=digest,
        feature_names=dataset.feature_names,
    )

    assert loaded.feature_normalizer == strategy.feature_normalizer
    assert loaded.decide(_observation()) is strategy.decide(_observation())
    assert loaded.policy.device.type == "cpu"


def test_real_ppo_fit_is_invariant_to_holdout_only_mutation() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    base = _content_verified_market(FeatureKind.LOG_RETURN)

    features = np.array(base.features, copy=True)
    features[:, 1, 0] = np.asarray([10_000.0, -20_000.0, 30_000.0, -40_000.0])
    multiplier = np.asarray([3.0, 7.0, 11.0, 17.0])
    price_fields: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close"):
        values = np.array(getattr(base, name), copy=True)
        values[:, 1] *= multiplier
        price_fields[name] = values
    funding = np.array(base.funding_rate, copy=True)
    funding[:, 1] = np.asarray([0.5, -0.75, 1.25, -1.5])

    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="signal", kind=FeatureKind.LOG_RETURN),),
        cross_asset_reference_symbol="BTCUSDT",
    )
    mutated = replace(
        base,
        identity_payload_json=None,
        features=features,
        funding_rate=funding,
        **price_fields,
    ).with_content_identity({"config": config.canonical_payload()})

    def fit(dataset: MarketDataset):
        return fit_ppo_strategy(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            total_timesteps=64,
            seed=149,
            training_layout="interleaved",
            rollout_steps_per_env=64,
            normalize_features=True,
        )

    original = fit(base)
    changed = fit(mutated)

    assert original.feature_normalizer is not None
    assert changed.feature_normalizer is not None
    assert original.feature_normalizer.mean == changed.feature_normalizer.mean
    assert original.feature_normalizer.scale == changed.feature_normalizer.scale

    original_state = original.policy.policy.state_dict()
    changed_state = changed.policy.policy.state_dict()
    assert original_state.keys() == changed_state.keys()
    for name in original_state:
        assert torch.equal(original_state[name], changed_state[name]), name
