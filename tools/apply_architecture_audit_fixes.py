from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing architecture audit anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_file(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n", encoding="utf-8")


def add_tests() -> None:
    write_file(
        "tests/architecture/test_architecture_audit_fixes.py",
        r'''
from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces

from trade_rl.integrations.sb3_serving import _SB3StructuredSequenceEnsemblePolicy
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    project_portfolio_targets,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.rl.policies import (
    SequenceAssetFeatureExtractor,
    SharedPerAssetActionHead,
    SharedPerAssetActorCriticPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _runner_namespace() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE_ROOT / "run_full_research.py"))


def test_training_policy_digest_uses_cli_policy_digest_and_fails_closed() -> None:
    resolve = _runner_namespace()["_training_policy_digest"]

    assert resolve({"policy_digest": "a" * 64}) == "a" * 64
    with pytest.raises(ValueError, match="policy_digest"):
        resolve({"artifact_digest": "b" * 64})
    with pytest.raises(ValueError, match="policy_digest"):
        resolve({"policy_digest": "A" * 64})


def test_confirmation_recheck_loads_existing_generation_context(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(EXAMPLE_ROOT / "recheck_confirmation.py"))
    load_context = namespace["_load_existing_context"]
    work_root = tmp_path / "generation"
    walk_forward = tmp_path / "artifacts" / "walk-forward"
    work_root.mkdir()
    walk_forward.mkdir(parents=True)
    summary = {
        "production_status": "NO-GO",
        "training": {"policy_digest": "b" * 64},
        "walk_forward": {"artifact_path": str(walk_forward)},
    }
    (work_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    loaded_summary, loaded_walk_forward, digest = load_context(
        work_root,
        repository_root=ROOT,
    )

    assert loaded_summary == summary
    assert loaded_walk_forward == walk_forward
    assert digest == "b" * 64


def _sequence_policy() -> SharedPerAssetActorCriticPolicy:
    n_symbols = 2
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    components: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -10.0, 10.0, shape=(n_symbols, 8), dtype=np.float32
        ),
        "asset_state": spaces.Box(
            -10.0, 10.0, shape=(n_symbols, 4), dtype=np.float32
        ),
        "global_state": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(n_symbols,), dtype=np.float32),
    }
    for timeframe in timeframes:
        shape = (n_symbols, 3, 2)
        components[f"sequence_{timeframe}_values"] = spaces.Box(
            -10.0, 10.0, shape=shape, dtype=np.float16
        )
        components[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        components[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 100.0, shape=shape, dtype=np.float16
        )
    observation_space = spaces.Dict(components)
    return SharedPerAssetActorCriticPolicy(
        observation_space,
        spaces.Box(-1.0, 1.0, shape=(n_symbols,), dtype=np.float32),
        lambda _: 1e-3,
        net_arch={"pi": [11], "vf": [13]},
        features_extractor_class=SequenceAssetFeatureExtractor,
        features_extractor_kwargs={
            "feature_counts": feature_counts,
            "window_lengths": window_lengths,
            "snapshot_width": 8,
            "asset_state_width": 4,
            "global_width": 3,
            "n_symbols": n_symbols,
            "d_model": 16,
            "attention_heads": 4,
            "attention_layers": 1,
            "dropout": 0.0,
        },
        shared_actor_n_symbols=n_symbols,
        shared_actor_d_model=16,
        shared_actor_global_dim=128,
        shared_actor_net_arch=(11,),
        log_std_init=-0.5,
    )


def _policy_observations(policy: SharedPerAssetActorCriticPolicy) -> dict[str, torch.Tensor]:
    observations: dict[str, torch.Tensor] = {}
    for key, space in policy.observation_space.spaces.items():
        value = np.zeros((2, *space.shape), dtype=space.dtype)
        if key.endswith("_available"):
            value.fill(1)
        observations[key] = torch.as_tensor(value)
    observations["active"][:, 0] = 1.0
    observations["active"][:, 1] = 0.0
    return observations


def test_shared_actor_uses_explicit_activity_and_one_shared_exploration_scale() -> None:
    head = SharedPerAssetActionHead(
        n_symbols=2,
        token_dim=3,
        context_dim=8,
        hidden_dims=(5,),
    ).eval()
    contexts = torch.randn(1, 2, 8)
    contexts[:, :, -1] = torch.tensor([[1.0, 0.0]])
    with torch.no_grad():
        output = head(contexts.reshape(1, -1))
    assert output[0, 1].item() == 0.0

    policy = _sequence_policy()
    assert tuple(policy.log_std.shape) == (1,)
    observations = _policy_observations(policy)
    distribution = policy.get_distribution(observations)
    stochastic = distribution.get_actions(deterministic=False)
    deterministic = distribution.get_actions(deterministic=True)
    assert torch.count_nonzero(stochastic[:, 1]) == 0
    assert torch.count_nonzero(deterministic[:, 1]) == 0

    active_equivalent = torch.tensor([[0.2, 0.0], [-0.1, 0.0]])
    inactive_changed = torch.tensor([[0.2, 0.8], [-0.1, -0.7]])
    _, first_log_prob, _ = policy.evaluate_actions(observations, active_equivalent)
    _, second_log_prob, _ = policy.evaluate_actions(observations, inactive_changed)
    torch.testing.assert_close(first_log_prob, second_log_prob)


def test_oracle_portfolio_projection_matches_supported_runtime_limits() -> None:
    risk = PortfolioRiskConfig(
        max_abs_weight=0.4,
        max_net_exposure=0.5,
        max_position_to_market_notional=0.1,
    )
    targets = np.asarray([[[0.8, 0.8], [-0.8, 0.2]]], dtype=np.float64)
    projected = project_portfolio_targets(
        targets,
        portfolio_value=np.asarray([100.0]),
        market_notional=np.asarray([50.0, 20.0]),
        config=risk,
    )

    assert projected.shape == targets.shape
    assert np.max(np.abs(projected[..., 0])) <= pytest.approx(0.05)
    assert np.max(np.abs(projected[..., 1])) <= pytest.approx(0.02)
    assert np.max(np.abs(projected.sum(axis=-1))) <= pytest.approx(0.5)

    with pytest.raises(ValueError, match="oracle portfolio risk"):
        OracleTeacherConfig(
            portfolio_risk=PortfolioRiskConfig(volatility_target=0.1)
        )


def test_structured_serving_rejects_feature_recipe_mismatch() -> None:
    policy = object.__new__(_SB3StructuredSequenceEnsemblePolicy)
    policy.dataset_reference = {
        "symbols": ["BTCUSDT"],
        "feature_names": ["return_1"],
        "global_feature_names": ["market_regime"],
        "bar_hours": 1.0,
        "feature_config_digest": "a" * 64,
    }
    policy.builder = SimpleNamespace(layout_digest=lambda _: "c" * 64)
    policy.sequence_normalizer = SimpleNamespace(sequence_schema_digest="c" * 64)
    dataset = SimpleNamespace(
        symbols=("BTCUSDT",),
        feature_names=("return_1",),
        global_feature_names=("market_regime",),
        bar_hours=1.0,
        feature_config_digest="b" * 64,
    )

    with pytest.raises(ValueError, match="feature recipe"):
        policy._validate_dataset(dataset)
''',
    )

    replace_once(
        "tests/rl/test_sequence_policy_core.py",
        """    contexts = torch.randn(2, 3, 9)\n    permutation = torch.tensor([2, 0, 1])\n""",
        """    contexts = torch.randn(2, 3, 9)\n    contexts[:, :, -1] = 1.0\n    permutation = torch.tensor([2, 0, 1])\n""",
    )
    replace_once(
        "tests/rl/test_sequence_policy_core.py",
        """def test_shared_actor_masks_inactive_zero_tokens() -> None:\n""",
        """def test_shared_actor_masks_explicitly_inactive_assets() -> None:\n""",
    )
    replace_once(
        "tests/rl/test_sequence_policy_core.py",
        """    contexts = torch.randn(1, 2, 7)\n    contexts[:, 1, :3] = 0.0\n""",
        """    contexts = torch.randn(1, 2, 7)\n    contexts[:, :, -1] = 1.0\n    contexts[:, 1, -1] = 0.0\n""",
    )
    replace_once(
        "tests/rl/test_sequence_policy_core.py",
        "assert output.shape == (1, 160)",
        "assert output.shape == (1, 161)",
    )
    replace_once(
        "tests/rl/test_sequence_policy_core.py",
        'assert policy.action_distribution_name == "squashed_diag_gaussian"',
        'assert policy.action_distribution_name == "masked_shared_squashed_diag_gaussian"',
    )


def add_implementation() -> None:
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        """def _load_json(path: Path) -> dict[str, Any]:\n    payload = json.loads(path.read_text(encoding=\"utf-8\"))\n    if not isinstance(payload, dict):\n        raise ValueError(f\"JSON payload must be an object: {path}\")\n    return dict(payload)\n\n\n""",
        """def _load_json(path: Path) -> dict[str, Any]:\n    payload = json.loads(path.read_text(encoding=\"utf-8\"))\n    if not isinstance(payload, dict):\n        raise ValueError(f\"JSON payload must be an object: {path}\")\n    return dict(payload)\n\n\ndef _training_policy_digest(payload: object) -> str:\n    if not isinstance(payload, dict):\n        raise ValueError(\"training result must be a JSON object\")\n    value = payload.get(\"policy_digest\")\n    if not isinstance(value, str) or re.fullmatch(r\"[0-9a-f]{64}\", value) is None:\n        raise ValueError(\"training result policy_digest is missing or invalid\")\n    return value\n\n\n""",
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '    expected_policy_digest = training.get("artifact_digest")\n',
        "    expected_policy_digest = _training_policy_digest(training)\n",
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        """        expected_policy_digest=(\n            str(expected_policy_digest)\n            if isinstance(expected_policy_digest, str)\n            else None\n        ),\n""",
        """        expected_policy_digest=expected_policy_digest,\n""",
    )

    write_file(
        "examples/binance-multitimeframe/recheck_confirmation.py",
        r'''
#!/usr/bin/env python3
"""Re-evaluate a completed research generation with fresh confirmation evidence."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any


def _runner_namespace() -> dict[str, Any]:
    runner = Path(__file__).with_name("run_full_research.py")
    return runpy.run_path(str(runner))


def _load_existing_context(
    work_root: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path, str]:
    root = work_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"research generation is missing: {root}")
    namespace = _runner_namespace()
    load_json = namespace["_load_json"]
    policy_digest = namespace["_training_policy_digest"]
    summary = load_json(root / "summary.json")
    training = summary.get("training")
    digest = policy_digest(training)
    walk_forward = summary.get("walk_forward")
    if not isinstance(walk_forward, dict):
        raise ValueError("summary walk_forward result is missing")
    raw_path = walk_forward.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("summary walk_forward artifact_path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_dir():
        raise FileNotFoundError(f"walk-forward artifact is missing: {path}")
    return summary, path, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    summary, walk_forward_path, policy_digest = _load_existing_context(
        args.work_root,
        repository_root=repository_root,
    )
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    exit_code = finalize(
        work_root=args.work_root.resolve(),
        walk_forward_path=walk_forward_path,
        summary=summary,
        strict=True,
        require_confirmation=True,
        expected_policy_digest=policy_digest,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
''',
    )

    replace_once(
        "trade_rl/rl/policies.py",
        "from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution\n",
        """from stable_baselines3.common.distributions import (\n    SquashedDiagGaussianDistribution,\n    TanhBijector,\n)\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        "features_dim=n_symbols * d_model + d_model + 128,",
        "features_dim=n_symbols * d_model + d_model + 128 + n_symbols,",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """        globals_ = self.global_encoder(observations[\"global_state\"].float())\n        ordered_assets = asset_tokens.reshape(asset_tokens.shape[0], -1)\n        return torch.cat((ordered_assets, pooled_assets, globals_), dim=-1)\n""",
        """        globals_ = self.global_encoder(observations[\"global_state\"].float())\n        ordered_assets = asset_tokens.reshape(asset_tokens.shape[0], -1)\n        active = observations[\"active\"].float()\n        return torch.cat((ordered_assets, pooled_assets, globals_, active), dim=-1)\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        "expected = n_symbols * token_dim + token_dim + global_dim",
        "expected = n_symbols * token_dim + token_dim + global_dim + n_symbols",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        "self.actor_context_dim = 2 * token_dim + global_dim",
        "self.actor_context_dim = 2 * token_dim + global_dim + 1",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """    def _parts(\n        self, features: torch.Tensor\n    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n        asset_width = self.n_symbols * self.token_dim\n        tokens = features[:, :asset_width].reshape(-1, self.n_symbols, self.token_dim)\n        pooled = features[:, asset_width : asset_width + self.token_dim]\n        globals_ = features[:, asset_width + self.token_dim :]\n        return tokens, pooled, globals_\n\n    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:\n        tokens, pooled, globals_ = self._parts(features)\n        pooled_per_asset = pooled[:, None, :].expand(-1, self.n_symbols, -1)\n        global_per_asset = globals_[:, None, :].expand(-1, self.n_symbols, -1)\n        contexts = torch.cat((tokens, pooled_per_asset, global_per_asset), dim=-1)\n        return contexts.reshape(features.shape[0], -1)\n\n    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:\n        _, pooled, globals_ = self._parts(features)\n        return self.critic_net(torch.cat((pooled, globals_), dim=-1))\n""",
        """    def _parts(\n        self, features: torch.Tensor\n    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n        asset_width = self.n_symbols * self.token_dim\n        pooled_start = asset_width\n        global_start = pooled_start + self.token_dim\n        active_start = global_start + self.global_dim\n        tokens = features[:, :asset_width].reshape(-1, self.n_symbols, self.token_dim)\n        pooled = features[:, pooled_start:global_start]\n        globals_ = features[:, global_start:active_start]\n        active = features[:, active_start:]\n        return tokens, pooled, globals_, active\n\n    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:\n        tokens, pooled, globals_, active = self._parts(features)\n        pooled_per_asset = pooled[:, None, :].expand(-1, self.n_symbols, -1)\n        global_per_asset = globals_[:, None, :].expand(-1, self.n_symbols, -1)\n        contexts = torch.cat(\n            (tokens, pooled_per_asset, global_per_asset, active.unsqueeze(-1)),\n            dim=-1,\n        )\n        return contexts.reshape(features.shape[0], -1)\n\n    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:\n        _, pooled, globals_, _ = self._parts(features)\n        return self.critic_net(torch.cat((pooled, globals_), dim=-1))\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """    def forward(self, actor_latent: torch.Tensor) -> torch.Tensor:\n        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)\n        token = contexts[:, :, : self.token_dim]\n        active = token.abs().sum(dim=-1) > 0.0\n        means = self.shared_head(contexts).squeeze(-1)\n        return means * active.to(dtype=means.dtype)\n\n\nclass SharedPerAssetActorCriticPolicy(MultiInputActorCriticPolicy):\n""",
        """    def active_mask(self, actor_latent: torch.Tensor) -> torch.Tensor:\n        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)\n        return contexts[:, :, -1] > 0.5\n\n    def forward(self, actor_latent: torch.Tensor) -> torch.Tensor:\n        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)\n        active = self.active_mask(actor_latent)\n        means = self.shared_head(contexts).squeeze(-1)\n        return means * active.to(dtype=means.dtype)\n\n\nclass MaskedSharedSquashedDiagGaussianDistribution(\n    SquashedDiagGaussianDistribution\n):\n    \"\"\"One shared exploration scale with inactive dimensions excluded.\"\"\"\n\n    def __init__(self, action_dim: int) -> None:\n        super().__init__(action_dim)\n        self.active_mask: torch.Tensor | None = None\n\n    def set_active_mask(self, active_mask: torch.Tensor) -> None:\n        mask = active_mask.to(dtype=torch.bool)\n        if mask.ndim != 2 or mask.shape[1] != self.action_dim:\n            raise ValueError(\"active action mask does not match action dimensions\")\n        self.active_mask = mask\n\n    def _masked(self, actions: torch.Tensor) -> torch.Tensor:\n        if self.active_mask is None:\n            raise RuntimeError(\"active action mask is not configured\")\n        return actions * self.active_mask.to(dtype=actions.dtype)\n\n    def sample(self) -> torch.Tensor:\n        return self._masked(super().sample())\n\n    def mode(self) -> torch.Tensor:\n        return self._masked(super().mode())\n\n    def log_prob(\n        self,\n        actions: torch.Tensor,\n        gaussian_actions: torch.Tensor | None = None,\n    ) -> torch.Tensor:\n        if self.active_mask is None:\n            raise RuntimeError(\"active action mask is not configured\")\n        if gaussian_actions is None:\n            gaussian_actions = TanhBijector.inverse(actions)\n        per_dimension = self.distribution.log_prob(gaussian_actions)\n        per_dimension -= torch.log(1 - actions**2 + self.epsilon)\n        return (\n            per_dimension * self.active_mask.to(dtype=per_dimension.dtype)\n        ).sum(dim=1)\n\n\nclass SharedPerAssetActorCriticPolicy(MultiInputActorCriticPolicy):\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '    action_distribution_name = "squashed_diag_gaussian"',
        '    action_distribution_name = "masked_shared_squashed_diag_gaussian"',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """    def _build(self, lr_schedule: Any) -> None:\n        self.action_dist = SquashedDiagGaussianDistribution(self.shared_actor_n_symbols)\n        super()._build(lr_schedule)\n        context_dim = 2 * self.shared_actor_d_model + self.shared_actor_global_dim\n""",
        """    def _build(self, lr_schedule: Any) -> None:\n        self.action_dist = MaskedSharedSquashedDiagGaussianDistribution(\n            self.shared_actor_n_symbols\n        )\n        super()._build(lr_schedule)\n        context_dim = 2 * self.shared_actor_d_model + self.shared_actor_global_dim + 1\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """        if self.ortho_init:\n            self.action_net.apply(partial(self.init_weights, gain=0.01))\n        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]\n""",
        """        if self.ortho_init:\n            self.action_net.apply(partial(self.init_weights, gain=0.01))\n        self.log_std = nn.Parameter(\n            torch.full((1,), float(self.log_std_init), device=self.device)\n        )\n        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        """    def _get_constructor_parameters(self) -> dict[str, Any]:\n""",
        """    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:\n        if not isinstance(\n            self.action_dist, MaskedSharedSquashedDiagGaussianDistribution\n        ):\n            raise RuntimeError(\"shared policy action distribution is invalid\")\n        self.action_dist.set_active_mask(self.action_net.active_mask(latent_pi))\n        mean_actions = self.action_net(latent_pi)\n        return self.action_dist.proba_distribution(mean_actions, self.log_std)\n\n    def _get_constructor_parameters(self) -> dict[str, Any]:\n""",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '    "AssetSetFeatureExtractor",\n',
        '    "AssetSetFeatureExtractor",\n    "MaskedSharedSquashedDiagGaussianDistribution",\n',
    )

    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "from trade_rl.data.market import MarketDataset\n",
        """from trade_rl.data.market import MarketDataset\nfrom trade_rl.risk.portfolio import PortfolioRiskConfig\n""",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        """    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)\n    positions: tuple[float, ...] = (-1.0, 0.0, 1.0)\n""",
        """    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)\n    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)\n    positions: tuple[float, ...] = (-1.0, 0.0, 1.0)\n""",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        """        cost = self.execution_cost\n        if (\n""",
        """        if not isinstance(self.portfolio_risk, PortfolioRiskConfig):\n            raise ValueError(\"oracle portfolio_risk must be PortfolioRiskConfig\")\n        if any(\n            value is not None\n            for value in (\n                self.portfolio_risk.volatility_target,\n                self.portfolio_risk.max_abs_beta,\n                self.portfolio_risk.max_stress_loss,\n            )\n        ):\n            raise ValueError(\n                \"oracle portfolio risk does not support covariance, beta, or stress inputs\"\n            )\n        cost = self.execution_cost\n        if (\n""",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        """def _transition_matrices(\n""",
        """def project_portfolio_targets(\n    targets: np.ndarray,\n    *,\n    portfolio_value: np.ndarray,\n    market_notional: np.ndarray,\n    config: PortfolioRiskConfig,\n) -> np.ndarray:\n    \"\"\"Vectorized maintained portfolio projection for oracle transitions.\"\"\"\n\n    weights = np.asarray(targets, dtype=np.float64).copy()\n    values = np.asarray(portfolio_value, dtype=np.float64).reshape(-1)\n    liquidity = np.asarray(market_notional, dtype=np.float64).reshape(-1)\n    if weights.ndim != 3 or weights.shape[0] != values.size:\n        raise ValueError(\"oracle portfolio target batch does not match portfolio values\")\n    if weights.shape[2] != liquidity.size:\n        raise ValueError(\"oracle portfolio target batch does not match liquidity\")\n    if (\n        not np.isfinite(weights).all()\n        or not np.isfinite(values).all()\n        or not np.isfinite(liquidity).all()\n        or np.any(values <= 0.0)\n        or np.any(liquidity < 0.0)\n    ):\n        raise ValueError(\"oracle portfolio projection inputs are invalid\")\n    if config.max_abs_weight is not None:\n        weights = np.clip(weights, -config.max_abs_weight, config.max_abs_weight)\n    if config.max_position_to_market_notional is not None:\n        caps = (\n            liquidity[None, None, :]\n            * config.max_position_to_market_notional\n            / values[:, None, None]\n        )\n        weights = np.clip(weights, -caps, caps)\n    if config.max_net_exposure is not None:\n        net = np.abs(weights.sum(axis=2, keepdims=True))\n        scale = np.minimum(\n            1.0,\n            config.max_net_exposure / np.maximum(net, _EPSILON),\n        )\n        weights *= scale\n    return weights\n\n\ndef _transition_matrices(\n""",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        """    execution_index = close_index + 1\n    requested_targets = _effective_target_matrix(config, current_weights, targets)\n    desired_delta = requested_targets - current_weights[:, None, :]\n""",
        """    execution_index = close_index + 1\n    requested_targets = _effective_target_matrix(config, current_weights, targets)\n    prices = dataset.open[execution_index]\n    market_notional = dataset.market_notional(\n        execution_index,\n        prices,\n        volume=dataset.volume[close_index],\n    )\n    requested_targets = project_portfolio_targets(\n        requested_targets,\n        portfolio_value=np.maximum(open_equity, _EPSILON),\n        market_notional=market_notional,\n        config=config.portfolio_risk,\n    )\n    desired_delta = requested_targets - current_weights[:, None, :]\n""",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        """    requested = np.abs(desired_delta) * open_equity[:, None, None]\n    prices = dataset.open[execution_index]\n    market_notional = dataset.market_notional(\n        execution_index,\n        prices,\n        volume=dataset.volume[close_index],\n    )\n""",
        """    requested = np.abs(desired_delta) * open_equity[:, None, None]\n""",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        '    "oracle_target_path",\n',
        '    "oracle_target_path",\n    "project_portfolio_targets",\n',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        """                    teacher_config = OracleTeacherConfig(\n                        execution_cost=unwrapped_teacher.config.execution_cost,\n""",
        """                    teacher_config = OracleTeacherConfig(\n                        execution_cost=unwrapped_teacher.config.execution_cost,\n                        portfolio_risk=unwrapped_teacher.portfolio_risk.config,\n""",
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '                "feature_names": dataset.feature_names,\n',
        '                "feature_config_digest": dataset.feature_config_digest,\n                "feature_names": dataset.feature_names,\n',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '                "dataset_id": dataset.dataset_id,\n                "schema_version": "dataset_reference_v1",\n',
        '                "dataset_id": dataset.dataset_id,\n                "feature_config_digest": dataset.feature_config_digest,\n                "schema_version": "dataset_reference_v2",\n',
    )
    replace_once(
        "trade_rl/integrations/sb3_serving.py",
        """        expected_bar_hours = self.dataset_reference.get(\"bar_hours\")\n        if dataset.symbols != expected_symbols:\n""",
        """        expected_bar_hours = self.dataset_reference.get(\"bar_hours\")\n        expected_feature_config_digest = self.dataset_reference.get(\n            \"feature_config_digest\"\n        )\n        if (\n            not isinstance(expected_feature_config_digest, str)\n            or len(expected_feature_config_digest) != 64\n            or any(\n                character not in \"0123456789abcdef\"\n                for character in expected_feature_config_digest\n            )\n        ):\n            raise ValueError(\n                \"dataset reference feature_config_digest is missing or invalid\"\n            )\n        if dataset.symbols != expected_symbols:\n""",
    )
    replace_once(
        "trade_rl/integrations/sb3_serving.py",
        """        if dataset.global_feature_names != expected_globals:\n            raise ValueError(\n                \"serving rolling dataset global feature order does not match training\"\n            )\n        if (\n""",
        """        if dataset.global_feature_names != expected_globals:\n            raise ValueError(\n                \"serving rolling dataset global feature order does not match training\"\n            )\n        if dataset.feature_config_digest != expected_feature_config_digest:\n            raise ValueError(\n                \"serving rolling dataset feature recipe does not match training\"\n            )\n        if (\n""",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit(
            "usage: apply_architecture_audit_fixes.py tests|implementation"
        )
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
