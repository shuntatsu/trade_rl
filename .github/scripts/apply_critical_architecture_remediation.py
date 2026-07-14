from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected snippet not found in {path}: {old[:80]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8") if file.exists() else ""
    if marker in text:
        return
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text + content, encoding="utf-8")


# ResidualMarketEnv: contract multiplier propagation and portfolio risk integration.
replace_once(
    "trade_rl/rl/environment.py",
    "from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget\n",
    "from trade_rl.risk.portfolio import PortfolioRiskModel\n"
    "from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget\n",
)
replace_once(
    "trade_rl/rl/environment.py",
    "        pre_trade_risk: PreTradeRisk | None = None,\n        normalizer: ObservationNormalizer | None = None,\n",
    "        pre_trade_risk: PreTradeRisk | None = None,\n"
    "        portfolio_risk: PortfolioRiskModel | None = None,\n"
    "        normalizer: ObservationNormalizer | None = None,\n",
)
replace_once(
    "trade_rl/rl/environment.py",
    "        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()\n        self.normalizer = normalizer\n",
    "        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()\n"
    "        self.portfolio_risk = portfolio_risk or PortfolioRiskModel()\n"
    "        self.normalizer = normalizer\n",
)
replace_once(
    "trade_rl/rl/environment.py",
    '            "pre_trade_risk": asdict(self.pre_trade_risk.config),\n',
    '            "portfolio_risk": asdict(self.portfolio_risk.config),\n'
    '            "pre_trade_risk": asdict(self.pre_trade_risk.config),\n',
)
replace_once(
    "trade_rl/rl/environment.py",
    "        self.hybrid = BookState.zero(\n            dataset.n_symbols,\n            self.config.initial_capital,\n            initial_prices,\n        )\n",
    "        self.hybrid = BookState.zero(\n"
    "            dataset.n_symbols,\n"
    "            self.config.initial_capital,\n"
    "            initial_prices,\n"
    "            contract_multipliers=dataset.contract_multipliers,\n"
    "        )\n",
)
replace_once(
    "trade_rl/rl/environment.py",
    "            max_gross=self.pre_trade_risk.config.max_gross,\n        )\n",
    "            max_gross=self.pre_trade_risk.config.max_gross,\n"
    "            contract_multipliers=self.dataset.contract_multipliers,\n"
    "        )\n",
)
replace_once(
    "trade_rl/rl/environment.py",
    "            if supplied.quantities.shape != (self.dataset.n_symbols,):\n                raise ValueError(\"initial_book does not match dataset symbols\")\n",
    "            if supplied.quantities.shape != (self.dataset.n_symbols,):\n"
    "                raise ValueError(\"initial_book does not match dataset symbols\")\n"
    "            if not np.array_equal(\n"
    "                np.asarray(supplied.contract_multipliers),\n"
    "                self.dataset.contract_multipliers,\n"
    "            ):\n"
    "                raise ValueError(\n"
    "                    \"initial_book contract multipliers do not match dataset\"\n"
    "                )\n",
)
replace_once(
    "trade_rl/rl/environment.py",
    "    def _constrain_target(\n        self,\n        proposal: np.ndarray,\n        book: BookState,\n    ) -> RiskConstrainedTarget:\n        target = np.asarray(proposal, dtype=np.float64).reshape(-1).copy()\n        if target.shape != (self.dataset.n_symbols,) or not np.isfinite(target).all():\n            raise ValueError(\"proposal does not match dataset symbols\")\n        return self.pre_trade_risk.constrain(\n            target,\n            current=book.weights,\n            drawdown=self._drawdown(book),\n        )\n",
    "    def _market_notional(self, index: int) -> np.ndarray:\n"
    "        prices = self.dataset.close[index]\n"
    "        volume = self.dataset.volume[index]\n"
    "        return np.asarray(\n"
    "            [\n"
    "                float(value)\n"
    "                if str(getattr(unit, \"value\", unit)) == \"quote_notional\"\n"
    "                else float(price * value)\n"
    "                for price, value, unit in zip(\n"
    "                    prices, volume, self.dataset.volume_units, strict=True\n"
    "                )\n"
    "            ],\n"
    "            dtype=np.float64,\n"
    "        )\n\n"
    "    def _constrain_target(\n"
    "        self,\n"
    "        proposal: np.ndarray,\n"
    "        book: BookState,\n"
    "    ) -> RiskConstrainedTarget:\n"
    "        target = np.asarray(proposal, dtype=np.float64).reshape(-1).copy()\n"
    "        if target.shape != (self.dataset.n_symbols,) or not np.isfinite(target).all():\n"
    "            raise ValueError(\"proposal does not match dataset symbols\")\n"
    "        pretrade = self.pre_trade_risk.constrain(\n"
    "            target,\n"
    "            current=book.weights,\n"
    "            drawdown=self._drawdown(book),\n"
    "        )\n"
    "        portfolio = self.portfolio_risk.constrain(\n"
    "            pretrade.weights,\n"
    "            portfolio_value=max(book.portfolio_value, 1e-12),\n"
    "            market_notional=self._market_notional(self.current_index),\n"
    "        )\n"
    "        final_weights = np.asarray(portfolio.weights, dtype=np.float64)\n"
    "        constrained_turnover = float(np.abs(final_weights - book.weights).sum())\n"
    "        reasons = tuple(\n"
    "            dict.fromkeys(\n"
    "                (*pretrade.reasons, *(f\"portfolio:{item}\" for item in portfolio.reasons))\n"
    "            )\n"
    "        )\n"
    "        return RiskConstrainedTarget(\n"
    "            weights=final_weights,\n"
    "            requested_turnover=pretrade.requested_turnover,\n"
    "            constrained_turnover=constrained_turnover,\n"
    "            was_constrained=bool(reasons),\n"
    "            reasons=reasons,\n"
    "            risk_scale=pretrade.risk_scale,\n"
    "            projection_l1=float(np.abs(target - final_weights).sum()),\n"
    "            turnover_overridden=pretrade.turnover_overridden,\n"
    "        )\n",
)

# Training run configuration and factories.
replace_once(
    "trade_rl/workflows/training_run.py",
    "from dataclasses import asdict, dataclass, replace\n",
    "from dataclasses import asdict, dataclass, field, replace\n",
)
replace_once(
    "trade_rl/workflows/training_run.py",
    "from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig\n",
    "from trade_rl.risk.portfolio import (\n"
    "    PortfolioRiskConfig,\n"
    "    PortfolioRiskModel,\n"
    ")\n"
    "from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig\n",
)
replace_once(
    "trade_rl/workflows/training_run.py",
    "    alpha_contract: AlphaContract\n    alpha_artifact: Path | None = None\n",
    "    alpha_contract: AlphaContract\n"
    "    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)\n"
    "    alpha_artifact: Path | None = None\n",
)
replace_once(
    "trade_rl/workflows/training_run.py",
    "            risk=PreTradeRiskConfig(**_mapping(payload.get(\"risk\"), field=\"risk\")),\n            reward=reward,\n",
    "            risk=PreTradeRiskConfig(**_mapping(payload.get(\"risk\"), field=\"risk\")),\n"
    "            reward=reward,\n"
    "            portfolio_risk=PortfolioRiskConfig(\n"
    "                **_mapping(payload.get(\"portfolio_risk\"), field=\"portfolio_risk\")\n"
    "            ),\n",
)
replace_once(
    "trade_rl/workflows/training_run.py",
    '            "risk": asdict(self.risk),\n            "reward": asdict(self.reward),\n',
    '            "portfolio_risk": asdict(self.portfolio_risk),\n'
    '            "risk": asdict(self.risk),\n'
    '            "reward": asdict(self.reward),\n',
)
replace_once(
    "trade_rl/workflows/training_run.py",
    "            pre_trade_risk=PreTradeRisk(config.risk),\n            config=config.environment,\n",
    "            pre_trade_risk=PreTradeRisk(config.risk),\n"
    "            portfolio_risk=PortfolioRiskModel(config.portfolio_risk),\n"
    "            config=config.environment,\n",
)

# Environment experiment identity.
replace_once(
    "trade_rl/rl/configuration.py",
    "from dataclasses import asdict, dataclass\n",
    "from dataclasses import asdict, dataclass, field\n",
)
replace_once(
    "trade_rl/rl/configuration.py",
    "from trade_rl.risk.pretrade import PreTradeRiskConfig\n",
    "from trade_rl.risk.portfolio import PortfolioRiskConfig\n"
    "from trade_rl.risk.pretrade import PreTradeRiskConfig\n",
)
replace_once(
    "trade_rl/rl/configuration.py",
    "    trend: TrendConfig\n    alpha_artifact_digest: str | None = None\n",
    "    trend: TrendConfig\n"
    "    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)\n"
    "    alpha_artifact_digest: str | None = None\n",
)
replace_once(
    "trade_rl/rl/configuration.py",
    '            "risk": asdict(self.risk),\n            "schema_version": self.schema_version,\n',
    '            "portfolio_risk": asdict(self.portfolio_risk),\n'
    '            "risk": asdict(self.risk),\n'
    '            "schema_version": self.schema_version,\n',
)
replace_once(
    "trade_rl/rl/configuration.py",
    "        trend: TrendConfig,\n        alpha_artifact_digest: str | None = None,\n",
    "        trend: TrendConfig,\n"
    "        portfolio_risk: PortfolioRiskConfig | None = None,\n"
    "        alpha_artifact_digest: str | None = None,\n",
)
replace_once(
    "trade_rl/rl/configuration.py",
    '            "risk": asdict(risk),\n            "schema_version": "environment_experiment_manifest_v1",\n',
    '            "portfolio_risk": asdict(portfolio_risk or PortfolioRiskConfig()),\n'
    '            "risk": asdict(risk),\n'
    '            "schema_version": "environment_experiment_manifest_v1",\n',
)
replace_once(
    "trade_rl/rl/configuration.py",
    "            trend=trend,\n            alpha_artifact_digest=alpha_artifact_digest,\n",
    "            trend=trend,\n"
    "            portfolio_risk=portfolio_risk or PortfolioRiskConfig(),\n"
    "            alpha_artifact_digest=alpha_artifact_digest,\n",
)

# Walk-forward and training environment composition.
for path in (
    "trade_rl/workflows/walk_forward_evaluation.py",
    "trade_rl/workflows/market_walk_forward.py",
):
    replace_once(
        path,
        "from trade_rl.risk.pretrade import PreTradeRisk\n",
        "from trade_rl.risk.portfolio import PortfolioRiskModel\n"
        "from trade_rl.risk.pretrade import PreTradeRisk\n",
    )
    replace_once(
        path,
        "        pre_trade_risk=PreTradeRisk(run.risk),\n",
        "        pre_trade_risk=PreTradeRisk(run.risk),\n"
        "        portfolio_risk=PortfolioRiskModel(run.portfolio_risk),\n",
    )

replace_once(
    "trade_rl/workflows/market_walk_forward_config.py",
    '                    "risk": asdict(item.run.risk),\n                    "reward": asdict(item.run.reward),\n',
    '                    "portfolio_risk": asdict(item.run.portfolio_risk),\n'
    '                    "risk": asdict(item.run.risk),\n'
    '                    "reward": asdict(item.run.reward),\n',
)

# Registry copies external attestations beside staged and installed bundle versions.
replace_once(
    "trade_rl/serving/registry.py",
    "from trade_rl.serving.bundle import ServingBundle, load_serving_bundle\n",
    "from trade_rl.release.attestation import default_attestation_path\n"
    "from trade_rl.serving.bundle import ServingBundle, load_serving_bundle\n",
)
replace_once(
    "trade_rl/serving/registry.py",
    "        else:\n            stage = self.staging_root / digest\n            if stage.exists():\n                shutil.rmtree(stage)\n            shutil.copytree(source, stage)\n            staged = load_serving_bundle(stage)\n            self._require_activatable(staged)\n            if staged.manifest.bundle_digest != digest:\n                shutil.rmtree(stage, ignore_errors=True)\n                raise ValueError(\"staged bundle digest changed during registry copy\")\n            os.replace(stage, destination)\n            _fsync_directory(self.versions_root)\n            installed = load_serving_bundle(destination)\n",
    "        else:\n"
    "            stage = self.staging_root / digest\n"
    "            stage_attestation = default_attestation_path(stage)\n"
    "            destination_attestation = default_attestation_path(destination)\n"
    "            source_attestation = default_attestation_path(source)\n"
    "            if stage.exists():\n"
    "                shutil.rmtree(stage)\n"
    "            stage_attestation.unlink(missing_ok=True)\n"
    "            try:\n"
    "                shutil.copytree(source, stage)\n"
    "                if source_attestation.is_file():\n"
    "                    shutil.copy2(source_attestation, stage_attestation)\n"
    "                staged = load_serving_bundle(stage)\n"
    "                self._require_activatable(staged)\n"
    "                if staged.manifest.bundle_digest != digest:\n"
    "                    raise ValueError(\n"
    "                        \"staged bundle digest changed during registry copy\"\n"
    "                    )\n"
    "                os.replace(stage, destination)\n"
    "                if stage_attestation.is_file():\n"
    "                    os.replace(stage_attestation, destination_attestation)\n"
    "                _fsync_directory(self.versions_root)\n"
    "                installed = load_serving_bundle(destination)\n"
    "            except Exception:\n"
    "                shutil.rmtree(stage, ignore_errors=True)\n"
    "                stage_attestation.unlink(missing_ok=True)\n"
    "                raise\n",
)
replace_once(
    "trade_rl/serving/registry.py",
    "        return ServingBundle(root=destination, manifest=installed.manifest)\n",
    "        return installed\n",
)

# Focused regression tests are generated as a new file.
append_once(
    "tests/test_critical_architecture_remediation.py",
    "def test_environment_uses_dataset_contract_multipliers",
    '''from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import create_bundle
from trade_rl.data.market import MarketDataset
from trade_rl.release.attestation import (
    ReleaseAttestation,
    default_attestation_path,
    write_release_attestation,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.serving.bundle import load_serving_bundle
from trade_rl.serving.registry import ServingRegistry
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset(*, multiplier: float = 0.1) -> MarketDataset:
    n_bars = 24
    close = np.linspace(100.0, 123.0, n_bars).reshape(-1, 1)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("PERP",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 4), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("a", "b", "c", "d"),
        periods_per_year=8_760,
        contract_multipliers=np.array([multiplier]),
    )


def _environment(
    dataset: MarketDataset,
    *,
    portfolio_risk: PortfolioRiskModel | None = None,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        portfolio_risk=portfolio_risk,
        config=ResidualMarketEnvConfig(
            initial_capital=1_000.0,
            episode_bars=8,
            decision_every=1,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_environment_uses_dataset_contract_multipliers() -> None:
    dataset = _dataset()
    environment = _environment(dataset)
    environment.reset(seed=0, options={"start_idx": 8})

    _, _, terminated, truncated, _ = environment.step(
        np.zeros(environment.action_spec.size, dtype=np.float32)
    )

    assert not terminated
    assert not truncated
    np.testing.assert_array_equal(
        environment.hybrid.contract_multipliers,
        dataset.contract_multipliers,
    )


def test_restore_rejects_contract_multiplier_mismatch() -> None:
    dataset = _dataset()
    environment = _environment(dataset)
    incompatible = BookState.zero(1, 1_000.0, dataset.close[8])

    with pytest.raises(ValueError, match="contract multipliers"):
        environment.reset(
            seed=0,
            options={
                "start_idx": 8,
                "initial_state_mode": "restore",
                "initial_book": incompatible,
            },
        )


def test_portfolio_risk_changes_final_target_and_environment_identity() -> None:
    dataset = _dataset(multiplier=1.0)
    unrestricted = _environment(dataset)
    restricted = _environment(
        dataset,
        portfolio_risk=PortfolioRiskModel(
            PortfolioRiskConfig(max_abs_weight=0.05)
        ),
    )
    assert unrestricted.environment_digest != restricted.environment_digest
    restricted.reset(seed=0, options={"start_idx": 8})

    _, _, _, _, info = restricted.step(
        np.zeros(restricted.action_spec.size, dtype=np.float32)
    )

    risk = info["hybrid_risk"]
    assert np.max(np.abs(risk.weights)) <= 0.05 + 1e-12
    assert "portfolio:max_abs_weight" in risk.reasons


def _write_external_attestation(source: Path) -> ReleaseAttestation:
    manifest = load_serving_bundle(source).manifest
    attestation = ReleaseAttestation.create(
        bundle_digest=manifest.bundle_digest,
        dataset_id=manifest.dataset_id,
        selection_evaluation_digest="1" * 64,
        gate_evaluation_digest="2" * 64,
        gate_evidence_digest="3" * 64,
        selected_policy_digest=manifest.policy_digest,
        git_commit="e" * 40,
        dependency_digest="4" * 64,
        approver="architecture-audit",
        approved_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    write_release_attestation(default_attestation_path(source), attestation)
    return attestation


def test_registry_installs_external_release_attestation(tmp_path: Path) -> None:
    source = create_bundle(tmp_path / "source", release_digest=None)
    expected = _write_external_attestation(source)
    registry = ServingRegistry(tmp_path / "registry")

    active = registry.activate(source)
    reloaded = registry.active_bundle()

    assert active.release is not None
    assert active.release.digest == expected.digest
    assert reloaded.release is not None
    assert reloaded.release.digest == expected.digest
    assert default_attestation_path(reloaded.root).is_file()
''',
)
