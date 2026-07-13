from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def replace(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Preserve margin-deficit severity through economic termination and liquidation.
replace(
    "trade_rl/simulation/accounting.py",
    "    maintenance_requirement: float = 0.0\n    insolvent: bool = False\n",
    "    maintenance_requirement: float = 0.0\n    margin_deficit: float = 0.0\n    insolvent: bool = False\n",
)
replace(
    "trade_rl/simulation/accounting.py",
    '            ("maintenance_requirement", self.maintenance_requirement),\n',
    '            ("maintenance_requirement", self.maintenance_requirement),\n'
    '            ("margin_deficit", self.margin_deficit),\n',
)
replace(
    "trade_rl/simulation/accounting.py",
    "            or self.maintenance_requirement < 0.0\n",
    "            or self.maintenance_requirement < 0.0\n"
    "            or self.margin_deficit < 0.0\n",
)
replace(
    "trade_rl/simulation/accounting.py",
    "            maintenance_requirement=self.maintenance_requirement,\n"
    "            insolvent=self.insolvent,\n",
    "            maintenance_requirement=self.maintenance_requirement,\n"
    "            margin_deficit=self.margin_deficit,\n"
    "            insolvent=self.insolvent,\n",
)
replace(
    "trade_rl/simulation/accounting.py",
    "        self.maintenance_requirement = requirement\n        if (\n",
    "        self.maintenance_requirement = requirement\n"
    "        self.margin_deficit = max(\n"
    "            0.0,\n"
    "            requirement - max(self.portfolio_value, 0.0),\n"
    "        )\n"
    "        if (\n",
)

replace(
    "trade_rl/simulation/execution.py",
    "        maintenance_required = self.cost.maintenance_margin_rate * gross_notional\n"
    '        if self.cost.margin_mode == "isolated" and gross_notional > 0.0:\n',
    "        maintenance_required = self.cost.maintenance_margin_rate * gross_notional\n"
    "        book.margin_deficit = max(\n"
    "            book.margin_deficit,\n"
    "            maintenance_required - collateral_equity,\n"
    "            0.0,\n"
    "        )\n"
    '        if self.cost.margin_mode == "isolated" and gross_notional > 0.0:\n',
)
replace(
    "trade_rl/simulation/execution.py",
    "            if np.any(isolated_equity + _TOLERANCE < isolated_required):\n"
    "                book.terminate(EconomicTerminationReason.MARGIN_CALL)\n",
    "            isolated_deficit = np.maximum(\n"
    "                isolated_required - isolated_equity,\n"
    "                0.0,\n"
    "            )\n"
    "            book.margin_deficit = max(\n"
    "                book.margin_deficit,\n"
    "                float(np.max(isolated_deficit, initial=0.0)),\n"
    "            )\n"
    "            if np.any(isolated_deficit > _TOLERANCE):\n"
    "                book.terminate(EconomicTerminationReason.MARGIN_CALL)\n",
)

# Add a continuous margin-deficit reward component.
rewards = Path("trade_rl/rl/rewards.py")
text = rewards.read_text(encoding="utf-8")
substitutions = [
    (
        "    terminal_equity_weight: float = 1.0\n"
        "    equity_floor_fraction: float = 1e-9\n",
        "    terminal_equity_weight: float = 1.0\n"
        "    margin_deficit_weight: float = 1.0\n"
        "    equity_floor_fraction: float = 1e-9\n",
    ),
    (
        '            ("terminal_equity_weight", self.terminal_equity_weight),\n'
        '            ("equity_floor_fraction", self.equity_floor_fraction),\n',
        '            ("terminal_equity_weight", self.terminal_equity_weight),\n'
        '            ("margin_deficit_weight", self.margin_deficit_weight),\n'
        '            ("equity_floor_fraction", self.equity_floor_fraction),\n',
    ),
    (
        "            self.terminal_equity_weight,\n        )\n",
        "            self.terminal_equity_weight,\n"
        "            self.margin_deficit_weight,\n"
        "        )\n",
    ),
    (
        "    terminal_equity_shortfall: float\n    absolute_component: float\n",
        "    terminal_equity_shortfall: float\n"
        "    margin_deficit: float\n"
        "    absolute_component: float\n",
    ),
    (
        "    terminal_penalty: float\n    unscaled_total: float\n",
        "    terminal_penalty: float\n"
        "    margin_penalty: float\n"
        "    unscaled_total: float\n",
    ),
    (
        "        projection_distance: float = 0.0,\n"
        "        hybrid_equity_fraction: float = 1.0,\n",
        "        projection_distance: float = 0.0,\n"
        "        hybrid_margin_deficit_fraction: float = 0.0,\n"
        "        hybrid_equity_fraction: float = 1.0,\n",
    ),
    (
        '            ("projection_distance", projection_distance),\n'
        '            ("hybrid_equity_fraction", hybrid_equity_fraction),\n',
        '            ("projection_distance", projection_distance),\n'
        '            ("hybrid_margin_deficit_fraction", hybrid_margin_deficit_fraction),\n'
        '            ("hybrid_equity_fraction", hybrid_equity_fraction),\n',
    ),
    (
        "        if projection_distance < 0.0:\n"
        '            raise ValueError("projection_distance must be non-negative")\n',
        "        if projection_distance < 0.0:\n"
        '            raise ValueError("projection_distance must be non-negative")\n'
        "        if hybrid_margin_deficit_fraction < 0.0:\n"
        "            raise ValueError(\n"
        '                "hybrid_margin_deficit_fraction must be non-negative"\n'
        "            )\n",
    ),
    (
        "        terminal_penalty = self.config.terminal_equity_weight * terminal_shortfall\n"
        "        unscaled_total = (\n",
        "        terminal_penalty = self.config.terminal_equity_weight * terminal_shortfall\n"
        "        margin_deficit = float(hybrid_margin_deficit_fraction)\n"
        "        margin_penalty = self.config.margin_deficit_weight * margin_deficit\n"
        "        unscaled_total = (\n",
    ),
    (
        "            - terminal_penalty\n        )\n",
        "            - terminal_penalty\n"
        "            - margin_penalty\n"
        "        )\n",
    ),
    (
        "            terminal_equity_shortfall=terminal_shortfall,\n"
        "            absolute_component=absolute_component,\n",
        "            terminal_equity_shortfall=terminal_shortfall,\n"
        "            margin_deficit=margin_deficit,\n"
        "            absolute_component=absolute_component,\n",
    ),
    (
        "            terminal_penalty=terminal_penalty,\n"
        "            unscaled_total=unscaled_total,\n",
        "            terminal_penalty=terminal_penalty,\n"
        "            margin_penalty=margin_penalty,\n"
        "            unscaled_total=unscaled_total,\n",
    ),
]
for old, new in substitutions:
    if old not in text:
        raise RuntimeError(f"reward replacement not found: {old[:100]!r}")
    text = text.replace(old, new, 1)
rewards.write_text(text, encoding="utf-8")

replace(
    "trade_rl/rl/environment.py",
    "            projection_distance=projection_distance,\n"
    "            hybrid_equity_fraction=max(self.hybrid.portfolio_value, 0.0)\n",
    "            projection_distance=projection_distance,\n"
    "            hybrid_margin_deficit_fraction=(\n"
    "                self.hybrid.margin_deficit / self.config.initial_capital\n"
    "            ),\n"
    "            hybrid_equity_fraction=max(self.hybrid.portfolio_value, 0.0)\n",
)

# Make serving identity binding fail closed by default.
runtime = dedent(
    '''
    """Thread-safe serving runtime with validated fail-closed hot swaps."""

    from __future__ import annotations

    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    from threading import RLock
    from typing import Protocol

    import numpy as np

    from trade_rl.domain.common import require_sha256
    from trade_rl.domain.selection import PolicyMode
    from trade_rl.rl.actions import ACTION_SCHEMA
    from trade_rl.rl.observations import OBSERVATION_SCHEMA
    from trade_rl.serving.bundle import ServingBundle, load_serving_bundle


    class LoadedPolicy(Protocol):
        def predict(self, observation: np.ndarray) -> np.ndarray: ...


    class PolicyLoader(Protocol):
        def load(self, bundle: ServingBundle) -> LoadedPolicy: ...


    class _BaselineIdentityPolicy:
        def __init__(self, action_size: int) -> None:
            self.action_size = action_size

        def predict(self, observation: np.ndarray) -> np.ndarray:
            del observation
            return np.zeros(self.action_size, dtype=np.float32)


    @dataclass(frozen=True, slots=True)
    class RuntimeIdentityContract:
        """Exact deployment identity required before a bundle can activate."""

        environment_digest: str
        action_names: tuple[str, ...]
        action_spec_digest: str
        normalizer_digest: str | None
        alpha_artifact_digest: str | None = None
        factor_artifact_digest: str | None = None

        def __post_init__(self) -> None:
            require_sha256(self.environment_digest, field="environment_digest")
            require_sha256(self.action_spec_digest, field="action_spec_digest")
            if not self.action_names or any(not name for name in self.action_names):
                raise ValueError("action_names must be non-empty")
            if len(set(self.action_names)) != len(self.action_names):
                raise ValueError("action_names must be unique")
            for field_name, value in (
                ("normalizer_digest", self.normalizer_digest),
                ("alpha_artifact_digest", self.alpha_artifact_digest),
                ("factor_artifact_digest", self.factor_artifact_digest),
            ):
                if value is not None:
                    require_sha256(value, field=field_name)


    @dataclass(frozen=True, slots=True)
    class RuntimeSnapshot:
        bundle_digest: str
        dataset_id: str
        action_schema: str
        action_size: int
        action_names: tuple[str, ...]
        action_spec_digest: str | None
        observation_schema: str
        observation_size: int
        environment_digest: str
        initial_capital: float
        policy_mode: PolicyMode
        policy_digest: str | None
        signal_digest: str
        selection_digest: str
        release_digest: str | None
        alpha_artifact_digest: str | None
        factor_artifact_digest: str | None
        normalizer_digest: str | None
        bundle_created_at: datetime


    class ServingRuntime:
        """Validate and fully load a replacement before swapping live state."""

        def __init__(
            self,
            policy_loader: PolicyLoader | None = None,
            *,
            allow_unreleased: bool = False,
            identity_contract: RuntimeIdentityContract | None = None,
            allow_unbound_identity: bool = False,
            expected_environment_digest: str | None = None,
            expected_action_names: tuple[str, ...] | None = None,
            expected_action_spec_digest: str | None = None,
            expected_normalizer_digest: str | None = None,
            expected_alpha_artifact_digest: str | None = None,
            expected_factor_artifact_digest: str | None = None,
        ) -> None:
            legacy_values = (
                expected_environment_digest,
                expected_action_names,
                expected_action_spec_digest,
                expected_normalizer_digest,
                expected_alpha_artifact_digest,
                expected_factor_artifact_digest,
            )
            if identity_contract is not None and any(
                value is not None for value in legacy_values
            ):
                raise ValueError(
                    "identity_contract cannot be combined with legacy expected fields"
                )
            if identity_contract is None and any(
                value is not None for value in legacy_values
            ):
                if (
                    expected_environment_digest is None
                    or expected_action_names is None
                    or expected_action_spec_digest is None
                ):
                    raise ValueError(
                        "legacy serving identity requires environment, action names, "
                        "and action spec"
                    )
                identity_contract = RuntimeIdentityContract(
                    environment_digest=expected_environment_digest,
                    action_names=expected_action_names,
                    action_spec_digest=expected_action_spec_digest,
                    normalizer_digest=expected_normalizer_digest,
                    alpha_artifact_digest=expected_alpha_artifact_digest,
                    factor_artifact_digest=expected_factor_artifact_digest,
                )
            if not isinstance(allow_unbound_identity, bool):
                raise ValueError("allow_unbound_identity must be a boolean")
            if identity_contract is None and not allow_unbound_identity:
                raise ValueError(
                    "serving runtime requires an explicit identity contract"
                )
            self.policy_loader = policy_loader
            self.allow_unreleased = allow_unreleased
            self.identity_contract = identity_contract
            self.allow_unbound_identity = allow_unbound_identity
            self._lock = RLock()
            self._snapshot: RuntimeSnapshot | None = None
            self._policy: LoadedPolicy | None = None

        @staticmethod
        def _snapshot_for(bundle: ServingBundle) -> RuntimeSnapshot:
            manifest = bundle.manifest
            return RuntimeSnapshot(
                bundle_digest=manifest.bundle_digest,
                dataset_id=manifest.dataset_id,
                action_schema=manifest.action_schema,
                action_size=manifest.action_size,
                action_names=manifest.action_names,
                action_spec_digest=manifest.action_spec_digest,
                observation_schema=manifest.observation_schema,
                observation_size=manifest.observation_size,
                environment_digest=manifest.environment_digest,
                initial_capital=manifest.initial_capital,
                policy_mode=manifest.policy_mode,
                policy_digest=manifest.policy_digest,
                signal_digest=manifest.signal_digest,
                selection_digest=manifest.selection_digest,
                release_digest=manifest.release_digest,
                alpha_artifact_digest=manifest.alpha_artifact_digest,
                factor_artifact_digest=manifest.factor_artifact_digest,
                normalizer_digest=manifest.normalizer_digest,
                bundle_created_at=manifest.created_at,
            )

        @staticmethod
        def _validate_identity(
            manifest: object,
            contract: RuntimeIdentityContract,
        ) -> None:
            comparisons = (
                (
                    getattr(manifest, "environment_digest"),
                    contract.environment_digest,
                    "environment identity",
                ),
                (
                    getattr(manifest, "action_names"),
                    contract.action_names,
                    "action names",
                ),
                (
                    getattr(manifest, "action_spec_digest"),
                    contract.action_spec_digest,
                    "action spec",
                ),
                (
                    getattr(manifest, "normalizer_digest"),
                    contract.normalizer_digest,
                    "normalizer",
                ),
                (
                    getattr(manifest, "alpha_artifact_digest"),
                    contract.alpha_artifact_digest,
                    "alpha artifact",
                ),
                (
                    getattr(manifest, "factor_artifact_digest"),
                    contract.factor_artifact_digest,
                    "factor artifact",
                ),
            )
            for observed, expected, label in comparisons:
                if observed != expected:
                    raise ValueError(
                        f"serving bundle {label} does not match runtime"
                    )

        def activate(self, root: Path) -> RuntimeSnapshot:
            bundle = load_serving_bundle(root)
            manifest = bundle.manifest
            if manifest.release_digest is None and not self.allow_unreleased:
                raise ValueError(
                    "serving bundle requires an approved release identity"
                )
            if manifest.action_schema != ACTION_SCHEMA:
                raise ValueError(
                    "serving bundle action schema does not match runtime action schema"
                )
            if manifest.observation_schema != OBSERVATION_SCHEMA:
                raise ValueError(
                    "serving bundle observation schema does not match runtime schema"
                )
            contract = self.identity_contract
            if contract is not None:
                self._validate_identity(manifest, contract)
            elif not self.allow_unbound_identity:
                raise RuntimeError("serving identity contract was not configured")

            if manifest.policy_mode is PolicyMode.BASELINE_ONLY:
                candidate_policy: LoadedPolicy = _BaselineIdentityPolicy(
                    manifest.action_size
                )
            else:
                loader = self.policy_loader
                if loader is None:
                    raise RuntimeError(
                        "residual policy bundle requires a policy loader"
                    )
                candidate_policy = loader.load(bundle)

            candidate_snapshot = self._snapshot_for(bundle)
            with self._lock:
                self._policy = candidate_policy
                self._snapshot = candidate_snapshot
            return candidate_snapshot

        def snapshot(self) -> RuntimeSnapshot:
            with self._lock:
                snapshot = self._snapshot
            if snapshot is None:
                raise RuntimeError("serving runtime has no active snapshot")
            return snapshot

        def predict(self, observation: np.ndarray) -> np.ndarray:
            vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("observation must be a non-empty finite vector")
            with self._lock:
                policy = self._policy
                snapshot = self._snapshot
            if policy is None or snapshot is None:
                raise RuntimeError("serving runtime has no active policy")
            if vector.shape != (snapshot.observation_size,):
                raise ValueError("observation violates the active observation schema")
            raw_action = np.asarray(
                policy.predict(vector),
                dtype=np.float32,
            ).reshape(-1)
            if raw_action.shape != (snapshot.action_size,) or not np.isfinite(
                raw_action
            ).all():
                raise ValueError("policy output violates the residual action schema")
            if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
                raise ValueError(
                    "policy output violates the residual action schema bounds"
                )
            return raw_action.copy()
    '''
).lstrip()
Path("trade_rl/serving/runtime.py").write_text(runtime, encoding="utf-8")

replace(
    "trade_rl/serving/__init__.py",
    "    RuntimeSnapshot,\n    ServingRuntime,\n",
    "    RuntimeIdentityContract,\n    RuntimeSnapshot,\n    ServingRuntime,\n",
)
replace(
    "trade_rl/serving/__init__.py",
    '    "RuntimeSnapshot",\n',
    '    "RuntimeIdentityContract",\n    "RuntimeSnapshot",\n',
)

helpers = Path("tests/serving/helpers.py")
text = helpers.read_text(encoding="utf-8")
text = text.replace(
    "from trade_rl.serving.bundle import "
    "ServingBundleManifest, write_serving_bundle_manifest\n",
    "from trade_rl.serving.bundle import "
    "ServingBundleManifest, write_serving_bundle_manifest\n"
    "from trade_rl.serving.runtime import RuntimeIdentityContract\n",
)
text += dedent(
    '''

    def runtime_identity_contract(
        *,
        environment_digest: str = "d" * 64,
        action_names: tuple[str, ...] = ACTION_NAMES,
        action_spec_digest: str = ACTION_SPEC_DIGEST,
        normalizer_digest: str | None = "9" * 64,
        alpha_artifact_digest: str | None = None,
        factor_artifact_digest: str | None = None,
    ) -> RuntimeIdentityContract:
        return RuntimeIdentityContract(
            environment_digest=environment_digest,
            action_names=action_names,
            action_spec_digest=action_spec_digest,
            normalizer_digest=normalizer_digest,
            alpha_artifact_digest=alpha_artifact_digest,
            factor_artifact_digest=factor_artifact_digest,
        )
    '''
)
helpers.write_text(text, encoding="utf-8")

runtime_test = dedent(
    '''
    from __future__ import annotations

    from pathlib import Path

    import numpy as np
    import pytest

    from tests.serving.helpers import (
        ACTION_NAMES,
        OBSERVATION_SIZE,
        create_bundle,
        runtime_identity_contract,
    )
    from trade_rl.domain.selection import PolicyMode
    from trade_rl.serving.bundle import ServingBundle
    from trade_rl.serving.runtime import LoadedPolicy, ServingRuntime


    class ConstantPolicy:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def predict(self, observation: np.ndarray) -> np.ndarray:
            return self.value.copy()


    class Loader:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def load(self, bundle: ServingBundle) -> LoadedPolicy:
            return ConstantPolicy(self.value)


    def test_runtime_requires_bound_identity_by_default() -> None:
        with pytest.raises(ValueError, match="explicit identity contract"):
            ServingRuntime()


    def test_baseline_bundle_returns_dynamic_zero_identity_action(
        tmp_path: Path,
    ) -> None:
        runtime = ServingRuntime(identity_contract=runtime_identity_contract())
        snapshot = runtime.activate(create_bundle(tmp_path / "baseline"))
        action = runtime.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
        assert snapshot.action_names == ACTION_NAMES
        np.testing.assert_array_equal(
            action,
            np.zeros(len(ACTION_NAMES), dtype=np.float32),
        )


    def test_runtime_fails_closed_on_identity_mismatch(tmp_path: Path) -> None:
        runtime = ServingRuntime(
            identity_contract=runtime_identity_contract(
                environment_digest="f" * 64
            )
        )
        with pytest.raises(ValueError, match="environment identity"):
            runtime.activate(create_bundle(tmp_path / "bundle"))


    def test_runtime_rejects_wrong_shape_nonfinite_and_out_of_bounds_actions(
        tmp_path: Path,
    ) -> None:
        for name, value in (
            ("shape", np.array([0.0])),
            ("finite", np.array([0.0, np.nan, 0.0])),
            ("bounds", np.array([0.0, 1.1, 0.0])),
        ):
            runtime = ServingRuntime(
                policy_loader=Loader(value),
                identity_contract=runtime_identity_contract(),
            )
            runtime.activate(
                create_bundle(
                    tmp_path / name,
                    policy_mode=PolicyMode.RESIDUAL_POLICY,
                )
            )
            with pytest.raises(ValueError, match="action schema"):
                runtime.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    '''
).lstrip()
Path("tests/serving/test_runtime.py").write_text(runtime_test, encoding="utf-8")

reward_tests = Path("tests/rl/test_reward_v2.py")
reward_text = reward_tests.read_text(encoding="utf-8")
reward_text += dedent(
    '''

    def test_margin_deficit_penalty_is_continuous() -> None:
        tracker = RewardTracker(
            RewardConfig(scale=100.0, margin_deficit_weight=2.0)
        )
        mild = tracker.step(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
            hybrid_margin_deficit_fraction=0.01,
        )
        severe = tracker.step(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
            hybrid_margin_deficit_fraction=0.10,
        )
        assert mild.margin_penalty == pytest.approx(0.02)
        assert severe.margin_penalty == pytest.approx(0.20)
        assert severe.scaled_total < mild.scaled_total
    '''
)
reward_tests.write_text(reward_text, encoding="utf-8")
