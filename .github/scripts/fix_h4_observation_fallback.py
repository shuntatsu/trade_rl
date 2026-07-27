from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/rl/environment_observation.py")
source = path.read_text(encoding="utf-8")
old = '''    def compact_observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> dict[str, np.ndarray]:
        """Build current structured state without sequence policy channels."""

        if self.sequence_observation_builder is None:
            raise RuntimeError("compact observation requires a sequence contract")
        _, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        structured = build_structured_current_observation(
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
        )
        structured["decision_index"] = np.asarray(
            [runtime.current_index],
            dtype=np.int64,
        )
        return structured

    def observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self.sequence_observation_builder is None:
            _, current = self.flat_pair(
                runtime,
                trends=trends,
                alpha=alpha,
                factor_basis=factor_basis,
                pre_trade_risk=pre_trade_risk,
            )
            return current
        structured = self.compact_observation(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        if self.sequence_policy_plane is not None:
            structured.update(
                self.sequence_policy_plane.components(runtime.current_index)
            )
            return structured
        sequence = self.sequence_observation_builder.build(
            self.dataset,
            index=runtime.current_index,
        )
        sequence_components = build_structured_policy_observation(
            sequence=sequence,
            current_flat=np.concatenate(
                (
                    structured["current_snapshot"].reshape(-1),
                    structured["asset_state"].reshape(-1),
                    structured["global_state"].reshape(-1),
                )
            ),
            layout=self.layout,
            n_features=self.dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        for key, value in sequence_components.items():
            if key.startswith("sequence_"):
                structured[key] = value
        return structured
'''
new = '''    def compact_observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> dict[str, np.ndarray]:
        """Build current structured state without sequence policy channels."""

        if self.sequence_observation_builder is None:
            raise RuntimeError("compact observation requires a sequence contract")
        _, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        structured = build_structured_current_observation(
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
        )
        structured["decision_index"] = np.asarray(
            [runtime.current_index],
            dtype=np.int64,
        )
        return structured

    def observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> np.ndarray | dict[str, np.ndarray]:
        _, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        if self.sequence_observation_builder is None:
            return current
        if self.sequence_policy_plane is not None:
            structured = build_structured_current_observation(
                current_flat=current,
                layout=self.layout,
                n_features=self.dataset.n_features,
            )
            structured.update(
                self.sequence_policy_plane.components(runtime.current_index)
            )
        else:
            sequence = self.sequence_observation_builder.build(
                self.dataset,
                index=runtime.current_index,
            )
            structured = build_structured_policy_observation(
                sequence=sequence,
                current_flat=current,
                layout=self.layout,
                n_features=self.dataset.n_features,
                sequence_normalizer=self.sequence_normalizer,
            )
        structured["decision_index"] = np.asarray(
            [runtime.current_index],
            dtype=np.int64,
        )
        return structured
'''
count = source.count(old)
if count != 1:
    raise SystemExit(f"observation fallback seam changed: expected one match, got {count}")
path.write_text(source.replace(old, new), encoding="utf-8")
