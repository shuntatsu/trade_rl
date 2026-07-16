from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 4 anchor in {path}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    append_once(
        "tests/data/test_cross_asset_features.py",
        "test_degenerate_cross_asset_statistics_are_unavailable",
        '''

def test_degenerate_cross_asset_statistics_are_unavailable() -> None:
    returns = np.full((8, 2), 0.01, dtype=np.float64)
    available = np.ones_like(returns, dtype=np.bool_)
    ages = np.zeros_like(returns)
    for kind in (
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
    ):
        result = calculate_cross_asset_feature_events(
            _spec(kind, lookback=4, min_periods=2),
            aligned_returns=returns,
            return_available=available,
            return_age_hours=ages,
            symbols=("BTCUSDT", "ETHUSDT"),
            reference_symbol="BTCUSDT",
        )
        assert not result.valid.any()
        np.testing.assert_array_equal(result.values, np.zeros_like(result.values))
''',
    )
    append_once(
        "tests/rl/test_sequence_normalization.py",
        "test_sequence_normalizer_records_channel_sample_counts",
        '''

def test_sequence_normalizer_records_channel_sample_counts() -> None:
    normalizer = SequenceFeatureNormalizer.fit(
        _dataset(), _builder(), train_start=96, train_end=120
    )

    assert normalizer.minimum_samples_per_channel == 1
    for timeframe, counts in normalizer.sample_count.items():
        assert counts.shape == normalizer.center[timeframe].shape
        assert np.all(counts > 0)
        assert np.issubdtype(counts.dtype, np.integer)
    assert "sample_count" in normalizer.digest_payload()


def test_sequence_normalizer_fails_closed_when_a_required_channel_has_no_events() -> None:
    dataset = _dataset()
    available = dataset.feature_available.copy()
    available[:, :, 3] = False
    missing = replace(dataset, feature_available=available, dataset_id="e" * 64)

    with np.testing.assert_raises_regex(ValueError, "1d.*15m__|1d.*sample|1d"):
        SequenceFeatureNormalizer.fit(
            missing,
            _builder(),
            train_start=96,
            train_end=120,
            minimum_samples_per_channel=1,
        )
''',
    )
    append_once(
        "tests/rl/test_normalization_semantics.py",
        "test_all_endogenous_state_fields_use_semantic_passthrough_scaling",
        '''

def test_all_endogenous_state_fields_use_semantic_passthrough_scaling() -> None:
    from trade_rl.rl.observations import observation_layout

    dataset = _dataset()
    layout = observation_layout(dataset, action_size=3, n_factors=2)
    passthrough = set(
        observation_passthrough_indices(dataset, action_size=3, n_factors=2)
    )
    asset_state_start = 4 * dataset.n_features
    assert set(range(asset_state_start, layout.per_symbol_width)).issubset(passthrough)
    global_base = dataset.n_symbols * layout.per_symbol_width
    endogenous_global_start = global_base + 4 * len(dataset.global_feature_names)
    assert set(
        range(endogenous_global_start, global_base + layout.global_width)
    ).issubset(passthrough)
''',
    )
    append_once(
        "tests/rl/test_observation_v2.py",
        "test_observation_v4_semantically_scales_age_basis_and_equity_state",
        '''

def test_observation_v4_semantically_scales_age_basis_and_equity_state() -> None:
    from trade_rl.rl.observations import ObservationExecutionState

    dataset = market()
    book = BookState.zero(2, 1_000.0, dataset.close[2])
    book.peak_value = 1_250.0
    result = build_observation(
        dataset=dataset,
        index=2,
        trends=TrendTargets(
            fast=np.zeros(2), base=np.zeros(2), slow=np.zeros(2)
        ),
        alpha=np.zeros(2),
        hybrid=book,
        shadow=book.clone(),
        start_index=0,
        end_index=5,
        hybrid_risk_scale=1.0,
        shadow_risk_scale=1.0,
        execution_state=ObservationExecutionState(
            requested_weights=np.zeros(2),
            fill_ratio=np.ones(2),
            unfilled_turnover=np.zeros(2),
            participation=np.zeros(2),
            execution_cost=np.zeros(2),
            position_age=np.array([24.0, 48.0]),
        ),
        previous_action=np.zeros(2),
        action_size=2,
    )
    layout = observation_layout(dataset, action_size=2)
    rows = result[: dataset.n_symbols * layout.per_symbol_width].reshape(
        dataset.n_symbols, layout.per_symbol_width
    )
    offset = 4 * dataset.n_features
    assert rows[0, offset + 14] == np.log1p(1.0)
    assert abs(rows[0, -1]) < 0.2
    global_values = result[dataset.n_symbols * layout.per_symbol_width :]
    endogenous = 4 * len(dataset.global_feature_names)
    assert global_values[endogenous] == np.log(0.8)
''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/data/cross_asset_features.py",
        '''                if spec.kind is FeatureKind.ROLLING_BETA_TO_BTC:
                    value = (
                        0.0
                        if btc_variance <= _EPSILON
                        else float(np.cov(asset_sample, btc_sample, ddof=0)[0, 1])
                        / btc_variance
                    )
                else:
                    asset_std = float(np.std(asset_sample))
                    btc_std = float(np.std(btc_sample))
                    value = (
                        0.0
                        if asset_std <= _EPSILON or btc_std <= _EPSILON
                        else float(np.corrcoef(asset_sample, btc_sample)[0, 1])
                    )
                    value = float(np.clip(value, -1.0, 1.0))
''',
        '''                if spec.kind is FeatureKind.ROLLING_BETA_TO_BTC:
                    if btc_variance <= _EPSILON:
                        continue
                    value = (
                        float(np.cov(asset_sample, btc_sample, ddof=0)[0, 1])
                        / btc_variance
                    )
                else:
                    asset_std = float(np.std(asset_sample))
                    btc_std = float(np.std(btc_sample))
                    if asset_std <= _EPSILON or btc_std <= _EPSILON:
                        continue
                    value = float(np.corrcoef(asset_sample, btc_sample)[0, 1])
                    value = float(np.clip(value, -1.0, 1.0))
''',
    )

    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        'SEQUENCE_NORMALIZER_SCHEMA = "sequence_feature_normalizer_v1"',
        'SEQUENCE_NORMALIZER_SCHEMA = "sequence_feature_normalizer_v2"',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        '''def _readonly_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy(order="C")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    result.setflags(write=False)
    return result
''',
        '''def _readonly_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy(order="C")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    result.setflags(write=False)
    return result


def _readonly_count_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1).copy(order="C")
    if result.size == 0 or np.any(result < 0):
        raise ValueError(f"{field} must be a non-empty non-negative count vector")
    result.setflags(write=False)
    return result
''',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "    sequence_schema_digest: str\n    clip: float = 10.0\n",
        "    sequence_schema_digest: str\n    sample_count: Mapping[str, np.ndarray] | None = None\n    minimum_samples_per_channel: int = 1\n    clip: float = 10.0\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "        if tuple(self.center) != clocks or tuple(self.scale) != clocks:\n            raise ValueError(\"sequence normalizer statistics must match feature clocks\")\n",
        "        if tuple(self.center) != clocks or tuple(self.scale) != clocks:\n            raise ValueError(\"sequence normalizer statistics must match feature clocks\")\n        if (\n            isinstance(self.minimum_samples_per_channel, bool)\n            or not isinstance(self.minimum_samples_per_channel, int)\n            or self.minimum_samples_per_channel <= 0\n        ):\n            raise ValueError(\"minimum_samples_per_channel must be positive\")\n        raw_counts = self.sample_count or {\n            timeframe: np.ones(len(self.feature_names[timeframe]), dtype=np.int64)\n            for timeframe in clocks\n        }\n        if tuple(raw_counts) != clocks:\n            raise ValueError(\"sequence sample counts must match feature clocks\")\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "        resolved_names: dict[str, tuple[str, ...]] = {}\n",
        "        resolved_names: dict[str, tuple[str, ...]] = {}\n        resolved_counts: dict[str, np.ndarray] = {}\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        '''            if np.any(scale <= 0.0):
                raise ValueError("sequence normalizer scale must be positive")
            resolved_names[timeframe] = names
''',
        '''            if np.any(scale <= 0.0):
                raise ValueError("sequence normalizer scale must be positive")
            counts = _readonly_count_vector(
                raw_counts[timeframe], field=f"{timeframe}.sample_count"
            )
            if counts.shape != center.shape:
                raise ValueError("sequence sample counts do not match channels")
            if np.any(counts < self.minimum_samples_per_channel):
                raise ValueError("sequence normalizer channel sample count is insufficient")
            resolved_names[timeframe] = names
            resolved_counts[timeframe] = counts
''',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "        object.__setattr__(self, \"scale\", MappingProxyType(resolved_scale))\n",
        "        object.__setattr__(self, \"scale\", MappingProxyType(resolved_scale))\n        object.__setattr__(self, \"sample_count\", MappingProxyType(resolved_counts))\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        '''            "scale": {
                key: tuple(float(value) for value in self.scale[key])
                for key in self.feature_names
            },
            "schema_version": self.schema_version,
''',
        '''            "scale": {
                key: tuple(float(value) for value in self.scale[key])
                for key in self.feature_names
            },
            "sample_count": {
                key: tuple(int(value) for value in self.sample_count[key])
                for key in self.feature_names
            },
            "minimum_samples_per_channel": self.minimum_samples_per_channel,
            "schema_version": self.schema_version,
''',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "        epsilon: float = 1e-8,\n    ) -> SequenceFeatureNormalizer:\n",
        "        epsilon: float = 1e-8,\n        minimum_samples_per_channel: int = 1,\n    ) -> SequenceFeatureNormalizer:\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "        scales: dict[str, np.ndarray] = {}\n",
        "        scales: dict[str, np.ndarray] = {}\n        sample_counts: dict[str, np.ndarray] = {}\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "            scale = np.ones(len(names), dtype=np.float64)\n",
        "            scale = np.ones(len(names), dtype=np.float64)\n            counts = np.zeros(len(names), dtype=np.int64)\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        '''                sample = sample[np.isfinite(sample)]
                if sample.size == 0:
                    continue
                median = float(np.median(sample))
''',
        '''                sample = sample[np.isfinite(sample)]
                counts[feature_index] = int(sample.size)
                if sample.size < minimum_samples_per_channel:
                    raise ValueError(
                        f"{timeframe} channel {names[feature_index]} has insufficient "
                        f"train samples: {sample.size} < {minimum_samples_per_channel}"
                    )
                median = float(np.median(sample))
''',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "            scales[timeframe] = scale\n",
        "            scales[timeframe] = scale\n            sample_counts[timeframe] = counts\n",
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        "            sequence_schema_digest=builder.layout_digest(dataset),\n            clip=clip,\n",
        "            sequence_schema_digest=builder.layout_digest(dataset),\n            sample_count=sample_counts,\n            minimum_samples_per_channel=minimum_samples_per_channel,\n            clip=clip,\n",
    )

    replace_once(
        "trade_rl/rl/observations.py",
        'OBSERVATION_SCHEMA = "baseline_residual_observation_v3"',
        'OBSERVATION_SCHEMA = "baseline_residual_observation_v4"',
    )
    replace_once(
        "trade_rl/rl/observations.py",
        '''        indices.append(base + 4 * n_features)  # asset_active
        indices.append(base + 4 * n_features + 1)  # tradable
        indices.append(base + 4 * n_features + n_factors + 15)  # borrow available
''',
        '''        state_start = base + 4 * n_features
        indices.extend(range(state_start, base + layout.per_symbol_width))
''',
    )
    replace_once(
        "trade_rl/rl/observations.py",
        '''    indices.extend(range(global_base + 3 * n_global, global_base + 4 * n_global))
    return tuple(sorted(set(indices)))
''',
        '''    indices.extend(range(global_base + 3 * n_global, global_base + 4 * n_global))
    endogenous_start = global_base + 4 * n_global
    indices.extend(range(endogenous_start, global_base + layout.global_width))
    return tuple(sorted(set(indices)))
''',
    )
    replace_once(
        "trade_rl/rl/observations.py",
        "            state.position_age,\n",
        "            np.log1p(state.position_age * dataset.bar_hours / 24.0),\n",
    )
    replace_once(
        "trade_rl/rl/observations.py",
        '''            dataset.resolved_array("borrow_rate")[index],
            dataset.resolved_array("mark_price")[index]
            / dataset.resolved_array("index_price")[index]
            - 1.0,
''',
        '''            np.tanh(dataset.resolved_array("borrow_rate")[index]),
            np.tanh(
                100.0
                * (
                    dataset.resolved_array("mark_price")[index]
                    / dataset.resolved_array("index_price")[index]
                    - 1.0
                )
            ),
''',
    )
    replace_once(
        "trade_rl/rl/observations.py",
        '''        math_log_value(hybrid_value),
        math_log_value(shadow_value),
''',
        '''        math_log_value(
            hybrid_value / max(hybrid.peak_value, hybrid_value, 1e-12)
        ),
        math_log_value(
            shadow_value / max(shadow.peak_value, shadow_value, 1e-12)
        ),
''',
    )
    replace_once(
        "trade_rl/rl/normalization.py",
        '    observation_schema: str = "baseline_residual_observation_v3"',
        '    observation_schema: str = "baseline_residual_observation_v4"',
    )
    replace_once(
        "trade_rl/rl/normalization.py",
        '        observation_schema: str = "baseline_residual_observation_v3",',
        '        observation_schema: str = "baseline_residual_observation_v4",',
    )

    for path in (
        "trade_rl/workflows/training_run.py",
        "trade_rl/workflows/market_walk_forward.py",
    ):
        replace_once(
            path,
            '''        "scale": {
            key: tuple(float(value) for value in normalizer.scale[key])
            for key in normalizer.feature_names
        },
        "schema_version": normalizer.schema_version,
''',
            '''        "scale": {
            key: tuple(float(value) for value in normalizer.scale[key])
            for key in normalizer.feature_names
        },
        "sample_count": {
            key: tuple(int(value) for value in normalizer.sample_count[key])
            for key in normalizer.feature_names
        },
        "minimum_samples_per_channel": normalizer.minimum_samples_per_channel,
        "schema_version": normalizer.schema_version,
''',
        )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task4_semantic_normalization.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
