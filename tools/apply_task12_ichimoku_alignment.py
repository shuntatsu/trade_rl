from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 12 anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    replace_once(
        "tests/data/test_indicator_features.py",
        '''from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
''',
        '''from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
)
''',
    )
    append_once(
        "tests/data/test_indicator_features.py",
        "test_all_ichimoku_features_are_invariant_to_mutated_future_bars",
        r'''
def test_all_ichimoku_features_are_invariant_to_mutated_future_bars() -> None:
    raw = _trend_series(120)
    index = 80
    future = slice(index + 1, None)
    mutated_high = raw.high.copy()
    mutated_low = raw.low.copy()
    mutated_close = raw.close.copy()
    mutated_high[future] *= 4.0
    mutated_low[future] *= 0.25
    mutated_close[future] *= 2.0

    for kind, lookback in (
        (FeatureKind.ICHIMOKU_TENKAN_DISTANCE, 9),
        (FeatureKind.ICHIMOKU_KIJUN_DISTANCE, 26),
        (FeatureKind.ICHIMOKU_CLOUD_POSITION, 52),
        (FeatureKind.ICHIMOKU_CLOUD_THICKNESS, 52),
    ):
        spec = FeatureSpec(
            name=kind.value,
            kind=kind,
            lookback=lookback,
            alignment=FeatureAlignment.UNSHIFTED_DECISION_TIME,
        )
        baseline = calculate_feature_events(
            spec,
            open_price=raw.open,
            high=raw.high,
            low=raw.low,
            close=raw.close,
            volume=raw.volume,
            funding_rate=raw.funding_rate,
            funding_available=raw.funding_available,
            row_present=np.ones(raw.timestamps.shape, dtype=np.bool_),
            active=np.ones(raw.timestamps.shape, dtype=np.bool_),
        )
        changed = calculate_feature_events(
            spec,
            open_price=raw.open,
            high=mutated_high,
            low=mutated_low,
            close=mutated_close,
            volume=raw.volume,
            funding_rate=raw.funding_rate,
            funding_available=raw.funding_available,
            row_present=np.ones(raw.timestamps.shape, dtype=np.bool_),
            active=np.ones(raw.timestamps.shape, dtype=np.bool_),
        )
        assert baseline[1][index]
        assert changed[1][index]
        np.testing.assert_allclose(
            baseline[0][index], changed[0][index], rtol=0.0, atol=0.0
        )
        assert baseline[2][index] == changed[2][index]


def test_feature_spec_identity_binds_unshifted_alignment() -> None:
    unbound = FeatureSpec(
        name="ichimoku",
        kind=FeatureKind.ICHIMOKU_CLOUD_POSITION,
        lookback=52,
    )
    aligned = FeatureSpec(
        name="ichimoku",
        kind=FeatureKind.ICHIMOKU_CLOUD_POSITION,
        lookback=52,
        alignment=FeatureAlignment.UNSHIFTED_DECISION_TIME,
    )
    assert aligned.alignment is FeatureAlignment.UNSHIFTED_DECISION_TIME
    assert aligned.canonical_payload()["alignment"] == "unshifted_decision_time"
    assert aligned.canonical_payload() != unbound.canonical_payload()
''',
    )

    replace_once(
        "tests/integrations/test_binance_multitimeframe.py",
        '''from typing import Any

from trade_rl.integrations.binance import (
''',
        '''from typing import Any

from trade_rl.data.contracts import FeatureAlignment, FeatureKind
from trade_rl.integrations.binance import (
''',
    )
    replace_once(
        "tests/integrations/test_binance_multitimeframe.py",
        '''    assert "15m__upper_wick_ratio" in names
    assert "1d__upper_wick_ratio" not in names
''',
        '''    assert "15m__upper_wick_ratio" in names
    assert "1d__upper_wick_ratio" not in names
    ichimoku_kinds = {
        FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
        FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
        FeatureKind.ICHIMOKU_CLOUD_POSITION,
        FeatureKind.ICHIMOKU_CLOUD_THICKNESS,
    }
    ichimoku = tuple(spec for spec in specs if spec.kind in ichimoku_kinds)
    assert len(ichimoku) == 16
    assert all(
        spec.alignment is FeatureAlignment.UNSHIFTED_DECISION_TIME
        for spec in ichimoku
    )
    assert all(
        spec.alignment is None for spec in specs if spec.kind not in ichimoku_kinds
    )
''',
    )

    append_once(
        "tests/workflows/test_training_run_config.py",
        "test_training_dataset_reference_declares_unshifted_ichimoku_alignment",
        r'''
def test_training_dataset_reference_declares_unshifted_ichimoku_alignment() -> None:
    from trade_rl.workflows.training_run import _feature_alignment_payload

    names = (
        "15m__log_return_1bar",
        "15m__ichimoku_tenkan_distance_9bar",
        "1h__ichimoku_cloud_position_9_26_52",
    )
    assert _feature_alignment_payload(names) == {
        "15m__ichimoku_tenkan_distance_9bar": "unshifted_decision_time",
        "1h__ichimoku_cloud_position_9_26_52": "unshifted_decision_time",
    }
''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/data/contracts.py",
        '''class NormalizationMode(StrEnum):
    NONE = "none"
    ROLLING_ZSCORE = "rolling_zscore"
''',
        '''class NormalizationMode(StrEnum):
    NONE = "none"
    ROLLING_ZSCORE = "rolling_zscore"


class FeatureAlignment(StrEnum):
    """Semantic placement of one feature on the decision-time axis."""

    UNSHIFTED_DECISION_TIME = "unshifted_decision_time"
''',
    )
    replace_once(
        "trade_rl/data/contracts.py",
        '''    max_staleness_hours: float = 24.0
    timeframe: str | None = None
''',
        '''    max_staleness_hours: float = 24.0
    timeframe: str | None = None
    alignment: FeatureAlignment | None = None
''',
    )
    replace_once(
        "trade_rl/data/contracts.py",
        '''        if self.timeframe is not None:
            require_non_empty(self.timeframe, field="feature timeframe")
            timeframe_hours(self.timeframe)
''',
        '''        if self.timeframe is not None:
            require_non_empty(self.timeframe, field="feature timeframe")
            timeframe_hours(self.timeframe)
        if self.alignment is not None:
            try:
                resolved_alignment = FeatureAlignment(
                    getattr(self.alignment, "value", self.alignment)
                )
            except ValueError as error:
                raise ValueError("feature alignment is unsupported") from error
            object.__setattr__(self, "alignment", resolved_alignment)
''',
    )
    replace_once(
        "trade_rl/data/contracts.py",
        '''        if self.timeframe is not None:
            payload["timeframe"] = self.timeframe
        return payload
''',
        '''        if self.timeframe is not None:
            payload["timeframe"] = self.timeframe
        if self.alignment is not None:
            payload["alignment"] = self.alignment.value
        return payload
''',
    )

    replace_once(
        "trade_rl/data/config.py",
        '''from trade_rl.data.contracts import (
    FeatureKind,
''',
        '''from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
''',
    )
    replace_once(
        "trade_rl/data/config.py",
        '''            "timeframe",
        },
''',
        '''            "timeframe",
            "alignment",
        },
''',
    )
    replace_once(
        "trade_rl/data/config.py",
        '''        normalization = NormalizationMode(
            _optional_string(
                item.get("normalization"),
                field=f"{field}.normalization",
            )
            or NormalizationMode.NONE.value
        )
''',
        '''        normalization = NormalizationMode(
            _optional_string(
                item.get("normalization"),
                field=f"{field}.normalization",
            )
            or NormalizationMode.NONE.value
        )
        raw_alignment = _optional_string(
            item.get("alignment"), field=f"{field}.alignment"
        )
        alignment = None if raw_alignment is None else FeatureAlignment(raw_alignment)
''',
    )
    replace_once(
        "trade_rl/data/config.py",
        '''        timeframe=_optional_string(item.get("timeframe"), field=f"{field}.timeframe"),
    )
''',
        '''        timeframe=_optional_string(item.get("timeframe"), field=f"{field}.timeframe"),
        alignment=alignment,
    )
''',
    )

    replace_once(
        "trade_rl/integrations/binance.py",
        '''from trade_rl.data.contracts import (
    FeatureKind,
''',
        '''from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
''',
    )
    replace_once(
        "trade_rl/integrations/binance.py",
        '''                    kind=kind,
                    timeframe=native,
                    lookback=lookback,
''',
        '''                    kind=kind,
                    timeframe=native,
                    alignment=(
                        FeatureAlignment.UNSHIFTED_DECISION_TIME
                        if kind
                        in {
                            FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
                            FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
                            FeatureKind.ICHIMOKU_CLOUD_POSITION,
                            FeatureKind.ICHIMOKU_CLOUD_THICKNESS,
                        }
                        else None
                    ),
                    lookback=lookback,
''',
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''def _ensemble_payload(manifest: PolicyEnsembleManifest) -> dict[str, object]:
    return asdict(manifest)
''',
        '''def _ensemble_payload(manifest: PolicyEnsembleManifest) -> dict[str, object]:
    return asdict(manifest)


def _feature_alignment_payload(
    feature_names: tuple[str, ...],
) -> dict[str, str]:
    return {
        name: "unshifted_decision_time"
        for name in feature_names
        if "__ichimoku_" in name or name.startswith("ichimoku_")
    }
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''                "feature_names": dataset.feature_names,
                "global_feature_names": dataset.global_feature_names,
                "schema_version": "dataset_reference_v3",
''',
        '''                "feature_names": dataset.feature_names,
                "feature_alignments": _feature_alignment_payload(dataset.feature_names),
                "global_feature_names": dataset.global_feature_names,
                "schema_version": "dataset_reference_v4",
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task12_ichimoku_alignment.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
