from pathlib import Path

path = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = path.read_text(encoding="utf-8")

old = "from dataclasses import dataclass\nfrom typing import Any, Mapping\n"
new = "from dataclasses import dataclass\nfrom types import SimpleNamespace\nfrom typing import Any, Mapping\n"
if text.count(old) != 1:
    raise SystemExit("import target drifted")
text = text.replace(old, new)

old = "from trade_rl.data.identity import content_and_arrays_digest\n"
new = """from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import (
    UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    universal_feature_schema_digest_from_names,
)
"""
if text.count(old) != 1:
    raise SystemExit("data import target drifted")
text = text.replace(old, new)

old = """    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
)
"""
new = """    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
    forward_log_return_label,
)
"""
if text.count(old) != 1:
    raise SystemExit("learning import target drifted")
text = text.replace(old, new)

old = """from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
"""
new = """from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
"""
if text.count(old) != 1:
    raise SystemExit("binding import target drifted")
text = text.replace(old, new)

old = """    feature_schema_digest: str
    context_digest: str
    decision_indices: np.ndarray
"""
new = """    feature_schema_digest: str
    context_digest: str
    reference_equity_mode: str
    reference_equity: float
    decision_indices: np.ndarray
"""
if text.count(old) != 1:
    raise SystemExit("sample fields target drifted")
text = text.replace(old, new)

old = """        names = tuple(self.feature_names)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
"""
new = """        if self.reference_equity_mode != "initial_capital":
            raise ValueError(
                "causal alpha reference_equity_mode must be initial_capital"
            )
        if not np.isfinite(self.reference_equity) or self.reference_equity <= 0.0:
            raise ValueError("causal alpha reference_equity must be positive")
        names = tuple(self.feature_names)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
"""
if text.count(old) != 1:
    raise SystemExit("sample validation target drifted")
text = text.replace(old, new)

old = """                "feature_names": names,
                "feature_schema_digest": self.feature_schema_digest,
                "schema_version": _CAUSAL_ALPHA_SYMBOL_SAMPLES_SCHEMA,
"""
new = """                "feature_names": names,
                "feature_schema_digest": self.feature_schema_digest,
                "reference_equity": float(self.reference_equity),
                "reference_equity_mode": self.reference_equity_mode,
                "schema_version": _CAUSAL_ALPHA_SYMBOL_SAMPLES_SCHEMA,
"""
if text.count(old) != 1:
    raise SystemExit("sample digest target drifted")
text = text.replace(old, new)

marker = "\ndef _validated_sample_scope(\n"
if text.count(marker) != 1:
    raise SystemExit("sample builder insertion marker drifted")
implementation = r'''

def _prefix_forward_label(
    dataset: Any,
    *,
    decision_index: int,
    horizon_hours: float,
    signal_delay_decisions: int,
    decision_bars: int,
    train_stop: int,
) -> tuple[float, int]:
    bars_for_hours = getattr(dataset, "bars_for_hours", None)
    if not callable(bars_for_hours):
        raise TypeError("causal alpha dataset cannot resolve label horizons")
    horizon_bars = int(bars_for_hours(horizon_hours))
    execution_start = decision_index + signal_delay_decisions * decision_bars + 1
    label_end = execution_start + horizon_bars - 1
    if execution_start >= train_stop or label_end >= train_stop:
        return float("nan"), -1
    label = forward_log_return_label(
        dataset,
        decision_index=decision_index,
        horizon_hours=horizon_hours,
        signal_delay_decisions=signal_delay_decisions,
        decision_bars=decision_bars,
    )
    if label.label_end_index != label_end:
        raise RuntimeError("causal alpha label timing drifted")
    return label.value, label.label_end_index


def build_causal_alpha_symbol_samples(
    *,
    environment: Any,
    binding: InstrumentDatasetBinding,
    instrument_context_provider: Any,
    train_range: tuple[int, int],
    feature_schema_digest: str,
) -> CausalAlphaSymbolSamples:
    """Extract one train-symbol causal table without action-dependent context."""

    if not isinstance(binding, InstrumentDatasetBinding):
        raise TypeError("causal alpha binding must be InstrumentDatasetBinding")
    if binding.split != "train":
        raise ValueError("causal alpha sample extraction requires a train binding")
    if not callable(instrument_context_provider):
        raise TypeError("causal alpha instrument context provider must be callable")
    dataset = getattr(environment, "dataset", None)
    if dataset is None:
        raise TypeError("causal alpha environment must expose its dataset")
    if tuple(getattr(dataset, "symbols", ())) != (binding.concrete_symbol,):
        raise ValueError("causal alpha dataset symbol does not match train binding")
    if getattr(dataset, "dataset_id", None) != binding.source_dataset_id:
        raise ValueError("causal alpha dataset identity does not match train binding")
    if getattr(dataset, "n_symbols", None) != 1:
        raise ValueError("causal alpha sample extraction requires one symbol")
    market_feature_names = tuple(getattr(dataset, "feature_names", ()))
    expected_schema = universal_feature_schema_digest_from_names(market_feature_names)
    if feature_schema_digest != expected_schema:
        raise ValueError("causal alpha feature schema digest does not match dataset")
    provider_schema_digest = getattr(instrument_context_provider, "schema_digest", None)
    provider_digest = getattr(instrument_context_provider, "digest", None)
    for field, value in (
        ("instrument context schema digest", provider_schema_digest),
        ("instrument context provider digest", provider_digest),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"causal alpha {field} is unavailable")
    initial_capital = float(getattr(environment, "initial_capital", np.nan))
    if not np.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("causal alpha environment initial_capital must be positive")
    decision_bars = getattr(environment, "decision_bars", None)
    if (
        isinstance(decision_bars, bool)
        or not isinstance(decision_bars, int)
        or decision_bars <= 0
    ):
        raise ValueError("causal alpha decision_bars must be positive")
    config = getattr(environment, "config", None)
    signal_delay_decisions = getattr(config, "signal_delay_decisions", None)
    if signal_delay_decisions not in {0, 1}:
        raise ValueError("causal alpha signal delay must be zero or one decision")
    start, stop, _ = _train_range(environment, train_range)

    market_features = np.asarray(getattr(dataset, "features", None), dtype=np.float64)
    market_available = np.asarray(
        getattr(dataset, "feature_available", None), dtype=np.bool_
    )
    expected_market_shape = (
        int(getattr(dataset, "n_bars", 0)),
        1,
        len(market_feature_names),
    )
    if market_features.shape != expected_market_shape:
        raise ValueError("causal alpha market feature shape is invalid")
    if market_available.shape != expected_market_shape:
        raise ValueError("causal alpha market availability shape is invalid")
    if not np.isfinite(market_features).all():
        raise ValueError("causal alpha market features must be finite")
    active = np.asarray(getattr(dataset, "asset_active", None), dtype=np.bool_)
    tradable = np.asarray(getattr(dataset, "tradable", None), dtype=np.bool_)
    if active.shape != expected_market_shape[:2] or tradable.shape != active.shape:
        raise ValueError("causal alpha active/tradable masks are invalid")

    decision_values: list[int] = []
    feature_rows: list[np.ndarray] = []
    availability_rows: list[np.ndarray] = []
    labels_24h: list[float] = []
    ends_24h: list[int] = []
    labels_72h: list[float] = []
    ends_72h: list[int] = []
    for index in range(start, stop):
        if not bool(active[index, 0] and tradable[index, 0]):
            continue
        proxy = SimpleNamespace(
            dataset=dataset,
            current_index=index,
            config=config,
            hybrid=SimpleNamespace(portfolio_value=initial_capital),
        )
        context = np.asarray(
            instrument_context_provider(proxy, binding), dtype=np.float64
        )
        expected_context_shape = (1, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
        if context.shape != expected_context_shape or not np.isfinite(context).all():
            raise ValueError("causal alpha instrument context shape is invalid")
        decision_values.append(index)
        feature_rows.append(
            np.concatenate((market_features[index, 0], context[0]), axis=0)
        )
        availability_rows.append(
            np.concatenate(
                (
                    market_available[index, 0],
                    np.ones(context.shape[1], dtype=np.bool_),
                ),
                axis=0,
            )
        )
        label_24h, end_24h = _prefix_forward_label(
            dataset,
            decision_index=index,
            horizon_hours=24.0,
            signal_delay_decisions=int(signal_delay_decisions),
            decision_bars=decision_bars,
            train_stop=stop,
        )
        label_72h, end_72h = _prefix_forward_label(
            dataset,
            decision_index=index,
            horizon_hours=72.0,
            signal_delay_decisions=int(signal_delay_decisions),
            decision_bars=decision_bars,
            train_stop=stop,
        )
        labels_24h.append(label_24h)
        ends_24h.append(end_24h)
        labels_72h.append(label_72h)
        ends_72h.append(end_72h)
    if not decision_values:
        raise ValueError("causal alpha train range contains no active tradable samples")
    context_digest = content_digest(
        {
            "binding_instrument_descriptor_digest": binding.instrument_descriptor_digest,
            "provider_digest": provider_digest,
            "provider_schema_digest": provider_schema_digest,
            "reference_equity": initial_capital,
            "reference_equity_mode": "initial_capital",
            "schema_version": "causal_alpha_signal_context_v1",
        }
    )
    feature_names = (
        *market_feature_names,
        *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    )
    return CausalAlphaSymbolSamples(
        symbol=binding.concrete_symbol,
        dataset_id=binding.source_dataset_id,
        feature_names=feature_names,
        feature_schema_digest=feature_schema_digest,
        context_digest=context_digest,
        reference_equity_mode="initial_capital",
        reference_equity=initial_capital,
        decision_indices=np.asarray(decision_values, dtype=np.int64),
        features=np.asarray(feature_rows, dtype=np.float64),
        feature_available=np.asarray(availability_rows, dtype=np.bool_),
        labels_24h=np.asarray(labels_24h, dtype=np.float64),
        label_end_indices_24h=np.asarray(ends_24h, dtype=np.int64),
        labels_72h=np.asarray(labels_72h, dtype=np.float64),
        label_end_indices_72h=np.asarray(ends_72h, dtype=np.int64),
    )
'''
text = text.replace(marker, implementation + marker)

old = """    "build_causal_alpha_episode_batch",
    "build_chronological_episode_partition",
"""
new = """    "build_causal_alpha_episode_batch",
    "build_causal_alpha_symbol_samples",
    "build_chronological_episode_partition",
"""
if text.count(old) != 1:
    raise SystemExit("all export target drifted")
text = text.replace(old, new)
path.write_text(text, encoding="utf-8")

test = Path("tests/workflows/test_universal_causal_alpha_fitting.py")
text = test.read_text(encoding="utf-8")
old = """        context_digest=content_digest(f"context:{symbol}"),
        decision_indices=decisions,
"""
new = """        context_digest=content_digest(f"context:{symbol}"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
"""
if text.count(old) != 1:
    raise SystemExit("primary synthetic sample target drifted")
text = text.replace(old, new)
old = """        context_digest=original.context_digest,
        decision_indices=original.decision_indices,
"""
new = """        context_digest=original.context_digest,
        reference_equity_mode=original.reference_equity_mode,
        reference_equity=original.reference_equity,
        decision_indices=original.decision_indices,
"""
if text.count(old) != 1:
    raise SystemExit("changed synthetic sample target drifted")
test.write_text(text.replace(old, new), encoding="utf-8")
