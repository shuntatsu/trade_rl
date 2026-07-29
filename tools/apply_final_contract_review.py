from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(
            f"{relative}: expected one anchor, observed {source.count(old)}"
        )
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def write_red_tests() -> None:
    relative = "tests/rl/test_environment_info_service.py"
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    marker = 'info["sampled_policy_action"], action_path.policy_target'
    if marker in source:
        return
    anchor = '    assert info["constraint_costs"] is costs\n'
    addition = anchor + textwrap.dedent(
        '''
        np.testing.assert_array_equal(
            info["sampled_policy_action"], action_path.policy_target
        )
        np.testing.assert_array_equal(
            info["effective_filled_weights"], action_path.filled_weight
        )
        assert not np.array_equal(
            info["sampled_policy_action"], info["submitted_target"]
        )
        '''
    )
    if anchor not in source:
        raise RuntimeError("environment telemetry RED anchor not found")
    path.write_text(source.replace(anchor, addition, 1), encoding="utf-8")


def patch_postgres_metadata() -> None:
    relative = "trade_rl/integrations/postgres_market_dataset.py"
    old = textwrap.dedent(
        '''
        def _metadata_number(
            metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
        ) -> float:
            raw = metadata.get(symbol, {}).get(field, 0.0)
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise ValueError(f"metadata {symbol}.{field} must be numeric")
            resolved = float(raw)
            if not math.isfinite(resolved) or resolved < 0.0:
                raise ValueError(f"metadata {symbol}.{field} must be finite and non-negative")
            return resolved
        '''
    )
    new = textwrap.dedent(
        '''
        def _metadata_entry(
            metadata: Mapping[str, Mapping[str, object]], symbol: str
        ) -> Mapping[str, object]:
            entry = metadata.get(symbol)
            if not isinstance(entry, Mapping):
                raise ValueError(f"metadata {symbol} must be an object")
            return entry


        def _metadata_number(
            metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
        ) -> float:
            entry = _metadata_entry(metadata, symbol)
            if field not in entry:
                raise ValueError(f"metadata {symbol}.{field} is required")
            raw = entry[field]
            if isinstance(raw, bool) or not isinstance(raw, str | int | float):
                raise ValueError(f"metadata {symbol}.{field} must be numeric")
            try:
                resolved = float(raw)
            except ValueError as error:
                raise ValueError(f"metadata {symbol}.{field} must be numeric") from error
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"metadata {symbol}.{field} must be finite and positive")
            return resolved


        def _metadata_datetime(
            metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
        ) -> datetime:
            entry = _metadata_entry(metadata, symbol)
            raw = entry.get(field)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"metadata {symbol}.{field} is required")
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    f"metadata {symbol}.{field} must be an ISO-8601 timestamp"
                ) from error
            return _aware_utc(parsed, field=f"metadata {symbol}.{field}")


        def _metadata_optional_datetime(
            metadata: Mapping[str, Mapping[str, object]], symbol: str, field: str
        ) -> datetime | None:
            entry = _metadata_entry(metadata, symbol)
            raw = entry.get(field)
            if raw is None:
                return None
            if not isinstance(raw, str) or not raw:
                raise ValueError(
                    f"metadata {symbol}.{field} must be an ISO-8601 timestamp"
                )
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    f"metadata {symbol}.{field} must be an ISO-8601 timestamp"
                ) from error
            return _aware_utc(parsed, field=f"metadata {symbol}.{field}")
        '''
    )
    replace_once(relative, old, new)
    replace_once(
        relative,
        textwrap.dedent(
            '''
                    InstrumentContract(
                        symbol=symbol,
                        listed_at=start,
                        volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            '''
        ),
        textwrap.dedent(
            '''
                    InstrumentContract(
                        symbol=symbol,
                        listed_at=_metadata_datetime(metadata, symbol, "listed_at"),
                        delisted_at=_metadata_optional_datetime(
                            metadata, symbol, "delisted_at"
                        ),
                        volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            '''
        ),
    )


def patch_postgres_observability() -> None:
    relative = "trade_rl/integrations/postgres_market_dataset.py"
    old = textwrap.dedent(
        '''
            economics = build_market_economic_semantics(
                timestamps=timestamps,
                instruments=instruments,
                row_present=np.ones(price_shape, dtype=np.bool_),
                raw_tradable=np.ones(price_shape, dtype=np.bool_),
                source_information_available=np.ones(price_shape, dtype=np.bool_),
                available_at=available_at,
                close=raw["close"],
                funding_event_count=funding_counts,
            )

            log_returns = np.zeros(price_shape, dtype=np.float64)
            log_returns[1:] = np.log(raw["close"][1:] / raw["close"][:-1])
            global_features = np.zeros((n_bars, 4), dtype=np.float32)
            global_features[:, 0] = economics.symbol_active.mean(axis=1)
            global_features[:, 1] = (economics.tradable & economics.information_available).mean(
                axis=1
            )
            global_features[:, 2] = np.mean(log_returns, axis=1, dtype=np.float64)
            global_features[:, 3] = np.std(log_returns, axis=1, dtype=np.float64)
            global_available = np.ones((n_bars, 4), dtype=np.bool_)
            global_available[0, 2:] = False
        '''
    )
    new = textwrap.dedent(
        '''
            economics = build_market_economic_semantics(
                timestamps=timestamps,
                instruments=instruments,
                row_present=np.ones(price_shape, dtype=np.bool_),
                raw_tradable=np.ones(price_shape, dtype=np.bool_),
                source_information_available=np.ones(price_shape, dtype=np.bool_),
                available_at=available_at,
                close=raw["close"],
                funding_event_count=funding_counts,
            )
            observable_features = economics.information_available[:, :, None]
            feature_available &= observable_features
            features = np.where(feature_available, features, np.float32(0.0))
            feature_age = np.where(feature_available, feature_age, np.float32(0.0))
            feature_staleness = np.where(
                feature_available, feature_staleness, np.float32(1.0)
            )

            log_returns = np.zeros(price_shape, dtype=np.float64)
            return_available = np.zeros(price_shape, dtype=np.bool_)
            contiguous = (
                economics.information_available[1:]
                & economics.information_available[:-1]
            )
            candidate_returns = np.log(raw["close"][1:] / raw["close"][:-1])
            np.copyto(log_returns[1:], candidate_returns, where=contiguous)
            return_available[1:] = contiguous
            global_features = np.zeros((n_bars, 4), dtype=np.float32)
            global_features[:, 0] = economics.symbol_active.mean(axis=1)
            global_features[:, 1] = (
                economics.tradable & economics.information_available
            ).mean(axis=1)
            global_available = np.ones((n_bars, 4), dtype=np.bool_)
            for index in range(n_bars):
                sample = log_returns[index, return_available[index]]
                if sample.size:
                    global_features[index, 2] = float(np.mean(sample))
                    global_features[index, 3] = float(np.std(sample))
                else:
                    global_available[index, 2:] = False
        '''
    )
    replace_once(relative, old, new)


def patch_action_telemetry() -> None:
    replace_once(
        "trade_rl/rl/environment_info.py",
        textwrap.dedent(
            '''
                    "sampled_policy_action": np.asarray(
                        request.submitted_target,
                        dtype=np.float64,
                    ).copy(),
                    "effective_filled_weights": np.asarray(
                        request.hybrid.weights,
                        dtype=np.float64,
                    ).copy(),
            '''
        ),
        textwrap.dedent(
            '''
                    "sampled_policy_action": np.asarray(
                        action_path.policy_target,
                        dtype=np.float64,
                    ).copy(),
                    "effective_filled_weights": np.asarray(
                        action_path.filled_weight,
                        dtype=np.float64,
                    ).copy(),
            '''
        ),
    )


def patch_economic_validation() -> None:
    relative = "trade_rl/data/economic_semantics.py"
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    if "from datetime import UTC" not in source:
        source = source.replace(
            "from dataclasses import dataclass, fields\n",
            "from dataclasses import dataclass, fields\nfrom datetime import UTC\n",
            1,
        )
    source = source.replace(
        'contract.listed_at.astimezone(__import__("datetime").UTC)',
        "contract.listed_at.astimezone(UTC)",
    ).replace(
        'contract.delisted_at.astimezone(__import__("datetime").UTC)',
        "contract.delisted_at.astimezone(UTC)",
    )
    marker = 'raise ValueError("max_participation_rate must be within [0, 1]")'
    if marker not in source:
        anchor = textwrap.dedent(
            '''
                    for item in fields(self):
                        value = getattr(self, item.name)
                        if value.flags.writeable:
                            raise ValueError(f"economic array {item.name} must be immutable")
            '''
        )
        addition = anchor + textwrap.dedent(
            '''
                    if np.any(self.max_participation_rate < 0.0) or np.any(
                        self.max_participation_rate > 1.0
                    ):
                        raise ValueError("max_participation_rate must be within [0, 1]")
                    for name in (
                        "spread_rate",
                        "borrow_rate",
                        "minimum_notional",
                        "lot_size",
                        "tick_size",
                    ):
                        if np.any(getattr(self, name) < 0.0):
                            raise ValueError(f"economic array {name} must be non-negative")
                    for name in ("mark_price", "index_price"):
                        if np.any(getattr(self, name) <= 0.0):
                            raise ValueError(
                                f"economic array {name} must be strictly positive"
                            )
            '''
        )
        if anchor not in source:
            raise RuntimeError("economic validation anchor not found")
        source = source.replace(anchor, addition, 1)
    path.write_text(source, encoding="utf-8")


def apply_implementation() -> None:
    patch_postgres_metadata()
    patch_postgres_observability()
    patch_action_telemetry()
    patch_economic_validation()
    patch_path = ROOT / "research-contract-hardening.patch"
    if patch_path.exists():
        patch_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("tests", "implementation"))
    arguments = parser.parse_args()
    if arguments.phase == "tests":
        write_red_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
