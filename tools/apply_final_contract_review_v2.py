from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: expected one anchor, observed {source.count(old)}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def patch_postgres() -> None:
    path = ROOT / "trade_rl/integrations/postgres_market_dataset.py"
    old_helper = textwrap.dedent(
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
    new_helper = textwrap.dedent(
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
    replace_once(path, old_helper, new_helper)
    replace_once(
        path,
        "            listed_at=start,\n            volume_unit=VolumeUnit.QUOTE_NOTIONAL,\n",
        "            listed_at=_metadata_datetime(metadata, symbol, \"listed_at\"),\n"
        "            delisted_at=_metadata_optional_datetime(\n"
        "                metadata, symbol, \"delisted_at\"\n"
        "            ),\n"
        "            volume_unit=VolumeUnit.QUOTE_NOTIONAL,\n",
    )
    source = path.read_text(encoding="utf-8")
    start = source.index("    economics = build_market_economic_semantics(\n")
    end = source.index("    normalization_digest = content_and_arrays_digest(\n", start)
    replacement = textwrap.dedent(
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
    path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")


def patch_environment_info() -> None:
    path = ROOT / "trade_rl/rl/environment_info.py"
    replace_once(
        path,
        "                request.submitted_target,\n                dtype=np.float64,\n"
        "            ).copy(),\n            \"effective_filled_weights\": np.asarray(\n"
        "                request.hybrid.weights,\n",
        "                action_path.policy_target,\n                dtype=np.float64,\n"
        "            ).copy(),\n            \"effective_filled_weights\": np.asarray(\n"
        "                action_path.filled_weight,\n",
    )


def patch_economic_semantics() -> None:
    path = ROOT / "trade_rl/data/economic_semantics.py"
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
    if "max_participation_rate must be within [0, 1]" not in source:
        anchor = (
            "        for item in fields(self):\n"
            "            value = getattr(self, item.name)\n"
            "            if value.flags.writeable:\n"
            "                raise ValueError(f\"economic array {item.name} must be immutable\")\n"
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


def main() -> None:
    patch_postgres()
    patch_environment_info()
    patch_economic_semantics()
    patch_file = ROOT / "research-contract-hardening.patch"
    if patch_file.exists():
        patch_file.unlink()


if __name__ == "__main__":
    main()
