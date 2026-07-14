from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Dataset artifact directories are closed, immutable two-file objects.
replace_once(
    "trade_rl/data/artifact_codec.py",
    '''_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)


def _sha256_bytes(payload: bytes) -> str:
''',
    '''_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)


def verify_exact_artifact_files(root: Path) -> None:
    expected = {DATASET_MANIFEST_NAME, DATASET_ARRAYS_NAME}
    if not root.is_dir():
        raise FileNotFoundError(f"dataset artifact directory is missing: {root}")
    actual: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink():
            raise ValueError(f"dataset artifact contains symlink: {entry.name}")
        if not entry.is_file():
            raise ValueError(f"dataset artifact contains non-file entry: {entry.name}")
        actual.add(entry.name)
    undeclared = sorted(actual - expected)
    missing = sorted(expected - actual)
    if undeclared:
        raise ValueError(f"dataset artifact contains undeclared files: {undeclared}")
    if missing:
        raise FileNotFoundError(f"dataset artifact is missing files: {missing}")


def _sha256_bytes(payload: bytes) -> str:
''',
)
replace_once(
    "trade_rl/data/artifact_codec.py",
    '''    manifest_path = root / DATASET_MANIFEST_NAME
    arrays_path = root / DATASET_ARRAYS_NAME
''',
    '''    verify_exact_artifact_files(root)
    manifest_path = root / DATASET_MANIFEST_NAME
    arrays_path = root / DATASET_ARRAYS_NAME
''',
)

# Feature age is elapsed timestamp time, not a duplicated normalized staleness score.
replace_once(
    "trade_rl/data/builder.py",
    '''def _carry_feature(
    event_values: np.ndarray,
    event_valid: np.ndarray,
    active: np.ndarray,
    *,
    bar_hours: float,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(event_values, dtype=np.float64)
    available = np.zeros_like(event_valid, dtype=np.bool_)
    staleness = np.ones_like(event_values, dtype=np.float64)
    last_index: int | None = None
    last_value = 0.0
    for index in range(len(event_values)):
        if not active[index]:
            last_index = None
            last_value = 0.0
            continue
        if event_valid[index]:
            last_index = index
            last_value = float(event_values[index])
        if last_index is None:
            continue
        age_hours = (index - last_index) * bar_hours
        normalized = min(age_hours / max_staleness_hours, 1.0)
        staleness[index] = normalized
        if age_hours <= max_staleness_hours + 1e-12:
            values[index] = last_value
            available[index] = True
    return values, available, staleness
''',
    '''def _carry_feature(
    event_values: np.ndarray,
    event_valid: np.ndarray,
    active: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_staleness_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(event_values, dtype=np.float64)
    available = np.zeros_like(event_valid, dtype=np.bool_)
    age = np.full_like(event_values, max_staleness_hours, dtype=np.float64)
    staleness = np.ones_like(event_values, dtype=np.float64)
    timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    last_index: int | None = None
    last_value = 0.0
    for index in range(len(event_values)):
        if not active[index]:
            last_index = None
            last_value = 0.0
            continue
        if event_valid[index]:
            last_index = index
            last_value = float(event_values[index])
        if last_index is None:
            continue
        age_hours = float(timestamp_ns[index] - timestamp_ns[last_index]) / _NS_PER_HOUR
        normalized = min(age_hours / max_staleness_hours, 1.0)
        age[index] = age_hours
        staleness[index] = normalized
        if age_hours <= max_staleness_hours + 1e-12:
            values[index] = last_value
            available[index] = True
    return values, available, age, staleness
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''        feature_available = np.zeros_like(features, dtype=np.bool_)
        feature_staleness = np.ones_like(features, dtype=np.float64)
''',
    '''        feature_available = np.zeros_like(features, dtype=np.bool_)
        feature_age_hours = np.ones_like(features, dtype=np.float64)
        feature_staleness = np.ones_like(features, dtype=np.float64)
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''                values, available, staleness = _carry_feature(
                    event_values,
                    event_valid,
                    symbol_active[:, symbol_index],
                    bar_hours=self.config.bar_hours,
                    max_staleness_hours=spec.max_staleness_hours,
                )
                features[:, symbol_index, feature_index] = values
                feature_available[:, symbol_index, feature_index] = available
                feature_staleness[:, symbol_index, feature_index] = staleness
''',
    '''                values, available, age_hours, staleness = _carry_feature(
                    event_values,
                    event_valid,
                    symbol_active[:, symbol_index],
                    timestamps,
                    max_staleness_hours=spec.max_staleness_hours,
                )
                features[:, symbol_index, feature_index] = values
                feature_available[:, symbol_index, feature_index] = available
                feature_age_hours[:, symbol_index, feature_index] = age_hours
                feature_staleness[:, symbol_index, feature_index] = staleness
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''        global_features = global_features.astype(np.float32)
        feature_staleness = feature_staleness.astype(np.float32)
''',
    '''        global_features = global_features.astype(np.float32)
        feature_age_hours = feature_age_hours.astype(np.float32)
        feature_staleness = feature_staleness.astype(np.float32)
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''                ("feature_available", feature_available),
                ("feature_staleness", feature_staleness),
''',
    '''                ("feature_available", feature_available),
                ("feature_age_hours", feature_age_hours),
                ("feature_staleness", feature_staleness),
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''            feature_available=feature_available,
            feature_staleness=feature_staleness,
''',
    '''            feature_available=feature_available,
            feature_staleness_hours=feature_age_hours,
            feature_staleness=feature_staleness,
''',
)

# Verified identities can be independently recomputed; quantity conversions honor multipliers.
replace_once(
    "trade_rl/data/market.py",
    '''    @property
    def identity_verified(self) -> bool:
        return self.identity_payload_json is not None

    def identity_contract_payload(self) -> dict[str, object]:
''',
    '''    def recomputed_dataset_id(self) -> str:
        if self.identity_payload_json is None:
            raise ValueError("dataset has no canonical identity payload")
        return compute_market_dataset_id(
            parse_identity_json(self.identity_payload_json),
            self.identity_arrays(),
        )

    @property
    def identity_verified(self) -> bool:
        return (
            self.identity_payload_json is not None
            and self.recomputed_dataset_id() == self.dataset_id
        )

    def identity_contract_payload(self) -> dict[str, object]:
''',
)
replace_once(
    "trade_rl/data/market.py",
    '''    def market_notional(
        self,
        index: int,
        prices: np.ndarray | None = None,
    ) -> np.ndarray:
''',
    '''    def elapsed_year_fraction(self, start_index: int, end_index: int) -> float:
        return self.elapsed_hours(start_index, end_index) / _HOURS_PER_YEAR

    def quantity_notional(
        self,
        index: int,
        quantities: np.ndarray,
        prices: np.ndarray | None = None,
    ) -> np.ndarray:
        if not 0 <= index < self.n_bars:
            raise IndexError("quantity-notional index is outside the dataset")
        quantity_vector = np.asarray(quantities, dtype=np.float64).reshape(-1)
        price_vector = (
            self.resolved_array("mark_price")[index]
            if prices is None
            else np.asarray(prices, dtype=np.float64).reshape(-1)
        )
        if (
            quantity_vector.shape != (self.n_symbols,)
            or price_vector.shape != (self.n_symbols,)
            or not np.isfinite(quantity_vector).all()
            or not np.isfinite(price_vector).all()
            or np.any(price_vector <= 0.0)
        ):
            raise ValueError("quantities and prices must match symbols and be finite")
        multipliers = self.resolved_array("contract_multipliers")
        return quantity_vector * price_vector * multipliers

    def notional_to_quantity(
        self,
        index: int,
        notionals: np.ndarray,
        prices: np.ndarray | None = None,
    ) -> np.ndarray:
        notional_vector = np.asarray(notionals, dtype=np.float64).reshape(-1)
        price_vector = (
            self.open[index]
            if prices is None
            else np.asarray(prices, dtype=np.float64).reshape(-1)
        )
        if (
            not 0 <= index < self.n_bars
            or notional_vector.shape != (self.n_symbols,)
            or price_vector.shape != (self.n_symbols,)
            or not np.isfinite(notional_vector).all()
            or not np.isfinite(price_vector).all()
            or np.any(price_vector <= 0.0)
        ):
            raise ValueError("notionals and prices must match symbols and be finite")
        multipliers = self.resolved_array("contract_multipliers")
        return notional_vector / (price_vector * multipliers)

    def market_notional(
        self,
        index: int,
        prices: np.ndarray | None = None,
        *,
        volume: np.ndarray | None = None,
    ) -> np.ndarray:
''',
)
replace_once(
    "trade_rl/data/market.py",
    '''        result = np.empty(self.n_symbols, dtype=np.float64)
        for symbol_index, unit in enumerate(self.volume_units):
            raw = self.volume[index, symbol_index]
''',
    '''        raw_volume = (
            self.volume[index]
            if volume is None
            else np.asarray(volume, dtype=np.float64).reshape(-1)
        )
        if (
            raw_volume.shape != (self.n_symbols,)
            or not np.isfinite(raw_volume).all()
            or np.any(raw_volume < 0.0)
        ):
            raise ValueError("market volume must match symbols and be non-negative")
        result = np.empty(self.n_symbols, dtype=np.float64)
        for symbol_index, unit in enumerate(self.volume_units):
            raw = raw_volume[symbol_index]
''',
)

# Full multiplier-aware execution supersedes the temporary fail-closed guard.
replace_once(
    "trade_rl/simulation/execution.py",
    '''        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()
        if not np.allclose(
            dataset.resolved_array("contract_multipliers"),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "non-unit contract multiplier accounting is not supported; "
                "execution fails closed"
            )
        self._rng = np.random.default_rng(self.cost.random_seed)
''',
    '''        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()
        self._rng = np.random.default_rng(self.cost.random_seed)
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    "        requested_notional_vector = requested_delta * price_vector\n",
    '''        requested_notional_vector = self.dataset.quantity_notional(
            market_index,
            requested_delta,
            price_vector,
        )
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    '''        capacity_notional = self._capacity_notional(
            price_vector,
            capacity_volume_vector,
        )
''',
    '''        capacity_notional = self.dataset.market_notional(
            market_index,
            price_vector,
            volume=capacity_volume_vector,
        )
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    '''        filled_delta = filled_notional_vector / price_vector
        next_quantities = self._round_quantities(
            book.quantities + filled_delta,
            index=market_index,
        )
        filled_notional_vector = (next_quantities - book.quantities) * price_vector
''',
    '''        filled_delta = self.dataset.notional_to_quantity(
            market_index,
            filled_notional_vector,
            price_vector,
        )
        next_quantities = self._round_quantities(
            book.quantities + filled_delta,
            index=market_index,
        )
        filled_notional_vector = self.dataset.quantity_notional(
            market_index,
            next_quantities - book.quantities,
            price_vector,
        )
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    '''        borrow_amount = float(
            np.sum(short_values * self.dataset.resolved_array("borrow_rate")[index])
            / self.dataset.periods_per_year
            * self.cost.borrow_rate_multiplier
        )
''',
    '''        previous_index = max(0, index - 1)
        year_fraction = self.dataset.elapsed_year_fraction(previous_index, index)
        borrow_amount = float(
            np.sum(short_values * self.dataset.resolved_array("borrow_rate")[index])
            * year_fraction
            * self.cost.borrow_rate_multiplier
        )
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    '''        if book.weights.shape != (self.dataset.n_symbols,):
            raise ValueError("book weights shape does not match market symbols")

        resolved_target = _target_weights(
''',
    '''        if book.weights.shape != (self.dataset.n_symbols,):
            raise ValueError("book weights shape does not match market symbols")
        if not np.array_equal(
            np.asarray(book.contract_multipliers),
            self.dataset.resolved_array("contract_multipliers"),
        ):
            raise ValueError("book contract multipliers do not match market dataset")

        resolved_target = _target_weights(
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    '''                desired_quantities = (
                    resolved_target * decision_equity / self.dataset.open[next_index]
                )
                requested_by_symbol = np.abs(
                    (desired_quantities - result_book.quantities)
                    * self.dataset.open[next_index]
                )
''',
    '''                desired_quantities = self.dataset.notional_to_quantity(
                    next_index,
                    resolved_target * decision_equity,
                    self.dataset.open[next_index],
                )
                requested_by_symbol = np.abs(
                    self.dataset.quantity_notional(
                        next_index,
                        desired_quantities - result_book.quantities,
                        self.dataset.open[next_index],
                    )
                )
''',
)
replace_once(
    "trade_rl/simulation/execution.py",
    '''            cash_interest = result_book.apply_cash_interest(
                float(self.dataset.resolved_array("cash_rate")[next_index]),
                periods_per_year=self.dataset.periods_per_year,
            )
''',
    '''            cash_interest = result_book.apply_cash_interest(
                float(self.dataset.resolved_array("cash_rate")[next_index]),
                year_fraction=self.dataset.elapsed_year_fraction(
                    close_index,
                    next_index,
                ),
            )
''',
)

# Elapsed-time metadata controls risk-metric annualization.
replace_once(
    "trade_rl/evaluation/metrics.py",
    "    annualization = math.sqrt(returns.periods_per_year)\n",
    "    annualization = math.sqrt(returns.annualization_periods_per_year)\n",
)

# Preserve the underlying digest error for fail-closed tamper diagnostics.
replace_once(
    "trade_rl/serving/normalizer.py",
    '''    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("serving normalizer sidecar is invalid") from error
''',
    '''    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"serving normalizer sidecar is invalid: {error}") from error
''',
)

# Regression now verifies complete multiplier accounting rather than temporary refusal.
replace_once(
    "tests/architecture/test_complete_architecture_hardening.py",
    '''def test_non_unit_contract_multiplier_fails_closed() -> None:
    with pytest.raises(ValueError, match="contract multiplier"):
        MarketExecutor(dataset(contract_multiplier=0.1), ExecutionCostConfig.zero())
''',
    '''def test_non_unit_contract_multiplier_uses_quantity_semantics() -> None:
    value = dataset(contract_multiplier=0.1)
    book = BookState.zero(
        1,
        1_000.0,
        value.close[0],
        contract_multipliers=value.contract_multipliers,
    )
    result = MarketExecutor(value, ExecutionCostConfig.zero()).execute_interval(
        book,
        np.array([0.5]),
        start_index=0,
        bars=1,
    )
    assert result.book.weights[0] == pytest.approx(0.5)
''',
)
