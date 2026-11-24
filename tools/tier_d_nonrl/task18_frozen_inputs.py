"""Adapted Task18 frozen model/forecast/cost plan, not historical authorization.

The format has a bounded canonical JSON header followed by recipe-sized
little-endian arrays. The pre-stop external file pin is checked before
decoding. Restoration does not refit, recost, predict or open a Dataset.
The trusted owner must still authenticate the producer, source and original
Dataset artifact and independently authorize any future execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.tier_d_nonrl.task18_costs import (
    ACTOR_ROWS,
    COST_SCENARIOS,
    COST_SHAPE,
    HORIZONS,
    HOUR_NS,
    build_cost_rates,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl._validation import require_git_sha, require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import read_verified_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnRidgeModel,
    fit_simple_return_ridge,
)

_PROTOCOL = build_task18_protocol()
_PLAN = _PROTOCOL.index_plan
_PROTOCOL_DIGEST = _PROTOCOL.adapted_protocol_digest
_CONFIGS = _PROTOCOL.configurations
_SYMBOLS = _PROTOCOL.dataset.symbols
_FEATURES = _PROTOCOL.dataset.feature_names
_FEATURE_INDICES = tuple(range(len(_FEATURES)))
_VINTAGES = _PLAN.vintage_start_indices
_FIRST = _PLAN.actor_start_inclusive
_STOP = _PLAN.actor_stop_exclusive
_START_NS = int(
    np.datetime64(_PROTOCOL.dataset.development_start, "ns").astype(np.int64)
)
_ARMS = {arm.arm_id: arm for arm in _PROTOCOL.arms}
_SCHEMA = "task18_adapted_frozen_inputs_v1"
_FRAME = struct.Struct(">8sI")
_MAGIC = b"TD18FRZ1"
_MAX_HEADER_BYTES = 128 * 1024
_ARRAYS = (
    ("forecasts", (len(_CONFIGS), len(_SYMBOLS), ACTOR_ROWS), "<f8"),
    ("cost_rates", COST_SHAPE, "<f8"),
    ("snapshots", (len(_SYMBOLS), ACTOR_ROWS, len(_FEATURES) + 1), "<f8"),
    ("source_available_at_ns", (len(_SYMBOLS), ACTOR_ROWS), "<i8"),
)
_ARRAY_BYTES = sum(math.prod(shape) * 8 for _, shape, _ in _ARRAYS)
_MAX_FILE_BYTES = _FRAME.size + _MAX_HEADER_BYTES + _ARRAY_BYTES


def _clock(index: int) -> int:
    return _START_NS + index * HOUR_NS


@dataclass(frozen=True, slots=True)
class FrozenInputBinding:
    """Externally expected identity/provenance claims, not producer authentication."""

    scope: str
    dataset_id: str
    manifest_sha256: str
    arrays_sha256: str
    source_commit: str
    source_manifest_sha256: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        require_git_sha(self.source_commit)
        for name in (
            "dataset_id",
            "manifest_sha256",
            "arrays_sha256",
            "source_manifest_sha256",
            "runtime_sha256",
        ):
            require_sha256(getattr(self, name), field=name)
        ref = _PROTOCOL.historical_reference
        pins = (self.dataset_id, self.manifest_sha256, self.arrays_sha256)
        historical = (ref.dataset_id, ref.manifest_sha256, ref.arrays_sha256)
        if self.scope == "SYNTHETIC_ONLY":
            if any(a == b for a, b in zip(pins, historical, strict=True)):
                raise ValueError(
                    "synthetic plan must not reuse historical Dataset pins"
                )
        elif self.scope == "CONSUMED_DEVELOPMENT_ADAPTED":
            if pins != historical:
                raise ValueError(
                    "adapted historical plan requires exact original Dataset pins"
                )
        else:
            raise ValueError("unsupported frozen-input scope")


@dataclass(frozen=True, slots=True)
class FrozenInputsPin:
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256(self.sha256, field="frozen input SHA-256")
        if (
            type(self.size_bytes) is not int
            or not _FRAME.size + _ARRAY_BYTES < self.size_bytes <= _MAX_FILE_BYTES
        ):
            raise ValueError("frozen input size exceeds recipe bounds")


@dataclass(frozen=True, slots=True)
class FrozenDecisionInputs:
    """One exact frozen decision; caller must match actual native snapshot."""

    arm_id: str
    decision_index: int
    decision_time_ns: int
    source_available_at_ns: int
    horizon_end_ns: int
    vintage_start_index: int
    training_scope_sha256: str
    feature_values: tuple[float, ...]
    decision_close: float
    expected_simple_return: float
    fit_prefix_marginal_variance: float
    buy_cost: float
    sell_cost: float
    exit_cost: float
    funding_return: float
    borrow_return: float
    cash_return: float


def _immutable_dense(
    value: np.ndarray, shape: tuple[int, ...], dtype: str
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.shape != shape
        or value.dtype.kind != np.dtype(dtype).kind
        or value.dtype.itemsize != 8
        or not np.isfinite(value).all()
    ):
        raise ValueError("frozen dense array has invalid shape, dtype or values")
    return np.frombuffer(
        np.ascontiguousarray(value, dtype=dtype).tobytes(), dtype=dtype
    ).reshape(shape)


@dataclass(frozen=True, slots=True)
class FrozenTask18Inputs:
    binding: FrozenInputBinding
    models: tuple[SimpleReturnRidgeModel, ...]
    forecasts: np.ndarray
    cost_rates: np.ndarray
    snapshots: np.ndarray
    source_available_at_ns: np.ndarray

    def __post_init__(self) -> None:
        if type(self.binding) is not FrozenInputBinding:
            raise ValueError("frozen plan needs exact typed source binding")
        if type(self.models) is not tuple or len(self.models) != len(_CONFIGS) * len(
            _VINTAGES
        ):
            raise ValueError("frozen plan needs all twelve ordered models")
        scopes: dict[tuple[int, int], SimpleReturnRidgeModel] = {}
        for c, config in enumerate(_CONFIGS):
            for v, activation in enumerate(_VINTAGES):
                model = self.models[c * len(_VINTAGES) + v]
                if (
                    type(model) is not SimpleReturnRidgeModel
                    or model.horizon_hours != config.horizon_hours
                    or model.alpha != config.alpha
                    or int(model.fit_cutoff.astype(np.int64)) != _clock(activation) - 1
                    or model.feature_indices != _FEATURE_INDICES
                    or model.feature_names != _FEATURES
                    or model.n_samples
                    > len(_SYMBOLS) * (activation - config.horizon_hours)
                ):
                    raise ValueError("frozen model does not match its recipe slot")
                # Alpha variants share the same training rows, scaling and
                # marginal variance, not the fitted coefficients.
                other = scopes.setdefault((config.horizon_hours, activation), model)
                if (
                    model.training_scope_sha256 != other.training_scope_sha256
                    or model.n_samples != other.n_samples
                    or model.fit_prefix_marginal_variance
                    != other.fit_prefix_marginal_variance
                    or not np.array_equal(model.feature_mean, other.feature_mean)
                    or not np.array_equal(model.feature_scale, other.feature_scale)
                    or model.intercept != other.intercept
                ):
                    raise ValueError(
                        "alpha variants differ in their frozen training scope"
                    )
        if len({m.training_scope_sha256 for m in scopes.values()}) != len(scopes):
            raise ValueError("different horizon/vintage slots share a training scope")
        for name, shape, dtype in _ARRAYS:
            object.__setattr__(
                self, name, _immutable_dense(getattr(self, name), shape, dtype)
            )
        if np.any(self.forecasts < -1.0) or np.any(self.snapshots[..., -1] <= 0.0):
            raise ValueError("frozen forecast or decision close is outside its domain")
        times = np.arange(_FIRST, _STOP, dtype=np.int64) * HOUR_NS + _START_NS
        # MarketDataset requires availability >= event time, and the strict
        # Task18 actor requires information to be available on its own row.
        if not np.all(self.source_available_at_ns == times[None, :]):
            raise ValueError(
                "frozen source publication differs from its decision clock"
            )
        costs = self.cost_rates
        econ = _PROTOCOL.economics
        if (
            np.any(costs[:, 0, ..., 0] < econ.base_fee_rate + econ.base_spread_rate)
            or np.any(
                costs[:, 0, ..., 0]
                > (
                    econ.base_fee_rate
                    + econ.base_spread_rate
                    + econ.impact_coefficient
                    * math.sqrt(econ.maximum_participation_rate)
                )
            )
            or not np.array_equal(costs[:, 1, ..., 0], 2.0 * costs[:, 0, ..., 0])
            or not np.array_equal(costs[:, 1, ..., 1], costs[:, 0, ..., 1])
            or not np.array_equal(costs[0, ..., 0], costs[1, ..., 0])
            or not np.allclose(
                costs[1, ..., 1], 3.0 * costs[0, ..., 1], rtol=1e-15, atol=0.0
            )
        ):
            raise ValueError("frozen cost scenario or horizon invariants differ")

    @property
    def execution_authorized(self) -> bool:
        return False

    def lookup_nonrl(
        self, arm_id: str, decision_index: int, *, decision_time_ns: int
    ) -> FrozenDecisionInputs:
        """O(1) array lookup, never infer, fit, recost, hash or substitute a row.

        A future runner must also compare the returned frozen snapshot with
        the actual native book/Dataset at that index. This is not that runner.
        """
        arm = _ARMS.get(arm_id) if type(arm_id) is str else None
        if (
            arm is None
            or arm.role != "NONRL"
            or type(decision_index) is not int
            or not _FIRST <= decision_index < _STOP
            or type(decision_time_ns) is not int
            or decision_time_ns != _clock(decision_index)
        ):
            raise ValueError("lookup requires an exact NONRL actor decision")
        c = next(
            i for i, config in enumerate(_CONFIGS) if config.name == arm.configuration
        )
        symbol = _SYMBOLS.index(arm.symbol)
        horizon = _CONFIGS[c].horizon_hours
        scenario = COST_SCENARIOS.index(
            "COST2L0" if arm.scenario == "COST2L0" else "BASEL0"
        )
        v = bisect_right(_VINTAGES, decision_index) - 1
        model = self.models[c * len(_VINTAGES) + v]
        offset = decision_index - _FIRST
        rate, funding = self.cost_rates[
            HORIZONS.index(horizon), scenario, symbol, offset
        ]
        snapshot = self.snapshots[symbol, offset]
        return FrozenDecisionInputs(
            arm_id=arm_id,
            decision_index=decision_index,
            decision_time_ns=decision_time_ns,
            source_available_at_ns=int(self.source_available_at_ns[symbol, offset]),
            horizon_end_ns=decision_time_ns + horizon * HOUR_NS,
            vintage_start_index=_VINTAGES[v],
            training_scope_sha256=model.training_scope_sha256,
            feature_values=tuple(float(x) for x in snapshot[:-1]),
            decision_close=float(snapshot[-1]),
            expected_simple_return=float(self.forecasts[c, symbol, offset]),
            fit_prefix_marginal_variance=model.fit_prefix_marginal_variance,
            buy_cost=float(rate),
            sell_cost=float(rate),
            exit_cost=float(rate),
            funding_return=float(funding),
            borrow_return=0.0,
            cash_return=0.0,
        )


def prepare_frozen_inputs(
    dataset: MarketDataset, *, binding: FrozenInputBinding
) -> FrozenTask18Inputs:
    """Prepare the adapted plan from an already admitted in-memory Dataset.

    No file/market access or execution permission is granted by this API.
    The original archived plan schema cannot be re-labelled as this schema.
    """
    if (
        type(binding) is not FrozenInputBinding
        or not isinstance(dataset, MarketDataset)
        or dataset.dataset_id != binding.dataset_id
        or not dataset.identity_verified
        or dataset.symbols != _SYMBOLS
        or dataset.feature_names != _FEATURES
        or dataset.n_bars != _PROTOCOL.dataset.row_count
        or not np.array_equal(
            dataset.timestamps.astype("datetime64[ns]").astype(np.int64),
            _START_NS
            + np.arange(_PROTOCOL.dataset.row_count, dtype=np.int64) * HOUR_NS,
        )
    ):
        raise ValueError("Task18 input Dataset identity, roster or clock differs")
    if (
        not dataset.feature_available[_FIRST:_STOP].all()
        or not np.isfinite(dataset.features[_FIRST:_STOP]).all()
        or not dataset.resolved_array("information_available")[_FIRST:_STOP].all()
    ):
        raise ValueError("Task18 frozen actor snapshots must be fully available")
    costs = build_cost_rates(dataset)
    models: list[SimpleReturnRidgeModel] = []
    forecasts = np.empty(_ARRAYS[0][1], dtype=np.float64)
    for c, config in enumerate(_CONFIGS):
        for activation, end in zip(_VINTAGES, (*_VINTAGES[1:], _STOP), strict=True):
            model = fit_simple_return_ridge(
                dataset,
                feature_indices=_FEATURE_INDICES,
                fit_symbol_indices=tuple(range(len(_SYMBOLS))),
                fit_cutoff=np.datetime64(_clock(activation) - 1, "ns"),
                horizon_hours=config.horizon_hours,
                alpha=config.alpha,
            )
            models.append(model)
            for row in range(activation, end):
                for symbol in range(len(_SYMBOLS)):
                    forecasts[c, symbol, row - _FIRST] = model.predict_simple_return(
                        dataset.features[row, symbol],
                        decision_time=dataset.timestamps[row],
                        feature_available=dataset.feature_available[row, symbol],
                        feature_names=_FEATURES,
                    )
    snapshots = np.concatenate(
        (
            dataset.features[_FIRST:_STOP].astype(np.float64),
            dataset.close[_FIRST:_STOP, :, None],
        ),
        axis=2,
    ).transpose(1, 0, 2)
    source_times = (
        dataset.resolved_array("available_at")[_FIRST:_STOP]
        .astype("datetime64[ns]")
        .astype(np.int64)
        .T
    )
    return FrozenTask18Inputs(
        binding, tuple(models), forecasts, costs, snapshots, source_times
    )


def _model_mapping(model: SimpleReturnRidgeModel) -> dict[str, object]:
    return {
        "feature_indices": model.feature_indices,
        "feature_names": model.feature_names,
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "coefficients": model.coefficients.tolist(),
        "intercept": model.intercept,
        "horizon_hours": model.horizon_hours,
        "alpha": model.alpha,
        "n_samples": model.n_samples,
        "fit_cutoff_ns": int(model.fit_cutoff.astype(np.int64)),
        "fit_prefix_marginal_variance": model.fit_prefix_marginal_variance,
        "training_scope_sha256": model.training_scope_sha256,
    }


def _header(plan: FrozenTask18Inputs) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "protocol_digest": _PROTOCOL_DIGEST,
        "return_unit": SimpleReturnRidgeModel.return_unit,
        "valuation_basis": SimpleReturnRidgeModel.valuation_basis,
        "variance_kind": SimpleReturnRidgeModel.variance_kind,
        "cost_basis": "declared_initial_notional_rates",
        "execution_authorized": False,
        "binding": asdict(plan.binding),
        "models": [_model_mapping(model) for model in plan.models],
    }


def encode_frozen_inputs(plan: FrozenTask18Inputs) -> bytes:
    if type(plan) is not FrozenTask18Inputs:
        raise ValueError("expected typed frozen plan")
    header = canonical_json_bytes(_header(plan))
    if len(header) > _MAX_HEADER_BYTES:
        raise ValueError("frozen input metadata exceeds bounded header")
    return (
        _FRAME.pack(_MAGIC, len(header))
        + header
        + b"".join(getattr(plan, name).tobytes(order="C") for name, _, _ in _ARRAYS)
    )


def _decode_model(value: object) -> SimpleReturnRidgeModel:
    if type(value) is not dict:
        raise ValueError("frozen model must be an object")
    data: dict[str, Any] = dict(value)
    cutoff = data.pop("fit_cutoff_ns")
    if type(cutoff) is not int:
        raise ValueError("frozen model cutoff requires integer nanoseconds")
    for name in ("feature_indices", "feature_names"):
        if type(data[name]) is not list:
            raise ValueError("frozen model layout must be an array")
        data[name] = tuple(data[name])
    for name in ("feature_mean", "feature_scale", "coefficients"):
        if type(data[name]) is not list or any(
            type(x) not in (int, float) for x in data[name]
        ):
            raise ValueError("frozen model vector must contain real numbers")
        data[name] = np.asarray(data[name], dtype=np.float64)
    return SimpleReturnRidgeModel(fit_cutoff=np.datetime64(cutoff, "ns"), **data)


def load_frozen_inputs(
    path: Path, *, expected_binding: FrozenInputBinding, expected_pin: FrozenInputsPin
) -> FrozenTask18Inputs:
    """Verify externally pinned bytes before bounded decoding and return exact values."""
    if (
        type(expected_binding) is not FrozenInputBinding
        or type(expected_pin) is not FrozenInputsPin
    ):
        raise ValueError("frozen restore requires typed external binding and pin")
    raw = read_verified_bytes(
        path,
        expected_digest=expected_pin.sha256,
        expected_size_bytes=expected_pin.size_bytes,
        field="Task18 frozen plan",
    )
    magic, header_size = _FRAME.unpack_from(raw)
    offset = _FRAME.size + header_size
    if (
        magic != _MAGIC
        or not 0 < header_size <= _MAX_HEADER_BYTES
        or offset + _ARRAY_BYTES != len(raw)
    ):
        raise ValueError("frozen input frame or payload length differs")
    header_raw = raw[_FRAME.size : offset]
    try:
        header = json.loads(header_raw.decode("utf-8"))
        if type(header) is not dict or header.get("binding") != asdict(
            expected_binding
        ):
            raise ValueError("frozen input context differs from external binding")
        if type(header.get("models")) is not list or len(header["models"]) != len(
            _CONFIGS
        ) * len(_VINTAGES):
            raise ValueError("frozen model roster differs")
        models = tuple(_decode_model(item) for item in header["models"])
        arrays: dict[str, np.ndarray] = {}
        for name, shape, dtype in _ARRAYS:
            count = math.prod(shape)
            arrays[name] = np.frombuffer(
                raw, dtype=dtype, count=count, offset=offset
            ).reshape(shape)
            offset += count * 8
        plan = FrozenTask18Inputs(binding=expected_binding, models=models, **arrays)
        if canonical_json_bytes(_header(plan)) != header_raw:
            raise ValueError(
                "frozen input header is not the exact canonical declaration"
            )
    except (KeyError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid frozen input metadata") from exc
    return plan


def publish_frozen_inputs(
    destination: Path, plan: FrozenTask18Inputs
) -> FrozenInputsPin:
    """Publish one complete plan to a fresh, exclusive-owner-controlled directory."""
    destination = Path(destination)
    if ".." in destination.parts or not destination.name:
        raise ValueError("invalid frozen plan destination")
    destination = destination.absolute()
    if (
        any(p.is_symlink() or p.is_junction() for p in reversed(destination.parents))
        or not destination.parent.is_dir()
    ):
        raise ValueError("frozen plan parent must be a regular non-aliased directory")
    if destination.exists() or destination.is_symlink() or destination.is_junction():
        raise FileExistsError("frozen plan destination already exists")
    raw = encode_frozen_inputs(plan)
    pin = FrozenInputsPin(hashlib.sha256(raw).hexdigest(), len(raw))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        atomic_write_bytes(staging / "frozen-plan.bin", raw)
        atomic_rename_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return pin
