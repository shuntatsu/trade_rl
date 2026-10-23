"""Consume pinned actor/reserve receipt rows through the owned market driver.

This is an offline comparison boundary. It does not authenticate an outer
launch, authorize research, or grant a Full-B4 result.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Literal, cast

import numpy as np

from tools.market_b4_oracle.artificial_component import (
    _FEATURE_NAMES,
    _SYMBOLS,
    ArtificialComponentInputs,
    artificial_component_binding,
    verify_artificial_component_forecast,
)
from tools.market_b4_oracle.decision_identity import (
    canonical_json_bytes as _identity_canonical_json_bytes,
)
from tools.market_b4_oracle.expected_intent import project_quantity
from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
from tools.market_b4_oracle.frozen_market_replay import (
    FrozenMarketBinding,
    FrozenMarketPolicy,
    FrozenMarketReplay,
    ObservedFill,
)
from tools.market_b4_oracle.outer_witness import (
    FrozenHRuntimePin,
    RootObservedAttempt,
    WitnessFilePin,
    implementation_source_identity,
)
from tools.market_b4_oracle.output_index import NativeOutputIndex
from tools.market_b4_oracle.receipt_stream import ReceiptScan
from tools.market_b4_oracle.reference_ledger import (
    ReferenceLedgerCoverage,
    _NativeReferenceLedger,
)
from tools.market_replay_oracle import (
    DEFAULT_PROTOCOL_SOURCE_PROFILE,
    DatasetContentPin,
    DeclaredProtocolPair,
    EventBatchPin,
    ExactFillAttemptBinding,
    NativeSourcePin,
    OrderEventContext,
    ProtocolPin,
    ProtocolSourceProfileId,
    VerifiedDatasetContent,
    load_declared_protocol_pair,
    load_pinned_exact_fill_attempt,
    load_pinned_v4,
    read_pinned_order_events,
    walk_bound_event_rows,
)

_MAX_FORECAST_BYTES = 64 * 1024**2
_FLOAT_TOLERANCE = 1e-10
_ARTIFICIAL_COMPONENT_OUTPUT_FILES = (
    "component-result.json",
    "forecasts/simple-return-stream.json",
    "operator-runtime.json",
    "receipts/cash.json.gz",
    "receipts/nonrl.json.gz",
)


@dataclass(frozen=True, slots=True)
class StaticInputReadback:
    """Paths and exact pins required to recheck a native H/v3 arm's inputs."""

    protocol_v3_path: Path
    protocol_v3_pin: ProtocolPin
    protocol_v4_path: Path
    protocol_v4_pin: ProtocolPin
    source_root: Path
    source_pin: NativeSourcePin
    dataset_root: Path
    dataset_pin: DatasetContentPin
    source_profile: ProtocolSourceProfileId = DEFAULT_PROTOCOL_SOURCE_PROFILE


@dataclass(frozen=True, slots=True)
class IndexedMarketComparison:
    """Comparison-only coverage; the outer native attempt is unauthenticated."""

    status: Literal[
        "PINNED_INPUT_COMPARISON_COMPLETE",
        "ARTIFICIAL_COMPARISON_COMPLETE",
    ]
    binding_digest: str
    account_id: str
    scenario: str
    role: str
    actor_rows: int
    reserve_rows: int
    actor_scan: ReceiptScan
    reserve_scan: ReceiptScan
    actor_fill_processing_indices: tuple[int, ...]
    reserve_fill_processing_indices: tuple[int, ...]
    plan_digest: str
    forecast_sha256: str
    runtime_environment_sha256: str | None
    output_index_rechecked: bool
    static_inputs_rechecked: bool
    outer_attempt_authenticated: Literal[False] = False
    root_attempt_observed: bool = False
    outer_start_witness_sha256: str | None = None
    outer_finish_witness_sha256: str | None = None
    full_b4_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True)
class ArtificialReferenceAccountDiagnostic:
    """Versioned arithmetic diagnostic over one fixed artificial H arm."""

    schema: Literal["issue810_artificial_reference_account_diagnostic_v1"]
    status: Literal["ARTIFICIAL_REFERENCE_ACCOUNT_ARITHMETIC_MATCH"]
    scope: Literal["ARTIFICIAL_COMPARISON_ONLY"]
    account_id: str
    scenario: str
    role: Literal["NONRL", "CASH"]
    dataset_id: str
    component_spec_sha256: str
    source_protocol_sha256: str
    binding_sha256: str
    actor_rows: int
    reserve_rows: int
    fill_count: int
    output_manifest_bytes: int
    output_manifest_sha256: str
    output_file_pins: tuple[tuple[str, int, str], ...]
    output_index_rechecked: Literal[True] = True
    independent_arithmetic_checked: Literal[True] = True
    outer_attempt_authenticated: Literal[False] = False
    root_attempt_observed: Literal[False] = False
    full_b4_authorized: Literal[False] = False
    research_authorized: Literal[False] = False
    economic_evidence: Literal[False] = False
    live_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True)
class _IndexedMarketArmConsumption:
    comparison: IndexedMarketComparison
    reference_account_coverage: ReferenceLedgerCoverage | None


@dataclass(frozen=True, slots=True)
class FrozenNativeMarketArm:
    """One arm selected in the exact frozen H/v3 matrix order."""

    configuration_id: str
    account_id: str
    scenario: str
    native_role: str
    role: Literal["NONRL", "CASH"]
    selected_symbol_index: int
    binding: FrozenMarketBinding
    policy: FrozenMarketPolicy
    receipt_name: str
    forecast_name: str


@dataclass(frozen=True, slots=True)
class IndexedMarketMatrixComparison:
    """One shared-index pass; comparison evidence only, never Full-B4 auth."""

    status: Literal["PINNED_INPUT_MATRIX_COMPARISON_COMPLETE"]
    arms: tuple[IndexedMarketComparison, ...]
    output_index_rechecks: int
    static_input_readbacks: int
    runtime_environment_sha256: str
    outer_attempt_authenticated: Literal[False] = False
    root_attempt_observed: bool = False
    full_b4_authorized: Literal[False] = False


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _read_indexed_bytes(index: NativeOutputIndex, name: str, maximum: int) -> bytes:
    return index.read_pinned_bytes(name, maximum=maximum)


def _validate_fixed_artificial_component_inputs(
    inputs: ArtificialComponentInputs,
) -> None:
    if type(inputs.content) is not VerifiedDatasetContent:
        raise ValueError("artificial inputs require exact verified Dataset content")
    if type(inputs.content.dataset_id) is not str:
        raise ValueError("artificial Dataset identity must be a string")
    spec_sha256 = _sha(inputs.spec_sha256, "artificial component spec digest")
    source_sha256 = _sha(
        inputs.source_protocol_sha256, "artificial source protocol digest"
    )
    expected_protocol: dict[str, object] = {
        "schema": "issue810_artificial_h_component_q_binding_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "dataset": {
            "dataset_id": inputs.content.dataset_id,
            "symbols": list(_SYMBOLS),
            "fit_symbols": [_SYMBOLS[0]],
            "prediction_symbols": list(_SYMBOLS),
            "feature_names": list(_FEATURE_NAMES),
        },
        "configurations": [
            {"id": "H24_RIDGE_A1", "horizon_hours": 24, "ridge_alpha": 1.0}
        ],
        "accounts": ["H24_RIDGE_A1_BTCUSDT"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "actor": {
            "start": 2903,
            "stop": 9430,
            "vintage_starts": [2903, 5087, 7295],
        },
        "terminal": {"stop_inclusive": 9502},
        "economics": {"reference_notional": 100000.0},
        "funding": {
            "coverage_bounds": {
                "lookback_hours": 24,
                "minimum_counted_events": 2,
                "maximum_counted_close_gap_hours": 9.0,
                "maximum_counted_close_age_hours": 8.0,
            }
        },
        "capital": {"per_account": 100000.0},
        "policy": {
            "risk": {
                "max_gross": 1.0,
                "max_abs_weight": 1.0,
                "max_turnover": None,
                "drawdown_start": 0.1,
                "drawdown_stop": 0.2,
                "emergency_turnover_override": True,
                "fail_closed_tolerance": 1e-10,
            }
        },
        "component_spec_sha256": spec_sha256,
        "source_protocol_sha256": source_sha256,
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
        "research_authorized": False,
    }
    expected_bytes = _canonical(expected_protocol)
    if (
        type(inputs.protocol_document_canonical) is not bytes
        or inputs.protocol_document_canonical != expected_bytes
        or _canonical(inputs.protocol_document) != expected_bytes
    ):
        raise ValueError("artificial inputs differ from the fixed comparison profile")


def _same_content(
    expected: VerifiedDatasetContent, observed: VerifiedDatasetContent
) -> bool:
    return (
        expected.dataset_id == observed.dataset_id
        and dict(expected.dataset) == dict(observed.dataset)
        and set(expected.arrays) == set(observed.arrays)
        and all(
            np.array_equal(expected.arrays[name], observed.arrays[name])
            for name in expected.arrays
        )
    )


def _readback_static_inputs(
    *,
    pins: StaticInputReadback,
    binding: FrozenMarketBinding,
    content: VerifiedDatasetContent,
    policy: FrozenMarketPolicy,
) -> tuple[DeclaredProtocolPair, ExactFillAttemptBinding, VerifiedDatasetContent]:
    if type(pins) is not StaticInputReadback or binding.comparison_only:
        raise ValueError("native exact readback requires a native arm and static pins")
    pair = load_declared_protocol_pair(
        pins.protocol_v3_path,
        pins.protocol_v3_pin,
        pins.protocol_v4_path,
        pins.protocol_v4_pin,
        source_profile=pins.source_profile,
    )
    if (
        binding.protocol_pair != pair
        or pair.native_pin.source_head != pins.source_pin.head_sha
    ):
        raise ValueError("declared protocol/source pins differ from the selected arm")
    frozen = load_pinned_exact_fill_attempt(
        pins.protocol_v4_path,
        pins.protocol_v4_pin,
        pins.source_root,
        pins.source_pin,
        pins.dataset_root,
        pins.dataset_pin,
        source_profile=pins.source_profile,
    )
    loaded = load_pinned_v4(pins.dataset_root, pins.dataset_pin)
    if not _same_content(content, loaded) or frozen.payload()["clock"] != {
        "actor_start": binding.window.actor_start,
        "actor_stop": binding.window.actor_stop,
        "settlement_stop": binding.window.terminal_stop,
    }:
        raise ValueError("reloaded source, Dataset or H/v3 clock differs from replay")
    rebound = FrozenMarketBinding.from_declared_protocol_pair(
        protocol_pair=pair,
        content=loaded,
        configuration_id=binding.configuration_id,
        scenario=binding.scenario,
        native_role=binding.native_role,
        selected_symbol_index=binding.selected_symbol_index,
    )
    expected_policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=loaded.dataset_id,
        profile=rebound.effective_profile,
        risk=rebound.protocol_document["policy"]["risk"],
    )
    if rebound != binding or expected_policy != policy:
        raise ValueError("reloaded native arm or execution policy differs from replay")
    if (
        load_declared_protocol_pair(
            pins.protocol_v3_path,
            pins.protocol_v3_pin,
            pins.protocol_v4_path,
            pins.protocol_v4_pin,
            source_profile=pins.source_profile,
        )
        != pair
    ):
        raise ValueError("declared protocol pair changed during static input readback")
    return pair, frozen, loaded


def _validate_native_event_rows(
    *,
    output_index: NativeOutputIndex,
    receipt_name: str,
    binding: FrozenMarketBinding,
    exact_fill_binding: ExactFillAttemptBinding,
) -> None:
    """Require a full strict-v4 chain over the original indexed receipt rows."""
    if (
        binding.comparison_only
        or type(exact_fill_binding) is not ExactFillAttemptBinding
    ):
        raise ValueError("native event readback requires its exact paired v4 binding")
    actor_reader = output_index.receipt_rows(receipt_name, phase="actor")
    reserve_reader = output_index.receipt_rows(receipt_name, phase="reserve")
    coverage = walk_bound_event_rows(
        exact_fill_binding,
        account=binding.account_id,
        scenario=binding.scenario,
        role=binding.native_role,
        actor_rows=actor_reader,
        closing_rows=reserve_reader,
    )
    actor_scan, reserve_scan = actor_reader.proof, reserve_reader.proof
    expected_actor_rows = binding.window.actor_advances
    expected_reserve_rows = binding.window.reserve_advances
    if coverage.rejected_native_rows != 0:
        raise ValueError("native event receipt contains rejected native facts")
    if (
        coverage.binding_digest != exact_fill_binding.digest
        or coverage.account != binding.account_id
        or coverage.scenario != binding.scenario
        or coverage.role != binding.native_role
        or coverage.expected_actor_rows != expected_actor_rows
        or coverage.expected_reserve_rows != expected_reserve_rows
        or coverage.observed_actor_rows != expected_actor_rows
        or coverage.observed_reserve_rows != expected_reserve_rows
        or not coverage.all_expected_batches_present
        or coverage.tail_native_missing
        or actor_scan.selected_rows != expected_actor_rows
        or actor_scan.actor_rows != expected_actor_rows
        or actor_scan.reserve_rows != expected_reserve_rows
        or reserve_scan.selected_rows != expected_reserve_rows
        or reserve_scan.actor_rows != expected_actor_rows
        or reserve_scan.reserve_rows != expected_reserve_rows
        or actor_scan.compressed_sha256 != reserve_scan.compressed_sha256
    ):
        raise ValueError(
            "native receipt lacks a complete, nonrejected strict-v4 event chain"
        )


def _json_pairs_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("frozen native matrix has a duplicate JSON key")
        result[key] = value
    return result


def _json_object_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw, object_pairs_hook=_json_pairs_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} is not bounded UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _read_bounded_gzip_json(
    output_index: NativeOutputIndex,
    name: str,
    *,
    compressed_limit: int,
    expanded_limit: int,
) -> dict[str, object]:
    compressed = _read_indexed_bytes(output_index, name, compressed_limit)
    try:
        stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = stream.decompress(compressed, expanded_limit + 1)
        if (
            len(raw) > expanded_limit
            or not stream.eof
            or stream.unused_data
            or stream.unconsumed_tail
        ):
            raise ValueError("frozen native matrix exceeds its bounded gzip contract")
        tail = stream.flush()
    except zlib.error as error:
        raise ValueError("frozen native matrix gzip is malformed") from error
    raw += tail
    if len(raw) > expanded_limit:
        raise ValueError("frozen native matrix exceeds its expanded byte limit")
    return _json_object_bytes(raw, "frozen native matrix")


def derive_frozen_native_market_roster(
    *,
    protocol_pair: DeclaredProtocolPair,
    content: VerifiedDatasetContent,
) -> tuple[FrozenNativeMarketArm, ...]:
    """Derive the frozen 48-arm H/v3 roster from its validated declaration."""
    if type(protocol_pair) is not DeclaredProtocolPair:
        raise ValueError("native matrix requires the exact declared H/v3 and H/v4 pair")
    protocol_pair.validate()
    if type(content) is not VerifiedDatasetContent:
        raise ValueError("native matrix requires caller-pinned Dataset content")
    protocol = _json_object_bytes(protocol_pair.native_canonical, "native protocol")
    dataset = protocol.get("dataset")
    policy_document = protocol.get("policy")
    configurations = protocol.get("configurations")
    scenarios = protocol.get("scenarios")
    roles = protocol.get("roles")
    if (
        type(dataset) is not dict
        or type(policy_document) is not dict
        or type(policy_document.get("risk")) is not dict
        or type(configurations) is not list
        or type(scenarios) is not list
        or type(roles) is not list
        or tuple(scenarios) != ("BASEL0", "COST2L0", "BASEL2")
        or tuple(roles) != ("nonrl", "cash")
        or protocol.get("scope") != "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
    ):
        raise ValueError("native protocol differs from the closed 48-arm roster")
    symbols = dataset.get("fit_symbols")
    risk = cast(dict[str, object], policy_document["risk"])
    if type(symbols) is not list or tuple(symbols) != ("BTCUSDT", "ETHUSDT"):
        raise ValueError("native matrix requires the exact two-symbol roster")
    if len(configurations) != 4:
        raise ValueError("native matrix requires four frozen forecast configurations")
    budget = protocol.get("budget")
    actor = protocol.get("actor")
    terminal = protocol.get("terminal")
    if (
        type(budget) is not dict
        or budget.get("native_arms") != 48
        or budget.get("advances_per_arm") != 6599
        or type(actor) is not dict
        or actor.get("rows") != 6527
        or type(terminal) is not dict
        or terminal.get("advances") != 72
    ):
        raise ValueError("native protocol actor/reserve coverage differs from 6,599")

    arms: list[FrozenNativeMarketArm] = []
    for configuration in configurations:
        if type(configuration) is not dict:
            raise ValueError("native configuration roster is malformed")
        configuration_id = configuration.get("id")
        if type(configuration_id) is not str:
            raise ValueError("native configuration id is malformed")
        for symbol_index, symbol in enumerate(symbols):
            if type(symbol) is not str:
                raise ValueError("native symbol roster is malformed")
            account_id = f"{configuration_id}_{symbol}"
            for scenario in scenarios:
                if type(scenario) is not str:
                    raise ValueError("native scenario roster is malformed")
                for native_role in roles:
                    if type(native_role) is not str:
                        raise ValueError("native role roster is malformed")
                    binding = FrozenMarketBinding.from_declared_protocol_pair(
                        protocol_pair=protocol_pair,
                        content=content,
                        configuration_id=configuration_id,
                        scenario=scenario,
                        native_role=native_role,
                        selected_symbol_index=symbol_index,
                    )
                    policy = FrozenMarketPolicy.from_effective_profile(
                        dataset_id=content.dataset_id,
                        profile=binding.effective_profile,
                        risk=risk,
                    )
                    arms.append(
                        FrozenNativeMarketArm(
                            configuration_id,
                            account_id,
                            scenario,
                            native_role,
                            "NONRL" if native_role == "nonrl" else "CASH",
                            symbol_index,
                            binding,
                            policy,
                            f"receipts/{account_id}_{scenario}_{native_role}.json.gz",
                            f"forecasts/{configuration_id}.json",
                        )
                    )
    if (
        len(arms) != 48
        or len({(arm.account_id, arm.scenario, arm.native_role) for arm in arms}) != 48
    ):
        raise ValueError("native protocol did not derive exactly 48 unique arms")
    return tuple(arms)


def _frozen_native_plan_digests(
    *,
    output_index: NativeOutputIndex,
    protocol_pair: DeclaredProtocolPair,
    roster: tuple[FrozenNativeMarketArm, ...],
    content: VerifiedDatasetContent,
    expected_runtime_environment_sha256: str,
) -> dict[tuple[str, str, str], str]:
    freeze = _json_object_bytes(
        _read_indexed_bytes(output_index, "freeze.json", maximum=1024 * 1024),
        "native freeze record",
    )
    expected_freeze_keys = {
        "schema",
        "protocol_digest",
        "provenance_digest",
        "matrix_digest",
        "matrix_file",
        "costs_digest",
        "costs_file",
        "forecasts",
        "pairs",
    }
    if (
        set(freeze) != expected_freeze_keys
        or freeze.get("schema") != "tier_d_frozen_inputs_v1"
        or freeze.get("protocol_digest")
        != hashlib.sha256(protocol_pair.native_canonical).hexdigest()
    ):
        raise ValueError("native frozen-input record differs from H/v3 protocol")

    matrix_file = freeze.get("matrix_file")
    costs_file = freeze.get("costs_file")
    forecasts = freeze.get("forecasts")
    if (
        type(matrix_file) is not dict
        or type(costs_file) is not dict
        or type(forecasts) is not dict
        or set(matrix_file) != {"path", "sha256", "bytes"}
        or set(costs_file) != {"path", "sha256", "bytes"}
        or matrix_file.get("path") != "native-plan.json.gz"
        or costs_file.get("path") != "costs.json.gz"
    ):
        raise ValueError("native frozen matrix/cost file pins are malformed")
    matrix_digest = _sha(freeze.get("matrix_digest"), "matrix_digest")
    _sha(freeze.get("costs_digest"), "costs_digest")
    matrix_name = cast(str, matrix_file["path"])
    matrix_pin = output_index.pin(matrix_name)
    if matrix_pin.bytes != matrix_file.get(
        "bytes"
    ) or matrix_pin.sha256 != matrix_file.get("sha256"):
        raise ValueError("native frozen matrix pin differs from the output index")
    costs_pin = output_index.pin(cast(str, costs_file["path"]))
    if costs_pin.bytes != costs_file.get("bytes") or costs_pin.sha256 != costs_file.get(
        "sha256"
    ):
        raise ValueError("native frozen cost pin differs from the output index")
    if set(forecasts) != {arm.configuration_id for arm in roster} or any(
        type(row) is not dict or set(row) != {"path", "sha256", "bytes", "dataset_id"}
        for row in forecasts.values()
    ):
        raise ValueError("native forecast roster differs from the four configurations")
    for arm in roster:
        row = cast(dict[str, object], forecasts[arm.configuration_id])
        if (
            row.get("path") != arm.forecast_name
            or row.get("dataset_id") != content.dataset_id
            or output_index.pin(arm.forecast_name).bytes != row.get("bytes")
            or output_index.pin(arm.forecast_name).sha256 != row.get("sha256")
        ):
            raise ValueError("native forecast pin differs from the frozen roster")

    matrix = _read_bounded_gzip_json(
        output_index,
        matrix_name,
        compressed_limit=256 * 1024**2,
        expanded_limit=512 * 1024**2,
    )
    if (
        hashlib.sha256(_canonical(matrix)).hexdigest() != matrix_digest
        or matrix.get("schema") != "allocation_settled_nonrl_comparison_plan_v1"
        or matrix.get("accounts")
        != list(dict.fromkeys(arm.account_id for arm in roster))
        or matrix.get("scenarios")
        != list(dict.fromkeys(arm.scenario for arm in roster))
    ):
        raise ValueError("native pre-execution matrix differs from the frozen roster")

    expected_pairs = {
        (arm.configuration_id, arm.account_id, arm.scenario, arm.selected_symbol_index)
        for arm in roster
    }
    pair_rows = freeze.get("pairs")
    if type(pair_rows) is not list or any(
        type(row) is not dict
        or set(row) != {"configuration", "account", "scenario", "symbol_index"}
        for row in pair_rows
    ):
        raise ValueError("native frozen account/scenario pair roster is malformed")
    observed_pairs = {
        (
            row["configuration"],
            row["account"],
            row["scenario"],
            row["symbol_index"],
        )
        for row in cast(list[dict[str, object]], pair_rows)
    }
    if len(pair_rows) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError(
            "native frozen account/scenario pairs differ from the protocol"
        )

    by_key = {(arm.account_id, arm.scenario, arm.native_role): arm for arm in roster}
    groups = matrix.get("groups")
    if type(groups) is not list or len(groups) != 24:
        raise ValueError("native pre-execution matrix must contain exactly 24 pairs")
    plan_digests: dict[tuple[str, str, str], str] = {}
    for group in groups:
        if (
            type(group) is not dict
            or group.get("schema") != "allocation_terminal_nonrl_controls_v1"
            or set(group) != {"schema", "plans"}
            or type(group.get("plans")) is not dict
        ):
            raise ValueError("native terminal pair plan is malformed")
        plans = cast(dict[str, object], group["plans"])
        if set(plans) != {"nonrl", "cash"}:
            raise ValueError("native terminal pair omits a frozen role")
        for native_role, plan_value in plans.items():
            if type(plan_value) is not dict:
                raise ValueError("native terminal plan is not an object")
            plan = cast(dict[str, object], plan_value)
            native_plan = plan.get("native_plan")
            if type(native_plan) is not dict:
                raise ValueError("native terminal plan lacks its native arm")
            common = native_plan.get("common")
            if type(common) is not dict:
                raise ValueError("native terminal plan lacks its frozen account")
            account_id = common.get("account_id")
            scenario = common.get("scenario")
            key = (account_id, scenario, native_role)
            if (
                type(account_id) is not str
                or type(scenario) is not str
                or key not in by_key
                or key in plan_digests
            ):
                raise ValueError("native terminal plan duplicates or leaves the roster")
            arm = by_key[key]
            digest = hashlib.sha256(_canonical(plan)).hexdigest()
            replay = FrozenMarketReplay(
                content=content,
                policy=arm.policy,
                binding=arm.binding,
                reserve_plan_digest=digest,
            )
            _metadata_arm(
                {"plan": plan, "plan_digest": digest},
                replay,
                digest,
                expected_runtime_environment_sha256=expected_runtime_environment_sha256,
            )
            expected_forecast = cast(
                dict[str, object], forecasts[arm.configuration_id]
            )["sha256"]
            if common.get("forecast_context_digest") != expected_forecast:
                raise ValueError(
                    "native plan forecast identity differs from its frozen file"
                )
            plan_digests[key] = digest
    if set(plan_digests) != set(by_key):
        raise ValueError("native terminal plan file does not cover exactly 48 arms")
    return plan_digests


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric receipt data")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric receipt data")
    return result


def _same_float(actual: object, expected: float, name: str) -> None:
    observed = _number(actual, name)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"native {name} differs from the owned replay")


def _event_context(
    replay: FrozenMarketReplay, processing_index: int
) -> OrderEventContext:
    decision_index = processing_index - 1
    timestamps = replay.content.arrays["timestamps"]
    symbol = replay.selected_symbol_index
    lot = (
        max(
            float(replay.content.arrays["lot_size"][processing_index, symbol]),
            replay.policy.lot_size_floor,
        )
        * replay.policy.lot_size_factor
    )
    return OrderEventContext(
        dataset_id=replay.content.dataset_id,
        execution_policy_digest=replay.policy.execution_policy_digest,
        symbol_index=symbol,
        decision_index=decision_index,
        decision_time_ns=int(timestamps[decision_index].astype("int64")),
        processing_index=processing_index,
        processing_time_ns=int(timestamps[processing_index].astype("int64")),
        effective_lot_size=lot,
    )


def _parsed_events(
    native: dict[str, object], context: OrderEventContext
) -> tuple[tuple[dict[str, object], ...], tuple[ObservedFill, ...]]:
    values = native.get("order_events")
    if type(values) is not list:
        raise ValueError("native row requires a bounded order_events array")
    # This parser pin covers only our canonical projection of already parsed
    # receipt events. The original gzip and output-manifest pins remain the
    # source evidence; this derived pin is never represented as the raw batch.
    raw = _canonical(values)
    evidence = read_pinned_order_events(
        raw,
        EventBatchPin(len(raw), hashlib.sha256(raw).hexdigest()),
        context,
    )
    events = tuple(item.payload() for item in evidence)
    fills = tuple(
        ObservedFill(
            order_id=cast(str, event["order_id"]),
            processing_index=cast(int, event["processing_index"]),
            exact_quantity=evidence[index].exact_filled_quantity,
            execution_price=_number(event["execution_price"], "fill execution price"),
            filled_lot_count=(
                cast(int, event["filled_lot_count"])
                if "filled_lot_count" in event
                else None
            ),
            lot_size=(
                _number(event["lot_size"], "fill lot size")
                if "lot_size" in event
                else 0.0
            ),
        )
        for index, event in enumerate(events)
        if event["event_type"] in {"filled", "partial_fill"}
    )
    return events, fills


def _native_bar(
    row: object,
    *,
    phase: Literal["actor", "reserve"],
    replay: FrozenMarketReplay,
    processing_index: int,
) -> dict[str, object]:
    if type(row) is not dict:
        raise ValueError("receipt row must be an object")
    decision_index = processing_index - 1
    times = replay.content.arrays["timestamps"]
    decision_time = int(times[decision_index].astype("int64"))
    processing_time = int(times[processing_index].astype("int64"))
    if phase == "actor":
        if row.get("decision_index") != decision_index:
            raise ValueError(
                "actor receipt decision index differs from the bound clock"
            )
        if row.get("decision_time_ns") != decision_time:
            raise ValueError("actor receipt decision time differs from the bound clock")
        if row.get("native_validation_failure") is not None:
            raise ValueError("actor receipt contains native validation failure")
        expected_terminated = processing_index == replay.window.actor_stop
        if (
            row.get("terminated") is not expected_terminated
            or row.get("truncated") is not False
        ):
            raise ValueError(
                "actor receipt termination differs from its bound phase edge"
            )
    elif (
        row.get("start_index") != decision_index
        or row.get("processing_index") != processing_index
        or row.get("processing_time_ns") != processing_time
        or row.get("clock_valid") is not True
    ):
        raise ValueError("reserve receipt row differs from the bound clock")
    native = row.get("native")
    if type(native) is not dict:
        raise ValueError("receipt row is missing its native state projection")
    native_clock = (
        (
            ("decision_index", decision_index),
            ("decision_time_ns", decision_time),
            ("processing_index", processing_index),
            ("processing_time_ns", processing_time),
            ("symbol_index", replay.selected_symbol_index),
        )
        if phase == "actor"
        else ()
    )
    for field, expected in native_clock:
        if native.get(field) != expected:
            raise ValueError(
                f"native row {field} differs from the bound clock: "
                f"{native.get(field)!r} != {expected!r}"
            )
    execution = native.get("execution")
    if (
        type(execution) is not dict
        or type(execution.get("next_index")) is not int
        or type(execution.get("bars_advanced")) is not int
        or execution.get("next_index") != processing_index
        or execution.get("bars_advanced") != 1
    ):
        raise ValueError("native execution row differs from one-bar advancement")
    return cast(dict[str, object], native)


def _compare_book(
    book: object,
    replay: FrozenMarketReplay,
    *,
    processing_index: int,
    interval_return: object | None = None,
) -> None:
    if type(book) is not dict:
        raise ValueError("native book is missing its row projection")
    snapshot = replay.account_snapshot
    if (
        type(book.get("as_of_index")) is not int
        or book.get("as_of_index") != processing_index
        or book.get("as_of_dataset_id") != replay.content.dataset_id
        or type(book.get("exact_quantities")) is not list
        or len(cast(list[object], book["exact_quantities"])) != 2
    ):
        raise ValueError("native book clock or exact quantity shape differs")
    exact_values = cast(list[object], book["exact_quantities"])
    if any(type(item) is not str for item in exact_values):
        raise ValueError("exact quantity must be a string")
    exact = tuple(Fraction(cast(str, item)) for item in exact_values)
    if exact != snapshot.exact_quantities:
        raise ValueError("native exact quantities differ from the owned replay")
    for field, expected in (
        ("cash", snapshot.cash),
        ("equity", snapshot.wealth),
        ("peak_value", snapshot.peak_value),
        ("max_drawdown", snapshot.max_drawdown),
        ("total_cost", snapshot.total_cost),
        ("funding_pnl", snapshot.funding_pnl),
        ("borrow_cost", snapshot.borrow_cost),
        ("turnover_total", snapshot.turnover_total),
        ("margin_used", snapshot.margin_used),
        ("maintenance_margin", snapshot.maintenance_margin),
        ("maintenance_requirement", snapshot.maintenance_requirement),
        ("margin_deficit", snapshot.margin_deficit),
    ):
        _same_float(book.get(field), expected, field)
    if (
        type(book.get("fill_count")) is not int
        or type(book.get("rebalance_events")) is not int
        or book.get("fill_count") != snapshot.fill_count
        or book.get("rebalance_events") != snapshot.rebalance_events
        or book.get("insolvent") is not snapshot.insolvent
        or book.get("termination_reason") != snapshot.termination_reason
    ):
        raise ValueError(
            "native book counters or termination differ from the owned replay"
        )
    for field, expected_values in (
        ("quantities", snapshot.projected_quantities),
        ("mark_prices", snapshot.marks),
        ("contract_multipliers", snapshot.multipliers),
    ):
        values = book.get(field)
        if type(values) is not list or len(values) != 2:
            raise ValueError(f"native book {field} must have two entries")
        for actual, wanted in zip(values, expected_values, strict=True):
            _same_float(actual, wanted, field)
    if interval_return is not None:
        if not snapshot.returns_history:
            raise ValueError("owned replay did not record its interval return")
        _same_float(interval_return, snapshot.returns_history[-1], "interval return")


def _compare_orders(
    native: dict[str, object],
    replay: FrozenMarketReplay,
    advance: object,
    prior_terminal_ids: set[str],
) -> None:
    from tools.market_b4_oracle.frozen_market_replay import ReplayAdvance

    if type(advance) is not ReplayAdvance:
        raise ValueError("owned replay did not return its exact row advance")
    active = native.get("active_orders")
    reasons = native.get("terminal_order_reasons")
    if type(active) is not list or type(reasons) is not list:
        raise ValueError("native row is missing active or terminal order projections")
    observed_active: dict[str, dict[str, object]] = {}
    for value in active:
        if type(value) is not dict or type(value.get("intent")) is not dict:
            raise ValueError("native active order shape is invalid")
        order_id = value["intent"].get("order_id")
        if type(order_id) is not str or order_id in observed_active:
            raise ValueError("native active order identity is invalid or duplicated")
        observed_active[order_id] = value
    owned_active = replay._active
    if set(observed_active) != set(owned_active):
        raise ValueError("native active order ids differ from owned order state")
    for order_id, order in owned_active.items():
        actual = observed_active[order_id]
        intent = actual.get("intent")
        if type(intent) is not dict or intent != order.intent.canonical_payload():
            raise ValueError("native active order intent differs from owned state")
        if (
            actual.get("status") != order.status
            or actual.get("evidence_version") != order.evidence_version
            or actual.get("last_processed_index") != order.last_processed_index
            or actual.get("trigger_index") is not None
            or actual.get("terminal_reason") is not None
            or actual.get("exact_cumulative_filled_quantity")
            != str(order.exact_filled_quantity)
        ):
            raise ValueError(
                "native active order status differs from owned order state"
            )
        _same_float(
            actual.get("remaining_quantity"),
            order.remaining_quantity,
            "remaining quantity",
        )
        _same_float(
            actual.get("cumulative_filled_quantity"),
            project_quantity(order.exact_filled_quantity),
            "cumulative filled quantity",
        )
        _same_float(
            actual.get("cumulative_filled_notional"),
            order.cumulative_filled_notional,
            "cumulative filled notional",
        )

    new_terminal = {
        order_id: order.terminal_reason
        for order_id, order in replay._terminal.items()
        if order_id not in prior_terminal_ids
    }
    observed_reasons: dict[str, object] = {}
    for value in reasons:
        if type(value) is not dict:
            raise ValueError("native terminal order reason must be an object")
        order_id, reason = value.get("order_id"), value.get("reason")
        if type(order_id) is not str or order_id in observed_reasons:
            raise ValueError("native terminal order identity is invalid or duplicated")
        observed_reasons[order_id] = reason
    if observed_reasons != new_terminal:
        raise ValueError("native terminal order updates differ from owned order state")


def _compare_event_effects(
    events: tuple[dict[str, object], ...],
    observed_fills: tuple[ObservedFill, ...],
    replay: FrozenMarketReplay,
    advance: object,
) -> None:
    from tools.market_b4_oracle.frozen_market_replay import ReplayAdvance

    if type(advance) is not ReplayAdvance:
        raise ValueError("owned replay did not return its exact row advance")
    submitted = tuple(
        cast(str, event["order_id"])
        for event in events
        if event["event_type"] == "submitted"
    )
    if submitted != advance.new_intent_ids:
        raise ValueError("native submissions differ from owned order transitions")
    cancelled = tuple(
        cast(str, event["order_id"])
        for event in events
        if event["event_type"] == "cancelled"
    )
    if cancelled != advance.cancelled_order_ids:
        raise ValueError("native cancellations differ from owned order transitions")
    expected_fills = tuple(fill.observed_projection() for fill in advance.owned_fills)
    if observed_fills != expected_fills:
        raise ValueError("native fill events differ from owned raw-row fills")

    last_event: dict[str, dict[str, object]] = {}
    for event in events:
        order_id = event.get("order_id")
        if type(order_id) is not str or order_id not in replay._all_orders:
            raise ValueError("native event references an unknown owned order")
        last_event[order_id] = event
    for order_id, event in last_event.items():
        order = replay._all_orders[order_id]
        if event.get("new_status") != order.status:
            raise ValueError("native event status differs from owned order state")
        _same_float(
            event.get("remaining_quantity"),
            order.remaining_quantity,
            "event remaining quantity",
        )


def _compare_native_fill_diagnostics(
    events: tuple[dict[str, object], ...],
    replay: FrozenMarketReplay,
    advance: object,
) -> None:
    from tools.market_b4_oracle.frozen_market_replay import OwnedFill, ReplayAdvance

    if type(advance) is not ReplayAdvance:
        raise ValueError("owned replay did not return its exact row advance")
    fill_events = tuple(
        event
        for event in events
        if event.get("event_type") in {"filled", "partial_fill"}
    )
    if not fill_events:
        return
    if (
        len(fill_events) != 1
        or len(advance.owned_fills) != 1
        or type(advance.owned_fills[0]) is not OwnedFill
    ):
        raise ValueError("native fill diagnostics require one owned active order")

    schema_version = fill_events[0].get("schema_version")
    if schema_version not in {"order_event_v1", "order_event_v2"}:
        raise ValueError("native fill diagnostics require a supported event schema")
    schema_label = cast(str, schema_version).removeprefix("order_event_")

    selected = replay.selected_symbol_index
    volume_units = replay.content.dataset.get("volume_units")
    if (
        type(volume_units) is not tuple
        or len(volume_units) != 2
        or volume_units[selected] != "quote_notional"
    ):
        raise ValueError("native fill diagnostics require quote-notional capacity")

    arrays = replay.content.arrays
    fill = advance.owned_fills[0]
    processing = fill.processing_index
    if processing != advance.processing_index or processing <= 0:
        raise ValueError("native fill diagnostics require the owned processing row")
    # The frozen native profile consumes prior-bar quote volume and fixes the
    # capacity pool from the minimum of policy and processing-row participation.
    prior_volume = float(arrays["volume"][processing - 1, selected])
    row_limit = float(arrays["max_participation_rate"][processing, selected])
    initial_capacity = prior_volume * min(
        replay.policy.max_participation_rate, row_limit
    )
    event = fill_events[0]
    capacity_after = max(0.0, initial_capacity - fill.filled_notional)
    _same_float(
        event.get("filled_notional"),
        fill.filled_notional,
        f"{schema_label} fill notional",
    )
    _same_float(
        event.get("capacity_before"),
        initial_capacity,
        f"{schema_label} capacity before",
    )
    _same_float(
        event.get("capacity_after"),
        capacity_after,
        f"{schema_label} capacity after",
    )
    _same_float(
        event.get("participation_rate"),
        fill.participation_rate,
        f"{schema_label} participation rate",
    )


def _native_actor_identity_hashes(
    native: dict[str, object],
    *,
    expected_context: dict[str, object],
    expected_proposal: dict[str, object],
) -> dict[str, object]:
    proposal = native.get("proposal")
    if type(proposal) is not dict or type(proposal.get("decision")) is not dict:
        raise ValueError("native actor proposal or decision is malformed")
    decision = cast(dict[str, object], proposal["decision"])
    baseline_value = decision.get("baseline")
    if type(baseline_value) is not dict:
        raise ValueError("native actor baseline is malformed")
    baseline = cast(dict[str, object], baseline_value)
    allocator_value, context_value = baseline.get("allocator"), baseline.get("context")
    inputs_value = baseline.get("inputs")
    if (
        type(allocator_value) is not dict
        or type(context_value) is not dict
        or type(inputs_value) is not dict
    ):
        raise ValueError("native actor baseline context or allocator is malformed")
    allocator = cast(dict[str, object], allocator_value)
    context = cast(dict[str, object], context_value)
    inputs = cast(dict[str, object], inputs_value)
    canonical_context = dict(context)
    canonical_inputs = dict(inputs)
    expected_inputs_value = expected_proposal.get("inputs")
    if type(expected_inputs_value) is not dict:
        raise ValueError("owned actor proposal inputs are malformed")
    expected_inputs = cast(dict[str, object], expected_inputs_value)
    for values, expected_values, names, label in (
        (
            canonical_context,
            expected_context,
            ("decision_time",),
            "native actor context time",
        ),
        (
            canonical_inputs,
            expected_inputs,
            ("decision_time", "available_at", "horizon_end"),
            "native actor input time",
        ),
    ):
        for name in names:
            observed_time = values.get(name)
            expected_time = expected_values.get(name)
            if type(observed_time) is not int or type(expected_time) is not str:
                raise ValueError(f"{label} is not in the native timestamp form")
            try:
                expected_ns = int(np.datetime64(expected_time, "ns").astype(np.int64))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{label} is not a valid owned timestamp") from error
            if observed_time != expected_ns:
                raise ValueError(f"{label} differs from owned replay")
            values[name] = expected_time
    current = _number(context.get("current_weight"), "native current weight")
    lower = _number(allocator.get("lower_weight"), "native lower bound")
    upper = _number(allocator.get("upper_weight"), "native upper bound")
    maximum_turnover = allocator.get("max_turnover")
    if maximum_turnover is not None:
        turnover = _number(maximum_turnover, "native maximum turnover")
        lower = max(lower, current - turnover)
        upper = min(upper, current + turnover)

    baseline_digest = _sha(baseline.get("decision_digest"), "native baseline digest")
    context_digest = _sha(context.get("state_digest"), "native context state digest")
    action_contract = decision.get("action_contract")
    if type(action_contract) is not dict:
        raise ValueError("native actor decision is missing its action contract")
    decision_payload = {
        "schema": "allocation_decision_v1",
        "baseline": baseline_digest,
        "action_contract": {
            "schema": "allocation_action_v1",
            **cast(dict[str, object], action_contract),
        },
        "feasible_bounds": (float(lower), float(upper)),
        "feature_names": decision.get("feature_names"),
        "feature_values": decision.get("feature_values"),
        "max_drawdown": decision.get("max_drawdown"),
        "initial_capital": decision.get("initial_capital"),
        "remaining_steps": decision.get("remaining_steps"),
        "pending_gross": decision.get("pending_gross"),
        "pending_count": decision.get("pending_count"),
    }
    decision_digest = hashlib.sha256(
        _identity_canonical_json_bytes(decision_payload)
    ).hexdigest()
    action_payload = {
        "schema": "allocation_action_proposal_v1",
        "decision": decision_digest,
        "raw_action": proposal.get("raw_action"),
        "action": proposal.get("action"),
        "raw_target_weight": proposal.get("raw_target_weight"),
        "target_weight": proposal.get("target_weight"),
        "is_hold": proposal.get("is_hold"),
    }
    action_digest = hashlib.sha256(
        _identity_canonical_json_bytes(action_payload)
    ).hexdigest()
    baseline_target = _number(baseline.get("target_weight"), "native baseline target")
    proposed_target = _number(proposal.get("target_weight"), "native proposed target")
    return {
        "baseline": baseline_digest,
        "context_state": context_digest,
        "decision": decision_digest,
        "action": action_digest,
        "selected": baseline_digest
        if proposed_target == baseline_target
        else action_digest,
        "context_bytes": _identity_canonical_json_bytes(canonical_context),
        "proposal_bytes": _identity_canonical_json_bytes(
            {
                "inputs": canonical_inputs,
                "context": canonical_context,
                "allocator": allocator,
                "target": baseline_target,
                "objective_value": _number(
                    baseline.get("objective_value"), "native objective value"
                ),
            }
        ),
        "decision_bytes": _identity_canonical_json_bytes(decision_payload),
        "action_bytes": _identity_canonical_json_bytes(action_payload),
    }


def _compare_actor_identity(native: dict[str, object], advance: object) -> None:
    from tools.market_b4_oracle.frozen_market_replay import ReplayAdvance

    if type(advance) is not ReplayAdvance or advance.actor_identity is None:
        raise ValueError("actor row lacks an owned identity result")
    if (
        type(advance.actor_context_bytes) is not bytes
        or type(advance.actor_proposal_bytes) is not bytes
    ):
        raise ValueError("actor row lacks immutable owned identity payloads")
    expected_context = json.loads(advance.actor_context_bytes)
    expected_proposal = json.loads(advance.actor_proposal_bytes)
    if type(expected_context) is not dict or type(expected_proposal) is not dict:
        raise ValueError("owned actor identity payloads are malformed")
    observed = _native_actor_identity_hashes(
        native,
        expected_context=expected_context,
        expected_proposal=expected_proposal,
    )
    comparisons = (
        ("actor_proposal_digest", advance.actor_proposal_digest, observed["baseline"]),
        (
            "actor context state digest",
            advance.actor_context_state_digest,
            observed["context_state"],
        ),
        ("actor decision digest", advance.actor_decision_digest, observed["decision"]),
        ("actor action digest", advance.actor_action_digest, observed["action"]),
        ("actor identity", advance.actor_identity, observed["selected"]),
    )
    for digest_name, expected_digest, actual_digest in comparisons:
        if expected_digest != actual_digest:
            raise ValueError(f"native {digest_name} differs from owned replay")
    payload_comparisons = (
        (
            "actor context content",
            advance.actor_context_bytes,
            observed["context_bytes"],
        ),
        (
            "actor proposal content",
            advance.actor_proposal_bytes,
            observed["proposal_bytes"],
        ),
        (
            "actor decision content",
            advance.actor_decision_bytes,
            observed["decision_bytes"],
        ),
        ("actor action content", advance.actor_action_bytes, observed["action_bytes"]),
    )
    for payload_name, expected_payload, actual_payload in payload_comparisons:
        if (
            type(expected_payload) is not bytes
            or type(actual_payload) is not bytes
            or expected_payload != actual_payload
        ):
            raise ValueError(f"native {payload_name} differs from owned replay")


def _metadata_arm(
    metadata: dict[str, object],
    replay: FrozenMarketReplay,
    reserve_plan_digest: str,
    *,
    expected_runtime_environment_sha256: str | None = None,
    expected_forecast_sha256: str | None = None,
) -> None:
    plan = metadata.get("plan")
    if type(plan) is not dict:
        raise ValueError("receipt metadata is missing its frozen plan")
    plan_digest = _sha(metadata.get("plan_digest"), "plan_digest")
    if (
        plan_digest != reserve_plan_digest
        or hashlib.sha256(_canonical(plan)).hexdigest() != plan_digest
    ):
        raise ValueError("receipt plan digest differs from the frozen reserve plan")
    native_plan = plan.get("native_plan")
    if type(native_plan) is not dict:
        raise ValueError("receipt plan is missing its native arm")
    common, cell = native_plan.get("common"), native_plan.get("cell")
    if type(common) is not dict or type(cell) is not dict:
        raise ValueError("receipt plan does not contain the bound native arm")
    if (
        common.get("account_id") != replay.binding.account_id
        or common.get("dataset_id") != replay.content.dataset_id
        or common.get("scenario") != replay.binding.scenario
        or common.get("symbol_index") != replay.selected_symbol_index
        or common.get("range") != [replay.window.actor_start, replay.window.actor_stop]
        or cell.get("kind") != replay.binding.native_role
    ):
        raise ValueError("receipt plan arm differs from the selected frozen binding")
    source_clocks = [
        int(value.astype("int64"))
        for value in replay.content.arrays["timestamps"][
            replay.window.actor_start : replay.window.actor_stop + 1
        ]
    ]
    profile = replay.binding.effective_profile
    runtime_profile = profile.get("runtime_profile")
    common_profile_ok = (
        common.get("source_clocks_ns") == source_clocks
        and common.get("horizon_steps") == replay.window.actor_advances
        and common.get("initial_capital") == replay.initial_capital
        and common.get("economics_digest") == replay.policy.execution_policy_digest
        and type(runtime_profile) is dict
        and common.get("risk_digest") == runtime_profile.get("risk_digest")
    )
    runtime = cell.get("runtime")
    runtime_map = cast(dict[str, object], runtime) if type(runtime) is dict else {}
    runtime_recipe = runtime_map.get("recipe")
    actual_cost = runtime_map.get("cost_payload")
    observation = (
        runtime_recipe.get("observation") if type(runtime_recipe) is dict else None
    )
    action = runtime_recipe.get("action") if type(runtime_recipe) is dict else None
    if (
        not common_profile_ok
        or type(runtime) is not dict
        or type(runtime_recipe) is not dict
        or type(observation) is not dict
        or type(action) is not dict
        or type(runtime_profile) is not dict
        or runtime_recipe.get("runtime_profile") != runtime_profile
        or actual_cost != profile.get("cost_policy")
        or observation.get("episode_steps") != replay.window.actor_advances
        or action.get("mode")
        != ("residual" if replay.binding.native_role == "nonrl" else "direct")
        or type(runtime_map.get("source_context_digest")) is not str
    ):
        raise ValueError(
            "receipt runtime/cost/risk profile differs from frozen binding"
        )
    if expected_forecast_sha256 is not None and common.get(
        "forecast_context_digest"
    ) != _sha(expected_forecast_sha256, "expected forecast context digest"):
        raise ValueError("receipt plan forecast identity differs from Q's fixed bytes")
    _sha(common.get("source_digest"), "native source digest")
    _sha(common.get("implementation_digest"), "native implementation digest")
    runtime_environment_sha256 = _sha(
        common.get("runtime_environment_digest"), "native runtime digest"
    )
    if (
        expected_runtime_environment_sha256 is not None
        and runtime_environment_sha256
        != _sha(
            expected_runtime_environment_sha256,
            "expected H runtime environment digest",
        )
    ):
        raise ValueError("native runtime differs from the root-pinned H runtime")
    _sha(
        runtime_map.get("source_context_digest"), "native runtime source context digest"
    )
    if hashlib.sha256(_canonical(runtime_recipe)).hexdigest() != cell.get(
        "runtime_recipe_digest"
    ):
        raise ValueError("native runtime recipe differs from its digest")
    native_receipt = metadata.get("native_receipt")
    if (
        type(native_receipt) is not dict
        or native_receipt.get("status") != "completed"
        or native_receipt.get("reset_count") != 1
        or native_receipt.get("plan") != native_plan
        or native_receipt.get("common_digest")
        != hashlib.sha256(_canonical(common)).hexdigest()
        or native_receipt.get("cell_plan_digest")
        != hashlib.sha256(_canonical(cell)).hexdigest()
    ):
        raise ValueError("embedded native receipt differs from its frozen plan")
    if (
        metadata.get("status") != "completed"
        or metadata.get("settlement_complete") is not True
        or metadata.get("full_close_coverage") is not True
        or metadata.get("failure") is not None
        or metadata.get("final_active_orders") != []
    ):
        raise ValueError("native receipt metadata is incomplete or unsuccessful")


def _assert_runtime_pin_current(
    runtime_pin: FrozenHRuntimePin, source_root: Path
) -> str:
    if (
        type(runtime_pin) is not FrozenHRuntimePin
        or runtime_pin.source_root != source_root
        or runtime_pin.python_executable
        != source_root / ".venv" / "Scripts" / "python.exe"
    ):
        raise ValueError("native comparison requires the exact frozen-H runtime pin")
    probe = _json_object_bytes(
        runtime_pin.probe_facts_canonical, "frozen-H runtime probe facts"
    )
    project_pins = probe.get("project_inputs_sha256")
    if type(project_pins) is not dict or set(project_pins) != {
        "pyproject.toml",
        "uv.lock",
    }:
        raise ValueError("frozen-H runtime pin omits its project lock inputs")
    from tools.market_b4_oracle.outer_witness import _read_source_file

    for name, expected in project_pins.items():
        _, raw = _read_source_file(source_root / name)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("frozen-H runtime project inputs changed")
    _, executable = _read_source_file(runtime_pin.python_executable)
    if hashlib.sha256(executable).hexdigest() != runtime_pin.python_executable_sha256:
        raise ValueError("frozen-H Python executable changed")
    return _sha(
        runtime_pin.runtime_environment_sha256,
        "frozen-H runtime environment digest",
    )


def _consume_indexed_market_arm(
    *,
    output_index: NativeOutputIndex,
    receipt_name: str,
    forecast_name: str,
    content: VerifiedDatasetContent,
    policy: FrozenMarketPolicy,
    binding: FrozenMarketBinding,
    role: Literal["NONRL", "CASH"],
    reserve_plan_digest: str,
    artificial_inputs: object | None = None,
    static_inputs: StaticInputReadback | None = None,
    native_event_binding: ExactFillAttemptBinding | None = None,
    expected_runtime_environment_sha256: str | None = None,
    whole_index_recheck: bool,
    static_input_recheck: bool,
    reference_account_diagnostic: bool = False,
) -> _IndexedMarketArmConsumption:
    """Replay every indexed receipt row and publish only complete comparison scope.

    A non-artificial binding requires H/v3, H/v4, native source and Dataset
    pins for before/after readback. The 32+8 fixture is explicitly synthetic.
    """
    if type(output_index) is not NativeOutputIndex:
        raise ValueError("requires an exact immutable native output index")
    if type(content) is not VerifiedDatasetContent:
        raise ValueError("requires caller-verified Dataset content")
    if (
        type(binding) is not FrozenMarketBinding
        or type(policy) is not FrozenMarketPolicy
    ):
        raise ValueError("requires exact frozen binding and policy types")
    if type(reference_account_diagnostic) is not bool:
        raise ValueError("reference-account diagnostic opt-in must be an exact bool")
    if reference_account_diagnostic and not binding.comparison_only:
        raise ValueError("reference-account diagnostic requires comparison-only input")
    if role != {"nonrl": "NONRL", "cash": "CASH"}.get(binding.native_role):
        raise ValueError("selected role differs from the frozen arm")
    if (not binding.comparison_only and static_inputs is None) or (
        binding.comparison_only and static_inputs is not None
    ):
        raise ValueError("native pins and artificial bindings must remain distinct")
    if type(reserve_plan_digest) is not str:
        raise ValueError("reserve plan digest must be an explicit lowercase hash")
    reserve_plan_digest = _sha(reserve_plan_digest, "reserve_plan_digest")
    if expected_runtime_environment_sha256 is not None:
        expected_runtime_environment_sha256 = _sha(
            expected_runtime_environment_sha256,
            "expected H runtime environment digest",
        )
    binding.assert_content(content)
    expected_policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=content.dataset_id,
        profile=binding.effective_profile,
        risk=binding.protocol_document["policy"]["risk"],
    )
    if policy != expected_policy:
        raise ValueError("execution policy differs from the selected frozen arm")

    protocol = binding.protocol_document
    expected_artificial_forecast_sha256: str | None = None
    if protocol.get("schema") == "issue810_artificial_h_component_q_binding_v1":
        if (
            type(artificial_inputs) is not ArtificialComponentInputs
            or artificial_inputs.content is not content
            or binding.protocol_canonical
            != artificial_inputs.protocol_document_canonical
            or forecast_name != "forecasts/simple-return-stream.json"
        ):
            raise ValueError(
                "artificial comparison requires its exact Q-owned source and forecast"
            )
    elif artificial_inputs is not None:
        raise ValueError("Q artificial inputs cannot bind a different protocol")

    static_before: (
        tuple[
            DeclaredProtocolPair,
            ExactFillAttemptBinding,
            VerifiedDatasetContent,
        ]
        | None
    ) = None
    if static_inputs is not None and static_input_recheck:
        static_before = _readback_static_inputs(
            pins=static_inputs, binding=binding, content=content, policy=policy
        )
    if whole_index_recheck:
        output_index.recheck()
    if not binding.comparison_only:
        if static_before is not None:
            if (
                native_event_binding is not None
                and native_event_binding.digest != static_before[1].digest
            ):
                raise ValueError("native event binding differs from static readback")
            native_event_binding = static_before[1]
        if type(native_event_binding) is not ExactFillAttemptBinding:
            raise ValueError("native arm requires the paired exact-fill event binding")
        _validate_native_event_rows(
            output_index=output_index,
            receipt_name=receipt_name,
            binding=binding,
            exact_fill_binding=native_event_binding,
        )
    elif native_event_binding is not None:
        raise ValueError("comparison-only arms cannot carry native event bindings")
    if protocol.get("schema") == "issue810_artificial_h_component_q_binding_v1":
        expected_artificial_forecast_sha256 = verify_artificial_component_forecast(
            output_index=output_index,
            inputs=cast(ArtificialComponentInputs, artificial_inputs),
        )
    forecast_pin = output_index.pin(forecast_name)
    forecast_bytes = _read_indexed_bytes(
        output_index, forecast_name, _MAX_FORECAST_BYTES
    )
    if (
        expected_artificial_forecast_sha256 is not None
        and hashlib.sha256(forecast_bytes).hexdigest()
        != expected_artificial_forecast_sha256
    ):
        raise ValueError("artificial forecast changed after exact recipe validation")
    forecast_index = FrozenForecastIndex(
        forecast_bytes,
        pinned_bytes=forecast_pin.bytes,
        pinned_sha256=forecast_pin.sha256,
    )
    replay = FrozenMarketReplay(
        content=content,
        policy=policy,
        binding=binding,
        reserve_plan_digest=reserve_plan_digest,
    )
    reference_ledger = (
        _NativeReferenceLedger(content=content, policy=policy, binding=binding)
        if not binding.comparison_only or reference_account_diagnostic
        else None
    )
    configurations = protocol.get("configurations")
    configuration_rows = (
        [
            row
            for row in configurations
            if type(row) is dict and row.get("id") == binding.configuration_id
        ]
        if type(configurations) is list
        else []
    )
    dataset = protocol.get("dataset")
    if (
        len(configuration_rows) != 1
        or type(dataset) is not dict
        or type(dataset.get("feature_names")) is not list
        or type(dataset.get("fit_symbols")) is not list
    ):
        raise ValueError("frozen protocol is missing the selected forecast arm")
    configuration = cast(dict[str, object], configuration_rows[0])
    feature_names = tuple(cast(list[str], dataset["feature_names"]))
    prediction_roster = (
        dataset.get("prediction_symbols")
        if binding.comparison_only
        and protocol.get("schema") == "issue810_artificial_h_component_q_binding_v1"
        else dataset.get("fit_symbols")
    )
    if type(prediction_roster) is not list or any(
        type(value) is not str for value in prediction_roster
    ):
        raise ValueError("frozen protocol has no exact prediction-symbol roster")
    prediction_symbols = tuple(cast(list[str], prediction_roster))
    horizon_hours = configuration.get("horizon_hours")
    if (
        type(horizon_hours) is not int
        or horizon_hours * 3600 != binding.expected_horizon_seconds
    ):
        raise ValueError("forecast horizon differs from the selected protocol config")
    expected_vintage_count: int | None = None
    if not binding.comparison_only:
        actor = protocol.get("actor")
        budget = protocol.get("budget")
        if (
            type(actor) is not dict
            or type(actor.get("vintage_starts")) is not list
            or not actor["vintage_starts"]
            or type(budget) is not dict
            or budget.get("regression_features") != len(feature_names)
        ):
            raise ValueError("native protocol lacks a closed forecast fit roster")
        expected_vintage_count = len(cast(list[object], actor["vintage_starts"]))
    forecast_index.assert_bound_arm_scope(
        source=replay._decision_snapshot(replay.window.actor_start),
        expected_alpha=_number(configuration.get("ridge_alpha"), "ridge alpha"),
        expected_horizon_hours=horizon_hours,
        expected_feature_indices=tuple(range(len(feature_names))),
        expected_prediction_symbols=prediction_symbols,
        expected_vintage_count=expected_vintage_count,
    )

    actor_reader = output_index.receipt_rows(receipt_name, phase="actor")
    actor_seen = 0
    actor_fill_processing_indices: list[int] = []
    metadata: dict[str, object] | None = None
    for row in actor_reader:
        actor_seen += 1
        if actor_seen > replay.window.actor_advances:
            raise ValueError("actor receipt rows differ from bound window")
        processing = replay.current_index + 1
        native = _native_bar(
            row, phase="actor", replay=replay, processing_index=processing
        )
        events, observed_fills = _parsed_events(
            native, _event_context(replay, processing)
        )
        prior_terminal = set(replay._terminal)
        advance = replay.advance(
            forecast_index=forecast_index,
            role=role,
            observed_fills=observed_fills,
        )
        if advance.phase != "actor" or advance.processing_index != processing:
            raise ValueError("owned replay left actor phase before its bound row")
        if advance.owned_fills:
            actor_fill_processing_indices.append(processing)
        _compare_actor_identity(native, advance)
        _compare_event_effects(events, observed_fills, replay, advance)
        if not binding.comparison_only:
            _compare_native_fill_diagnostics(events, replay, advance)
        _compare_orders(native, replay, advance, prior_terminal)
        execution = cast(dict[str, object], native["execution"])
        _compare_book(
            native.get("book"),
            replay,
            processing_index=processing,
            interval_return=execution.get("interval_net_return"),
        )
        if reference_ledger is not None:
            reference_ledger.apply_advance(
                advance=advance,
                observed_fills=observed_fills,
                observed_account=replay.account_snapshot,
            )
    actor_scan = actor_reader.proof
    metadata = actor_reader.metadata
    if (
        actor_seen != replay.window.actor_advances
        or actor_scan.actor_rows != replay.window.actor_advances
        or actor_scan.reserve_rows != replay.window.reserve_advances
        or actor_scan.selected_rows != replay.window.actor_advances
    ):
        raise ValueError("actor receipt rows differ from bound window")

    reserve_reader = output_index.receipt_rows(receipt_name, phase="reserve")
    reserve_seen = 0
    reserve_fill_processing_indices: list[int] = []
    for row in reserve_reader:
        reserve_seen += 1
        if reserve_seen > replay.window.reserve_advances:
            raise ValueError("reserve receipt rows differ from bound window")
        processing = replay.current_index + 1
        native = _native_bar(
            row, phase="reserve", replay=replay, processing_index=processing
        )
        events, observed_fills = _parsed_events(
            native, _event_context(replay, processing)
        )
        prior_terminal = set(replay._terminal)
        advance = replay.advance(
            forecast_index=forecast_index,
            role=role,
            observed_fills=observed_fills,
        )
        if advance.phase != "reserve" or advance.processing_index != processing:
            raise ValueError("owned replay left reserve phase before its bound row")
        if advance.owned_fills:
            reserve_fill_processing_indices.append(processing)
        _compare_event_effects(events, observed_fills, replay, advance)
        if not binding.comparison_only:
            _compare_native_fill_diagnostics(events, replay, advance)
        if advance.actor_identity is not None:
            raise ValueError("reserve submissions differ from owned reserve identity")
        _compare_orders(native, replay, advance, prior_terminal)
        execution = cast(dict[str, object], native["execution"])
        _compare_book(
            native.get("book"),
            replay,
            processing_index=processing,
            interval_return=execution.get("interval_net_return"),
        )
        if reference_ledger is not None:
            reference_ledger.apply_advance(
                advance=advance,
                observed_fills=observed_fills,
                observed_account=replay.account_snapshot,
            )
    reserve_scan = reserve_reader.proof
    if reserve_reader.metadata != metadata:
        raise ValueError("receipt metadata changed between actor and reserve scans")
    if (
        reserve_seen != replay.window.reserve_advances
        or reserve_scan.actor_rows != replay.window.actor_advances
        or reserve_scan.reserve_rows != replay.window.reserve_advances
        or reserve_scan.selected_rows != replay.window.reserve_advances
    ):
        raise ValueError("reserve receipt rows differ from bound window")

    reference_ledger_coverage = (
        reference_ledger.assert_complete() if reference_ledger is not None else None
    )
    completion = replay.completion()
    if (
        completion["status"] != "comparison_window_consumed"
        or completion["processed_actor_advances"] != replay.window.actor_advances
        or completion["processed_reserve_advances"] != replay.window.reserve_advances
        or completion["terminal_flat"] is not True
        or completion["no_active_orders"] is not True
    ):
        raise ValueError("owned replay did not finish flat with no active order")
    _metadata_arm(
        metadata,
        replay,
        reserve_plan_digest,
        expected_runtime_environment_sha256=expected_runtime_environment_sha256,
        expected_forecast_sha256=forecast_index.metadata.pinned_sha256,
    )
    _compare_book(
        metadata.get("final_book"),
        replay,
        processing_index=replay.window.terminal_stop,
    )

    forecast_after = _read_indexed_bytes(
        output_index, forecast_name, _MAX_FORECAST_BYTES
    )
    if forecast_after != forecast_bytes:
        raise ValueError("indexed forecast bytes changed during row consumption")
    forecast_check = FrozenForecastIndex(
        forecast_after,
        pinned_bytes=forecast_pin.bytes,
        pinned_sha256=forecast_pin.sha256,
    )
    if forecast_check.metadata != forecast_index.metadata:
        raise ValueError("forecast pin or decoded roster changed during consumption")
    if static_inputs is not None and static_input_recheck:
        static_after = _readback_static_inputs(
            pins=static_inputs, binding=binding, content=content, policy=policy
        )
        if static_before is None:
            raise ValueError(
                "static input readback was not captured before consumption"
            )
        if (
            static_after[0] != static_before[0]
            or static_after[1].digest != static_before[1].digest
        ):
            raise ValueError(
                "static source/protocol readback changed during consumption"
            )
    if whole_index_recheck:
        output_index.recheck()
    comparison = IndexedMarketComparison(
        status=(
            "ARTIFICIAL_COMPARISON_COMPLETE"
            if binding.comparison_only
            else "PINNED_INPUT_COMPARISON_COMPLETE"
        ),
        binding_digest=hashlib.sha256(binding.protocol_canonical).hexdigest(),
        account_id=binding.account_id,
        scenario=binding.scenario,
        role=binding.native_role,
        actor_rows=actor_scan.actor_rows,
        reserve_rows=reserve_scan.reserve_rows,
        actor_scan=actor_scan,
        reserve_scan=reserve_scan,
        actor_fill_processing_indices=tuple(actor_fill_processing_indices),
        reserve_fill_processing_indices=tuple(reserve_fill_processing_indices),
        plan_digest=_sha(metadata.get("plan_digest"), "plan_digest"),
        forecast_sha256=forecast_index.metadata.pinned_sha256,
        runtime_environment_sha256=expected_runtime_environment_sha256,
        output_index_rechecked=whole_index_recheck,
        static_inputs_rechecked=static_inputs is not None and static_input_recheck,
    )
    return _IndexedMarketArmConsumption(
        comparison=comparison,
        reference_account_coverage=reference_ledger_coverage,
    )


def consume_indexed_market_arm(
    *,
    output_index: NativeOutputIndex,
    receipt_name: str,
    forecast_name: str,
    content: VerifiedDatasetContent,
    policy: FrozenMarketPolicy,
    binding: FrozenMarketBinding,
    role: Literal["NONRL", "CASH"],
    reserve_plan_digest: str,
    artificial_inputs: object | None = None,
    static_inputs: StaticInputReadback | None = None,
    expected_runtime_environment_sha256: str | None = None,
) -> IndexedMarketComparison:
    """Replay one complete arm with independent before/after source checks."""
    result = _consume_indexed_market_arm(
        output_index=output_index,
        receipt_name=receipt_name,
        forecast_name=forecast_name,
        content=content,
        policy=policy,
        binding=binding,
        role=role,
        reserve_plan_digest=reserve_plan_digest,
        artificial_inputs=artificial_inputs,
        static_inputs=static_inputs,
        expected_runtime_environment_sha256=expected_runtime_environment_sha256,
        whole_index_recheck=True,
        static_input_recheck=True,
    )
    return result.comparison


def consume_artificial_reference_account_diagnostic(
    *,
    output_index: object,
    inputs: object,
    role: object,
    reserve_plan_digest: object,
) -> ArtificialReferenceAccountDiagnostic:
    """Recheck one complete fixed artificial H arm and its account arithmetic.

    This detached comparison adapter consumes only the exact H output roster
    and Q-owned artificial recipe. It does not authenticate the process that
    produced those files or authorize financial, research, or live use.
    """
    if type(inputs) is not ArtificialComponentInputs:
        raise ValueError("requires exact artificial component inputs")
    if type(output_index) is not NativeOutputIndex:
        raise ValueError("requires an exact immutable native output index")
    if type(role) is not str or role not in {"NONRL", "CASH"}:
        raise ValueError("role must be the exact NONRL or CASH label")
    if type(reserve_plan_digest) is not str:
        raise ValueError("reserve plan digest must be an explicit lowercase hash")
    reserve_plan_digest = _sha(reserve_plan_digest, "reserve_plan_digest")

    _validate_fixed_artificial_component_inputs(cast(ArtificialComponentInputs, inputs))

    if output_index.files != _ARTIFICIAL_COMPONENT_OUTPUT_FILES:
        raise ValueError("H output roster differs from the fixed artificial arm")
    receipt_name = f"receipts/{role.lower()}.json.gz"
    forecast_name = "forecasts/simple-return-stream.json"
    binding, policy = artificial_component_binding(
        inputs, cast(Literal["NONRL", "CASH"], role)
    )
    if (
        binding.comparison_only is not True
        or binding.native_role != role.lower()
        or binding.account_id != "H24_RIDGE_A1_BTCUSDT"
        or binding.configuration_id != "H24_RIDGE_A1"
        or binding.scenario != "BASEL0"
        or binding.selected_symbol_index != 0
        or binding.expected_horizon_seconds != 24 * 3600
        or binding.window.actor_start != 2903
        or binding.window.actor_stop != 9430
        or binding.window.terminal_stop != 9502
    ):
        raise ValueError("derived artificial binding differs from its fixed profile")

    pins = tuple(
        (name, output_index.pin(name).bytes, output_index.pin(name).sha256)
        for name in _ARTIFICIAL_COMPONENT_OUTPUT_FILES
    )
    consumed = _consume_indexed_market_arm(
        output_index=output_index,
        receipt_name=receipt_name,
        forecast_name=forecast_name,
        content=inputs.content,
        policy=policy,
        binding=binding,
        role=cast(Literal["NONRL", "CASH"], role),
        reserve_plan_digest=reserve_plan_digest,
        artificial_inputs=inputs,
        whole_index_recheck=True,
        static_input_recheck=False,
        reference_account_diagnostic=True,
    )
    comparison = consumed.comparison
    coverage = consumed.reference_account_coverage
    if (
        coverage is None
        or comparison.status != "ARTIFICIAL_COMPARISON_COMPLETE"
        or not comparison.output_index_rechecked
        or (comparison.actor_rows, comparison.reserve_rows) != (6527, 72)
        or (coverage.actor_rows, coverage.reserve_rows) != (6527, 72)
        or coverage.fill_count
        < len(comparison.actor_fill_processing_indices)
        + len(comparison.reserve_fill_processing_indices)
    ):
        raise ValueError(
            "fixed artificial arm did not complete diagnostic row coverage"
        )
    return ArtificialReferenceAccountDiagnostic(
        schema="issue810_artificial_reference_account_diagnostic_v1",
        status="ARTIFICIAL_REFERENCE_ACCOUNT_ARITHMETIC_MATCH",
        scope="ARTIFICIAL_COMPARISON_ONLY",
        account_id=binding.account_id,
        scenario=binding.scenario,
        role=cast(Literal["NONRL", "CASH"], role),
        dataset_id=inputs.content.dataset_id,
        component_spec_sha256=inputs.spec_sha256,
        source_protocol_sha256=inputs.source_protocol_sha256,
        binding_sha256=comparison.binding_digest,
        actor_rows=coverage.actor_rows,
        reserve_rows=coverage.reserve_rows,
        fill_count=coverage.fill_count,
        output_manifest_bytes=output_index.manifest_pin.bytes,
        output_manifest_sha256=output_index.manifest_pin.sha256,
        output_file_pins=pins,
    )


def consume_indexed_market_matrix(
    *,
    output_index: NativeOutputIndex,
    protocol_pair: DeclaredProtocolPair,
    content: VerifiedDatasetContent,
    static_inputs: StaticInputReadback,
    runtime_pin: FrozenHRuntimePin | None = None,
) -> IndexedMarketMatrixComparison:
    """Consume the frozen 48 arms under one shared output-index boundary.

    The native pre-execution plan supplies each expected terminal-plan digest;
    its common account, scenario, role, clock, horizon, capital and runtime
    profile are independently checked against the H/v3 binding before rows are
    consumed. One whole-index recheck brackets the matrix. Each file still uses
    the index's exact bounded pin checks while it is read. This remains
    comparison evidence: an outer process witness and completed repository
    quality gates are separate requirements.
    """
    if type(output_index) is not NativeOutputIndex:
        raise ValueError("native matrix requires the exact immutable output index")
    if type(static_inputs) is not StaticInputReadback:
        raise ValueError(
            "native matrix requires all exact H/v3, V4, source and Dataset pins"
        )
    if type(runtime_pin) is not FrozenHRuntimePin:
        raise ValueError("native matrix requires the independent frozen-H runtime pin")
    expected_runtime_environment_sha256 = _assert_runtime_pin_current(
        runtime_pin, static_inputs.source_root
    )
    roster = derive_frozen_native_market_roster(
        protocol_pair=protocol_pair,
        content=content,
    )
    first = roster[0]
    static_before = _readback_static_inputs(
        pins=static_inputs,
        binding=first.binding,
        content=content,
        policy=first.policy,
    )
    output_index.recheck()

    expected_receipts = {arm.receipt_name for arm in roster}
    expected_forecasts = {arm.forecast_name for arm in roster}
    actual_receipts = {
        name for name in output_index.files if name.startswith("receipts/")
    }
    actual_forecasts = {
        name for name in output_index.files if name.startswith("forecasts/")
    }
    if actual_receipts != expected_receipts or actual_forecasts != expected_forecasts:
        raise ValueError(
            "native output files do not cover exactly the frozen arm roster"
        )
    plan_digests = _frozen_native_plan_digests(
        output_index=output_index,
        protocol_pair=protocol_pair,
        roster=roster,
        content=content,
        expected_runtime_environment_sha256=expected_runtime_environment_sha256,
    )

    comparisons: list[IndexedMarketComparison] = []
    for arm in roster:
        key = (arm.account_id, arm.scenario, arm.native_role)
        try:
            digest = plan_digests[key]
        except KeyError as error:
            raise ValueError("native terminal plan omitted a frozen arm") from error
        arm_consumption = _consume_indexed_market_arm(
            output_index=output_index,
            receipt_name=arm.receipt_name,
            forecast_name=arm.forecast_name,
            content=content,
            policy=arm.policy,
            binding=arm.binding,
            role=arm.role,
            reserve_plan_digest=digest,
            static_inputs=static_inputs,
            native_event_binding=static_before[1],
            expected_runtime_environment_sha256=expected_runtime_environment_sha256,
            whole_index_recheck=False,
            static_input_recheck=False,
        )
        comparison = arm_consumption.comparison
        if (
            comparison.actor_rows != 6527
            or comparison.reserve_rows != 72
            or comparison.output_index_rechecked
            or comparison.static_inputs_rechecked
        ):
            raise ValueError(
                "native arm comparison did not complete its exact row window"
            )
        comparisons.append(comparison)

    static_after = _readback_static_inputs(
        pins=static_inputs,
        binding=first.binding,
        content=content,
        policy=first.policy,
    )
    if (
        static_after[0] != static_before[0]
        or static_after[1].digest != static_before[1].digest
        or not _same_content(content, static_after[2])
    ):
        raise ValueError(
            "native matrix source, protocol or Dataset changed during comparison"
        )
    if (
        _assert_runtime_pin_current(runtime_pin, static_inputs.source_root)
        != expected_runtime_environment_sha256
    ):
        raise ValueError("frozen-H runtime pin changed during matrix comparison")
    output_index.recheck()
    return IndexedMarketMatrixComparison(
        status="PINNED_INPUT_MATRIX_COMPARISON_COMPLETE",
        arms=tuple(comparisons),
        output_index_rechecks=2,
        static_input_readbacks=2,
        runtime_environment_sha256=expected_runtime_environment_sha256,
    )


def _validate_root_observed_attempt(
    *,
    output_index: NativeOutputIndex,
    content: VerifiedDatasetContent,
    binding: FrozenMarketBinding,
    artificial_inputs: object | None = None,
    static_inputs: StaticInputReadback | None,
    root_witness: RootObservedAttempt | None,
    expected_start_pin: WitnessFilePin | None,
    expected_finish_pin: WitnessFilePin | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        type(root_witness) is not RootObservedAttempt
        or type(expected_start_pin) is not WitnessFilePin
        or type(expected_finish_pin) is not WitnessFilePin
    ):
        raise ValueError("root-observed start and finish pins are required")
    output_root = output_index._root.resolve()
    if any(
        path.resolve().is_relative_to(output_root)
        or output_root.is_relative_to(path.resolve())
        for path in (root_witness.start_path, root_witness.finish_path)
    ):
        raise ValueError(
            "root-owned witness files must be outside the child output root"
        )
    start, finish = root_witness.readback(
        expected_start_pin=expected_start_pin,
        expected_finish_pin=expected_finish_pin,
    )
    if start.get("output_root") != str(output_root) or finish.get("output_root") != str(
        output_root
    ):
        raise ValueError("root-observed output root differs from the indexed receipt")
    source_start = start.get("implementation_source")
    source_finish = finish.get("implementation_source")
    if type(source_start) is not dict or type(source_finish) is not dict:
        raise ValueError("root-observed implementation closure is missing")
    current_source = implementation_source_identity()
    for source in (source_start, source_finish):
        if (
            source.get("closure_sha256") != current_source["closure_sha256"]
            or source.get("runtime") != current_source["runtime"]
            or source.get("repository_head") != root_witness.repository_head
            or source.get("closure_sha256") != root_witness.implementation_source_sha256
        ):
            raise ValueError("root-observed comparison source/runtime identity changed")
    authority = start.get("input_authority")
    if type(authority) is not dict:
        raise ValueError("root-observed frozen input authority is missing")
    expected_binding_digest = hashlib.sha256(binding.protocol_canonical).hexdigest()
    if (
        authority.get("binding_digest") != expected_binding_digest
        or type(authority.get("dataset")) is not dict
        or authority["dataset"].get("dataset_id") != content.dataset_id
    ):
        raise ValueError("root-observed binding or Dataset differs from the consumer")
    if binding.comparison_only:
        artificial_protocol = binding.protocol_document.get("schema") == (
            "issue810_artificial_h_component_q_binding_v1"
        )
        expected_scope = (
            "ARTIFICIAL_H_COMPONENT_ONLY"
            if artificial_protocol
            else "ARTIFICIAL_COMPARISON_ONLY"
        )
        if artificial_protocol and (
            type(artificial_inputs) is not ArtificialComponentInputs
            or authority.get("component_spec_sha256") != artificial_inputs.spec_sha256
            or authority.get("outer_attempt_authenticated") is not False
            or authority.get("full_b4_authorized") is not False
        ):
            raise ValueError("root-observed artificial plan differs from Q's spec")
        if (
            static_inputs is not None
            or authority.get("scope") != expected_scope
            or authority.get("protocols") is not None
            or authority.get("native_source") is not None
        ):
            raise ValueError(
                "artificial outer witness acquired native source authority"
            )
    else:
        if (
            static_inputs is None
            or authority.get("scope") != "NATIVE_H3_V4_STATIC_PINS"
        ):
            raise ValueError("native outer witness requires distinct H/v3 and V4 pins")
        pair = binding.protocol_pair
        if pair is None:
            raise ValueError(
                "native binding has no validated H/v3 and V4 protocol pair"
            )
        expected_protocols = {
            "h3": {
                "bytes": pair.native_pin.raw_bytes,
                "sha256": pair.native_pin.sha256,
                "source_head": pair.native_pin.source_head,
            },
            "v4": {
                "bytes": pair.comparison_pin.raw_bytes,
                "sha256": pair.comparison_pin.sha256,
                "source_head": pair.comparison_pin.source_head,
            },
        }
        expected_source = {
            "head_sha": static_inputs.source_pin.head_sha,
            "snapshot_sha256": static_inputs.source_pin.snapshot_sha256,
        }
        expected_dataset = {
            "dataset_id": static_inputs.dataset_pin.dataset_id,
            "manifest_bytes": static_inputs.dataset_pin.manifest_bytes,
            "manifest_sha256": static_inputs.dataset_pin.manifest_sha256,
            "arrays_bytes": static_inputs.dataset_pin.arrays_bytes,
            "arrays_sha256": static_inputs.dataset_pin.arrays_sha256,
        }
        if (
            authority.get("protocols") != expected_protocols
            or authority.get("native_source") != expected_source
            or authority.get("dataset") != expected_dataset
        ):
            raise ValueError("root-observed H/v3, V4, source or Dataset pins differ")
    manifest = finish.get("output_manifest")
    if type(manifest) is not dict or manifest != {
        "bytes": output_index.manifest_pin.bytes,
        "sha256": output_index.manifest_pin.sha256,
    }:
        raise ValueError("root-observed output-manifest pin differs from the index")
    child = finish.get("child")
    if type(child) is not dict or child.get("exit_code") != 0:
        raise ValueError("root-observed supervisor did not exit successfully")
    return start, finish


def consume_root_observed_market_arm(
    *,
    output_index: NativeOutputIndex,
    receipt_name: str,
    forecast_name: str,
    content: VerifiedDatasetContent,
    policy: FrozenMarketPolicy,
    binding: FrozenMarketBinding,
    role: Literal["NONRL", "CASH"],
    reserve_plan_digest: str,
    artificial_inputs: object | None = None,
    static_inputs: StaticInputReadback | None = None,
    root_witness: RootObservedAttempt | None = None,
    expected_start_pin: WitnessFilePin | None = None,
    expected_finish_pin: WitnessFilePin | None = None,
) -> IndexedMarketComparison:
    """Consume one full arm only after external root start/finish pins match."""
    start, finish = _validate_root_observed_attempt(
        output_index=output_index,
        content=content,
        binding=binding,
        artificial_inputs=artificial_inputs,
        static_inputs=static_inputs,
        root_witness=root_witness,
        expected_start_pin=expected_start_pin,
        expected_finish_pin=expected_finish_pin,
    )
    comparison = consume_indexed_market_arm(
        output_index=output_index,
        receipt_name=receipt_name,
        forecast_name=forecast_name,
        content=content,
        policy=policy,
        binding=binding,
        role=role,
        reserve_plan_digest=reserve_plan_digest,
        artificial_inputs=artificial_inputs,
        static_inputs=static_inputs,
    )
    if (
        comparison.actor_rows != binding.window.actor_advances
        or comparison.reserve_rows
        != (binding.window.terminal_stop - binding.window.actor_stop)
    ):
        raise ValueError("root-observed consumer did not finish the bound arm window")
    assert expected_start_pin is not None and expected_finish_pin is not None
    return replace(
        comparison,
        root_attempt_observed=True,
        outer_start_witness_sha256=expected_start_pin.sha256,
        outer_finish_witness_sha256=expected_finish_pin.sha256,
    )


def consume_root_observed_market_matrix(
    *,
    output_index: NativeOutputIndex,
    protocol_pair: DeclaredProtocolPair,
    content: VerifiedDatasetContent,
    static_inputs: StaticInputReadback,
    runtime_pin: FrozenHRuntimePin,
    root_witness: RootObservedAttempt,
    expected_start_pin: WitnessFilePin,
    expected_finish_pin: WitnessFilePin,
    expected_input_authority_canonical: bytes,
) -> IndexedMarketMatrixComparison:
    """Consume all 48 arms only after one external native-H witness matches."""
    if (
        type(expected_input_authority_canonical) is not bytes
        or len(expected_input_authority_canonical) > 1024 * 1024
    ):
        raise ValueError("root-observed matrix requires a bounded plan authority")
    expected_authority = _json_object_bytes(
        expected_input_authority_canonical, "root-observed plan authority"
    )
    if _canonical(expected_authority) != expected_input_authority_canonical:
        raise ValueError("root-observed plan authority is not canonical")
    roster = derive_frozen_native_market_roster(
        protocol_pair=protocol_pair,
        content=content,
    )
    if len(roster) != 48 or any(arm.binding.comparison_only for arm in roster):
        raise ValueError("root-observed native matrix requires the exact 48-arm roster")
    start, finish = _validate_root_observed_attempt(
        output_index=output_index,
        content=content,
        binding=roster[0].binding,
        static_inputs=static_inputs,
        root_witness=root_witness,
        expected_start_pin=expected_start_pin,
        expected_finish_pin=expected_finish_pin,
    )
    if _canonical(start.get("input_authority")) != expected_input_authority_canonical:
        raise ValueError("root-observed start authority differs from the frozen plan")
    comparison = consume_indexed_market_matrix(
        output_index=output_index,
        protocol_pair=protocol_pair,
        content=content,
        static_inputs=static_inputs,
        runtime_pin=runtime_pin,
    )
    if (
        comparison.status != "PINNED_INPUT_MATRIX_COMPARISON_COMPLETE"
        or len(comparison.arms) != 48
        or comparison.output_index_rechecks != 2
        or comparison.static_input_readbacks != 2
        or comparison.outer_attempt_authenticated is not False
        or comparison.full_b4_authorized is not False
    ):
        raise ValueError("root-observed matrix did not close its exact comparison")
    return replace(
        comparison,
        arms=tuple(
            replace(
                arm,
                root_attempt_observed=True,
                outer_start_witness_sha256=expected_start_pin.sha256,
                outer_finish_witness_sha256=expected_finish_pin.sha256,
            )
            for arm in comparison.arms
        ),
        root_attempt_observed=True,
    )


__all__ = [
    "IndexedMarketComparison",
    "IndexedMarketMatrixComparison",
    "StaticInputReadback",
    "consume_indexed_market_arm",
    "consume_indexed_market_matrix",
    "consume_root_observed_market_arm",
    "consume_root_observed_market_matrix",
]
