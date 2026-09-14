"""Pre-result contract for the training-only aggTrades flow diagnostic."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "aggtrades_flow_predictive_diagnostic_protocol_v1"
_PROVIDER_HEAD_SHA = "dd51da97845f4d8fe69c34b1c4e4859358911daf"
_PROVIDER_PARSER_BLOB_SHA = "15cfe8f63716451fdb8c08ae84fba66ca754c55e"
_ARCHIVE_ROSTER_PROTOCOL_DIGEST = (
    "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
)
_ARCHIVE_ROSTER_SEAL_RUN_ID = 34761985041
_ARCHIVE_ROSTER_SEAL_ARTIFACT_ID = 10319386570
_ARCHIVE_ROSTER_SEAL_ARTIFACT_DIGEST = (
    "dbc6ef286abb9e4c8530089328fb64b2cfaa5e98715a82740439470a45942e57"
)
_ARCHIVE_ROSTER_FRESH_VERIFIER_RUN_ID = 34762075059
_ARCHIVE_EVIDENCE_RUN_ID = 34766830666
_ARCHIVE_EVIDENCE_ARTIFACT_ID = 10320428830
_ARCHIVE_EVIDENCE_ARTIFACT_DIGEST = (
    "810e792636a696a0bc11e22ab53a21f378aaa0bfaeab8c3411123e9c6d9a5c35"
)
_ARCHIVE_EVIDENCE_FRESH_VERIFIER_RUN_ID = 34767391268
_CANONICAL_DATASET_ID = (
    "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
)
_CANONICAL_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_DATASET_CONTAINER_RUN_ID = 34702660287
_DATASET_CONTAINER_ARTIFACT_ID = 10300479733
_DATASET_CONTAINER_ARTIFACT_DIGEST = (
    "60127cc2f24c7e8b3dcb5c6157ca49a5dd2f5b60c44f1020e4d540405e34e1f4"
)
_MARKET = "usds-m"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_DISCOVERY_START = datetime(2021, 1, 1, tzinfo=UTC)
_DISCOVERY_STOP_EXCLUSIVE = datetime(2023, 1, 1, tzinfo=UTC)
_SAMPLE_MONTH_DAYS = (1,)
_ARCHIVE_URL_TEMPLATE = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{date}.zip"
)
_BUYER_TAKER_WHEN_BUYER_IS_MAKER = False
_SELLER_TAKER_WHEN_BUYER_IS_MAKER = True
_PREDICTOR_AVAILABILITY = "completed_utc_hour_end"
_PREDICTOR_FORMULA = "buy_minus_sell_over_total_taker_notional"
_LABEL_FORMULA = "log_open_t_plus_2_over_open_t_plus_1"
_LABEL_HORIZON_HOURS = 1
_REGRESSION_INTERCEPT = True
_SLOPE_REDUCTION = "chronological_math_fsum_centered_ols"
_MIN_ACCEPTED_DAYS_PER_SYMBOL = 20
_MIN_ACCEPTED_DAYS_PER_YEAR = 10
_MIN_VALID_OBSERVATIONS = 480
_MIN_VALID_OBSERVATIONS_PER_YEAR = 220
_FULL_SAMPLE_POSITIVE_SYMBOLS_REQUIRED = 4
_YEAR_POSITIVE_SYMBOLS_REQUIRED = 3
_REPLACEMENT_DATES_ALLOWED = False
_POST_2022_DATA_ALLOWED = False
_STRATEGY_OR_PNL_INPUT_ALLOWED = False
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "provider_head_sha",
        "provider_parser_blob_sha",
        "archive_roster_protocol_digest",
        "archive_roster_seal_run_id",
        "archive_roster_seal_artifact_id",
        "archive_roster_seal_artifact_digest",
        "archive_roster_fresh_verifier_run_id",
        "archive_evidence_run_id",
        "archive_evidence_artifact_id",
        "archive_evidence_artifact_digest",
        "archive_evidence_fresh_verifier_run_id",
        "canonical_dataset_id",
        "canonical_dataset_artifact_digest",
        "dataset_container_run_id",
        "dataset_container_artifact_id",
        "dataset_container_artifact_digest",
        "market",
        "symbols",
        "discovery_start",
        "discovery_stop_exclusive",
        "sample_month_days",
        "archive_url_template",
        "buyer_taker_when_buyer_is_maker",
        "seller_taker_when_buyer_is_maker",
        "predictor_availability",
        "predictor_formula",
        "label_formula",
        "label_horizon_hours",
        "regression_intercept",
        "slope_reduction",
        "min_accepted_days_per_symbol",
        "min_accepted_days_per_year",
        "min_valid_observations",
        "min_valid_observations_per_year",
        "full_sample_positive_symbols_required",
        "year_positive_symbols_required",
        "replacement_dates_allowed",
        "post_2022_data_allowed",
        "strategy_or_pnl_input_allowed",
    }
)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_datetime(value: object, *, field: str) -> datetime:
    text = _text(value, field=field)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    return _utc(result, field=field)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(_text(item, field=field) for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return result


def _integers(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(_integer(item, field=field) for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique integers")
    return result


@dataclass(frozen=True, slots=True)
class AggTradesFlowPredictiveDiagnosticProtocol:
    """Immutable discovery-only order-flow predictive diagnostic contract."""

    provider_head_sha: str
    provider_parser_blob_sha: str
    archive_roster_protocol_digest: str
    archive_roster_seal_run_id: int
    archive_roster_seal_artifact_id: int
    archive_roster_seal_artifact_digest: str
    archive_roster_fresh_verifier_run_id: int
    archive_evidence_run_id: int
    archive_evidence_artifact_id: int
    archive_evidence_artifact_digest: str
    archive_evidence_fresh_verifier_run_id: int
    canonical_dataset_id: str
    canonical_dataset_artifact_digest: str
    dataset_container_run_id: int
    dataset_container_artifact_id: int
    dataset_container_artifact_digest: str
    market: str
    symbols: tuple[str, ...]
    discovery_start: datetime
    discovery_stop_exclusive: datetime
    sample_month_days: tuple[int, ...]
    archive_url_template: str
    buyer_taker_when_buyer_is_maker: bool
    seller_taker_when_buyer_is_maker: bool
    predictor_availability: str
    predictor_formula: str
    label_formula: str
    label_horizon_hours: int
    regression_intercept: bool
    slope_reduction: str
    min_accepted_days_per_symbol: int
    min_accepted_days_per_year: int
    min_valid_observations: int
    min_valid_observations_per_year: int
    full_sample_positive_symbols_required: int
    year_positive_symbols_required: int
    replacement_dates_allowed: bool
    post_2022_data_allowed: bool
    strategy_or_pnl_input_allowed: bool
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        actual = (
            _git_sha(self.provider_head_sha, field="provider_head_sha"),
            _git_sha(self.provider_parser_blob_sha, field="provider_parser_blob_sha"),
            _sha256(
                self.archive_roster_protocol_digest,
                field="archive_roster_protocol_digest",
            ),
            _integer(self.archive_roster_seal_run_id, field="archive_roster_seal_run_id"),
            _integer(
                self.archive_roster_seal_artifact_id,
                field="archive_roster_seal_artifact_id",
            ),
            _sha256(
                self.archive_roster_seal_artifact_digest,
                field="archive_roster_seal_artifact_digest",
            ),
            _integer(
                self.archive_roster_fresh_verifier_run_id,
                field="archive_roster_fresh_verifier_run_id",
            ),
            _integer(self.archive_evidence_run_id, field="archive_evidence_run_id"),
            _integer(
                self.archive_evidence_artifact_id,
                field="archive_evidence_artifact_id",
            ),
            _sha256(
                self.archive_evidence_artifact_digest,
                field="archive_evidence_artifact_digest",
            ),
            _integer(
                self.archive_evidence_fresh_verifier_run_id,
                field="archive_evidence_fresh_verifier_run_id",
            ),
            _sha256(self.canonical_dataset_id, field="canonical_dataset_id"),
            _sha256(
                self.canonical_dataset_artifact_digest,
                field="canonical_dataset_artifact_digest",
            ),
            _integer(self.dataset_container_run_id, field="dataset_container_run_id"),
            _integer(
                self.dataset_container_artifact_id,
                field="dataset_container_artifact_id",
            ),
            _sha256(
                self.dataset_container_artifact_digest,
                field="dataset_container_artifact_digest",
            ),
            _text(self.market, field="market"),
            tuple(self.symbols),
            _utc(self.discovery_start, field="discovery_start"),
            _utc(self.discovery_stop_exclusive, field="discovery_stop_exclusive"),
            tuple(self.sample_month_days),
            _text(self.archive_url_template, field="archive_url_template"),
            _boolean(
                self.buyer_taker_when_buyer_is_maker,
                field="buyer_taker_when_buyer_is_maker",
            ),
            _boolean(
                self.seller_taker_when_buyer_is_maker,
                field="seller_taker_when_buyer_is_maker",
            ),
            _text(self.predictor_availability, field="predictor_availability"),
            _text(self.predictor_formula, field="predictor_formula"),
            _text(self.label_formula, field="label_formula"),
            _integer(self.label_horizon_hours, field="label_horizon_hours"),
            _boolean(self.regression_intercept, field="regression_intercept"),
            _text(self.slope_reduction, field="slope_reduction"),
            _integer(
                self.min_accepted_days_per_symbol,
                field="min_accepted_days_per_symbol",
            ),
            _integer(
                self.min_accepted_days_per_year,
                field="min_accepted_days_per_year",
            ),
            _integer(self.min_valid_observations, field="min_valid_observations"),
            _integer(
                self.min_valid_observations_per_year,
                field="min_valid_observations_per_year",
            ),
            _integer(
                self.full_sample_positive_symbols_required,
                field="full_sample_positive_symbols_required",
            ),
            _integer(
                self.year_positive_symbols_required,
                field="year_positive_symbols_required",
            ),
            _boolean(self.replacement_dates_allowed, field="replacement_dates_allowed"),
            _boolean(self.post_2022_data_allowed, field="post_2022_data_allowed"),
            _boolean(
                self.strategy_or_pnl_input_allowed,
                field="strategy_or_pnl_input_allowed",
            ),
            _text(self.schema_version, field="schema_version"),
        )
        preregistered = (
            _PROVIDER_HEAD_SHA,
            _PROVIDER_PARSER_BLOB_SHA,
            _ARCHIVE_ROSTER_PROTOCOL_DIGEST,
            _ARCHIVE_ROSTER_SEAL_RUN_ID,
            _ARCHIVE_ROSTER_SEAL_ARTIFACT_ID,
            _ARCHIVE_ROSTER_SEAL_ARTIFACT_DIGEST,
            _ARCHIVE_ROSTER_FRESH_VERIFIER_RUN_ID,
            _ARCHIVE_EVIDENCE_RUN_ID,
            _ARCHIVE_EVIDENCE_ARTIFACT_ID,
            _ARCHIVE_EVIDENCE_ARTIFACT_DIGEST,
            _ARCHIVE_EVIDENCE_FRESH_VERIFIER_RUN_ID,
            _CANONICAL_DATASET_ID,
            _CANONICAL_DATASET_ARTIFACT_DIGEST,
            _DATASET_CONTAINER_RUN_ID,
            _DATASET_CONTAINER_ARTIFACT_ID,
            _DATASET_CONTAINER_ARTIFACT_DIGEST,
            _MARKET,
            _SYMBOLS,
            _DISCOVERY_START,
            _DISCOVERY_STOP_EXCLUSIVE,
            _SAMPLE_MONTH_DAYS,
            _ARCHIVE_URL_TEMPLATE,
            _BUYER_TAKER_WHEN_BUYER_IS_MAKER,
            _SELLER_TAKER_WHEN_BUYER_IS_MAKER,
            _PREDICTOR_AVAILABILITY,
            _PREDICTOR_FORMULA,
            _LABEL_FORMULA,
            _LABEL_HORIZON_HOURS,
            _REGRESSION_INTERCEPT,
            _SLOPE_REDUCTION,
            _MIN_ACCEPTED_DAYS_PER_SYMBOL,
            _MIN_ACCEPTED_DAYS_PER_YEAR,
            _MIN_VALID_OBSERVATIONS,
            _MIN_VALID_OBSERVATIONS_PER_YEAR,
            _FULL_SAMPLE_POSITIVE_SYMBOLS_REQUIRED,
            _YEAR_POSITIVE_SYMBOLS_REQUIRED,
            _REPLACEMENT_DATES_ALLOWED,
            _POST_2022_DATA_ALLOWED,
            _STRATEGY_OR_PNL_INPUT_ALLOWED,
            _SCHEMA_VERSION,
        )
        if actual != preregistered:
            raise ValueError(
                "aggTrades flow diagnostic fields differ from preregistered protocol"
            )

    @property
    def planned_days(self) -> tuple[datetime, ...]:
        result: list[datetime] = []
        cursor = self.discovery_start.replace(day=1)
        while cursor < self.discovery_stop_exclusive:
            for day in self.sample_month_days:
                candidate = cursor.replace(day=day)
                if self.discovery_start <= candidate < self.discovery_stop_exclusive:
                    result.append(candidate)
            cursor = _next_month(cursor)
        return tuple(result)

    @property
    def planned_urls(self) -> tuple[str, ...]:
        return tuple(
            self.archive_url_template.format(
                symbol=symbol,
                date=day.strftime("%Y-%m-%d"),
            )
            for symbol in self.symbols
            for day in self.planned_days
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_head_sha": self.provider_head_sha,
            "provider_parser_blob_sha": self.provider_parser_blob_sha,
            "archive_roster_protocol_digest": self.archive_roster_protocol_digest,
            "archive_roster_seal_run_id": self.archive_roster_seal_run_id,
            "archive_roster_seal_artifact_id": self.archive_roster_seal_artifact_id,
            "archive_roster_seal_artifact_digest": (
                self.archive_roster_seal_artifact_digest
            ),
            "archive_roster_fresh_verifier_run_id": (
                self.archive_roster_fresh_verifier_run_id
            ),
            "archive_evidence_run_id": self.archive_evidence_run_id,
            "archive_evidence_artifact_id": self.archive_evidence_artifact_id,
            "archive_evidence_artifact_digest": self.archive_evidence_artifact_digest,
            "archive_evidence_fresh_verifier_run_id": (
                self.archive_evidence_fresh_verifier_run_id
            ),
            "canonical_dataset_id": self.canonical_dataset_id,
            "canonical_dataset_artifact_digest": self.canonical_dataset_artifact_digest,
            "dataset_container_run_id": self.dataset_container_run_id,
            "dataset_container_artifact_id": self.dataset_container_artifact_id,
            "dataset_container_artifact_digest": self.dataset_container_artifact_digest,
            "market": self.market,
            "symbols": list(self.symbols),
            "discovery_start": _iso(self.discovery_start),
            "discovery_stop_exclusive": _iso(self.discovery_stop_exclusive),
            "sample_month_days": list(self.sample_month_days),
            "archive_url_template": self.archive_url_template,
            "buyer_taker_when_buyer_is_maker": self.buyer_taker_when_buyer_is_maker,
            "seller_taker_when_buyer_is_maker": self.seller_taker_when_buyer_is_maker,
            "predictor_availability": self.predictor_availability,
            "predictor_formula": self.predictor_formula,
            "label_formula": self.label_formula,
            "label_horizon_hours": self.label_horizon_hours,
            "regression_intercept": self.regression_intercept,
            "slope_reduction": self.slope_reduction,
            "min_accepted_days_per_symbol": self.min_accepted_days_per_symbol,
            "min_accepted_days_per_year": self.min_accepted_days_per_year,
            "min_valid_observations": self.min_valid_observations,
            "min_valid_observations_per_year": self.min_valid_observations_per_year,
            "full_sample_positive_symbols_required": (
                self.full_sample_positive_symbols_required
            ),
            "year_positive_symbols_required": self.year_positive_symbols_required,
            "replacement_dates_allowed": self.replacement_dates_allowed,
            "post_2022_data_allowed": self.post_2022_data_allowed,
            "strategy_or_pnl_input_allowed": self.strategy_or_pnl_input_allowed,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_aggtrades_flow_predictive_diagnostic_protocol() -> (
    AggTradesFlowPredictiveDiagnosticProtocol
):
    return AggTradesFlowPredictiveDiagnosticProtocol(
        provider_head_sha=_PROVIDER_HEAD_SHA,
        provider_parser_blob_sha=_PROVIDER_PARSER_BLOB_SHA,
        archive_roster_protocol_digest=_ARCHIVE_ROSTER_PROTOCOL_DIGEST,
        archive_roster_seal_run_id=_ARCHIVE_ROSTER_SEAL_RUN_ID,
        archive_roster_seal_artifact_id=_ARCHIVE_ROSTER_SEAL_ARTIFACT_ID,
        archive_roster_seal_artifact_digest=_ARCHIVE_ROSTER_SEAL_ARTIFACT_DIGEST,
        archive_roster_fresh_verifier_run_id=_ARCHIVE_ROSTER_FRESH_VERIFIER_RUN_ID,
        archive_evidence_run_id=_ARCHIVE_EVIDENCE_RUN_ID,
        archive_evidence_artifact_id=_ARCHIVE_EVIDENCE_ARTIFACT_ID,
        archive_evidence_artifact_digest=_ARCHIVE_EVIDENCE_ARTIFACT_DIGEST,
        archive_evidence_fresh_verifier_run_id=_ARCHIVE_EVIDENCE_FRESH_VERIFIER_RUN_ID,
        canonical_dataset_id=_CANONICAL_DATASET_ID,
        canonical_dataset_artifact_digest=_CANONICAL_DATASET_ARTIFACT_DIGEST,
        dataset_container_run_id=_DATASET_CONTAINER_RUN_ID,
        dataset_container_artifact_id=_DATASET_CONTAINER_ARTIFACT_ID,
        dataset_container_artifact_digest=_DATASET_CONTAINER_ARTIFACT_DIGEST,
        market=_MARKET,
        symbols=_SYMBOLS,
        discovery_start=_DISCOVERY_START,
        discovery_stop_exclusive=_DISCOVERY_STOP_EXCLUSIVE,
        sample_month_days=_SAMPLE_MONTH_DAYS,
        archive_url_template=_ARCHIVE_URL_TEMPLATE,
        buyer_taker_when_buyer_is_maker=_BUYER_TAKER_WHEN_BUYER_IS_MAKER,
        seller_taker_when_buyer_is_maker=_SELLER_TAKER_WHEN_BUYER_IS_MAKER,
        predictor_availability=_PREDICTOR_AVAILABILITY,
        predictor_formula=_PREDICTOR_FORMULA,
        label_formula=_LABEL_FORMULA,
        label_horizon_hours=_LABEL_HORIZON_HOURS,
        regression_intercept=_REGRESSION_INTERCEPT,
        slope_reduction=_SLOPE_REDUCTION,
        min_accepted_days_per_symbol=_MIN_ACCEPTED_DAYS_PER_SYMBOL,
        min_accepted_days_per_year=_MIN_ACCEPTED_DAYS_PER_YEAR,
        min_valid_observations=_MIN_VALID_OBSERVATIONS,
        min_valid_observations_per_year=_MIN_VALID_OBSERVATIONS_PER_YEAR,
        full_sample_positive_symbols_required=_FULL_SAMPLE_POSITIVE_SYMBOLS_REQUIRED,
        year_positive_symbols_required=_YEAR_POSITIVE_SYMBOLS_REQUIRED,
        replacement_dates_allowed=_REPLACEMENT_DATES_ALLOWED,
        post_2022_data_allowed=_POST_2022_DATA_ALLOWED,
        strategy_or_pnl_input_allowed=_STRATEGY_OR_PNL_INPUT_ALLOWED,
    )


def _finite_betas(values: tuple[float, ...], *, field: str) -> tuple[float, ...]:
    if len(values) != len(_SYMBOLS):
        raise ValueError(f"{field} must align with the five-symbol roster")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain only finite values")
    return result


def flow_screen_status(
    *,
    full_sample_betas: tuple[float, ...],
    year_2021_betas: tuple[float, ...],
    year_2022_betas: tuple[float, ...],
) -> str:
    """Apply the frozen sign-only discovery screen after all validity gates pass."""

    full = _finite_betas(full_sample_betas, field="full_sample_betas")
    y2021 = _finite_betas(year_2021_betas, field="year_2021_betas")
    y2022 = _finite_betas(year_2022_betas, field="year_2022_betas")
    if (
        sum(value > 0.0 for value in full) >= _FULL_SAMPLE_POSITIVE_SYMBOLS_REQUIRED
        and sum(value > 0.0 for value in y2021) >= _YEAR_POSITIVE_SYMBOLS_REQUIRED
        and sum(value > 0.0 for value in y2022) >= _YEAR_POSITIVE_SYMBOLS_REQUIRED
    ):
        return "PASS_FLOW_SCREEN"
    return "NO_STABLE_FLOW_SIGNAL"


def load_aggtrades_flow_predictive_diagnostic_protocol(
    path: str | Path,
) -> AggTradesFlowPredictiveDiagnosticProtocol:
    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number is forbidden: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load aggTrades flow diagnostic protocol: {source}"
        ) from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("aggTrades flow diagnostic protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    keys = set(payload)
    missing = sorted(_PAYLOAD_FIELDS - keys)
    unknown = sorted(keys - _PAYLOAD_FIELDS)
    if missing or unknown:
        raise ValueError(
            "aggTrades flow diagnostic protocol keys differ from contract: "
            f"missing={missing}, unknown={unknown}"
        )
    return AggTradesFlowPredictiveDiagnosticProtocol(
        schema_version=_text(payload["schema_version"], field="schema_version"),
        provider_head_sha=_git_sha(payload["provider_head_sha"], field="provider_head_sha"),
        provider_parser_blob_sha=_git_sha(
            payload["provider_parser_blob_sha"], field="provider_parser_blob_sha"
        ),
        archive_roster_protocol_digest=_sha256(
            payload["archive_roster_protocol_digest"],
            field="archive_roster_protocol_digest",
        ),
        archive_roster_seal_run_id=_integer(
            payload["archive_roster_seal_run_id"], field="archive_roster_seal_run_id"
        ),
        archive_roster_seal_artifact_id=_integer(
            payload["archive_roster_seal_artifact_id"],
            field="archive_roster_seal_artifact_id",
        ),
        archive_roster_seal_artifact_digest=_sha256(
            payload["archive_roster_seal_artifact_digest"],
            field="archive_roster_seal_artifact_digest",
        ),
        archive_roster_fresh_verifier_run_id=_integer(
            payload["archive_roster_fresh_verifier_run_id"],
            field="archive_roster_fresh_verifier_run_id",
        ),
        archive_evidence_run_id=_integer(
            payload["archive_evidence_run_id"], field="archive_evidence_run_id"
        ),
        archive_evidence_artifact_id=_integer(
            payload["archive_evidence_artifact_id"],
            field="archive_evidence_artifact_id",
        ),
        archive_evidence_artifact_digest=_sha256(
            payload["archive_evidence_artifact_digest"],
            field="archive_evidence_artifact_digest",
        ),
        archive_evidence_fresh_verifier_run_id=_integer(
            payload["archive_evidence_fresh_verifier_run_id"],
            field="archive_evidence_fresh_verifier_run_id",
        ),
        canonical_dataset_id=_sha256(
            payload["canonical_dataset_id"], field="canonical_dataset_id"
        ),
        canonical_dataset_artifact_digest=_sha256(
            payload["canonical_dataset_artifact_digest"],
            field="canonical_dataset_artifact_digest",
        ),
        dataset_container_run_id=_integer(
            payload["dataset_container_run_id"], field="dataset_container_run_id"
        ),
        dataset_container_artifact_id=_integer(
            payload["dataset_container_artifact_id"],
            field="dataset_container_artifact_id",
        ),
        dataset_container_artifact_digest=_sha256(
            payload["dataset_container_artifact_digest"],
            field="dataset_container_artifact_digest",
        ),
        market=_text(payload["market"], field="market"),
        symbols=_strings(payload["symbols"], field="symbols"),
        discovery_start=_parse_datetime(
            payload["discovery_start"], field="discovery_start"
        ),
        discovery_stop_exclusive=_parse_datetime(
            payload["discovery_stop_exclusive"], field="discovery_stop_exclusive"
        ),
        sample_month_days=_integers(
            payload["sample_month_days"], field="sample_month_days"
        ),
        archive_url_template=_text(
            payload["archive_url_template"], field="archive_url_template"
        ),
        buyer_taker_when_buyer_is_maker=_boolean(
            payload["buyer_taker_when_buyer_is_maker"],
            field="buyer_taker_when_buyer_is_maker",
        ),
        seller_taker_when_buyer_is_maker=_boolean(
            payload["seller_taker_when_buyer_is_maker"],
            field="seller_taker_when_buyer_is_maker",
        ),
        predictor_availability=_text(
            payload["predictor_availability"], field="predictor_availability"
        ),
        predictor_formula=_text(payload["predictor_formula"], field="predictor_formula"),
        label_formula=_text(payload["label_formula"], field="label_formula"),
        label_horizon_hours=_integer(
            payload["label_horizon_hours"], field="label_horizon_hours"
        ),
        regression_intercept=_boolean(
            payload["regression_intercept"], field="regression_intercept"
        ),
        slope_reduction=_text(payload["slope_reduction"], field="slope_reduction"),
        min_accepted_days_per_symbol=_integer(
            payload["min_accepted_days_per_symbol"],
            field="min_accepted_days_per_symbol",
        ),
        min_accepted_days_per_year=_integer(
            payload["min_accepted_days_per_year"], field="min_accepted_days_per_year"
        ),
        min_valid_observations=_integer(
            payload["min_valid_observations"], field="min_valid_observations"
        ),
        min_valid_observations_per_year=_integer(
            payload["min_valid_observations_per_year"],
            field="min_valid_observations_per_year",
        ),
        full_sample_positive_symbols_required=_integer(
            payload["full_sample_positive_symbols_required"],
            field="full_sample_positive_symbols_required",
        ),
        year_positive_symbols_required=_integer(
            payload["year_positive_symbols_required"],
            field="year_positive_symbols_required",
        ),
        replacement_dates_allowed=_boolean(
            payload["replacement_dates_allowed"], field="replacement_dates_allowed"
        ),
        post_2022_data_allowed=_boolean(
            payload["post_2022_data_allowed"], field="post_2022_data_allowed"
        ),
        strategy_or_pnl_input_allowed=_boolean(
            payload["strategy_or_pnl_input_allowed"],
            field="strategy_or_pnl_input_allowed",
        ),
    )


__all__ = [
    "AggTradesFlowPredictiveDiagnosticProtocol",
    "canonical_aggtrades_flow_predictive_diagnostic_protocol",
    "flow_screen_status",
    "load_aggtrades_flow_predictive_diagnostic_protocol",
]
