"""Pure planning for the FS1 data acquisition (Issue #810): archive URLs and date ranges.

Nothing here touches the network, a clock, a file or a dataset, and no module of the
Binance integration is imported (that package imports a network client). The planner
rebuilds the Binance Vision archive layout from the pinned ``FS1_BUILD_POLICY``; a test
checks it against the maintained ``plan_binance_vision_cache`` and the acquisition
itself repeats that check before any request is sent.

The plan splits the FS1 span ``[start_time, end_time)`` at the registry's development
data end: months before it are reused read-only from the verified development archives
(never refetched), months from it on are the ones an authorized run fetches. The
exchange-rule snapshot is one extra REST request recorded in the seal record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from trade_rl.evaluation.forward_study.registry import Fs1Registry, payload_digest
from trade_rl.evaluation.forward_study.seal import AUTHORIZATION_PATTERN

VISION_ROOT: Final = "https://data.binance.vision/data"
CHECKSUM_SUFFIX: Final = ".CHECKSUM"
EXCHANGE_INFO_URL: Final = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# Digest of the FS1_BUILD_POLICY this tool was written against. A different policy is a
# different study (FS2); it needs a reviewed edit of this tool, never a silent pass.
IMPLEMENTED_BUILD_POLICY_DIGEST: Final = (
    "13d404e9a8b0a87d02cf7a90f9af2bbbad412d439dae42f730e4f416efd03a34"
)

KIND_KLINES: Final = "klines"
KIND_FUNDING: Final = "fundingRate"

# The only construction this tool can perform. Anything else in the policy is refused.
SUPPORTED_MARKET: Final = "usds_m"
SUPPORTED_BASE_INTERVAL: Final = "1h"
SUPPORTED_FEATURE_TIMEFRAMES: Final = ("4h", "1d")
SUPPORTED_GAP_POLICY: Final = "non_tradable_rows"
SUPPORTED_ECONOMICS_PROFILE: Final = "bot_overlay_only_assumptions_v1"
SUPPORTED_BAKED_IN_COSTS: Final = "zero; costs are applied by the replay"


# Output layout of an acquisition root (the root must not exist before the run).
CACHE_DIR: Final = "vision-cache"
EXCHANGE_INFO_DIR: Final = "exchange-info"
DATASET_DIR: Final = "dataset"
MANIFEST_NAME: Final = "fs1-dataset-seal-manifest.json"
RECEIPT_NAME: Final = "acquisition-receipt.json"
RECEIPT_SCHEMA: Final = "fs1_acquisition_receipt_v1"

# Non-archive entries of the manifest's ``source_manifest`` (archive entries are keyed by
# their Vision URL). They bind the receipt, and through it the builder commit and command.
SOURCE_KEY_RECEIPT: Final = "fs1:acquisition-receipt"
SOURCE_KEY_EXCHANGE_INFO: Final = "fs1:exchange-information-raw-payload"
SOURCE_KEY_DATASET_ARTIFACT: Final = "fs1:dataset-artifact-digest"
SOURCE_KEY_DEV_ROSTER: Final = "fs1:development-source-roster"

# Identity of the verified development Dataset whose archives FS1 reuses read-only
# (copies of ``dev_realdata_study.dataset_build.STUDY_*``; a test pins the equality
# without importing the builder here, because that module pulls in a network client).
DEV_DATASET_ID: Final = (
    "0b95547145a8f35d4deaaea99479ccf93244080bff52ef969203f4fa721de174"
)
DEV_ARTIFACT_DIGEST: Final = (
    "9f00e17c2f3de700f3a9ff08d41dea312ff61c143d85514b73e0cd84e03cd381"
)


def authorization_problems(authorization: str | None) -> tuple[str, ...]:
    """Reason codes for a missing or malformed Issue #810 comment URL (same pattern as
    the scoring evaluator)."""

    if not authorization:
        return ("authorization_missing",)
    if not AUTHORIZATION_PATTERN.match(authorization):
        return ("authorization_malformed",)
    return ()


@dataclass(frozen=True)
class BuildParams:
    """The dataset construction parameters, parsed from the pinned build policy."""

    market: str
    symbols: tuple[str, ...]
    base_interval: str
    feature_timeframes: tuple[str, ...]
    start: datetime
    end: datetime
    gap_policy: str
    economics_profile: str


def _utc(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _month_aligned(value: datetime) -> bool:
    return (
        value.day == 1
        and value.hour == value.minute == value.second == value.microsecond == 0
    )


def build_params_or_problems(
    registry: Fs1Registry,
    *,
    implemented_policy_digest: str = IMPLEMENTED_BUILD_POLICY_DIGEST,
) -> tuple[BuildParams | None, tuple[str, ...]]:
    """Parse ``registry.build_policy`` into build parameters, or list every mismatch.

    The policy must hash to the digest this tool was written against, and each field
    must be one of the few values the tool can build; the span must agree with the
    registry. Every deviation is a reason code; no field is silently defaulted.
    """

    policy: Mapping[str, object] = registry.build_policy
    problems: list[str] = []
    if payload_digest(dict(policy)) != implemented_policy_digest:
        problems.append("build_policy_digest_not_implemented")
    if policy.get("market") != SUPPORTED_MARKET:
        problems.append("build_policy_unsupported_market")
    symbols = policy.get("symbols")
    if not isinstance(symbols, list) or tuple(symbols) != registry.symbols:
        problems.append("build_policy_symbols_differ_from_registry")
    if policy.get("base_interval") != SUPPORTED_BASE_INTERVAL:
        problems.append("build_policy_unsupported_base_interval")
    timeframes = policy.get("feature_timeframes")
    if (
        not isinstance(timeframes, list)
        or tuple(timeframes) != SUPPORTED_FEATURE_TIMEFRAMES
    ):
        problems.append("build_policy_unsupported_feature_timeframes")
    if policy.get("gap_policy") != SUPPORTED_GAP_POLICY:
        problems.append("build_policy_unsupported_gap_policy")
    if policy.get("economics_profile") != SUPPORTED_ECONOMICS_PROFILE:
        problems.append("build_policy_unsupported_economics_profile")
    if policy.get("baked_in_fees_and_spread") != SUPPORTED_BAKED_IN_COSTS:
        problems.append("build_policy_unsupported_baked_in_costs")
    start = _utc(policy.get("start_time"))
    end = _utc(policy.get("end_time"))
    scoring_end = _utc(registry.scoring_end)
    development_end = _utc(registry.development_data_end)
    first_row_latest = _utc(registry.dataset_first_row_latest)
    if start is None or end is None or scoring_end is None or development_end is None:
        problems.append("build_policy_span_unparsable")
    elif first_row_latest is None:
        problems.append("build_policy_span_unparsable")
    else:
        if end != scoring_end:
            problems.append("build_policy_end_differs_from_scoring_end")
        if not _month_aligned(start) or not _month_aligned(end):
            problems.append("build_policy_span_not_month_aligned")
        if not _month_aligned(development_end):
            problems.append("development_data_end_not_month_aligned")
        if not start < development_end < end:
            problems.append("development_data_end_outside_span")
        if start > first_row_latest:
            problems.append("build_policy_start_after_first_row_latest")
    if problems or start is None or end is None:
        return None, tuple(dict.fromkeys(problems))
    assert isinstance(symbols, list) and isinstance(timeframes, list)
    return (
        BuildParams(
            market=SUPPORTED_MARKET,
            symbols=tuple(str(item) for item in symbols),
            base_interval=SUPPORTED_BASE_INTERVAL,
            feature_timeframes=tuple(str(item) for item in timeframes),
            start=start,
            end=end,
            gap_policy=SUPPORTED_GAP_POLICY,
            economics_profile=SUPPORTED_ECONOMICS_PROFILE,
        ),
        (),
    )


@dataclass(frozen=True)
class PlannedArchive:
    """One Binance Vision monthly archive and where it comes from."""

    url: str
    kind: str
    symbol: str
    period: str
    reuse_local: bool

    @property
    def relative_path(self) -> str:
        """Path below a Vision-mirroring archive root (the development layout)."""

        if self.kind == KIND_KLINES:
            return f"{self.kind}/{self.symbol}/1h/{self.symbol}-1h-{self.period}.zip"
        return f"{self.kind}/{self.symbol}/{self.symbol}-{self.kind}-{self.period}.zip"

    @property
    def checksum_url(self) -> str:
        return self.url + CHECKSUM_SUFFIX


def _months(start: datetime, end: datetime) -> list[datetime]:
    months: list[datetime] = []
    cursor = start
    while cursor < end:
        months.append(cursor)
        cursor = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12
            else cursor.replace(month=cursor.month + 1)
        )
    return months


def _url(kind: str, symbol: str, period: str, interval: str) -> str:
    stem = (
        f"{symbol}-{interval}-{period}"
        if kind == KIND_KLINES
        else (f"{symbol}-{kind}-{period}")
    )
    suffix = (
        f"{kind}/{symbol}/{interval}" if kind == KIND_KLINES else f"{kind}/{symbol}"
    )
    return f"{VISION_ROOT}/futures/um/monthly/{suffix}/{stem}.zip"


@dataclass(frozen=True)
class AcquisitionPlan:
    """Every archive of the FS1 span, in the order the maintained planner uses."""

    params: BuildParams
    development_end: datetime
    scoring_start: datetime
    archives: tuple[PlannedArchive, ...]

    @property
    def reused(self) -> tuple[PlannedArchive, ...]:
        return tuple(item for item in self.archives if item.reuse_local)

    @property
    def fetched(self) -> tuple[PlannedArchive, ...]:
        return tuple(item for item in self.archives if not item.reuse_local)

    @property
    def archive_urls(self) -> tuple[str, ...]:
        return tuple(item.url for item in self.archives)

    @property
    def fetch_urls(self) -> tuple[str, ...]:
        """Every request an authorized run sends to the archive host (zip + checksum),
        then the single REST request for the exchange rules."""

        urls: list[str] = []
        for item in self.fetched:
            urls.extend((item.url, item.checksum_url))
        return tuple(urls)

    def periods(self, items: tuple[PlannedArchive, ...]) -> tuple[str, ...]:
        return tuple(sorted({item.period for item in items}))


def plan_acquisition(
    params: BuildParams, *, development_end: datetime, scoring_start: datetime
) -> AcquisitionPlan:
    """Plan klines then funding archives for the span, ordered like the maintained
    ``plan_binance_vision_cache`` (symbol-major klines, then month-major funding)."""

    months = _months(params.start, params.end)
    archives: list[PlannedArchive] = []

    def planned(kind: str, symbol: str, month: datetime) -> PlannedArchive:
        period = month.strftime("%Y-%m")
        return PlannedArchive(
            url=_url(kind, symbol, period, params.base_interval),
            kind=kind,
            symbol=symbol,
            period=period,
            reuse_local=month < development_end,
        )

    for symbol in params.symbols:
        archives.extend(planned(KIND_KLINES, symbol, month) for month in months)
    for month in months:
        archives.extend(
            planned(KIND_FUNDING, symbol, month) for symbol in params.symbols
        )
    return AcquisitionPlan(
        params=params,
        development_end=development_end,
        scoring_start=scoring_start,
        archives=tuple(archives),
    )


def render_plan(plan: AcquisitionPlan, registry: Fs1Registry) -> tuple[str, ...]:
    """Human-readable plan: counts, date ranges and the exact URLs an authorized run
    would request. Nothing in the text has been fetched."""

    reused, fetched = plan.reused, plan.fetched
    reused_periods = plan.periods(reused)
    fetched_periods = plan.periods(fetched)
    lines = [
        "FS1 acquisition PLAN ONLY: nothing was fetched, read or written.",
        f"market {plan.params.market}, symbols {', '.join(plan.params.symbols)}, "
        f"base interval {plan.params.base_interval}, gap policy {plan.params.gap_policy}",
        f"span [{plan.params.start:%Y-%m-%d}, {plan.params.end:%Y-%m-%d}): "
        f"{len(plan.archives)} monthly archives (klines + fundingRate)",
        f"reuse read-only from the verified development archives (no fetch): "
        f"{len(reused)} archives, months {reused_periods[0]}..{reused_periods[-1]}, "
        f"before the development data end {plan.development_end:%Y-%m-%d}",
        f"fetch: {len(fetched)} archives plus {len(fetched)} {CHECKSUM_SUFFIX} files, "
        f"months {fetched_periods[0]}..{fetched_periods[-1]}, covering "
        f"[{plan.development_end:%Y-%m-%d}, {plan.params.end:%Y-%m-%d})",
        f"  warm-up only, never scored: [{registry.warmup_only_start[:10]}, "
        f"{registry.scoring_start[:10]})",
        f"  scored window: [{registry.scoring_start[:10]}, {registry.scoring_end[:10]})",
        f"exchange rules (one REST request): GET {EXCHANGE_INFO_URL}",
        "requests an authorized run would send:",
    ]
    lines.extend(f"  {url}" for url in plan.fetch_urls)
    lines.append(f"  {EXCHANGE_INFO_URL}")
    lines.append(
        "re-run with --allow-network (and the development inputs) to fetch; the "
        "settlement margin must have passed"
    )
    return tuple(lines)


__all__ = [
    "CACHE_DIR",
    "CHECKSUM_SUFFIX",
    "DATASET_DIR",
    "DEV_ARTIFACT_DIGEST",
    "DEV_DATASET_ID",
    "EXCHANGE_INFO_DIR",
    "EXCHANGE_INFO_URL",
    "IMPLEMENTED_BUILD_POLICY_DIGEST",
    "KIND_FUNDING",
    "KIND_KLINES",
    "MANIFEST_NAME",
    "RECEIPT_NAME",
    "RECEIPT_SCHEMA",
    "SOURCE_KEY_DATASET_ARTIFACT",
    "SOURCE_KEY_DEV_ROSTER",
    "SOURCE_KEY_EXCHANGE_INFO",
    "SOURCE_KEY_RECEIPT",
    "VISION_ROOT",
    "AcquisitionPlan",
    "BuildParams",
    "PlannedArchive",
    "authorization_problems",
    "build_params_or_problems",
    "plan_acquisition",
    "render_plan",
]
