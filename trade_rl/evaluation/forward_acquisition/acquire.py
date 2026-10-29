"""Authorized FS1 data acquisition, offline dataset build and manifest (Issue #810).

:func:`run_acquire` is the single entrypoint that turns an Issue #810 data-fetch
authorization into the FS1 dataset, its seal manifest and an acquisition receipt. It
scores nothing and never reads a price, return, feature or funding value for a decision:
the only dataset inspection is row presence and timestamps (the gap/regularity guard the
scoring evaluator repeats later).

Behaviour by mode:

* **Plan only** (no ``allow_network``): checks the authorization pattern, the unused
  output root and the pinned build policy, then returns the exact archive URLs and date
  ranges an authorized run would request. No network client is imported, no file is
  read or written.
* **Authorized run** (``allow_network``): additionally requires the settlement margin to
  have passed (a manifest fetched earlier would be refused by the scoring evaluator), a
  clean trade_rl checkout (the builder commit is provenance), and the verified
  development inputs. Every refusal that can be known before the first request is
  collected and raised together; nothing is created in that case.

The authorized run then (1) copies the development archives read-only into a fresh
transport cache after proving them against the verified development Dataset identity,
(2) fetches only the archives from the development data end onwards, plus their official
``.CHECKSUM`` files, and the exchange rules, (3) refuses on any missing or unexpected
archive or checksum mismatch, (4) builds the Dataset offline under ``FS1_BUILD_POLICY``,
(5) refuses on a gap the policy does not admit, (6) publishes the Dataset and writes the
receipt and the seal manifest. The manifest is written last; a refusal after step 2
leaves a partial root with no manifest, which must be inspected and removed by hand
(a new run needs a new root).

The public entrypoint takes no registry, digest, clock, code-identity or transport
argument; those exist only in the private factory :func:`_run_acquire`, which tests
replace.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.artifacts.publication import (
    load_published_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forward_acquisition.plan import (
    CACHE_DIR,
    DATASET_DIR,
    DEV_ARTIFACT_DIGEST,
    DEV_DATASET_ID,
    EXCHANGE_INFO_DIR,
    IMPLEMENTED_BUILD_POLICY_DIGEST,
    MANIFEST_NAME,
    RECEIPT_NAME,
    RECEIPT_SCHEMA,
    SOURCE_KEY_DATASET_ARTIFACT,
    SOURCE_KEY_DEV_ROSTER,
    SOURCE_KEY_EXCHANGE_INFO,
    SOURCE_KEY_RECEIPT,
    AcquisitionPlan,
    BuildParams,
    authorization_problems,
    build_params_or_problems,
    plan_acquisition,
    render_plan,
)
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    Fs1Registry,
)
from trade_rl.evaluation.forward_study.seal import (
    CodeIdentity,
    current_code_identity,
    manifest_bytes_from_dataset,
    parse_manifest,
)

Emit = Callable[[str], None]
_NS_PER_HOUR: Final = 3_600 * 1_000_000_000
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_DOWNLOADER_REUSE: Final = (
    "trade-rl/fs1-development-archive-reuse (local copy, no network)"
)


class AcquisitionRefusal(RuntimeError):
    """The acquisition will not (continue to) run; ``reasons`` lists every cause."""

    def __init__(self, reasons: tuple[str, ...], detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__("FS1 acquisition refused: " + ", ".join(reasons) + suffix)
        self.reasons = reasons
        self.detail = detail


@dataclass(frozen=True)
class DevReference:
    """The pinned identity of the verified development Dataset."""

    dataset_id: str
    artifact_digest: str


@dataclass(frozen=True)
class AcquisitionResult:
    """What a call produced; ``planned_only`` means nothing at all was fetched/written."""

    planned_only: bool
    plan: AcquisitionPlan
    plan_lines: tuple[str, ...] = ()
    output_root: Path | None = None
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    receipt_path: Path | None = None
    dataset_path: Path | None = None
    dataset_id: str | None = None
    fetched_utc: str | None = None


@dataclass(frozen=True)
class _Environment:
    """Everything the call takes from outside: private and test-only."""

    registry: Fs1Registry
    frozen_params_digest: str
    implemented_policy_digest: str
    dev_reference: DevReference
    clock: Callable[[], datetime]
    code_identity: Callable[[], CodeIdentity]
    network_transport: Callable[[Path], Any]


def _real_transport(cache_root: Path) -> Any:
    """The only place a network client is created (imported lazily, on demand)."""

    from trade_rl.integrations.binance import BinancePublicTransport

    return BinancePublicTransport(cache_root=cache_root, allow_network=True)


def _environment() -> _Environment:
    """Production wiring: the frozen registry, the real wall clock, git and transport."""

    return _Environment(
        registry=FS1_REGISTRY,
        frozen_params_digest=FROZEN_PARAMS_DIGEST,
        implemented_policy_digest=IMPLEMENTED_BUILD_POLICY_DIGEST,
        dev_reference=DevReference(DEV_DATASET_ID, DEV_ARTIFACT_DIGEST),
        clock=lambda: datetime.now(UTC),
        code_identity=current_code_identity,
        network_transport=_real_transport,
    )


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def settlement_instant(registry: Fs1Registry) -> datetime:
    """Scoring end plus the settlement margin: the earliest instant a manifest's fetch
    time may have (the scoring evaluator refuses a fetch before it)."""

    end = datetime.fromisoformat(registry.scoring_end).replace(tzinfo=UTC)
    return end + timedelta(hours=registry.settlement_margin_hours)


def parse_checksum(text: str, *, filename: str) -> str | None:
    """The SHA-256 of an official ``.CHECKSUM`` file (``<hex>  <file name>``), or None
    when the file does not have exactly that shape for ``filename``."""

    tokens = text.split()
    if len(tokens) != 2 or not _SHA256.match(tokens[0]):
        return None
    if tokens[1].lstrip("*") != filename:
        return None
    return tokens[0]


# ---------------------------------------------------------------------------
# Development inputs: reuse by identity, never refetch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DevReuse:
    dataset_id: str
    artifact_digest: str
    roster_digest: str
    roster: dict[str, dict[str, Any]]


def _dev_reuse_or_problems(
    env: _Environment,
    plan: AcquisitionPlan,
    *,
    dev_archive_root: Path | None,
    dev_source_evidence: Path | None,
    dev_dataset: Path | None,
) -> tuple[_DevReuse | None, tuple[str, ...]]:
    """Prove the local development archives belong to the verified development Dataset.

    The chain is: published artifact digest and Dataset id equal the pinned development
    identity and the Dataset's own content identity verifies; the identity payload binds
    ``raw_source_roster_digest``; the evidence file's roster hashes to it and lists
    exactly the planned development archives; every archive file equals its roster
    SHA-256 and size. Read-only: nothing under the three inputs is modified.
    """

    if dev_archive_root is None or dev_source_evidence is None or dev_dataset is None:
        return None, ("dev_inputs_missing",)
    problems: list[str] = []
    bound_digest: object = None
    dataset_id = artifact_digest = ""
    try:
        artifact, dataset = load_published_market_dataset_artifact(dev_dataset)
    except Exception:
        problems.append("dev_dataset_unreadable")
    else:
        dataset_id, artifact_digest = dataset.dataset_id, artifact.artifact_digest
        if not dataset.identity_verified:
            problems.append("dev_dataset_identity_unverified")
        if dataset.dataset_id != env.dev_reference.dataset_id:
            problems.append("dev_dataset_id_mismatch")
        if artifact.artifact_digest != env.dev_reference.artifact_digest:
            problems.append("dev_dataset_artifact_digest_mismatch")
        if dataset.identity_payload_json is not None:
            payload = json.loads(dataset.identity_payload_json)
            evidence = payload.get("metadata_evidence")
            if isinstance(evidence, dict):
                bound_digest = evidence.get("raw_source_roster_digest")
        if not isinstance(bound_digest, str) or not _SHA256.match(bound_digest):
            problems.append("dev_dataset_roster_digest_unbound")
    roster: dict[str, dict[str, Any]] = {}
    roster_digest = ""
    try:
        evidence_doc = json.loads(dev_source_evidence.read_text(encoding="utf-8"))
        items = evidence_doc["roster"]
        if not isinstance(items, list):
            raise TypeError("roster is not a list")
        roster_digest = content_digest(items)
        for item in items:
            roster[str(item["url"])] = {
                "url": str(item["url"]),
                "sha256": str(item["sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
    except (OSError, ValueError, KeyError, TypeError):
        problems.append("dev_source_evidence_unreadable")
    else:
        if isinstance(bound_digest, str) and roster_digest != bound_digest:
            problems.append("dev_roster_digest_mismatch")
        planned = {item.url for item in plan.reused}
        if set(roster) != planned:
            problems.append("dev_roster_urls_differ_from_plan")
        missing = mismatched = 0
        for item in plan.reused:
            entry = roster.get(item.url)
            path = dev_archive_root / item.relative_path
            if not path.is_file():
                missing += 1
            elif entry is not None:
                data = path.read_bytes()
                if (
                    hashlib.sha256(data).hexdigest() != entry["sha256"]
                    or len(data) != entry["size_bytes"]
                ):
                    mismatched += 1
        if missing:
            problems.append("dev_archive_missing")
        if mismatched:
            problems.append("dev_archive_hash_mismatch")
    if problems:
        return None, tuple(dict.fromkeys(problems))
    return _DevReuse(dataset_id, artifact_digest, roster_digest, roster), ()


# ---------------------------------------------------------------------------
# Cache: reuse copy, fetch, completeness, checksums
# ---------------------------------------------------------------------------


def _copy_development_archives(
    plan: AcquisitionPlan, *, dev_archive_root: Path, cache_root: Path
) -> None:
    from trade_rl.integrations.binance import (
        validate_cached_vision_payload,
        vision_cache_path,
    )

    for item in plan.reused:
        source = dev_archive_root / item.relative_path
        payload = source.read_bytes()
        destination = vision_cache_path(cache_root, item.url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            # Source file mtime: the sidecar schema requires a value, it is NOT a
            # download time (nothing is downloaded) and is not part of any identity.
            "acquired_at": datetime.fromtimestamp(
                source.stat().st_mtime, UTC
            ).isoformat(),
            "downloader": _DOWNLOADER_REUSE,
            "etag": None,
            "last_modified": None,
            "schema_version": "binance_vision_raw_cache_v1",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "url": item.url,
        }
        destination.write_bytes(payload)
        destination.with_suffix(".json").write_text(
            json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        validate_cached_vision_payload(item.url, destination)


def _cache_problems(
    plan: AcquisitionPlan, cache_root: Path
) -> tuple[tuple[str, ...], dict[str, Path]]:
    """Missing and unexpected cache members, plus the planned archives' cache files."""

    from trade_rl.integrations.binance import vision_cache_path

    expected = set(plan.archive_urls) | {item.checksum_url for item in plan.fetched}
    found: set[str] = set()
    for path in cache_root.rglob("*.bin"):
        sidecar = path.with_suffix(".json")
        try:
            found.add(str(json.loads(sidecar.read_text(encoding="utf-8"))["url"]))
        except (OSError, ValueError, KeyError):
            found.add(f"unlabelled:{path.name}")
    problems: list[str] = []
    if expected - found:
        problems.append("archives_missing")
    if found - expected:
        problems.append("archives_unexpected")
    paths = {url: vision_cache_path(cache_root, url) for url in plan.archive_urls}
    return tuple(problems), paths


def _checksum_problems(plan: AcquisitionPlan, cache_root: Path) -> tuple[str, ...]:
    from trade_rl.integrations.binance import vision_cache_path

    malformed = mismatched = 0
    for item in plan.fetched:
        zip_path = vision_cache_path(cache_root, item.url)
        checksum_path = vision_cache_path(cache_root, item.checksum_url)
        expected = parse_checksum(
            checksum_path.read_text(encoding="utf-8", errors="replace"),
            filename=item.url.rsplit("/", 1)[1],
        )
        if expected is None:
            malformed += 1
        elif _sha256(zip_path) != expected:
            mismatched += 1
    problems = []
    if malformed:
        problems.append("archive_checksum_file_malformed")
    if mismatched:
        problems.append("archive_checksum_mismatch")
    return tuple(problems)


# ---------------------------------------------------------------------------
# Dataset checks (row presence and timestamps only) and the build
# ---------------------------------------------------------------------------


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


def dataset_gap_problems(
    dataset: MarketDataset, registry: Fs1Registry
) -> tuple[str, ...]:
    """Gaps (and span errors) the build policy does not admit; reads only the
    timestamps and the tradable flags, never a price, return, feature or funding value.

    The ``non_tradable_rows`` policy admits a missing 1h bar as a non-tradable row.
    What it does not admit, and the scoring evaluator would refuse anyway, is an
    irregular timeline, a wrong first/last row or too little tradable time per symbol
    over ``[development_data_end, scoring_end)``.
    """

    problems: list[str] = []
    if tuple(dataset.symbols) != registry.symbols:
        problems.append("dataset_symbols_mismatch")
    stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if stamps.size < 2 or not np.all(np.diff(stamps) == _NS_PER_HOUR):
        problems.append("gap_timeline_not_regular_hourly")
        return tuple(problems)
    if int(stamps[0]) > _ns(registry.dataset_first_row_latest):
        problems.append("dataset_first_row_after_latest")
    if int(stamps[-1]) != _ns(registry.scoring_end):
        problems.append("dataset_last_row_not_scoring_end")
    window = (stamps >= _ns(registry.development_data_end)) & (
        stamps < _ns(registry.scoring_end)
    )
    if not window.any():
        problems.append("dataset_window_empty")
    else:
        fraction = np.asarray(dataset.tradable)[window].mean(axis=0)
        for symbol, value in zip(dataset.symbols, fraction, strict=True):
            if float(value) < registry.min_tradable_fraction:
                problems.append(f"gap_tradable_fraction_low_{symbol}")
    return tuple(problems)


def _tradable_fraction(
    dataset: MarketDataset, registry: Fs1Registry
) -> dict[str, float]:
    stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    window = (stamps >= _ns(registry.development_data_end)) & (
        stamps < _ns(registry.scoring_end)
    )
    fraction = np.asarray(dataset.tradable)[window].mean(axis=0)
    return {
        symbol: float(value)
        for symbol, value in zip(dataset.symbols, fraction, strict=True)
    }


def _build_dataset(
    params: BuildParams,
    *,
    cache_root: Path,
    exchange_info_dir: Path,
    evidence: dict[str, Any],
) -> tuple[Any, Any]:
    """Offline build, mirroring the development builder under the FS1 policy.

    Returns ``(build_result, aggregating_transport)``. The wrapped transport has
    ``allow_network=False``: a missing cache member fails the build instead of fetching.
    """

    from trade_rl.data.build.economics import ExecutionEconomicsProfile
    from trade_rl.integrations.binance import (
        BinanceMarket,
        BinancePublicTransport,
        BinanceTransportMode,
        FrozenBinanceExchangeInfoTransport,
        build_binance_market_dataset,
    )
    from trade_rl.integrations.binance.dataset import BinanceGapPolicy
    from trade_rl.integrations.binance.derived_klines import (
        BaseIntervalAggregatingTransport,
    )

    frozen = FrozenBinanceExchangeInfoTransport(exchange_info_dir, delegate=None)
    transport = BaseIntervalAggregatingTransport(
        BinancePublicTransport(cache_root=cache_root, allow_network=False),
        exchange_information=frozen,
        base_interval=params.base_interval,
    )
    built = build_binance_market_dataset(
        market=BinanceMarket.USDS_M,
        symbols=params.symbols,
        interval=params.base_interval,
        start_time=params.start,
        end_time=params.end,
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        feature_timeframes=params.feature_timeframes,
        metadata_evidence=evidence,
        execution_economics=ExecutionEconomicsProfile(name=params.economics_profile),
        gap_policy=BinanceGapPolicy(params.gap_policy),
    )
    return built, transport


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _preflight(
    env: _Environment,
    *,
    authorization: str | None,
    output_root: Path,
    allow_network: bool,
    dev_archive_root: Path | None,
    dev_source_evidence: Path | None,
    dev_dataset: Path | None,
) -> tuple[AcquisitionPlan | None, _DevReuse | None, datetime | None, tuple[str, ...]]:
    reasons: list[str] = list(authorization_problems(authorization))
    if output_root.exists():
        reasons.append("output_exists")
    registry = env.registry
    params, policy_problems = build_params_or_problems(
        registry, implemented_policy_digest=env.implemented_policy_digest
    )
    reasons.extend(policy_problems)
    if registry.digest() != env.frozen_params_digest:
        reasons.append("registry_digest_differs_from_frozen")
    plan: AcquisitionPlan | None = None
    if params is not None:
        development_end = datetime.fromisoformat(registry.development_data_end).replace(
            tzinfo=UTC
        )
        scoring_start = datetime.fromisoformat(registry.scoring_start).replace(
            tzinfo=UTC
        )
        plan = plan_acquisition(
            params, development_end=development_end, scoring_start=scoring_start
        )
    dev: _DevReuse | None = None
    now: datetime | None = None
    if allow_network and plan is not None:
        now = env.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("the clock must return a timezone-aware datetime")
        if now < settlement_instant(registry):
            reasons.append("settlement_not_reached")
        identity = env.code_identity()
        if identity.git_head is None:
            reasons.append("code_identity_unavailable")
        elif not identity.tracked_tree_clean:
            reasons.append("code_tree_dirty")
        dev, dev_problems = _dev_reuse_or_problems(
            env,
            plan,
            dev_archive_root=dev_archive_root,
            dev_source_evidence=dev_source_evidence,
            dev_dataset=dev_dataset,
        )
        reasons.extend(dev_problems)
    return plan, dev, now, tuple(dict.fromkeys(reasons))


def _run_acquire(
    env: _Environment,
    *,
    authorization: str | None,
    output_root: Path,
    allow_network: bool,
    dev_archive_root: Path | None,
    dev_source_evidence: Path | None,
    dev_dataset: Path | None,
    command_line: str,
    emit: Emit | None = None,
) -> AcquisitionResult:
    """Private factory behind :func:`run_acquire` (tests inject the environment here)."""

    say = emit or (lambda _message: None)
    plan, dev, _, reasons = _preflight(
        env,
        authorization=authorization,
        output_root=output_root,
        allow_network=allow_network,
        dev_archive_root=dev_archive_root,
        dev_source_evidence=dev_source_evidence,
        dev_dataset=dev_dataset,
    )
    if reasons or plan is None:
        raise AcquisitionRefusal(reasons or ("build_policy_unparsed",))
    registry = env.registry
    if not allow_network:
        return AcquisitionResult(
            planned_only=True, plan=plan, plan_lines=render_plan(plan, registry)
        )
    assert authorization is not None and dev is not None
    assert dev_archive_root is not None

    # The maintained planner and this tool's pure planner must agree before any request.
    from trade_rl.integrations.binance import (
        BinanceMarket,
        FrozenBinanceExchangeInfoTransport,
        plan_binance_vision_cache,
    )

    maintained = plan_binance_vision_cache(
        market=BinanceMarket.USDS_M,
        symbols=plan.params.symbols,
        intervals=(plan.params.base_interval,),
        start_time=plan.params.start,
        end_time=plan.params.end,
    )
    if maintained.urls != plan.archive_urls:
        raise AcquisitionRefusal(("plan_differs_from_maintained_planner",))

    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise AcquisitionRefusal(("output_exists",)) from None
    cache_root = output_root / CACHE_DIR
    exchange_dir = output_root / EXCHANGE_INFO_DIR
    cache_root.mkdir()
    say(f"copying {len(plan.reused)} development archives (read-only source)")
    _copy_development_archives(
        plan, dev_archive_root=dev_archive_root, cache_root=cache_root
    )

    transport = env.network_transport(cache_root)
    fetch_started = env.clock()
    say(f"fetching {len(plan.fetched)} archives and their checksums")
    from trade_rl.integrations.binance import (
        BinanceTransportError,
        sync_binance_vision_urls,
    )

    try:
        sync_binance_vision_urls(plan.fetch_urls, transport=transport)
    except (BinanceTransportError, OSError, RuntimeError) as exc:
        raise AcquisitionRefusal(
            ("fetch_failed",), f"{type(exc).__name__}: {exc}"[:300]
        ) from exc
    cache_problems, archive_paths = _cache_problems(plan, cache_root)
    if cache_problems:
        raise AcquisitionRefusal(cache_problems)
    checksum_problems = _checksum_problems(plan, cache_root)
    if checksum_problems:
        raise AcquisitionRefusal(checksum_problems)

    roster: list[dict[str, Any]] = []
    origins: dict[str, str] = {}
    for item in plan.archives:
        path = archive_paths[item.url]
        data = path.read_bytes()
        entry = {
            "url": item.url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
        if item.reuse_local:
            expected = dev.roster[item.url]
            if entry != expected:
                raise AcquisitionRefusal(("dev_archive_hash_mismatch",), item.url)
        roster.append(entry)
        origins[item.url] = "development_reuse" if item.reuse_local else "fetched"

    say("freezing exchange rules")
    try:
        snapshot = FrozenBinanceExchangeInfoTransport(
            exchange_dir, delegate=transport
        ).load_exchange_information_snapshot(market=BinanceMarket.USDS_M, mode="rest")
    except Exception as exc:
        raise AcquisitionRefusal(
            ("exchange_information_failed",), f"{type(exc).__name__}: {exc}"[:300]
        ) from exc

    evidence: dict[str, Any] = {
        "schema_version": "fs1_source_evidence_v1",
        "market": plan.params.market,
        "build_policy_digest": registry.build_policy_digest(),
        "exchange_information": {
            "source_uri": snapshot.source_uri,
            "retrieved_at": snapshot.retrieved_at.isoformat(),
            "raw_payload_sha256": snapshot.raw_payload_sha256,
            "rules_assumption": (
                "frozen snapshot applied statically; not historical point-in-time rules"
            ),
        },
        "raw_source_roster_digest": content_digest(roster),
        "raw_source_count": len(roster),
        "development_reuse": {
            "dataset_id": dev.dataset_id,
            "artifact_digest": dev.artifact_digest,
            "roster_digest": dev.roster_digest,
            "archive_count": len(plan.reused),
        },
        "feature_timeframes_derivation": (
            "4h and 1d bars aggregated from complete 1h Vision klines"
        ),
        "checksum_files": (
            "official .CHECKSUM verified for every fetched archive; reused development "
            "archives verified against the development roster"
        ),
    }
    say("building the dataset offline")
    try:
        built, transport_offline = _build_dataset(
            plan.params,
            cache_root=cache_root,
            exchange_info_dir=exchange_dir,
            evidence=evidence,
        )
    except Exception as exc:
        raise AcquisitionRefusal(
            ("dataset_build_failed",), f"{type(exc).__name__}: {exc}"[:300]
        ) from exc
    dataset: MarketDataset = built.dataset
    gap_problems = dataset_gap_problems(dataset, registry)
    if gap_problems:
        raise AcquisitionRefusal(gap_problems)

    destination = output_root / DATASET_DIR
    published = publish_market_dataset_artifact(destination, dataset)
    _, loaded = load_published_market_dataset_artifact(destination)
    if loaded.dataset_id != dataset.dataset_id or not loaded.identity_verified:
        raise AcquisitionRefusal(("published_dataset_identity_mismatch",))

    fetched_utc = _utc_text(fetch_started)
    identity = env.code_identity()
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "issue": 810,
        "command_line": command_line,
        "output_root": str(output_root),
        "builder": {
            "module": "trade_rl.evaluation.forward_acquisition.acquire",
            "git_head": identity.git_head,
            "tracked_tree_clean": identity.tracked_tree_clean,
        },
        "authorization_reference": authorization,
        "registry_params_digest": registry.digest(),
        "build_policy": dict(registry.build_policy),
        "build_policy_digest": registry.build_policy_digest(),
        "fetched_utc": fetched_utc,
        "exchange_information": evidence["exchange_information"],
        "development_reuse": evidence["development_reuse"],
        "source_archives": {
            "count": len(roster),
            "roster_digest": evidence["raw_source_roster_digest"],
            "fetched_count": len(plan.fetched),
            "roster": [{**entry, "origin": origins[entry["url"]]} for entry in roster],
        },
        "dataset": {
            "dataset_id": loaded.dataset_id,
            "artifact_digest": published.artifact_digest,
            "n_bars": loaded.n_bars,
            "symbols": list(loaded.symbols),
            "first_timestamp": str(loaded.timestamps[0]),
            "last_timestamp": str(loaded.timestamps[-1]),
            "gap_report": {key: dict(value) for key, value in built.gap_report.items()},
            "derived_incomplete_windows": {
                f"{symbol}:{interval}": count
                for (
                    symbol,
                    interval,
                ), count in transport_offline.dropped_windows.items()
            },
            "tradable_fraction_development_end_to_scoring_end": _tradable_fraction(
                loaded, registry
            ),
        },
        "economic_replay_executed": False,
        "scored_window_use": (
            "row presence and timestamps only (gap and regularity guard); no price, "
            "return, feature or funding value was read for any decision"
        ),
    }
    receipt_bytes = _canonical_bytes(receipt)
    receipt_path = output_root / RECEIPT_NAME
    receipt_path.write_bytes(receipt_bytes)

    source_manifest: dict[str, str] = {
        entry["url"]: entry["sha256"] for entry in roster
    }
    source_manifest[SOURCE_KEY_RECEIPT] = hashlib.sha256(receipt_bytes).hexdigest()
    source_manifest[SOURCE_KEY_EXCHANGE_INFO] = snapshot.raw_payload_sha256
    source_manifest[SOURCE_KEY_DATASET_ARTIFACT] = published.artifact_digest
    source_manifest[SOURCE_KEY_DEV_ROSTER] = dev.roster_digest
    sealed_utc = max(env.clock(), fetch_started)
    manifest_bytes = manifest_bytes_from_dataset(
        loaded,
        source_manifest=source_manifest,
        build_policy_digest=registry.build_policy_digest(),
        authorization_reference=authorization,
        fetched_utc=fetched_utc,
        sealed_utc=_utc_text(sealed_utc),
    )
    parse_manifest(manifest_bytes)
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    say(f"manifest sha256 {manifest_sha}")
    return AcquisitionResult(
        planned_only=False,
        plan=plan,
        output_root=output_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        receipt_path=receipt_path,
        dataset_path=destination,
        dataset_id=loaded.dataset_id,
        fetched_utc=fetched_utc,
    )


def run_acquire(
    *,
    authorization: str | None,
    output_root: Path,
    allow_network: bool,
    dev_archive_root: Path | None = None,
    dev_source_evidence: Path | None = None,
    dev_dataset: Path | None = None,
    command_line: str,
    emit: Emit | None = None,
) -> AcquisitionResult:
    """Plan (default) or perform the authorized FS1 acquisition.

    Always uses the frozen registry and digest, the real wall clock, the real git
    checkout and the real transport. Without ``allow_network`` it only plans. See the
    module docstring for the refusals.
    """

    return _run_acquire(
        _environment(),
        authorization=authorization,
        output_root=output_root,
        allow_network=allow_network,
        dev_archive_root=dev_archive_root,
        dev_source_evidence=dev_source_evidence,
        dev_dataset=dev_dataset,
        command_line=command_line,
        emit=emit,
    )


__all__ = [
    "AcquisitionRefusal",
    "AcquisitionResult",
    "DevReference",
    "dataset_gap_problems",
    "parse_checksum",
    "run_acquire",
    "settlement_instant",
]
