"""Offline, opt-in build of the Issue #810 P2 development Dataset.

The study Dataset (5 USDT-perpetual symbols, 1h bars 2021-01-01..2026-08-01,
zero fees baked in) is built from monthly Binance Vision archives that already
sit on local disk. Nothing here touches the network and nothing here runs an
economic replay.

Build policy (pinned; :data:`DATASET_BUILD_POLICY` and its digest are recorded in
every build receipt):

* ``gap_policy = non_tradable_rows``: a missing 1h bar becomes a
  ``row_present=False``, non-tradable row with forward-filled marks (the
  allocator holds); funding events inside a missing bar are not accrued.
* 4h and 1d feature bars are aggregated from the verified 1h bars. A 4h/1d
  window with any missing 1h bar keeps its slot as a price-carried placeholder
  whose ``available_at`` is the end of the range, so every recursive or rolling
  higher-clock feature that touches it is unavailable afterwards.
* Exchange rules are the frozen current-snapshot exchange information, applied
  statically (not historical point-in-time rules).
* Local archive bytes are re-keyed into the transport cache layout with their
  own SHA-256 sidecar; the official ``.CHECKSUM`` files are not available
  offline.

This builder was committed AFTER the first study run, as a faithful port of the
scratch script that produced the study Dataset. The proof that it is the same
procedure is the identical ``dataset_id`` and artifact digest it reproduces from
the same local inputs (see ``docs/research/current-status.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from trade_rl.artifacts import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
    FrozenBinanceExchangeInfoTransport,
    plan_binance_vision_cache,
    require_complete_binance_vision_cache,
    validate_cached_vision_payload,
    vision_cache_path,
)
from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
)

RECEIPT_SCHEMA: Final = "dev_5sym_dataset_build_receipt_v1"
EVIDENCE_SCHEMA: Final = "dev_5sym_local_rekey_source_evidence_v1"
STUDY_DATASET_ID: Final = (
    "0b95547145a8f35d4deaaea99479ccf93244080bff52ef969203f4fa721de174"
)
STUDY_ARTIFACT_DIGEST: Final = (
    "9f00e17c2f3de700f3a9ff08d41dea312ff61c143d85514b73e0cd84e03cd381"
)

START: Final = datetime(2021, 1, 1, tzinfo=UTC)
END: Final = datetime(2026, 8, 1, tzinfo=UTC)
SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MARKET: Final = BinanceMarket.USDS_M
BASE_INTERVAL: Final = "1h"
FEATURE_TIMEFRAMES: Final = ("4h", "1d")
ECONOMICS_PROFILE_NAME: Final = "bot_overlay_only_assumptions_v1"
DOWNLOADER: Final = "trade-rl/local-rekey from data/breakeven-2020-2026 (no network)"
_ARCHIVE_MARKER: Final = "/data/futures/um/monthly/"

DATASET_BUILD_POLICY: Final[dict[str, object]] = {
    "schema_version": "dev_5sym_dataset_build_policy_v1",
    "network": "never",
    "market": MARKET.value,
    "symbols": list(SYMBOLS),
    "base_interval": BASE_INTERVAL,
    "feature_timeframes": list(FEATURE_TIMEFRAMES),
    "start_time": START.isoformat(),
    "end_time": END.isoformat(),
    "gap_policy": BinanceGapPolicy.NON_TRADABLE_ROWS.value,
    "missing_base_bar": "non-tradable row (row_present=False), forward-filled marks",
    "missing_base_bar_funding": "funding events inside a missing bar are not accrued",
    "higher_clock_after_gap": (
        "window with any missing 1h bar is a price-carried placeholder with "
        "available_at=range end; recursive/rolling features touching it are unavailable"
    ),
    "higher_clock_source": "4h and 1d bars aggregated from complete 1h windows",
    "exchange_information": "frozen current snapshot applied statically",
    "economics_profile": ECONOMICS_PROFILE_NAME,
}


def build_policy_digest() -> str:
    """Digest of :data:`DATASET_BUILD_POLICY`; recorded in every receipt."""

    return content_digest(DATASET_BUILD_POLICY)


def _archive_path(archive_root: Path, url: str) -> Path:
    tail = url.split(_ARCHIVE_MARKER, 1)[1]
    return archive_root / tail


def rekey_local_archives(
    urls: Sequence[str],
    *,
    archive_root: Path,
    cache_root: Path,
) -> list[dict[str, object]]:
    """Copy local archives into the transport cache layout with SHA-256 sidecars."""

    roster: list[dict[str, object]] = []
    for url in urls:
        source = _archive_path(archive_root, url)
        payload = source.read_bytes()
        destination = vision_cache_path(cache_root, url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        # ``acquired_at`` is fixed by the Binance cache sidecar schema. Here it is the
        # local source file's mtime, NOT a download time (nothing is downloaded); it
        # is not part of the dataset_id or artifact digest.
        evidence = {
            "acquired_at": datetime.fromtimestamp(
                source.stat().st_mtime, UTC
            ).isoformat(),
            "downloader": DOWNLOADER,
            "etag": None,
            "last_modified": None,
            "schema_version": "binance_vision_raw_cache_v1",
            "sha256": digest,
            "size_bytes": len(payload),
            "url": url,
        }
        destination.write_bytes(payload)
        destination.with_suffix(".json").write_text(
            json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        validate_cached_vision_payload(url, destination)
        roster.append({"url": url, "sha256": digest, "size_bytes": len(payload)})
    return roster


def build_dev_dataset(
    *,
    archive_root: Path,
    exchange_info_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build and publish the study Dataset into a new ``output_dir``.

    Refuses a non-empty ``output_dir``. Returns the receipt that is also
    written to ``output_dir/dataset-receipt.json``.
    """

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "vision-cache"
    shutil.copytree(exchange_info_dir, output_dir / "exchange-info")

    plan = plan_binance_vision_cache(
        market=MARKET,
        symbols=SYMBOLS,
        intervals=(BASE_INTERVAL,),
        start_time=START,
        end_time=END,
    )
    missing = [
        url for url in plan.urls if not _archive_path(archive_root, url).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} local archives are missing, e.g. {missing[:3]}"
        )
    roster = rekey_local_archives(
        plan.urls, archive_root=archive_root, cache_root=cache_root
    )
    require_complete_binance_vision_cache(plan, cache_root=cache_root)

    frozen = FrozenBinanceExchangeInfoTransport(
        output_dir / "exchange-info", delegate=None
    )
    snapshot = frozen.load_exchange_information_snapshot(
        market=MARKET, mode=BinanceTransportMode.REST
    )
    transport = BaseIntervalAggregatingTransport(
        BinancePublicTransport(cache_root=cache_root, allow_network=False),
        exchange_information=frozen,
        base_interval=BASE_INTERVAL,
    )
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "market": MARKET.value,
        "exchange_information": {
            "source_uri": snapshot.source_uri,
            "retrieved_at": snapshot.retrieved_at.isoformat(),
            "raw_payload_sha256": snapshot.raw_payload_sha256,
            "rules_assumption": (
                "frozen current snapshot applied statically; "
                "not historical point-in-time rules"
            ),
        },
        "raw_source_roster_digest": content_digest(roster),
        "raw_source_count": len(roster),
        "feature_timeframes_derivation": (
            "4h and 1d native bars aggregated from verified 1h Vision klines "
            "(local archives had no 4h/1d)"
        ),
        "gap_handling": (
            "missing 1h klines (XRPUSDT 2022-02-26..28, 2022-04-01..02) are absent "
            "rows (row_present False); native 4h/1d placeholders have "
            "available_at=dataset end"
        ),
        "checksum_files": (
            "absent: only sidecar sha256 of local bytes; "
            "official .CHECKSUM not available offline"
        ),
    }
    built = build_binance_market_dataset(
        market=MARKET,
        symbols=SYMBOLS,
        interval=BASE_INTERVAL,
        start_time=START,
        end_time=END,
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        feature_timeframes=FEATURE_TIMEFRAMES,
        metadata_evidence=evidence,
        execution_economics=ExecutionEconomicsProfile(name=ECONOMICS_PROFILE_NAME),
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )
    dataset = built.dataset
    destination = output_dir / "dataset"
    published = publish_market_dataset_artifact(destination, dataset)
    inspect_published_market_dataset_artifact(destination)
    loaded = load_market_dataset_artifact(destination)
    if loaded.dataset_id != dataset.dataset_id or not loaded.identity_verified:
        raise RuntimeError("published Dataset does not reload to the built identity")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "build_policy": DATASET_BUILD_POLICY,
        "build_policy_digest": build_policy_digest(),
        "dataset_path": str(destination),
        "dataset_id": loaded.dataset_id,
        "artifact_digest": published.artifact_digest,
        "n_bars": loaded.n_bars,
        "symbols": list(loaded.symbols),
        "first_close_timestamp": str(loaded.timestamps[0]),
        "last_close_timestamp": str(loaded.timestamps[-1]),
        "n_features": len(loaded.feature_names),
        "feature_names": list(loaded.feature_names),
        "feature_timeframes": list(built.feature_timeframes),
        "sources_used": list(built.sources_used),
        "gap_report": {key: dict(value) for key, value in built.gap_report.items()},
        "derived_incomplete_windows": {
            f"{symbol}:{interval}": count
            for (symbol, interval), count in transport.dropped_windows.items()
        },
        "source_archive_count": len(roster),
        "raw_source_roster_digest": evidence["raw_source_roster_digest"],
        "source_digest": content_digest(evidence),
        "funding_events_by_symbol": loaded.resolved_array("funding_event_count")
        .sum(axis=0)
        .tolist(),
        "economic_replay_executed": False,
    }
    (output_dir / "dataset-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "source-evidence.json").write_text(
        json.dumps({"evidence": evidence, "roster": roster}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.evaluation.dev_realdata_study.dataset_build",
        description=(
            "Offline, opt-in build of the Issue #810 P2 development Dataset from "
            "local Binance Vision archives. Never uses the network."
        ),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="directory holding klines/<SYMBOL>/1h and fundingRate/<SYMBOL> zips",
    )
    parser.add_argument(
        "--exchange-info-dir",
        type=Path,
        required=True,
        help="frozen exchange-info directory to copy into the build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new empty directory; an existing non-empty one is refused",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_dev_dataset(
            archive_root=args.archive_root,
            exchange_info_dir=args.exchange_info_dir,
            output_dir=args.output_dir,
        )
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 2
    print(
        f"dataset_id={receipt['dataset_id']} "
        f"artifact_digest={receipt['artifact_digest']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
