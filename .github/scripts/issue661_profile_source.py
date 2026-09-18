"""Result-blind source capture/rederivation for Issue #661 market profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.artifacts import load_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.binance.market_order_profile import (
    build_usdm_market_order_profile,
)
from trade_rl.integrations.binance.metadata import BinanceExchangeInfoSnapshot

SOURCE_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
DATASET_ARTIFACT_DIGEST = (
    "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
)
SCHEMA_VERSION = "issue661_market_profile_source_v1"


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("retrieved_at must be a non-empty string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("retrieved_at must include timezone")
    return parsed.astimezone(UTC)


def _snapshot(raw: bytes, retrieved_at: datetime) -> BinanceExchangeInfoSnapshot:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("exchangeInfo response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("exchangeInfo response must be an object")
    return BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw,
        source_uri=SOURCE_URL,
        retrieved_at=retrieved_at.astimezone(UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validate_dataset(dataset: MarketDataset) -> None:
    if dataset.dataset_id != DATASET_ID:
        raise ValueError("dataset id differs from Issue 661 authority")
    if tuple(dataset.symbols) != SYMBOLS:
        raise ValueError("dataset symbol roster/order differs from Issue 661 authority")


def build_bundle(
    dataset: MarketDataset,
    raw: bytes,
    retrieved_at: datetime,
) -> dict[str, bytes]:
    """Derive the two frozen profiles from one exact raw response."""
    _validate_dataset(dataset)
    snapshot = _snapshot(raw, retrieved_at)
    profiles = {}
    for name, reduce_only_exits in (("control", False), ("treatment", True)):
        profiles[name] = build_usdm_market_order_profile(
            dataset,
            snapshot,
            selected_symbols=SYMBOLS,
            account_mode="one_way",
            reduce_only_exits=reduce_only_exits,
        )

    control_payload = profiles["control"].canonical_payload()
    treatment_payload = profiles["treatment"].canonical_payload()
    differing = {
        key
        for key in set(control_payload) | set(treatment_payload)
        if control_payload.get(key) != treatment_payload.get(key)
    }
    if differing != {"reduce_only_exits"}:
        raise ValueError("profile pair differs outside reduce_only_exits")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_uri": SOURCE_URL,
        "retrieved_at": _utc_text(retrieved_at),
        "raw_payload_sha256": snapshot.raw_payload_sha256,
        "dataset_id": dataset.dataset_id,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "selected_symbols": list(SYMBOLS),
        "account_mode": "one_way",
        "control_profile_digest": profiles["control"].digest,
        "treatment_profile_digest": profiles["treatment"].digest,
        "control_reduce_only_exits": False,
        "treatment_reduce_only_exits": True,
        "economic_execution_started": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    return {
        "exchange-info.raw.json": raw,
        "control-profile.json": canonical_json_bytes(control_payload),
        "treatment-profile.json": canonical_json_bytes(treatment_payload),
        "manifest.json": canonical_json_bytes(manifest),
    }


def write_bundle(root: Path, bundle: Mapping[str, bytes]) -> None:
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "exchange-info.raw.json",
        "control-profile.json",
        "treatment-profile.json",
        "manifest.json",
    ):
        raw = bundle.get(name)
        if not isinstance(raw, bytes):
            raise ValueError(f"bundle is missing bytes for {name}")
        with (destination / name).open("xb") as stream:
            stream.write(raw)


def load_and_verify_bundle(dataset: MarketDataset, root: Path) -> dict[str, bytes]:
    source = Path(root)
    raw = (source / "exchange-info.raw.json").read_bytes()
    try:
        manifest = json.loads((source / "manifest.json").read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("source manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be an object")
    rebuilt = build_bundle(dataset, raw, _parse_utc(manifest.get("retrieved_at")))
    actual = {name: (source / name).read_bytes() for name in rebuilt}
    if actual != rebuilt:
        raise ValueError(
            "published profile source bundle differs from independent rederivation"
        )
    return rebuilt


def capture_exact_response(
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> tuple[bytes, datetime]:
    """Issue exactly one GET to the frozen URL; no retry or fallback."""
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "trade-rl-issue661-source-capture/1"},
        method="GET",
    )
    with opener(request, timeout=30.0) as response:
        raw = response.read()
    retrieved_at = clock().astimezone(UTC)
    _snapshot(raw, retrieved_at)
    return raw, retrieved_at


def extract_dataset_only(artifact_zip: Path, destination: Path) -> Path:
    """Safely extract exactly one dataset/ subtree and nothing else."""
    root = Path(destination)
    if root.exists():
        raise FileExistsError(root)
    with zipfile.ZipFile(artifact_zip) as archive:
        infos = archive.infolist()
        candidates = []
        for info in infos:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
                raise ValueError("unsafe artifact member path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("symlink artifact member is forbidden")
            if len(pure.parts) >= 2 and pure.parts[-2:] == ("dataset", "manifest.json"):
                candidates.append(pure.parent)
        if len(candidates) != 1:
            raise ValueError("artifact must contain exactly one dataset manifest")
        dataset_prefix = candidates[0]
        members = [
            info
            for info in infos
            if not info.is_dir()
            and PurePosixPath(info.filename).parts[: len(dataset_prefix.parts)]
            == dataset_prefix.parts
        ]
        if not members:
            raise ValueError("dataset artifact subtree is empty")
        root.mkdir(parents=True)
        for info in members:
            pure = PurePosixPath(info.filename)
            relative = PurePosixPath(*pure.parts[len(dataset_prefix.parts) :])
            if not relative.parts:
                continue
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                output.write(source.read())
    if not (root / "manifest.json").is_file():
        raise ValueError("dataset manifest was not extracted")
    return root


def _load_dataset(path: Path) -> MarketDataset:
    dataset = load_market_dataset_artifact(path)
    _validate_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "rederive", "verify"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset_root)
    if args.action == "capture":
        if args.output is None:
            parser.error("capture requires --output")
        raw, retrieved_at = capture_exact_response()
        write_bundle(args.output, build_bundle(dataset, raw, retrieved_at))
        print("ISSUE661_PROFILE_SOURCE_CAPTURED=true")
        print("ECONOMIC_EXECUTION_STARTED=false")
        return
    if args.source_root is None:
        parser.error(f"{args.action} requires --source-root")
    bundle = load_and_verify_bundle(dataset, args.source_root)
    if args.action == "rederive":
        if args.output is None:
            parser.error("rederive requires --output")
        write_bundle(args.output, bundle)
        print("ISSUE661_PROFILE_SOURCE_REDERIVED=true")
        print("MARKET_DATA_NETWORK_ACCESS_PERFORMED=false")
        return
    print("ISSUE661_PROFILE_SOURCE_VERIFIED=true")
    print("MARKET_DATA_NETWORK_ACCESS_PERFORMED=false")


if __name__ == "__main__":
    main()
