"""Offline builder for the Issue #810 P2 development Dataset (synthetic inputs only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trade_rl.evaluation.dev_realdata_study import dataset_build
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.integrations.binance import (
    BinanceTransportError,
    plan_binance_vision_cache,
    validate_cached_vision_payload,
    vision_cache_path,
)


def test_pinned_ids_match_the_frozen_protocol() -> None:
    assert dataset_build.STUDY_DATASET_ID == PROTOCOL.dataset_id
    assert len(dataset_build.STUDY_ARTIFACT_DIGEST) == 64


def test_build_policy_pins_the_gap_rules_and_has_a_stable_digest() -> None:
    policy = dataset_build.DATASET_BUILD_POLICY

    assert policy["gap_policy"] == "non_tradable_rows"
    assert policy["network"] == "never"
    assert policy["feature_timeframes"] == ["4h", "1d"]
    assert "forward-filled" in str(policy["missing_base_bar"])
    assert "not accrued" in str(policy["missing_base_bar_funding"])
    assert "unavailable" in str(policy["higher_clock_after_gap"])
    assert "aggregated from complete 1h windows" in str(policy["higher_clock_source"])
    digest = dataset_build.build_policy_digest()
    assert digest == dataset_build.build_policy_digest()
    # Pinned: changing the gap policy text must be a deliberate, reviewed edit.
    assert digest == (
        "3d82d2daf28a341d333f8121b79221853dd379c8861ac7d80c03fd202e4e1a03"
    )


def _plan_urls() -> tuple[str, ...]:
    plan = plan_binance_vision_cache(
        market=dataset_build.MARKET,
        symbols=dataset_build.SYMBOLS,
        intervals=(dataset_build.BASE_INTERVAL,),
        start_time=dataset_build.START,
        end_time=dataset_build.END,
    )
    return plan.urls


def test_plan_covers_klines_and_funding_for_every_symbol() -> None:
    urls = _plan_urls()

    assert len(urls) == 670
    assert {url.split("/monthly/")[1].split("/")[0] for url in urls} == {
        "klines",
        "fundingRate",
    }
    assert all(
        any(f"/{symbol}/" in url for url in urls) for symbol in dataset_build.SYMBOLS
    )


def test_rekey_copies_bytes_with_a_verifiable_sha256_sidecar(tmp_path: Path) -> None:
    urls = _plan_urls()[:3]
    archive_root = tmp_path / "archives"
    for url in urls:
        source = dataset_build._archive_path(archive_root, url)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"payload:{url}".encode())
    cache_root = tmp_path / "cache"

    roster = dataset_build.rekey_local_archives(
        urls, archive_root=archive_root, cache_root=cache_root
    )

    assert [item["url"] for item in roster] == list(urls)
    for url, item in zip(urls, roster, strict=True):
        cached = vision_cache_path(cache_root, url)
        payload = f"payload:{url}".encode()
        assert cached.read_bytes() == payload
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
        assert item["size_bytes"] == len(payload)
        assert validate_cached_vision_payload(url, cached) == payload
        evidence = json.loads(cached.with_suffix(".json").read_text(encoding="utf-8"))
        assert evidence["sha256"] == item["sha256"]
        assert evidence["downloader"] == dataset_build.DOWNLOADER
        assert evidence["etag"] is None

    cached = vision_cache_path(cache_root, urls[0])
    cached.write_bytes(b"tampered")
    with pytest.raises(BinanceTransportError):
        validate_cached_vision_payload(urls[0], cached)


def test_build_refuses_missing_archives_and_non_empty_output(tmp_path: Path) -> None:
    exchange_info = tmp_path / "exchange-info"
    exchange_info.mkdir()
    output = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="local archives are missing"):
        dataset_build.build_dev_dataset(
            archive_root=tmp_path / "none",
            exchange_info_dir=exchange_info,
            output_dir=output,
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        dataset_build.build_dev_dataset(
            archive_root=tmp_path / "none",
            exchange_info_dir=exchange_info,
            output_dir=occupied,
        )
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "existing"


def test_cli_returns_two_for_a_non_empty_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("existing", encoding="utf-8")

    code = dataset_build.main(
        [
            "--archive-root",
            str(tmp_path),
            "--exchange-info-dir",
            str(tmp_path),
            "--output-dir",
            str(occupied),
        ]
    )

    assert code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
