from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset


def _load() -> ModuleType:
    path = Path(".github/scripts/issue661_profile_source.py")
    spec = importlib.util.spec_from_file_location("issue661_profile_source", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Issue 661 profile-source helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset(module: ModuleType) -> MarketDataset:
    symbols = module.SYMBOLS
    shape = (6, len(symbols))
    prices = np.full(shape, 100.0)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2024-01-01", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=prices,
        high=prices + 1,
        low=prices - 1,
        close=prices,
        volume=np.full(shape, 1000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8760,
        minimum_notional=np.full(shape, 5.0),
        lot_size=np.full(shape, 0.001),
    ).with_content_identity({"fixture": "issue661-profile-source"})
    return dataset


def _raw(module: ModuleType) -> bytes:
    symbols = []
    for name in module.SYMBOLS:
        symbols.append(
            {
                "symbol": name,
                "status": "TRADING",
                "baseAsset": name[:-4],
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "marginAsset": "USDT",
                "orderTypes": ["LIMIT", "MARKET"],
                "onboardDate": 1_600_000_000_000,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.001",
                        "minQty": "0.001",
                        "maxQty": "1000",
                    },
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "stepSize": "0.001",
                        "minQty": "0.001",
                        "maxQty": "1000",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        )
    return json.dumps(
        {"serverTime": 1, "symbols": symbols}, separators=(",", ":")
    ).encode()


def test_bundle_derives_both_profiles_from_one_raw_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    dataset = _dataset(module)
    monkeypatch.setattr(module, "DATASET_ID", dataset.dataset_id)
    captured_at = datetime(2026, 9, 18, 10, 0, tzinfo=UTC)

    bundle = module.build_bundle(dataset, _raw(module), captured_at)
    manifest = json.loads(bundle["manifest.json"])
    control = json.loads(bundle["control-profile.json"])
    treatment = json.loads(bundle["treatment-profile.json"])

    assert manifest["raw_payload_sha256"] == hashlib.sha256(_raw(module)).hexdigest()
    assert manifest["selected_symbols"] == list(module.SYMBOLS)
    assert manifest["control_reduce_only_exits"] is False
    assert manifest["treatment_reduce_only_exits"] is True
    assert manifest["network_request_count"] == 1
    assert manifest["first_successful_validated_response"] is True
    assert control["reduce_only_exits"] is False
    assert treatment["reduce_only_exits"] is True
    assert {
        key
        for key in set(control) | set(treatment)
        if control.get(key) != treatment.get(key)
    } == {"reduce_only_exits"}
    assert manifest["economic_execution_started"] is False
    assert manifest["unused_data_accessed"] is False


class _Response:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.raw


def test_source_parser_rejects_duplicate_json_keys() -> None:
    module = _load()
    duplicated = b'{"symbols":[],"symbols":[]}'
    with pytest.raises(ValueError, match="duplicate"):
        module._snapshot(duplicated, datetime(2026, 9, 18, 10, 0, tzinfo=UTC))


def test_capture_uses_exact_url_once_without_retry_or_fallback() -> None:
    module = _load()
    calls: list[tuple[str, float]] = []

    def opener(request, *, timeout: float):
        calls.append((request.full_url, timeout))
        return _Response(_raw(module))

    captured_at = datetime(2026, 9, 18, 10, 1, tzinfo=UTC)
    raw, retrieved_at = module.capture_exact_response(
        opener=opener, clock=lambda: captured_at
    )
    assert calls == [(module.SOURCE_URL, 30.0)]
    assert raw == _raw(module)
    assert retrieved_at == captured_at

    calls.clear()

    def bad_opener(request, *, timeout: float):
        calls.append((request.full_url, timeout))
        return _Response(b"not-json")

    with pytest.raises(ValueError, match="valid JSON"):
        module.capture_exact_response(opener=bad_opener, clock=lambda: captured_at)
    assert calls == [(module.SOURCE_URL, 30.0)]


def test_rederivation_requires_byte_identical_source_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    dataset = _dataset(module)
    monkeypatch.setattr(module, "DATASET_ID", dataset.dataset_id)
    bundle = module.build_bundle(
        dataset, _raw(module), datetime(2026, 9, 18, 10, 2, tzinfo=UTC)
    )
    root = tmp_path / "source"
    module.write_bundle(root, bundle)
    assert module.load_and_verify_bundle(dataset, root) == bundle

    changed = bytearray((root / "treatment-profile.json").read_bytes())
    changed[-2] ^= 1
    (root / "treatment-profile.json").write_bytes(bytes(changed))
    with pytest.raises(ValueError, match="rederivation"):
        module.load_and_verify_bundle(dataset, root)


def test_dataset_extraction_ignores_non_dataset_evidence(tmp_path: Path) -> None:
    module = _load()
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("outer/dataset/manifest.json", b"{}")
        target.writestr("outer/dataset/arrays.bin", b"data")
        target.writestr("outer/study/result.json", b'{"economic":true}')
    output = tmp_path / "dataset"
    module.extract_dataset_only(archive, output)
    assert (output / "manifest.json").read_bytes() == b"{}"
    assert (output / "arrays.bin").read_bytes() == b"data"
    assert not (tmp_path / "study").exists()


def test_dataset_extraction_rejects_unsafe_or_ambiguous_archives(
    tmp_path: Path,
) -> None:
    module = _load()
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as target:
        target.writestr("../dataset/manifest.json", b"{}")
    with pytest.raises(ValueError, match="unsafe"):
        module.extract_dataset_only(unsafe, tmp_path / "unsafe-out")

    ambiguous = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(ambiguous, "w") as target:
        target.writestr("a/dataset/manifest.json", b"{}")
        target.writestr("b/dataset/manifest.json", b"{}")
    with pytest.raises(ValueError, match="exactly one"):
        module.extract_dataset_only(ambiguous, tmp_path / "ambiguous-out")
