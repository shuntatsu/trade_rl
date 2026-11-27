"""Strict JSON readers admit only text representable by canonical UTF-8."""

import json
from pathlib import Path

import pytest

from tests.evaluation.test_candidate_run_config import raw_config
from trade_rl.artifacts.canonical import canonical_json_bytes, strict_json_loads
from trade_rl.data.build.config import load_market_build_request
from trade_rl.evaluation.runs.config import load_candidate_run_config


@pytest.mark.parametrize(
    "raw",
    [
        r'"\ud800"',
        r'"\udfff"',
        r'{"\ud800":1}',
        r'["\udc00"]',
        r'{"outer":[{"inner":"\ud800x"}]}',
        r'"\udc00\ud800"',
        r'"\ud83d\ude00\ud800"',
        '"' + chr(0xD800) + '"',
    ],
)
def test_strict_json_rejects_unpaired_surrogates(raw: str) -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        strict_json_loads(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r'"\ud83d\ude00"', chr(0x1F600)),
        (r'{"\ud83d\ude00":["\u65e5","\u0000"]}', {chr(0x1F600): ["日", chr(0)]}),
        (r'"\\ud800"', r"\ud800"),
        ('{"x":[null,true,1,0.125]}', {"x": [None, True, 1, 0.125]}),
    ],
)
def test_strict_json_preserves_valid_scalar_identity(
    raw: str, expected: object
) -> None:
    assert strict_json_loads(raw) == expected
    assert canonical_json_bytes(strict_json_loads(raw)) == canonical_json_bytes(
        expected
    )


def _build_payload(name: str) -> dict[str, object]:
    return {
        "source_root": ".",
        "base_timeframe": "1h",
        "features": [{"name": name, "kind": "log_return"}],
        "instruments": [{"symbol": "BTCUSDT", "listed_at": "2020-01-01T00:00:00Z"}],
    }


def _run_payload(name: str) -> dict[str, object]:
    return {**raw_config(), "signal_name": name, "feature_names": [name, "f1"]}


@pytest.mark.parametrize("name", [chr(0xD800), chr(0xDC00)])
@pytest.mark.parametrize("kind", ["dataset", "run"])
def test_config_files_refuse_unserializable_text(
    tmp_path: Path, name: str, kind: str
) -> None:
    path = tmp_path / "config.json"
    payload = _build_payload(name) if kind == "dataset" else _run_payload(name)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    loader = (
        load_market_build_request if kind == "dataset" else load_candidate_run_config
    )
    with pytest.raises(ValueError, match="UTF-8"):
        loader(path)


@pytest.mark.parametrize("kind", ["dataset", "run"])
def test_config_files_preserve_supplementary_unicode_identity(
    tmp_path: Path, kind: str
) -> None:
    name = "信号" + chr(0x1F600)
    path = tmp_path / "config.json"
    payload = _build_payload(name) if kind == "dataset" else _run_payload(name)
    values = []
    for escaped in (False, True):
        path.write_text(json.dumps(payload, ensure_ascii=escaped), encoding="utf-8")
        if kind == "dataset":
            request = load_market_build_request(path)
            assert request.config.features[0].name == name
            values.append(canonical_json_bytes(request.config.canonical_payload()))
        else:
            config = load_candidate_run_config(path)
            assert config.signal_name == name
            values.append(canonical_json_bytes(config.to_json_payload()))
    assert values[0] == values[1]
