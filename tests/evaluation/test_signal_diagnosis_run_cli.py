"""CLI-level proof that no Dataset array is read before every refusal has run.

The payload reader (``load_published_market_dataset_artifact``) is replaced by a spy.
It must never be reached when the artifact's metadata puts coverage outside the window,
when the G4 URL is missing or malformed, when the working tree is dirty, when LightGBM
is missing, or on any digest mismatch, and it must be reached exactly once for a valid
input (the positive control). No market value exists in these fixtures: the artifact is
a manifest plus an opaque byte string.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.evaluation.signal_diagnosis_run_support import CLEAN, DIRTY, protocol
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.signal_diagnosis.protocol import (
    ForbiddenEvaluationWindowError,
    SignalDiagnosisProtocolError,
)
from trade_rl.evaluation.signal_diagnosis_run import cli
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    admit_artifact_manifest,
    parse_instant_ns,
)
from trade_rl.evaluation.signal_diagnosis_run.runner import SignalDiagnosisRun

URL = "https://github.com/shuntatsu/trade_rl/issues/1146#issuecomment-1234567890"
PAYLOAD = b"opaque payload bytes, never parsed"
HOUR = 3_600_000_000_000


def _window_bars() -> int:
    data = protocol().data
    span = parse_instant_ns(data.evaluation_end) - parse_instant_ns(
        data.evaluation_start
    )
    return int(span // HOUR + 1)


def _build(
    root: Path, *, bars: int, payload: bytes = PAYLOAD, **overrides: Any
) -> SimpleNamespace:
    """Write an artifact directory and return a stand-in protocol pinning it."""

    data = protocol().data
    scalars: dict[str, Any] = {
        "feature_config_digest": "4" * 64,
        "symbols": list(data.symbols),
        "calendar_kind": "continuous",
        "periods_per_year": 8760,
    }
    body: dict[str, Any] = {
        "arrays": {"timestamps": {"dtype": "<M8[ns]", "shape": [bars]}},
        "arrays_digest": hashlib.sha256(PAYLOAD).hexdigest(),
        "arrays_file": "arrays.npz",
        "dataset": scalars,
        "dataset_id": "1" * 64,
        "schema_version": "market_dataset_artifact_v3",
    }
    body.update(overrides)
    manifest = {**body, "artifact_digest": content_digest(body)}
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "arrays.npz").write_bytes(payload)
    pinned = SimpleNamespace(
        dataset_id="1" * 64,
        artifact_digest=manifest["artifact_digest"],
        arrays_digest=hashlib.sha256(PAYLOAD).hexdigest(),
        feature_config_digest="4" * 64,
        symbols=data.symbols,
        evaluation_start=data.evaluation_start,
        evaluation_end=data.evaluation_end,
    )
    return SimpleNamespace(data=pinned)


class Spies:
    def __init__(self) -> None:
        self.loader_calls = 0
        self.run_calls = 0


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Spies:
    spy = Spies()

    def loader(path: Any) -> Any:
        spy.loader_calls += 1
        return SimpleNamespace(
            manifest_path=Path(path) / "manifest.json", artifact_digest="a" * 64
        ), SimpleNamespace()

    def run(*args: Any, **kwargs: Any) -> SignalDiagnosisRun:
        spy.run_calls += 1
        return SignalDiagnosisRun(
            "NO_SIGNAL", {"verdict.json": {"verdict": "NO_SIGNAL"}}, {}
        )

    monkeypatch.setattr(cli, "load_published_market_dataset_artifact", loader)
    monkeypatch.setattr(cli, "run_signal_diagnosis", run)
    monkeypatch.setattr(cli, "panel_from_dataset", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    return spy


def _argv(tmp_path: Path, url: str = URL) -> list[str]:
    return [
        "--dataset",
        str(tmp_path / "artifact"),
        "--output-dir",
        str(tmp_path / "out"),
        "--g4-authorization-url",
        url,
    ]


def _pin(monkeypatch: pytest.MonkeyPatch, pinned: SimpleNamespace) -> None:
    monkeypatch.setattr(cli, "default_signal_diagnosis_protocol", lambda: pinned)


def test_a_valid_artifact_reaches_the_reader_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies
) -> None:
    _pin(monkeypatch, _build(tmp_path / "artifact", bars=_window_bars()))
    assert cli.main(_argv(tmp_path)) == 0
    assert (spies.loader_calls, spies.run_calls) == (1, 1)
    assert (tmp_path / "out" / "verdict.json").is_file()


@pytest.mark.parametrize("extra_bars", [1, 24 * 40, 24 * 100])
def test_coverage_beyond_the_window_is_refused_without_reading_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies, extra_bars: int
) -> None:
    # One hour, the warm-up span and the sealed window of extra bars.
    _pin(
        monkeypatch,
        _build(tmp_path / "artifact", bars=_window_bars() + extra_bars),
    )
    assert cli.main(_argv(tmp_path)) == 2
    assert (spies.loader_calls, spies.run_calls) == (0, 0)
    assert not (tmp_path / "out").exists()


def test_a_missing_or_malformed_g4_url_never_reaches_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies
) -> None:
    _pin(monkeypatch, _build(tmp_path / "artifact", bars=_window_bars()))
    assert cli.main(_argv(tmp_path, url="https://example.com/x")) == 2
    with pytest.raises(SystemExit):  # the URL is a required argument
        cli.main(_argv(tmp_path)[:4])
    assert (spies.loader_calls, spies.run_calls) == (0, 0)


def test_a_dirty_tree_or_missing_lightgbm_never_reaches_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies
) -> None:
    _pin(monkeypatch, _build(tmp_path / "artifact", bars=_window_bars()))
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: DIRTY)
    assert cli.main(_argv(tmp_path)) == 2
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    assert cli.main(_argv(tmp_path)) == 2
    assert (spies.loader_calls, spies.run_calls) == (0, 0)


@pytest.mark.parametrize(
    "case", ["artifact_pin", "dataset_pin", "payload_bytes", "tampered_manifest"]
)
def test_any_digest_mismatch_never_reaches_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies, case: str
) -> None:
    root = tmp_path / "artifact"
    pinned = _build(root, bars=_window_bars())
    if case == "artifact_pin":
        pinned.data.artifact_digest = "9" * 64
    elif case == "dataset_pin":
        pinned.data.dataset_id = "9" * 64
    elif case == "payload_bytes":
        (root / "arrays.npz").write_bytes(PAYLOAD + b" tampered")
    else:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest["dataset_id"] = "5" * 64  # the manifest no longer hashes to itself
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _pin(monkeypatch, pinned)
    assert cli.main(_argv(tmp_path)) == 2
    assert (spies.loader_calls, spies.run_calls) == (0, 0)


@pytest.mark.parametrize(
    "override",
    [
        {
            "dataset": {
                "feature_config_digest": "4" * 64,
                "symbols": ["A"],
                "calendar_kind": "continuous",
                "periods_per_year": 8760,
            }
        },
        {
            "dataset": {
                "feature_config_digest": "4" * 64,
                "symbols": None,
                "calendar_kind": "session",
                "periods_per_year": 8760,
            }
        },
        {"arrays": {"other": {"shape": [3]}}},
    ],
)
def test_wrong_roster_clock_or_missing_timestamps_metadata_is_refused(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    pinned = _build(tmp_path / "artifact", bars=10)
    body = json.loads((tmp_path / "artifact" / "manifest.json").read_text("utf-8"))
    body.pop("artifact_digest")
    body.update(override)
    manifest = {**body, "artifact_digest": content_digest(body)}
    pinned.data.artifact_digest = manifest["artifact_digest"]
    with pytest.raises(SignalDiagnosisProtocolError):
        admit_artifact_manifest(manifest, pinned)  # type: ignore[arg-type]


def test_the_window_bound_is_exact_and_the_refusal_is_a_forbidden_window_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ok"
    pinned = _build(root, bars=_window_bars())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    admit_artifact_manifest(manifest, pinned)  # type: ignore[arg-type]
    over = tmp_path / "over"
    pinned_over = _build(over, bars=_window_bars() + 1)
    manifest_over = json.loads((over / "manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(ForbiddenEvaluationWindowError, match="before any array"):
        admit_artifact_manifest(manifest_over, pinned_over)  # type: ignore[arg-type]


def test_the_real_pins_refuse_a_foreign_artifact_directory(
    tmp_path: Path, spies: Spies
) -> None:
    (tmp_path / "artifact").mkdir()
    (tmp_path / "artifact" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifact" / "arrays.npz").write_bytes(b"x")
    assert cli.main(_argv(tmp_path)) == 2
    assert spies.loader_calls == 0
