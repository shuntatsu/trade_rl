"""CLI-level proof of manifest-first refusal and no-follow array file admission.

The in-memory Dataset decoder is replaced by a spy, while file-open probes
observe both ordinary Path.open and the descriptor-safe open_regular_binary.
Invalid manifest metadata must refuse before arrays I/O; digest-mismatched
payload bytes may be read once but must never be decoded. Valid metadata opens
arrays exactly once and decodes the same verified bytes. Synthetic link-swap
and linked-parent witnesses exercise the pre-open no-follow boundary on POSIX;
the portable fallback makes no descriptor-atomic claim. No fixture contains
a real market value: each Dataset payload is an opaque byte stand-in.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
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
        "calendar_kind": "continuous_24_7",
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
        self.loader_args: list[tuple[bytes, bytes]] = []
        self.run_calls = 0


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Spies:
    spy = Spies()

    def loader(manifest_bytes: bytes, arrays_bytes: bytes) -> Any:
        spy.loader_calls += 1
        spy.loader_args.append((manifest_bytes, arrays_bytes))
        return SimpleNamespace(), "a" * 64

    def run(*args: Any, **kwargs: Any) -> SignalDiagnosisRun:
        spy.run_calls += 1
        return SignalDiagnosisRun(
            "NO_SIGNAL", {"verdict.json": {"verdict": "NO_SIGNAL"}}, {}
        )

    monkeypatch.setattr(cli, "load_dataset_from_verified_bytes", loader)
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
    # the decoder is handed exactly the bytes that were verified
    root = tmp_path / "artifact"
    assert spies.loader_args == [
        ((root / "manifest.json").read_bytes(), (root / "arrays.npz").read_bytes())
    ]


def _arrays_opens(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[Path]:
    """Watch the actual payload file open, not merely the Dataset decoder."""

    observed: list[Path] = []
    original_open = Path.open
    arrays_path = root / "arrays.npz"

    def track_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == arrays_path:
            observed.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)

    # The production reader may use the repository's descriptor-guarded opener,
    # which deliberately bypasses Path.open on POSIX. Count both paths, never
    # weaken the zero-open / exactly-once array I/O assertions.
    guarded_open = getattr(cli, "open_regular_binary", None)
    if guarded_open is not None:

        @contextmanager
        def spy_guarded_open(path: Path, *, field: str) -> Any:
            if path == arrays_path:
                observed.append(path)
            with guarded_open(path, field=field) as handle:
                yield handle

        monkeypatch.setattr(cli, "open_regular_binary", spy_guarded_open)

    return observed


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or os.open not in os.supports_dir_fd,
    reason="descriptor-relative O_NOFOLLOW is a POSIX-specific guarantee",
)
@pytest.mark.parametrize("name", ("manifest.json", "arrays.npz"))
def test_preopen_leaf_symlink_swap_cannot_read_foreign_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A swap *after* a leaf check cannot redirect either artifact file read."""

    root = tmp_path / "artifact"
    root.mkdir()
    target = root / name
    target.write_bytes(PAYLOAD)
    foreign = tmp_path / "synthetic-foreign-stand-in"
    foreign.write_bytes(b"synthetic forbidden stand-in; never read")
    original_is_symlink = Path.is_symlink
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        was_symlink = original_is_symlink(path)
        if path == target and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(foreign)
        return was_symlink

    monkeypatch.setattr(Path, "is_symlink", swap_after_check)
    with pytest.raises((SignalDiagnosisProtocolError, OSError, ValueError)):
        cli._read_artifact_file(root, name)
    assert swapped, "the pre-open replacement was not exercised"


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or os.open not in os.supports_dir_fd,
    reason="descriptor-relative O_NOFOLLOW is a POSIX-specific guarantee",
)
def test_parent_swap_after_fd_capture_keeps_original_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renaming the path after a parent fd opens cannot redirect its child."""

    root = tmp_path / "artifact"
    root.mkdir()
    (root / "arrays.npz").write_bytes(PAYLOAD)
    replacement = tmp_path / "synthetic-foreign"
    replacement.mkdir()
    (replacement / "arrays.npz").write_bytes(b"synthetic forbidden stand-in")
    moved = tmp_path / "original-artifact"
    original_open = os.open
    swapped = False

    def race_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is not None and str(path) == root.name and not swapped:
            swapped = True
            root.rename(moved)
            root.symlink_to(replacement, target_is_directory=True)
        return fd

    monkeypatch.setattr(os, "open", race_open)
    observed = cli._read_artifact_file(root, "arrays.npz")
    assert swapped, "the artifact directory was not replaced after fd capture"
    assert root.is_symlink()
    assert observed == PAYLOAD  # the descriptor, not the replaced path, won


def test_symlinked_artifact_directory_cannot_redirect_array_read(
    tmp_path: Path,
) -> None:
    """A regular leaf below a linked parent is not an admissible payload."""

    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "arrays.npz").write_bytes(PAYLOAD)
    alias = tmp_path / "artifact"
    try:
        alias.symlink_to(real_root, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises((SignalDiagnosisProtocolError, OSError, ValueError)):
        cli._read_artifact_file(alias, "arrays.npz")


@pytest.mark.parametrize(
    "case",
    (
        "forbidden_window",
        "malformed_json",
        "tampered_manifest",
        "wrong_artifact_pin",
        "wrong_dataset_pin",
        "wrong_roster",
        "invalid_timestamps_shape",
    ),
)
def test_manifest_refusal_never_opens_arrays_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies, case: str
) -> None:
    """Even opaque FS1/unused bytes must not be opened on a manifest refusal."""

    root = tmp_path / "artifact"
    bars = _window_bars() + 1 if case == "forbidden_window" else _window_bars()
    pinned = _build(root, bars=bars)
    if case in {
        "malformed_json",
        "tampered_manifest",
        "wrong_roster",
        "invalid_timestamps_shape",
    }:
        manifest_path = root / "manifest.json"
        if case == "malformed_json":
            manifest_path.write_bytes(b'{"invalid":')
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if case == "tampered_manifest":
                manifest["dataset_id"] = "5" * 64
            else:
                if case == "wrong_roster":
                    manifest["dataset"]["symbols"] = ["OUTSIDE-ROSTER"]
                else:
                    manifest["arrays"]["timestamps"]["shape"] = ["not-a-bar-count"]
                body = {
                    key: val
                    for key, val in manifest.items()
                    if key != "artifact_digest"
                }
                manifest["artifact_digest"] = content_digest(body)
                pinned.data.artifact_digest = manifest["artifact_digest"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif case == "wrong_artifact_pin":
        pinned.data.artifact_digest = "9" * 64
    elif case == "wrong_dataset_pin":
        pinned.data.dataset_id = "9" * 64
    _pin(monkeypatch, pinned)
    opens = _arrays_opens(monkeypatch, root)

    assert cli.main(_argv(tmp_path)) == 2
    assert opens == [], f"{case} opened arrays.npz before its manifest was admitted"
    assert (spies.loader_calls, spies.run_calls) == (0, 0)
    assert not (tmp_path / "out").exists()


def test_admitted_arrays_are_opened_once_and_decode_from_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies
) -> None:
    root = tmp_path / "artifact"
    _pin(monkeypatch, _build(root, bars=_window_bars()))
    opens = _arrays_opens(monkeypatch, root)

    assert cli.main(_argv(tmp_path)) == 0
    assert opens == [root / "arrays.npz"]
    assert spies.loader_calls == 1
    assert len(spies.loader_args) == 1
    assert spies.loader_args[0][1] == PAYLOAD


def test_payload_digest_mismatch_reads_once_but_never_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies
) -> None:
    root = tmp_path / "artifact"
    _pin(monkeypatch, _build(root, bars=_window_bars()))
    (root / "arrays.npz").write_bytes(PAYLOAD + b" tampered")
    opens = _arrays_opens(monkeypatch, root)

    assert cli.main(_argv(tmp_path)) == 2
    assert opens == [root / "arrays.npz"]
    assert (spies.loader_calls, spies.run_calls) == (0, 0)
    assert not (tmp_path / "out").exists()


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
                "calendar_kind": "continuous_24_7",
                "periods_per_year": 8760,
            }
        },
        {
            "dataset": {
                "feature_config_digest": "4" * 64,
                "symbols": None,
                "calendar_kind": "session_calendar",
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


def _swap_after_admission(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    evil_manifest: bytes,
    evil_arrays: bytes,
) -> None:
    """Replace the artifact files right after the gate has verified their bytes."""

    real = cli.admit_artifact_bytes

    def racing(*args: Any, **kwargs: Any) -> str:
        digest = real(*args, **kwargs)
        (root / "manifest.json").write_bytes(evil_manifest)
        (root / "arrays.npz").write_bytes(evil_arrays)
        return digest

    monkeypatch.setattr(cli, "admit_artifact_bytes", racing)


def test_a_swap_between_verification_and_decode_is_never_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spies: Spies
) -> None:
    root = tmp_path / "artifact"
    _pin(monkeypatch, _build(root, bars=_window_bars()))
    verified = (
        (root / "manifest.json").read_bytes(),
        (root / "arrays.npz").read_bytes(),
    )
    evil = (b'{"dataset_id": "forbidden future data"}', b"swapped payload bytes")
    _swap_after_admission(monkeypatch, root, *evil)
    assert cli.main(_argv(tmp_path)) == 0
    assert spies.loader_args == [verified]  # the swapped bytes never reach the decoder
    assert (root / "arrays.npz").read_bytes() == evil[1]  # the swap did happen


def test_the_real_decoder_returns_the_verified_dataset_after_a_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    from tests.data.test_market_dataset_artifact import _dataset
    from trade_rl.data import write_market_dataset_files

    good = _dataset()
    evil_dataset = dataclasses.replace(
        good,
        open=good.open * 1.01,
        high=good.high * 1.01,
        low=good.low * 1.01,
        close=good.close * 1.01,
        dataset_id="0" * 64,
        identity_payload_json=None,
    ).with_content_identity()
    assert evil_dataset.dataset_id != good.dataset_id
    root, evil_root = tmp_path / "artifact", tmp_path / "evil"
    write_market_dataset_files(root, good)
    write_market_dataset_files(evil_root, evil_dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pinned = SimpleNamespace(
        dataset_id=manifest["dataset_id"],
        artifact_digest=manifest["artifact_digest"],
        arrays_digest=manifest["arrays_digest"],
        feature_config_digest=manifest["dataset"]["feature_config_digest"],
        symbols=tuple(manifest["dataset"]["symbols"]),
        evaluation_start=protocol().data.evaluation_start,
        evaluation_end=protocol().data.evaluation_end,
    )
    monkeypatch.setattr(
        cli, "default_signal_diagnosis_protocol", lambda: SimpleNamespace(data=pinned)
    )
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    seen: list[tuple[Any, dict[str, Any]]] = []

    def panel(dataset: Any, **kwargs: Any) -> Any:
        seen.append((dataset, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(cli, "panel_from_dataset", panel)
    monkeypatch.setattr(
        cli,
        "run_signal_diagnosis",
        lambda *a, **k: SignalDiagnosisRun("NO_SIGNAL", {"verdict.json": {}}, {}),
    )
    _swap_after_admission(
        monkeypatch,
        root,
        (evil_root / "manifest.json").read_bytes(),
        (evil_root / "arrays.npz").read_bytes(),
    )
    assert cli.main(_argv(tmp_path)) == 0
    dataset, kwargs = seen[0]
    assert dataset.dataset_id == good.dataset_id  # the verified bytes were decoded
    assert kwargs["artifact_digest"] == manifest["artifact_digest"]
    assert kwargs["arrays_digest"] == manifest["arrays_digest"]
    swapped = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert swapped["dataset_id"] == evil_dataset.dataset_id  # and the swap did happen
