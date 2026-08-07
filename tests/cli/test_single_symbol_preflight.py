from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from trade_rl import cli
from trade_rl.cli import maintained_single_symbol as preflight


def _action(*, count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        mode=SimpleNamespace(value="target_weight"),
        target_weight_count=count,
        names_for_symbols=lambda symbols: tuple(
            f"target_weight:{symbol}" for symbol in symbols
        ),
    )


def test_top_level_cli_rejects_non_btc_dataset_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_load_dataset",
        lambda path: SimpleNamespace(symbols=("ETHUSDT",), n_bars=100),
    )
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = cli.main(
        [
            "train",
            "run",
            "--config",
            "training.json",
            "--dataset",
            "dataset",
            "--output",
            "artifacts",
            "--run-id",
            "run-001",
        ],
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    payload = json.loads(errors.getvalue())
    assert payload["schema"] == "training_run_error_v1"
    assert payload["production_status"] == "NO-GO"
    assert "exactly BTCUSDT" in payload["error"]


def test_top_level_cli_dispatches_valid_btc_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_load_dataset",
        lambda path: SimpleNamespace(symbols=("BTCUSDT",), n_bars=100),
    )
    monkeypatch.setattr(
        preflight,
        "_load_training_config",
        lambda path: SimpleNamespace(action=_action()),
    )
    import trade_rl.cli.extended as extended

    observed: list[list[str]] = []

    def dispatch(arguments, *, stdout, stderr):
        observed.append(list(arguments))
        return 0

    monkeypatch.setattr(extended, "main", dispatch)

    exit_code = cli.main(
        [
            "train",
            "run",
            "--config=training.json",
            "--dataset=dataset",
            "--output",
            "artifacts",
            "--run-id",
            "run-001",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert observed == [
        [
            "train",
            "run",
            "--config=training.json",
            "--dataset=dataset",
            "--output",
            "artifacts",
            "--run-id",
            "run-001",
        ]
    ]


def test_training_preflight_rejects_three_action_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_load_dataset",
        lambda path: SimpleNamespace(symbols=("BTCUSDT",), n_bars=100),
    )
    monkeypatch.setattr(
        preflight,
        "_load_training_config",
        lambda path: SimpleNamespace(action=_action(count=3)),
    )

    with pytest.raises(ValueError, match="one BTCUSDT target-weight action"):
        preflight.require_maintained_single_symbol_cli(
            [
                "train",
                "run",
                "--config",
                "training.json",
                "--dataset",
                "dataset",
            ]
        )


def test_walk_forward_preflight_checks_every_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_load_dataset",
        lambda path: SimpleNamespace(symbols=("BTCUSDT",), n_bars=100),
    )
    candidates = (
        SimpleNamespace(run=SimpleNamespace(action=_action())),
        SimpleNamespace(run=SimpleNamespace(action=_action(count=3))),
    )
    monkeypatch.setattr(
        preflight,
        "_load_walk_forward_config",
        lambda path, *, n_bars: SimpleNamespace(candidates=candidates),
    )

    with pytest.raises(ValueError, match="one BTCUSDT target-weight action"):
        preflight.require_maintained_single_symbol_cli(
            [
                "walk-forward",
                "run",
                "--config",
                "walk-forward.json",
                "--dataset",
                "dataset",
            ]
        )
