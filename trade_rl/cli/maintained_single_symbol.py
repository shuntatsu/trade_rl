"""Fail-closed preflight for maintained training and walk-forward commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trade_rl.data.market import MarketDataset
    from trade_rl.rl.training_run_config import TrainingRunConfig
    from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig

_MAINTAINED_SYMBOLS = ("BTCUSDT",)


def _argument_value(arguments: Sequence[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(arguments):
        if value.startswith(prefix):
            return value[len(prefix) :]
        if value == name:
            if index + 1 >= len(arguments):
                return None
            return arguments[index + 1]
    return None


def _load_dataset(path: Path) -> MarketDataset:
    from trade_rl.data import load_market_dataset_artifact

    return load_market_dataset_artifact(path)


def _load_training_config(path: Path) -> TrainingRunConfig:
    from trade_rl.rl.training_run_config import TrainingRunConfig

    return TrainingRunConfig.from_json(path)


def _load_walk_forward_config(path: Path, *, n_bars: int) -> MarketWalkForwardConfig:
    from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig

    return MarketWalkForwardConfig.from_json(path, n_bars=n_bars)


def _require_dataset(arguments: Sequence[str]) -> MarketDataset | None:
    raw = _argument_value(arguments, "--dataset")
    if raw is None:
        return None
    dataset = _load_dataset(Path(raw))
    if dataset.symbols != _MAINTAINED_SYMBOLS:
        raise ValueError(
            "maintained training requires exactly BTCUSDT as the single symbol"
        )
    return dataset


def _require_training_config(arguments: Sequence[str]) -> None:
    raw = _argument_value(arguments, "--config")
    if raw is None:
        return
    config = _load_training_config(Path(raw))
    if (
        config.action.mode.value != "target_weight"
        or config.action.target_weight_count != 1
        or config.action.names_for_symbols(_MAINTAINED_SYMBOLS)
        != ("target_weight:BTCUSDT",)
    ):
        raise ValueError(
            "maintained training requires exactly one BTCUSDT target-weight action"
        )


def _require_walk_forward_config(arguments: Sequence[str], *, n_bars: int) -> None:
    raw = _argument_value(arguments, "--config")
    if raw is None:
        return
    config = _load_walk_forward_config(Path(raw), n_bars=n_bars)
    if not config.candidates:
        raise ValueError("maintained walk-forward requires at least one candidate")
    for candidate in config.candidates:
        run = candidate.run
        if (
            run.action.mode.value != "target_weight"
            or run.action.target_weight_count != 1
            or run.action.names_for_symbols(_MAINTAINED_SYMBOLS)
            != ("target_weight:BTCUSDT",)
        ):
            raise ValueError(
                "maintained walk-forward requires one BTCUSDT target-weight action"
            )


def require_maintained_single_symbol_cli(arguments: Sequence[str]) -> None:
    """Validate maintained artifact-producing commands before resource allocation."""

    command = tuple(arguments[:2])
    if command not in {("train", "run"), ("walk-forward", "run")}:
        return
    command_arguments = arguments[2:]
    dataset = _require_dataset(command_arguments)
    if dataset is None:
        return
    if command == ("train", "run"):
        _require_training_config(command_arguments)
    else:
        _require_walk_forward_config(command_arguments, n_bars=dataset.n_bars)


__all__ = ["require_maintained_single_symbol_cli"]
