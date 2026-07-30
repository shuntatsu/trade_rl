"""Execute exactly one resumable Binance symbol-triplet training stage."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from trade_rl.workflows.binance_metadata_modes import BinanceMetadataMode
from trade_rl.workflows.binance_symbol_triplet_stage_command import (
    execute_binance_symbol_triplet_stage_command,
)

_RESULT_SCHEMA = "binance_symbol_triplet_stage_command_result_v1"


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid ISO-8601 datetime: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetime must include an explicit timezone")
    return parsed.astimezone(UTC)


def _result_payload(result: Any | None) -> dict[str, object]:
    if result is None:
        return {
            "schema_version": _RESULT_SCHEMA,
            "status": "complete",
        }
    return {
        "completion_digest": result.completion.digest,
        "next_stage_index": result.cursor.next_stage_index,
        "run_id": result.training.run_id,
        "schema_version": _RESULT_SCHEMA,
        "stage_id": result.request.stage_id,
        "stage_index": result.request.stage_index,
        "status": result.training.status,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the current immutable Binance USDS-M symbol-triplet stage, "
            "then advance its Plan/Cursor only after validated final checkpoints."
        )
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--cursor-path", type=Path, required=True)
    parser.add_argument("--base-config-path", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-mode",
        choices=tuple(mode.value for mode in BinanceMetadataMode),
        required=True,
    )
    parser.add_argument("--start-time", type=_parse_utc, required=True)
    parser.add_argument("--end-time", type=_parse_utc, required=True)
    parser.add_argument("--conservative-static-path", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    execute: Callable[..., Any] = execute_binance_symbol_triplet_stage_command,
) -> int:
    arguments = _parser().parse_args(argv)
    result = execute(
        manifest_path=arguments.manifest_path,
        plan_path=arguments.plan_path,
        cursor_path=arguments.cursor_path,
        base_config_path=arguments.base_config_path,
        work_root=arguments.work_root,
        cache_root=arguments.cache_root,
        metadata_mode=arguments.metadata_mode,
        start_time=arguments.start_time,
        end_time=arguments.end_time,
        conservative_static_path=arguments.conservative_static_path,
    )
    print(json.dumps(_result_payload(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
