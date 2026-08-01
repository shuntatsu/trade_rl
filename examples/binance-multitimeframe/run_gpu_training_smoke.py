#!/usr/bin/env python3
"""Compatibility wrapper for the maintained GPU training smoke operation."""

from trade_rl.operations.gpu_training_smoke import (
    _load_torch_runtime,
    _load_training_performance,
    build_parser,
    build_smoke_config,
    main,
    run_gpu_training_smoke,
)

__all__ = [
    "_load_torch_runtime",
    "_load_training_performance",
    "build_parser",
    "build_smoke_config",
    "main",
    "run_gpu_training_smoke",
]


if __name__ == "__main__":
    raise SystemExit(main())
