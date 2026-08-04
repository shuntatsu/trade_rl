"""Compose the Oracle CUDA adapter with the maintained benchmark CLI."""

from __future__ import annotations

from collections.abc import Sequence

from trade_rl.integrations.oracle_solver import solve_torch_cuda_oracle_batch
from trade_rl.learning.oracle_solver import register_oracle_accelerator_backend
from trade_rl.operations.oracle_teacher_benchmark import main as _benchmark_main


def main(argv: Sequence[str] | None = None) -> int:
    """Register the CUDA adapter explicitly, then run the benchmark CLI."""

    register_oracle_accelerator_backend("cuda", solve_torch_cuda_oracle_batch)
    return _benchmark_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
