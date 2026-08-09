from __future__ import annotations

from trade_rl.integrations.nautilus.execution_probe import (
    run_flat_long_flat_execution_probe,
)


def main() -> None:
    result = run_flat_long_flat_execution_probe()
    print(result.digest())


if __name__ == "__main__":
    main()
