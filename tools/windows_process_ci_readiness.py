"""Child-free Windows CI readiness check for the process snapshot provider.

This preflight is a CI environment check, not an ownership observation. It
discards every snapshot and must finish before the native ownership test step
launches any supervised child. In-run process-tree sampling remains unchanged.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable

from tools.tier_d_nonrl.tier_d_process_tree import (
    ProcessUsage,
    windows_process_snapshot,
)

Snapshot = Callable[[], tuple[ProcessUsage, ...]]
Sleep = Callable[[float], None]
Log = Callable[[str], None]

_RETRY_DELAY_SECONDS = 1.0


def ensure_windows_cim_ready(
    *,
    snapshot: Snapshot = windows_process_snapshot,
    sleep: Sleep = time.sleep,
    log: Log = print,
) -> None:
    """Require one successful pre-child snapshot, retrying only one timeout."""
    for attempt in (1, 2):
        try:
            snapshot()
        except subprocess.TimeoutExpired:
            if attempt == 2:
                log(
                    "Windows CIM readiness preflight failed after two "
                    "timeouts; no ownership tests were started."
                )
                raise
            log(
                "Windows CIM readiness preflight timed out before ownership "
                "tests; retrying once."
            )
            sleep(_RETRY_DELAY_SECONDS)
        else:
            log(f"Windows CIM readiness preflight succeeded on attempt {attempt}.")
            return


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("Windows CIM readiness preflight requires Windows.")
    ensure_windows_cim_ready()


if __name__ == "__main__":
    main()
