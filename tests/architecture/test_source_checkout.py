from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.architecture.repository_paths import REPOSITORY_ROOT


def test_source_checkout_resolves_root_python_package_layout() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from trade_rl._source_checkout import source_checkout_root; "
                "print(source_checkout_root())"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve() == REPOSITORY_ROOT
