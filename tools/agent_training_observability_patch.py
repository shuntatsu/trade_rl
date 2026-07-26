from __future__ import annotations

from pathlib import Path

from agent_observability_backend import apply as apply_backend
from agent_observability_docs import apply as apply_docs
from agent_observability_frontend import apply as apply_frontend
from agent_observability_studio import apply as apply_studio


def main() -> None:
    apply_backend()
    apply_studio()
    apply_frontend()
    apply_docs()
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tools/agent_observability_backend.py",
        "tools/agent_observability_studio.py",
        "tools/agent_observability_frontend.py",
        "tools/agent_observability_docs.py",
    ):
        (root / relative).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
