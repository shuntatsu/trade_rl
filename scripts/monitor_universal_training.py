from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.operations.universal_training_monitor import (
    inspect_universal_training_generation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Universal training evidence")
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--container")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _docker_evidence(
    container: str | None,
) -> tuple[dict[str, object] | None, str | None]:
    if not container:
        return None, None
    inspect = subprocess.run(
        ["docker", "inspect", container, "--format", "{{json .State}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    logs = subprocess.run(
        ["docker", "logs", "--tail", "2000", container],
        check=False,
        capture_output=True,
        text=True,
    )
    state = json.loads(inspect.stdout) if inspect.returncode == 0 else None
    log = logs.stdout + logs.stderr if logs.returncode == 0 else None
    return state, log


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state, log = _docker_evidence(args.container)
    snapshot = inspect_universal_training_generation(
        args.generation_root,
        container_state=state,
        container_log=log,
    )
    payload = snapshot.to_json_dict()
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        args.output_root / "monitor-snapshot.json",
        canonical_json_bytes(payload) + b"\n",
    )
    reward_payload = {
        "inspected_at": snapshot.inspected_at,
        "members": [
            {
                "algorithm": member.algorithm,
                "seed": member.seed,
                "global_step": member.global_step,
                "reward_total": asdict(member.reward_total),
                "reward_growth": asdict(member.reward_growth),
                "drawdown": asdict(member.drawdown),
                "scalar_trends": {
                    tag: asdict(trend) for tag, trend in member.scalar_trends.items()
                },
            }
            for member in snapshot.members
        ],
        "schema_version": "universal_reward_trends_v1",
        "status": snapshot.status,
    }
    atomic_write_bytes(
        args.output_root / "reward-trends.json",
        canonical_json_bytes(reward_payload) + b"\n",
    )
    print(json.dumps(payload, sort_keys=True))
    return {"healthy": 0, "warning": 2, "failed": 3, "incomplete": 4}[snapshot.status]


if __name__ == "__main__":
    raise SystemExit(main())
