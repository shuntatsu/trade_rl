"""Command-line entry point for canonical M2 study bootstrap."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a canonical M2 Study bundle.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bootstrap once and emit its stable identity summary as JSON."""

    args = _parser().parse_args(argv)
    result = bootstrap_canonical_m2_study(args.config, args.output)
    print(
        json.dumps(
            {
                "bootstrap_digest": result.bootstrap_digest,
                "config_digest": result.config_digest,
                "dataset_artifact_digest": result.dataset_artifact_digest,
                "dataset_id": result.dataset_id,
                "root": str(result.root),
                "study_digest": result.study_digest,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module execution
    raise SystemExit(main())
