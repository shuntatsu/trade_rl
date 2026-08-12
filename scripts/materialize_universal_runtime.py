"""Materialize the manifest-bound Universal runtime input tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from trade_rl.integrations.postgres_indicator_artifacts import (
    IndicatorArtifactConnection,
)
from trade_rl.workflows.universal_runtime_preflight import (
    materialize_universal_runtime_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Universal runtime inputs")
    parser.add_argument("--postgres-url", required=True)
    parser.add_argument("--frozen-metadata-root", type=Path, required=True)
    parser.add_argument("--instrument-artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-artifact-root", type=Path, required=True)
    parser.add_argument("--normalizer-artifact-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import psycopg

    with psycopg.connect(args.postgres_url) as connection:
        manifest = materialize_universal_runtime_inputs(
            connection=cast(IndicatorArtifactConnection, connection),
            frozen_metadata_root=args.frozen_metadata_root,
            instrument_artifact_root=args.instrument_artifact_root,
            dataset_artifact_root=args.dataset_artifact_root,
            normalizer_artifact_root=args.normalizer_artifact_root,
            runtime_manifest_path=args.runtime_manifest,
        )
    print(
        json.dumps(
            {
                "cache_id": manifest.cache_id,
                "manifest_digest": manifest.manifest_digest,
                "runtime_manifest": str(args.runtime_manifest.resolve()),
                "shared_complete_row_count": manifest.shared_complete_row_count,
                "statistics_digest": manifest.statistics_digest,
                "test_symbols": list(manifest.test_symbols),
                "train_symbols": list(manifest.train_symbols),
                "validation_symbols": list(manifest.validation_symbols),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
