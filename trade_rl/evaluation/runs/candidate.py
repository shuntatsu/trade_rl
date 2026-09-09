"""Filesystem entry point for one immutable universal lean candidate run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.runs.artifact import (
    PublishedCandidateRun,
    publish_candidate_run,
)
from trade_rl.evaluation.runs.config import (
    load_candidate_run_config,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import execute_candidate_run
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance


def _provenance_execution_identity(payload: dict[str, object]) -> tuple[object, object]:
    return (
        payload.get("implementation_digest"),
        payload.get("runtime_environment_digest"),
    )


def run_candidate_artifact(
    *,
    dataset_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    research_context_digest: str | None = None,
) -> PublishedCandidateRun:
    """Resolve, execute, provenance-check, and publish one candidate suite Run."""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"candidate run destination already exists: {output}")

    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    config = load_candidate_run_config(config_path)
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=config,
    )

    before = build_candidate_run_provenance(
        research_context_digest=research_context_digest,
    )
    result = execute_candidate_run(dataset, spec)
    after = build_candidate_run_provenance(
        research_context_digest=research_context_digest,
    )
    if _provenance_execution_identity(before) != _provenance_execution_identity(after):
        raise RuntimeError("candidate run provenance changed during execution")

    return publish_candidate_run(output, result, after)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the universal lean candidate suite from filesystem artifacts."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--research-context-digest")
    args = parser.parse_args(argv)
    artifact = run_candidate_artifact(
        dataset_root=cast(str, args.dataset),
        config_path=cast(str, args.config),
        output_root=cast(str, args.output),
        research_context_digest=cast(str | None, args.research_context_digest),
    )
    print(artifact.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PublishedCandidateRun", "main", "run_candidate_artifact"]
