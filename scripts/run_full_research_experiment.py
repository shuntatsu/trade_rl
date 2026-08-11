from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.full_research_state import ResearchPhase, run_research_phase
from trade_rl.workflows.universal_research import (
    FullResearchAlgorithm,
    UniversalFullResearchPlan,
    UniversalResearchManifest,
    validate_full_research_start_inputs,
)
from trade_rl.workflows.universal_research_runtime import UniversalResearchStages


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return None if value is None else str(value)


def _load_manifest(path: Path) -> UniversalResearchManifest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    zero_shot_gate_passed = payload.get("zero_shot_gate_passed", False)
    if not isinstance(zero_shot_gate_passed, bool):
        raise ValueError("zero_shot_gate_passed must be a boolean")
    return UniversalResearchManifest(
        catalog_digest=str(payload["catalog_digest"]),
        split_manifest_digest=str(payload["split_manifest_digest"]),
        normalizer_digest=str(payload["normalizer_digest"]),
        feature_schema_digest=str(payload["feature_schema_digest"]),
        seed_manifest_digest=str(payload["seed_manifest_digest"]),
        architecture_name=UniversalArchitectureName(payload["architecture_name"]),
        checkpoint_digest=str(payload["checkpoint_digest"]),
        cost_model_digest=str(payload["cost_model_digest"]),
        required_pairs=tuple(str(value) for value in payload["required_pairs"]),
        completed_pairs=tuple(str(value) for value in payload["completed_pairs"]),
        bc_teacher_digest=_optional_str(payload, "bc_teacher_digest"),
        software_identity=_optional_str(payload, "software_identity"),
        universe_manifest_digest=_optional_str(payload, "universe_manifest_digest"),
        normalization_fit_scope_digest=_optional_str(
            payload, "normalization_fit_scope_digest"
        ),
        observation_contract_digest=_optional_str(
            payload, "observation_contract_digest"
        ),
        architecture_evidence_digest=_optional_str(
            payload, "architecture_evidence_digest"
        ),
        zero_shot_gate_digest=_optional_str(payload, "zero_shot_gate_digest"),
        zero_shot_gate_passed=zero_shot_gate_passed,
        paired_baseline_digest=_optional_str(payload, "paired_baseline_digest"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enter the universal U6 full-research state machine."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--phase",
        type=ResearchPhase,
        choices=tuple(ResearchPhase),
        default=ResearchPhase.DEVELOP,
    )
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--selection-authorization-digest", default=None)
    parser.add_argument("--fresh-confirmation-digest", default=None)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    validate_full_research_start_inputs(manifest)
    plan = UniversalFullResearchPlan.create(
        selected_architecture=manifest.architecture_name,
        zero_shot_gate_passed=manifest.zero_shot_gate_passed,
        algorithms=tuple(FullResearchAlgorithm),
    )
    stages = UniversalResearchStages(
        manifest=manifest,
        plan=plan,
        selection_authorization_digest=args.selection_authorization_digest,
        fresh_confirmation_digest=args.fresh_confirmation_digest,
    )
    work_root = (
        args.work_root
        if args.work_root is not None
        else args.manifest.parent / "full-research-state" / args.phase.value
    )
    result = run_research_phase(
        phase=args.phase,
        work_root=work_root,
        stages=stages,
    )
    print(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "status": result.status.value,
                "summary_path": str(result.summary_path),
                "summary": dict(result.summary),
            },
            sort_keys=True,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
