from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_research import (
    UniversalResearchManifest,
    validate_full_research_inputs,
)


def _load_manifest(path: Path) -> UniversalResearchManifest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
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
        bc_teacher_digest=(
            None
            if payload.get("bc_teacher_digest") is None
            else str(payload["bc_teacher_digest"])
        ),
        software_identity=(
            None
            if payload.get("software_identity") is None
            else str(payload["software_identity"])
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and enter the universal U6 full-research orchestration lane."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    validate_full_research_inputs(manifest)
    print(
        json.dumps(
            {
                "status": "ready",
                "manifest_digest": manifest.manifest_digest,
                "architecture": manifest.architecture_name.value,
                "pair_count": len(manifest.required_pairs),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
