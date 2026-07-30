from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "trade_rl/workflows/training_run.py"
    text = path.read_text(encoding="utf-8")
    old = '''    if proposal is not None:\n        proposal.require_execution_evidence_digest(execution_evidence.digest)\n    if run_kind == "research_selected_final":\n        metadata_promotion.require_promotable()\n        if execution_evidence_path is None:\n            raise ExecutionPromotionError(\n                "selected-final training requires explicit execution evidence"\n            )\n        validate_execution_promotion(\n'''
    new = '''    if run_kind == "research_selected_final":\n        metadata_promotion.require_promotable()\n        if execution_evidence_path is None:\n            raise ExecutionPromotionError(\n                "selected-final training requires explicit execution evidence"\n            )\n        assert proposal is not None\n        proposal.require_execution_evidence_digest(execution_evidence.digest)\n        validate_execution_promotion(\n'''
    if text.count(old) != 1:
        raise RuntimeError("selected-final evidence check block changed unexpectedly")
    path.write_text(text.replace(old, new), encoding="utf-8")
    for relative in (
        "scripts/agent_reorder_execution_evidence_check.py",
        ".github/workflows/agent-reorder-execution-evidence.yml",
        ".agent/reorder-execution-evidence-trigger",
    ):
        candidate = ROOT / relative
        if candidate.exists():
            candidate.unlink()


if __name__ == "__main__":
    main()
