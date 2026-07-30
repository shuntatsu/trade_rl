from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    selection = "trade_rl/workflows/selection_authorization.py"
    replace_once(
        selection,
        '''    resume_checkpoint_digests: tuple[tuple[int, str], ...]\n    schema_version: str = SELECTION_PROPOSAL_SCHEMA\n''',
        '''    resume_checkpoint_digests: tuple[tuple[int, str], ...]\n    execution_evidence_digest: str | None = None\n    schema_version: str = SELECTION_PROPOSAL_SCHEMA\n''',
    )
    replace_once(
        selection,
        '''        require_git_sha(self.git_commit)\n''',
        '''        if self.execution_evidence_digest is not None:\n            require_sha256(\n                self.execution_evidence_digest,\n                field="execution_evidence_digest",\n            )\n        require_git_sha(self.git_commit)\n''',
    )
    replace_once(
        selection,
        '''    def digest_payload(self) -> dict[str, object]:\n        return {\n            "candidate_config_digest": self.candidate_config_digest,\n            "dataset_id": self.dataset_id,\n            "dependency_digest": self.dependency_digest,\n            "execution_sensitivity_digest": self.execution_sensitivity_digest,\n            "git_commit": self.git_commit,\n            "resume_checkpoint_digests": self.resume_checkpoint_digests,\n            "schema_version": self.schema_version,\n            "seeds": self.seeds,\n            "selected_configuration": self.selected_configuration,\n            "walk_forward_run_digest": self.walk_forward_run_digest,\n            "gate_evidence_digest": self.gate_evidence_digest,\n        }\n\n''',
        '''    def digest_payload(self) -> dict[str, object]:\n        payload: dict[str, object] = {\n            "candidate_config_digest": self.candidate_config_digest,\n            "dataset_id": self.dataset_id,\n            "dependency_digest": self.dependency_digest,\n            "execution_sensitivity_digest": self.execution_sensitivity_digest,\n            "git_commit": self.git_commit,\n            "resume_checkpoint_digests": self.resume_checkpoint_digests,\n            "schema_version": self.schema_version,\n            "seeds": self.seeds,\n            "selected_configuration": self.selected_configuration,\n            "walk_forward_run_digest": self.walk_forward_run_digest,\n            "gate_evidence_digest": self.gate_evidence_digest,\n        }\n        if self.execution_evidence_digest is not None:\n            payload["execution_evidence_digest"] = self.execution_evidence_digest\n        return payload\n\n    def require_execution_evidence_digest(self, evidence_digest: str) -> None:\n        """Require selected-final evidence to match the signed proposal exactly."""\n\n        require_sha256(evidence_digest, field="execution_evidence_digest")\n        if self.execution_evidence_digest is None:\n            raise ValueError("selection proposal lacks execution evidence identity")\n        if self.execution_evidence_digest != evidence_digest:\n            raise ValueError("selection proposal execution evidence digest mismatch")\n\n''',
    )
    replace_once(
        selection,
        '''        resume_checkpoint_digests: tuple[tuple[int, str], ...],\n    ) -> SelectionProposal:\n        resolved_seeds = _seeds(seeds)\n        resolved_resume = _resume_digests(resume_checkpoint_digests)\n''',
        '''        resume_checkpoint_digests: tuple[tuple[int, str], ...],\n        execution_evidence_digest: str | None = None,\n    ) -> SelectionProposal:\n        resolved_seeds = _seeds(seeds)\n        resolved_resume = _resume_digests(resume_checkpoint_digests)\n        if execution_evidence_digest is not None:\n            require_sha256(\n                execution_evidence_digest,\n                field="execution_evidence_digest",\n            )\n''',
    )
    replace_once(
        selection,
        '''            "walk_forward_run_digest": walk_forward_run_digest,\n            "gate_evidence_digest": gate_evidence_digest,\n        }\n        return cls(\n''',
        '''            "walk_forward_run_digest": walk_forward_run_digest,\n            "gate_evidence_digest": gate_evidence_digest,\n        }\n        if execution_evidence_digest is not None:\n            payload["execution_evidence_digest"] = execution_evidence_digest\n        return cls(\n''',
    )
    replace_once(
        selection,
        '''            resume_checkpoint_digests=resolved_resume,\n            schema_version=SELECTION_PROPOSAL_SCHEMA,\n''',
        '''            resume_checkpoint_digests=resolved_resume,\n            execution_evidence_digest=execution_evidence_digest,\n            schema_version=SELECTION_PROPOSAL_SCHEMA,\n''',
    )
    replace_once(
        selection,
        '''    def to_mapping(self) -> dict[str, object]:\n        return asdict(self)\n''',
        '''    def to_mapping(self) -> dict[str, object]:\n        payload = asdict(self)\n        if self.execution_evidence_digest is None:\n            payload.pop("execution_evidence_digest")\n        return payload\n''',
    )
    replace_once(
        selection,
        '''                resume_checkpoint_digests=_strict_resume(\n                    raw["resume_checkpoint_digests"]\n                ),\n                schema_version=_string(raw, "schema_version"),\n''',
        '''                resume_checkpoint_digests=_strict_resume(\n                    raw["resume_checkpoint_digests"]\n                ),\n                execution_evidence_digest=_optional_string(\n                    raw,\n                    "execution_evidence_digest",\n                ),\n                schema_version=_string(raw, "schema_version"),\n''',
    )
    replace_once(
        selection,
        '''def _datetime(raw: Mapping[str, object], field: str) -> datetime:\n''',
        '''def _optional_string(raw: Mapping[str, object], field: str) -> str | None:\n    value = raw.get(field)\n    if value is None:\n        return None\n    if not isinstance(value, str):\n        raise ValueError(f"{field} must be a string or null")\n    return value\n\n\ndef _datetime(raw: Mapping[str, object], field: str) -> datetime:\n''',
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''        if (\n            execution_evidence.execution_policy_digest\n            != expected_execution_policy_digest\n        ):\n            raise ValueError("execution evidence policy digest mismatch")\n    if run_kind == "research_selected_final":\n''',
        '''        if (\n            execution_evidence.execution_policy_digest\n            != expected_execution_policy_digest\n        ):\n            raise ValueError("execution evidence policy digest mismatch")\n    if proposal is not None:\n        proposal.require_execution_evidence_digest(execution_evidence.digest)\n    if run_kind == "research_selected_final":\n''',
    )

    e2e = "tests/e2e/test_research_to_serving_v2.py"
    replace_once(
        e2e,
        '''    config = normalize_training_run_config(TrainingRunConfig.from_json(config_path))\n\n    selection_private = Ed25519PrivateKey.from_private_bytes(b"\\x51" * 32)\n''',
        '''    config = normalize_training_run_config(TrainingRunConfig.from_json(config_path))\n    execution_cost = config.environment.execution_cost\n    execution_evidence = ExecutionEvidence(\n        dataset_id=dataset.dataset_id,\n        execution_policy_digest=execution_cost.execution_policy_digest,\n        path_mode=execution_cost.path_mode,\n        processing_bar_volume_capacity=(\n            execution_cost.processing_bar_volume_capacity\n        ),\n        partial_fill_carry=execution_cost.partial_fill_carry,\n        trigger_volume_fractions=execution_cost.trigger_volume_fractions,\n        order_event_count=1,\n        complete_order_evidence=True,\n        sensitivity_path_modes=("optimistic", "neutral", "conservative"),\n    )\n\n    selection_private = Ed25519PrivateKey.from_private_bytes(b"\\x51" * 32)\n''',
    )
    replace_once(
        e2e,
        '''        execution_sensitivity_digest="3" * 64,\n        dataset_id=dataset.dataset_id,\n''',
        '''        execution_sensitivity_digest="3" * 64,\n        execution_evidence_digest=execution_evidence.digest,\n        dataset_id=dataset.dataset_id,\n''',
    )
    replace_once(
        e2e,
        '''    execution_evidence_path = tmp_path / "execution-evidence.json"\n    execution_cost = config.environment.execution_cost\n    write_execution_evidence(\n        execution_evidence_path,\n        ExecutionEvidence(\n            dataset_id=dataset.dataset_id,\n            execution_policy_digest=execution_cost.execution_policy_digest,\n            path_mode=execution_cost.path_mode,\n            processing_bar_volume_capacity=(\n                execution_cost.processing_bar_volume_capacity\n            ),\n            partial_fill_carry=execution_cost.partial_fill_carry,\n            trigger_volume_fractions=execution_cost.trigger_volume_fractions,\n            order_event_count=1,\n            complete_order_evidence=True,\n            sensitivity_path_modes=("optimistic", "neutral", "conservative"),\n        ),\n    )\n''',
        '''    execution_evidence_path = tmp_path / "execution-evidence.json"\n    write_execution_evidence(execution_evidence_path, execution_evidence)\n''',
    )

    tests = "tests/workflows/test_selection_authorization.py"
    replace_once(
        tests,
        '''def test_selection_loader_rejects_seed_type_coercion(tmp_path: Path) -> None:\n''',
        '''def test_selection_proposal_binds_exact_execution_evidence() -> None:\n    proposal = SelectionProposal.create(\n        walk_forward_run_digest="a" * 64,\n        gate_evidence_digest="b" * 64,\n        execution_sensitivity_digest="c" * 64,\n        execution_evidence_digest="9" * 64,\n        dataset_id="d" * 64,\n        selected_configuration="ppo-15m-target",\n        candidate_config_digest="e" * 64,\n        seeds=(0, 1, 2),\n        git_commit="f" * 40,\n        dependency_digest="1" * 64,\n        resume_checkpoint_digests=(),\n    )\n\n    proposal.require_execution_evidence_digest("9" * 64)\n    with pytest.raises(ValueError, match="execution evidence digest mismatch"):\n        proposal.require_execution_evidence_digest("8" * 64)\n\n\ndef test_selection_proposal_without_execution_identity_fails_closed() -> None:\n    with pytest.raises(ValueError, match="lacks execution evidence"):\n        _proposal().require_execution_evidence_digest("9" * 64)\n\n\ndef test_selection_loader_rejects_seed_type_coercion(tmp_path: Path) -> None:\n''',
    )

    for relative in (
        "scripts/agent_bind_execution_evidence.py",
        ".github/workflows/agent-bind-execution-evidence.yml",
        ".agent/bind-execution-evidence-trigger",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
