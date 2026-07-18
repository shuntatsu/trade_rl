from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    source = target.read_text(encoding="utf-8")
    if old not in source:
        raise SystemExit(f"postfix marker is missing in {path}: {old[:80]!r}")
    target.write_text(source.replace(old, new, 1), encoding="utf-8")


replace_once(
    "tests/architecture/test_architecture_remediation_v2.py",
    "NOW = datetime(2026, 7, 16, tzinfo=UTC)\n",
    '''NOW = datetime(2026, 7, 16, tzinfo=UTC)\nCONFIRMATION_PRIVATE_KEY = generate_private_key()\nCONFIRMATION_PUBLIC_KEY = PublicVerificationKey(\n    key_id="confirmation-key",\n    public_key=public_key_bytes(CONFIRMATION_PRIVATE_KEY),\n    purpose="fresh-confirmation",\n    valid_from=NOW - timedelta(days=1),\n    valid_until=NOW + timedelta(days=365),\n)\n''',
)
replace_once(
    "tests/serving/helpers.py",
    'TEST_CLOCK = lambda: datetime(2026, 7, 14, tzinfo=UTC)\n',
    'def TEST_CLOCK() -> datetime:\n    return datetime(2026, 7, 14, tzinfo=UTC)\n',
)
replace_once(
    "trade_rl/serving/runtime.py",
    "from typing import Any, Protocol\n",
    "from typing import Any, Callable, Protocol\n",
)
replace_once(
    "trade_rl/serving/runtime.py",
    "from trade_rl.release.attestation import ReleaseAttestation\n",
    "from trade_rl.release.asymmetric import PublicVerificationKey\n"
    "from trade_rl.release.attestation import ReleaseAttestation\n",
)
replace_once(
    "trade_rl/artifacts/provenance.py",
    "    for relative in included_roots:\n        path = repository / relative\n",
    "    for included_root in included_roots:\n        path = repository / included_root\n",
)
replace_once(
    "trade_rl/artifacts/provenance.py",
    '''        relative = path.relative_to(repository).as_posix().encode("utf-8")\n        digest.update(len(relative).to_bytes(8, "big"))\n        digest.update(relative)\n''',
    '''        relative_bytes = path.relative_to(repository).as_posix().encode("utf-8")\n        digest.update(len(relative_bytes).to_bytes(8, "big"))\n        digest.update(relative_bytes)\n''',
)
replace_once(
    "trade_rl/workflows/selection_authorization.py",
    "        return cls(digest=content_digest(payload), **payload)\n",
    '''        return cls(\n            digest=content_digest(payload),\n            walk_forward_run_digest=walk_forward_run_digest,\n            gate_evidence_digest=gate_evidence_digest,\n            execution_sensitivity_digest=execution_sensitivity_digest,\n            dataset_id=dataset_id,\n            selected_configuration=selected_configuration,\n            candidate_config_digest=candidate_config_digest,\n            seeds=resolved_seeds,\n            git_commit=git_commit,\n            dependency_digest=dependency_digest,\n            resume_checkpoint_digests=resolved_resume,\n            schema_version=SELECTION_PROPOSAL_SCHEMA,\n        )\n''',
)
replace_once(
    "trade_rl/serving/bundle.py",
    '''        values = {\n            "dataset_id": dataset_id,\n            "action_schema": action_schema,\n            "observation_schema": observation_schema,\n            "observation_size": observation_size,\n            "environment_digest": environment_digest,\n            "initial_capital": initial_capital,\n            "policy_mode": policy_mode,\n            "policy_digest": policy_digest,\n            "signal_digest": signal_digest,\n            "selection_digest": selection_digest,\n            "training_run_digest": training_run_digest,\n            "run_kind": resolved_run_kind,\n            "selection_proposal_digest": selection_proposal_digest,\n            "selection_authorization_digest": selection_authorization_digest,\n            "walk_forward_run_digest": walk_forward_run_digest,\n            "gate_evidence_digest": gate_evidence_digest,\n            "confirmation_evidence_digest": confirmation_evidence_digest,\n            "files": ordered,\n            "created_at": created_at,\n            "action_size": action_size,\n            "action_names": action_names,\n            "action_spec_digest": action_spec_digest,\n            "alpha_artifact_digest": alpha_artifact_digest,\n            "factor_artifact_digest": factor_artifact_digest,\n            "normalizer_digest": normalizer_digest,\n            "schema_version": SERVING_BUNDLE_SCHEMA,\n        }\n''',
    "",
)
replace_once(
    "trade_rl/serving/bundle.py",
    "        return cls(bundle_digest=content_digest(payload), **values)\n",
    '''        return cls(\n            bundle_digest=content_digest(payload),\n            dataset_id=dataset_id,\n            action_schema=action_schema,\n            observation_schema=observation_schema,\n            observation_size=observation_size,\n            environment_digest=environment_digest,\n            initial_capital=initial_capital,\n            policy_mode=policy_mode,\n            policy_digest=policy_digest,\n            signal_digest=signal_digest,\n            selection_digest=selection_digest,\n            training_run_digest=training_run_digest,\n            run_kind=resolved_run_kind,\n            selection_proposal_digest=selection_proposal_digest,\n            selection_authorization_digest=selection_authorization_digest,\n            walk_forward_run_digest=walk_forward_run_digest,\n            gate_evidence_digest=gate_evidence_digest,\n            confirmation_evidence_digest=confirmation_evidence_digest,\n            files=ordered,\n            created_at=created_at,\n            action_size=action_size,\n            action_names=action_names,\n            action_spec_digest=action_spec_digest,\n            alpha_artifact_digest=alpha_artifact_digest,\n            factor_artifact_digest=factor_artifact_digest,\n            normalizer_digest=normalizer_digest,\n            schema_version=SERVING_BUNDLE_SCHEMA,\n        )\n''',
)
replace_once(
    "trade_rl/serving/package.py",
    '''def _integer(value: object, *, field: str) -> int:\n    if isinstance(value, bool) or not isinstance(value, int):\n        raise ValueError(f"{field} must be an integer")\n    return value\n\n\n''',
    '''def _integer(value: object, *, field: str) -> int:\n    if isinstance(value, bool) or not isinstance(value, int):\n        raise ValueError(f"{field} must be an integer")\n    return value\n\n\ndef _number(value: object, *, field: str) -> float:\n    if isinstance(value, bool) or not isinstance(value, (int, float)):\n        raise ValueError(f"{field} must be numeric")\n    return float(value)\n\n\n''',
)
replace_once(
    "trade_rl/serving/package.py",
    '            initial_capital=float(ensemble_raw.get("initial_capital")),\n',
    '''            initial_capital=_number(\n                ensemble_raw.get("initial_capital"),\n                field="ensemble.initial_capital",\n            ),\n''',
)
