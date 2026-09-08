Status: Active

# Controlled Experiment Loop v1 — Implementation Plan Amendment

This amendment is normative for the active implementation plan.

## Evidence / analysis identity separation

Task 3 `EvidenceSet` is raw Study evidence. Its fingerprint MUST depend only on the frozen semantic configuration, Study seed roster, research-context digest, and exact semantic identities of the generated Candidate Runs.

`EvidenceSet` MUST NOT contain or digest statistical analysis output.

Use this interface:

```python
@dataclass(frozen=True, slots=True)
class EvidenceSet:
    fingerprint: str
    semantic_config_digest: str
    ppo_seeds: tuple[int, ...]
    run_digests: tuple[tuple[int, str], ...]
    research_context_digest: str
```

Task 4 analysis is a separate immutable derived artifact whose payload binds the source EvidenceSet fingerprint(s). Changing or correcting an analysis implementation must not silently create a new raw-evidence identity.

This separation preserves provenance and permits recomputation/validation of analysis from unchanged evidence.

## Task 6 workflow artifact boundary

Task 6 preserves the same separation on disk. A baseline or candidate workflow node stores the raw EvidenceSet under an `evidence/` directory and stores derived within-EvidenceSet analysis as a sibling `analysis.json`. The analysis file is never inserted into the raw EvidenceSet directory because `load_evidence_set()` treats that directory as an exact immutable raw-evidence artifact.

Use these workflow paths:

```text
baseline/
  evidence/{manifest.json,runs/...}
  analysis.json

experiments/0001/candidate/
  evidence/{manifest.json,runs/...}
  analysis.json
```

`ExperimentComparison` is a public immutable domain contract. It therefore belongs under `trade_rl/evaluation/experiments/contracts/` rather than inside `workflow.py`. The contract binds at least the Study/Experiment identities, baseline and candidate EvidenceSet fingerprints, verification digest, baseline/candidate analysis digests, and the factor-effect comparison payload/digest. `workflow.py` owns only lifecycle orchestration and append-only publication of that contract.

This is a responsibility-boundary clarification only; it does not change the approved Study state machine, factor semantics, evidence identity, or final-data boundary.
