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