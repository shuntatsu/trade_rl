# Current Docs and Agent Routing Cleanup Plan

> **For agentic workers:** use test-first architecture contracts for the final docs tree and `superpowers:verification-before-completion` before any completion claim.

**Goal:** Replace completed migration/spec documentation with a small current-only documentation tree and explicit human/agent routing, while preserving all durable architecture/research contracts and Git history.

**Exact base:** `21efa93ddc6073fbd31c7000461801a699797254` (verified Phase 4A final head).

## Quality Contract

### Objective
- Make the repository answer “what should a human or agent read now?” unambiguously.
- Preserve current architecture/research rules from the lean redesign and completed boundary spec in durable current docs.
- Treat Git history, not an in-tree archive, as the record of completed plans/specs.

### Non-goals
- No production/test behavior change beyond docs-tree architecture tests.
- No Controlled Experiment Loop implementation or design.
- No strategy/model/risk/execution/evaluation semantic change.
- No docs site generator, YAML metadata system, or auto-index framework unless a concrete future need justifies it; this cleanup stays lean.
- No modification to `LICENSE`, `LICENSES/*`, or package licensing metadata.

### Final docs tree

```text
AGENTS.md
docs/
  README.md
  AGENTS.md
  architecture/
    lean-core.md
    package-boundaries.md
  research/
    current-status.md
```

`docs/specs/` and `docs/plans/` are absent at final Phase 4B because there is no active spec/plan after this cleanup. Future active work may recreate them temporarily.

### Acceptance Criteria
1. Root `AGENTS.md` routes agents to `docs/README.md` and `docs/AGENTS.md`.
2. `docs/README.md` is the single human docs index and links all four substantive/current routing docs.
3. `docs/AGENTS.md` defines read order, update matrix, retention policy, folder policy, verification expectations, and points legal work to `LICENSES/`.
4. `docs/architecture/lean-core.md` preserves current lean-core invariants: causal data, one execution/accounting authority, strategy/risk separation, universal symbol-agnostic strategy contract, immutable artifacts, no mandatory DB/UI/teacher path.
5. `docs/architecture/package-boundaries.md` documents the actual post-Phase-4A package ownership and dependency direction.
6. `docs/research/current-status.md` preserves the current research objective, five candidates + three controls, universal fit/evaluation rules, per-symbol evaluation, M1/M2/M3 status, development/final/stress protocol, current candidate CLI, no-winner rule, and explicit no-profitability/no-production claims.
7. Root `README.md` links `docs/README.md` and no longer points at `docs/trade_rl_lean_redesign_20260908.md`.
8. All completed migration plans, inventories, amendments, completed package-boundary specs, and `docs/trade_rl_lean_redesign_20260908.md` are removed after their durable current content is represented.
9. No `docs/history/` or `docs/archive/` is introduced.
10. Final `docs/` contains exactly the five files listed above; no completed plan/spec remains.
11. `LICENSE`, every `LICENSES/*` file, and `pyproject.toml` licensing metadata are byte-identical to the exact base.
12. Final exact HEAD passes Ruff, Format, Mypy, full tests, package identity, and normal PR CI.

### Invariants
- Git history remains sufficient to recover removed plans/specs; deleting them from current tree is not history destruction.
- Legal/provenance records are permanent compliance material and never treated as disposable docs history.
- Current docs describe current code, not the migration journey.
- A future active spec/plan may exist only while independently normative/in progress; once implemented, durable content moves to architecture/research docs and the completed file is deleted.

### Failure Modes
- unique current contract is deleted with the old redesign/spec;
- root README or agent instructions link a removed file;
- package-boundary doc describes a pre-refactor path;
- research status accidentally claims profitability or completion of real-data comparison;
- historical migration detail leaks into current architecture docs and recreates a museum;
- legal/provenance files are modified or deleted;
- docs architecture test is weakened to allow stale extras.

### Risk
- Documentation loss/misdirection: Medium; affects agents/humans and future changes.
- Legal/provenance regression: High; must be independently compared byte-for-byte.
- Runtime software regression: Low because production code is out of scope, but full CI remains mandatory.

### Test Oracle
- exact final docs-file set;
- root README and AGENTS link/routing assertions;
- required agent-policy phrases/targets;
- manual source-to-current-doc preservation crosswalk below;
- base/final SHA comparison for legal files;
- full repository CI.

### Required Test Layers
- Architecture tests for docs tree and routing.
- Static checks through existing Ruff/Format/Mypy CI.
- Full regression tests even though production code is untouched.
- Falsification: exact base/final changed-file review and legal-file blob comparison.

### Quality Gate
- Do not call Phase 4B complete until the final docs set is exact, the content-preservation crosswalk is reviewed, no stale links remain, legal files are identical, full CI is Green on the exact helper-free HEAD, and the PR remains Draft.

## Durable-content crosswalk

The old `docs/trade_rl_lean_redesign_20260908.md` and completed boundary spec may be removed only after these destinations exist:

| Old durable subject | Current destination |
|---|---|
| objective / lean structure / deletion philosophy | `architecture/lean-core.md`, `research/current-status.md` |
| strategy contract / quantity hold | `architecture/lean-core.md` |
| strategy vs risk responsibilities | `architecture/lean-core.md` |
| causal data contract | `architecture/lean-core.md` |
| execution/accounting contract | `architecture/lean-core.md` |
| package ownership / public API policy | `architecture/package-boundaries.md` |
| five candidates + three controls | `research/current-status.md` |
| universal Ridge/LightGBM/PPO fit scope | `research/current-status.md` |
| independent per-symbol evaluation | `research/current-status.md` |
| M1/M2/M3 status | `research/current-status.md` |
| development run command/output | `research/current-status.md` |
| decision/no-winner protocol | `research/current-status.md` |
| final unused-data/stress protocol | `research/current-status.md` |
| docs retention / active plan/spec rules | `docs/README.md`, `docs/AGENTS.md` |
| legal/provenance retention | `docs/AGENTS.md`, `LICENSES/` |

## TDD sequence

1. Add architecture test requiring the exact final docs tree and routing; observe RED while the old docs remain and new docs are absent.
2. Create root/docs agent routing and the three current substantive docs.
3. Update root README to the new docs entry point.
4. Review the crosswalk against the generated current docs.
5. Delete every completed plan/spec/inventory/amendment and the old monolithic redesign, including this implementation plan itself.
6. Run targeted architecture tests, then full CI.
7. Falsify by comparing the exact base/final changed files and legal blob SHAs.
8. Keep Draft; do not merge without explicit authorization.
