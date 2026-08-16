# Causal Alpha V3 Signal Forensics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic read-only analyzer for persisted Causal Alpha V3 Signal V2 artifacts that explains fit-, episode-, and symbol-level instability without refitting or weakening any gate.

**Architecture:** Add one workflow module that strictly loads/validates a V3 run root and produces an immutable forensic report, plus one thin `scripts/` CLI that writes canonical JSON. Reuse current Signal V2/run-manifest/config parsers as schema authorities. Do not touch `trade_rl/reporting/*` because open PR #410 owns general run reporting.

**Tech Stack:** Python 3.12, standard library statistics/filesystem/json, existing `trade_rl` content digests and V3 contracts, pytest, Ruff, Mypy, import-linter, vulture.

## Global Constraints

- Source run artifacts are read-only.
- No data rebuild, model refit, environment replay, threshold change, bootstrap change, candidate re-ranking, or promotion behavior.
- One chronological `(contract_start, contract_stop)` interval remains one independent episode regardless of symbol count.
- Missing underlying 24h/72h predictions or model coefficients must be reported unavailable rather than inferred from digests.
- Output schema is `causal_alpha_v3_signal_forensics_v1`, deterministic, content-addressed, research-only, and `promotion_eligible=false`.
- Strict identity/schema/digest/path validation is fail-closed.

---

### Task 1: Define RED forensic contract tests

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py`
- Design: `docs/implementation-plans/specs/2026-08-16-causal-alpha-v3-signal-forensics-design.md`

**Interfaces:**
- Consumes existing `CausalAlphaV3RunManifestV2`, `CausalAlphaV3ResearchConfig`, `CausalAlphaV3SignalScopeMetric`, `signal_scope_metric_from_payload`.
- Produces desired public API: `load_causal_alpha_v3_signal_forensics(root: Path) -> CausalAlphaV3SignalForensicsReport`.

- [ ] **Step 1: Write fixture helpers that create a complete synthetic V2 run root**

Create a two-symbol, two-fit, four-episode run. Build real V3 dataclasses and serialize their current payloads. Each episode cluster must share one `fit_digest` across symbols, while consecutive episode clusters use different fit digests. Create `signal/rejection.json` whose evidence `metric_digests` references the serialized leaves.

- [ ] **Step 2: Write a failing aggregation test**

Assert the wished-for report exposes:

```python
report.schema_version == "causal_alpha_v3_signal_forensics_v1"
report.raw_scope_count == 16
report.independent_episode_count == 4
report.promotion_eligible is False
```

For the stronger synthetic fit, assert exact fit summary raw means, episode means, early/late trend, slope sign, fit-digest unique/transition counts, and candidate/ridge metadata. Assert each episode uses cross-symbol means and each symbol summary uses only that symbol's four rows.

- [ ] **Step 3: Write a failing paired-fit comparison test**

For two fit configs with common episode intervals, assert exact mean episode-level deltas and win fractions for rank IC, spread, and direction-accuracy excess.

- [ ] **Step 4: Write failing truthfulness tests for unavailable diagnostics**

Assert the fixed report entries mark these analyses unavailable with non-empty reasons: `horizon_24h_vs_72h`, `coefficient_cosine_similarity`, `coefficient_sign_flip_rate`, `prediction_distribution`, `residual_rmse_by_episode`.

- [ ] **Step 5: Write failing fail-closed tests**

Cover at least:

```text
corrupt Signal leaf artifact_digest
wrong run_manifest_digest
wrong signal record path
unknown symbol
unknown fit config
missing symbol from one chronological cluster
mixed fit_digest inside one chronological cluster
rejection metric_digests inconsistent with loaded leaves
rejection outer artifact_digest mismatch
```

Each must raise `ValueError` (or `FileNotFoundError` for required missing files) before a report is returned.

- [ ] **Step 6: Verify RED on CI**

Open a Draft PR with tests/spec/plan but without the production forensics module. Expected pytest failure is import/collection failure because `trade_rl.workflows.universal_causal_alpha_v3_signal_forensics` does not exist. Static checks unrelated to that import should remain clean. Record exact RED head/run in the PR body.

---

### Task 2: Implement strict loading and aggregation

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py`

**Interfaces:**
- `load_causal_alpha_v3_signal_forensics(root: Path) -> CausalAlphaV3SignalForensicsReport`
- `CausalAlphaV3SignalForensicsReport.to_payload(include_digest: bool = True) -> dict[str, object]`

- [ ] **Step 1: Implement source contract loading**

Load `run-manifest.json` with `CausalAlphaV3RunManifestV2.from_payload`; load `authored-config.json` with `CausalAlphaV3ResearchConfig.from_mapping`; require `config.digest == manifest.config_digest`; require `signal/records` exists and contains records.

- [ ] **Step 2: Implement strict Signal leaf loading**

Parse every JSON leaf with `signal_scope_metric_from_payload`. Require run identity, allowed manifest symbol, allowed authored fit digest, exact canonical path `signal/records/<fit>/<symbol>/<episode>.json`, and unique metric identity.

- [ ] **Step 3: Implement chronological cluster validation**

For each fit, group by `metric.cluster_identity`; require the cluster symbol set exactly equals `manifest.train_symbols`, one metric per symbol, one shared `fit_digest`, and consistent episode ordering. Do not use raw symbol copies as independent episode count.

- [ ] **Step 4: Implement numeric summary helpers**

Use deterministic population statistics:

```python
count, mean, pstdev, min, max, negative_fraction
```

Direction metric is always `direction_accuracy - 0.5`. Trend uses chronological cross-symbol episode means. For `n > 1`, split at `n // 2`; early is the first half and late is the remainder. Least-squares slope uses episode ordinal `0..n-1`; for one episode slope is `0.0` and early/late are that value.

- [ ] **Step 5: Implement fit, episode, symbol, and paired-fit summaries**

Fit summaries include candidate names and one ridge strength from authored config. Episode summaries include cross-symbol means, negative-symbol counts, and cohort sample totals/means. Symbol summaries include per-symbol distributions. Paired comparisons are computed only from common chronological episode identities and report mean deltas plus left-fit win fractions.

- [ ] **Step 6: Implement optional rejection validation**

If `signal/rejection.json` exists, require exact outer schema/safety fields and verify its content digest. For every fit result with evidence, require the run digest, raw/independent counts, pass flag, and `metric_digests` set/length to agree with the loaded leaves for that fit. Rejection evidence is validation/context only; do not recalculate gate thresholds.

- [ ] **Step 7: Implement immutable report contract**

Bind manifest/config/rejection identities, sorted summaries, unavailable-analysis reasons, `research_only=true`, `promotion_eligible=false`, and schema version into the report content digest. Repeated loads of unchanged bytes must produce equal payload and digest.

- [ ] **Step 8: Run targeted tests**

Run:

```bash
pytest -q tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py
ruff check trade_rl/workflows/universal_causal_alpha_v3_signal_forensics.py tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py
ruff format --check trade_rl/workflows/universal_causal_alpha_v3_signal_forensics.py tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py
mypy trade_rl/workflows/universal_causal_alpha_v3_signal_forensics.py
```

Expected: all targeted checks pass.

---

### Task 3: Add read-only CLI and source-mutation regression

**Files:**
- Create: `scripts/analyze_universal_causal_alpha_v3_signal.py`
- Create: `tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py`

**Interfaces:**
- CLI positional: `run_root: Path`
- CLI optional: `--output PATH`
- stdout/output: canonical compact sorted JSON + newline.

- [ ] **Step 1: Write CLI RED tests**

Import the script via `importlib.util.spec_from_file_location`. Assert no-output mode writes one JSON object to stdout and output mode writes the same payload to the requested file.

- [ ] **Step 2: Add source immutability test**

Hash/read all files under the synthetic source run root before and after analysis/CLI execution and assert byte-for-byte equality. The only permitted new file is outside the source root when `--output` points elsewhere.

- [ ] **Step 3: Implement the thin CLI**

Follow `scripts/analyze_causal_alpha_checkpoint.py`: parse arguments, call the workflow API, serialize `report.to_payload()` using `ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`, append newline, then print or write to the requested destination.

- [ ] **Step 4: Verify CLI tests and static checks**

Run:

```bash
pytest -q tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py
ruff check scripts/analyze_universal_causal_alpha_v3_signal.py tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py
ruff format --check scripts/analyze_universal_causal_alpha_v3_signal.py tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py
```

Expected: pass.

---

### Task 4: Document usage and verify architecture boundaries

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Design/plan docs above remain authoritative.

- [ ] **Step 1: Add one concise Signal Forensics usage section**

Document the command for an existing V3 run, emphasize no refit/replay, describe fit/episode/symbol and paired-fit outputs, and state that 24h/72h and coefficient diagnostics are unavailable for current V2 leaves.

- [ ] **Step 2: Verify no overlap with PR #410 package**

Search the final diff and ensure no `trade_rl/reporting/*` changes and no general run-stage state abstraction was introduced.

- [ ] **Step 3: Run architecture/dead-code checks**

Run repository import-linter and `vulture trade_rl tests --min-confidence 100`; any new finding in this feature must be fixed or justified before completion.

---

### Task 5: Falsification review and exact-head verification

**Files:** final diff only.

- [ ] **Step 1: Falsification review**

Try to prove the analyzer wrong by checking: symbol copies cannot inflate episode count; different intervals with same local episode index remain distinct; missing symbols fail; fit identity cannot mix; a copied record cannot cross run identity; a renamed/misplaced record fails; corrupt digest fails; rejection cannot reference foreign/missing metric digests; report does not include source path in its content identity; no unsupported 24h/72h/coefficients are synthesized.

- [ ] **Step 2: Self-review final diff**

Review requirement compliance, responsibility boundary, naming, complexity, error handling, deterministic ordering, filesystem side effects, compatibility, documentation, dead code, and unrelated changes.

- [ ] **Step 3: Run full repository quality gates on exact HEAD**

Require the repository CI jobs used by current main/PRs: Ruff, format, Mypy, import architecture, dead-code report, recovery/serving smoke, full pytest + branch coverage, critical coverage, package/uv identity, Windows compatibility, Ubuntu compatibility, training image/runtime probe, Studio checks, and PostgreSQL Catalog workflow where triggered.

- [ ] **Step 4: Inspect exact final artifacts/results**

Record final HEAD, full pytest pass/skip counts, coverage percentage, CI job conclusions, PostgreSQL conclusion, final changed files, and PR mergeability. Do not call the work complete if the exact final HEAD is not what CI verified.

- [ ] **Step 5: Keep PR Draft**

Update PR body with What/Why/Acceptance Criteria/Design/RED-GREEN/Tests/Falsification/CI/Risks. Do not merge without an explicit user request.
