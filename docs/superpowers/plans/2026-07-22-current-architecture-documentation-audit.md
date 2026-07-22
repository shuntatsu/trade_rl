# Current Architecture Documentation Sync and Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize every maintained user-facing document with the current `main` implementation, add executable documentation contracts, perform an evidence-based post-merge architecture audit, and publish a dated remediation roadmap without mixing behavioral fixes into the documentation PR.

**Architecture:** Treat the maintained documentation as a versioned interface to the implementation. Add one focused pytest contract that detects stale schema names, missing layer declarations, obsolete execution-capacity wording, PostgreSQL storage ambiguity, and Live Training capability ambiguity. Update documents by responsibility, then produce a dated audit report from code inspection and exact-head verification evidence. Historical specs and plans remain immutable.

**Tech Stack:** Markdown, Python 3.12, pytest, pathlib, import-linter, Ruff, Mypy, npm/Vitest/TypeScript/Vite, Docker Compose, PostgreSQL 16, GitHub Actions.


## Post-rebase execution note

The branch was rebased conceptually against `main` commit `6bec98e43599c98fb4b86a1522ab455f5acd396b` before Task 1 execution. PR #78 already completed the compatibility-execution migration, telemetry layer enforcement, indexed strict telemetry parsing, duplicate-stream rejection, canonical JSON unification, PostgreSQL responsibility split, and `ResidualMarketEnv` decomposition. Steps below that describe those items as open findings are superseded by this note. The executable documentation test derives schema constants and the complete layer list from source instead of hard-coding the pre-#78 baseline.

## Global Constraints

- Work only on branch `docs/current-architecture-sync-20260722` until the documentation/audit PR is reviewed.
- Do not modify historical files under `docs/superpowers/specs/` or `docs/superpowers/plans/`, except this implementation plan and the already-approved design for this work.
- Do not change model behavior, execution behavior, release behavior, selection thresholds, or profitability gates in this PR.
- Keep production status, direct exchange routing, and profitability explicitly `NO-GO`.
- Document the maintained observation contract as `baseline_residual_observation_v5`.
- Document the maintained serving candidate as bundle v5.
- Document stateful order intent, latency, eligibility, trigger, shared processing-bar capacity, partial-fill carry, time in force, cancellation/replacement, rejection, expiry, deterministic event evidence, execution-promotion evidence, and replay evidence.
- Do not retain the obsolete universal statement that all next-open capacity always uses the previous completed bar volume.
- Describe PostgreSQL as a metadata/provenance/cache/dependency/lifecycle catalog; immutable numerical and model payloads remain filesystem artifacts.
- Describe Live Training telemetry as exploratory visualization only, never as exchange activity, selection evidence, sealed evaluation, profitability evidence, or production authorization.
- Match the Architecture responsibility order to `.importlinter` exactly, including the enforced `trade_rl.telemetry` layer below artifacts and above domain.
- Every audit finding must use `CONFIRMED`, `RISK`, or `NOT_FOUND`, plus priority `P0`, `P1`, `P2`, or `P3`.
- Record volatile test counts, workflow run IDs, artifact IDs, commit SHAs, and image IDs only in the dated verification report.
- Every commit must remain independently reviewable and pass its focused validation before the next task.

---

## File Responsibility Map

- `tests/test_current_documentation_contract.py`: executable assertions that maintained documents expose the current schema, layer, storage, and capability contracts.
- `README.md`: concise English capability boundary, project entry points, current high-level execution and serving contracts.
- `README.ja.md`: Japanese capability boundary and current responsibility overview.
- `START.md`: minimal reproducible dataset-to-training path and current causal/stateful execution explanation.
- `docs/ARCHITECTURE.md`: authoritative responsibility map, dependency order, data flow, execution, observation, selection, artifact, serving, Studio, catalog, and explicit non-capabilities.
- `docs/RESEARCH_STATUS.md`: dated empirical interpretation, current NO-GO reasons, stateful execution interpretation, and separation between software verification and profitability.
- `docs/BINANCE.md`: Binance ingestion, metadata modes, volume semantics, conservative execution evidence, and limitations.
- `docs/operations/docker-gpu-full-training.md`: actual container phases, evidence boundaries, PostgreSQL/catalog relationship, retries, and non-production interpretation.
- `studio/README.md`: Studio and Live Training semantics, seed/environment/episode identity limits, checkpoint evidence separation, and read-only serving boundary.
- `docs/verification/2026-07-22-post-merge-architecture-audit.md`: exact audit commit, commands, evidence, findings, non-findings, priorities, and remediation PR sequence.

---

### Task 1: Add Executable Documentation Contracts

**Files:**
- Create: `tests/test_current_documentation_contract.py`
- Inspect: `.importlinter`
- Inspect: `trade_rl/rl/observations.py`
- Inspect: `trade_rl/serving/bundle.py`

**Interfaces:**
- Consumes: repository-root Markdown files and `.importlinter` as text.
- Produces: pytest assertions that later document tasks must satisfy.

- [ ] **Step 1: Create the failing documentation contract test**

Create `tests/test_current_documentation_contract.py` with the following complete content:

```python
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "README.ja.md",
    ROOT / "START.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "RESEARCH_STATUS.md",
    ROOT / "docs" / "BINANCE.md",
    ROOT / "docs" / "operations" / "docker-gpu-full-training.md",
    ROOT / "studio" / "README.md",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _constant(path: Path, name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', _text(path), flags=re.MULTILINE)
    assert match is not None, f"missing {name} in {path.relative_to(ROOT)}"
    return match.group(1)


def _configured_layers() -> tuple[str, ...]:
    text = _text(ROOT / ".importlinter")
    pattern = (
        r"\[importlinter:contract:layers\].*?^layers\s*=\s*\n"
        r"(?P<body>(?:    trade_rl\.[^\n]+\n)+)"
    )
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None
    return tuple(line.strip() for line in match.group("body").splitlines())


def test_maintained_documents_exist() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if not path.is_file()
    ]
    assert missing == []


def test_current_schema_contracts_are_documented() -> None:
    observation_schema = _constant(
        ROOT / "trade_rl" / "rl" / "observations.py", "OBSERVATION_SCHEMA"
    )
    bundle_schema = _constant(
        ROOT / "trade_rl" / "serving" / "bundle.py", "SERVING_BUNDLE_SCHEMA"
    )
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    readme = _text(ROOT / "README.md")
    readme_ja = _text(ROOT / "README.ja.md")
    for document in (architecture, readme, readme_ja):
        assert observation_schema in document
        assert bundle_schema in document
    assert (
        "pending-order" in architecture.lower()
        or "pending order" in architecture.lower()
    )
    assert "observation schema v3" not in readme.lower()
    assert "observation schema v3" not in readme_ja.lower()
    assert "observation schema v3" not in architecture.lower()


def test_architecture_layer_order_matches_import_linter() -> None:
    architecture = _text(ROOT / "docs" / "ARCHITECTURE.md")
    configured = _configured_layers()
    assert "trade_rl.telemetry" in configured
    positions = tuple(architecture.index(layer) for layer in configured)
    assert positions == tuple(sorted(positions))
    telemetry_start = architecture.index("trade_rl.telemetry")
    telemetry_context = architecture[telemetry_start : telemetry_start + 360].lower()
    for stale in (
        "outside the enforced",
        "not listed in the enforced",
        "not currently governed",
        "missing enforcement",
    ):
        assert stale not in telemetry_context


def test_obsolete_universal_capacity_statement_is_absent() -> None:
    obsolete_patterns = (
        r"last completed bar(?:'s)? volume as its capacity proxy",
        r"previous completed bar(?:'s)? volume as (?:the|its) capacity proxy",
        r"前バー.*volume.*capacity proxy",
    )
    for path in MAINTAINED_DOCUMENTS:
        text = _text(path)
        for pattern in obsolete_patterns:
            assert re.search(pattern, text, flags=re.IGNORECASE) is None, path


def test_postgres_is_not_described_as_payload_storage() -> None:
    combined = "\n".join(
        _text(path)
        for path in (
            ROOT / "README.md",
            ROOT / "README.ja.md",
            ROOT / "docs" / "ARCHITECTURE.md",
        )
    ).lower()
    for phrase in ("metadata catalog", "filesystem artifact"):
        assert phrase in combined
    forbidden_phrases = (
        "model blob",
        "checkpoint blob",
        "postgresql is the numerical source",
    )
    for forbidden in forbidden_phrases:
        assert forbidden not in combined


def test_live_training_boundary_is_explicit() -> None:
    studio = _text(ROOT / "studio" / "README.md")
    readme = _text(ROOT / "README.md")
    assert "not exchange activity" in readme.lower()
    assert "not model-selection evidence" in readme.lower()
    assert "not sealed evaluation" in readme.lower()
    assert "not profitability evidence" in readme.lower()
    assert "取引所注文ではありません" in studio
    assert "モデル選択" in studio
    assert "Sealed" in studio
    assert "収益性" in studio
    assert "NO-GO" in studio


def test_remediated_findings_are_not_described_as_current() -> None:
    current_documents = "\n".join(
        _text(path)
        for path in (
            ROOT / "README.md",
            ROOT / "README.ja.md",
            ROOT / "docs" / "ARCHITECTURE.md",
            ROOT / "docs" / "RESEARCH_STATUS.md",
            ROOT / "studio" / "README.md",
        )
    ).lower()
    for stale in (
        "telemetry is not yet placed",
        "telemetry` is not yet placed",
        "outside the enforced layer stack",
        "scan the jsonl file from the beginning",
        "coerced with python truthiness",
        "discovery order instead of being rejected",
        "execute_interval` remains a separate compatibility",
        "baseline reward pre-roll currently uses the compatibility execution path",
    ):
        assert stale not in current_documents


def test_internal_markdown_links_resolve() -> None:
    link_pattern = re.compile(
        r"\[[^\]]+\]\((?!https?://|#|mailto:)([^)#]+)(?:#[^)]+)?\)"
    )
    broken: list[str] = []
    for document in MAINTAINED_DOCUMENTS:
        text = _text(document)
        for target in link_pattern.findall(text):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert broken == []
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest -q tests/test_current_documentation_contract.py
```

Expected: failures for stale observation/bundle wording, explicit Live Training boundaries, and findings already remediated by PR #78. The layer test derives the current order from `.importlinter`; a collection or syntax failure is not an acceptable RED state.

- [ ] **Step 3: Commit the RED test**

```bash
git add tests/test_current_documentation_contract.py
git commit -m "test: define maintained documentation contracts"
```

---

### Task 2: Synchronize Project Entry Documents

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`

**Interfaces:**
- Consumes: current constants and behavior from `trade_rl/rl/observations.py`, `trade_rl/serving/bundle.py`, `trade_rl/simulation/orders.py`, `trade_rl/simulation/stateful_execution.py`, `.importlinter`, and `trade_rl/catalog/`.
- Produces: concise English and Japanese entry documents consistent with the detailed contracts.

- [ ] **Step 1: Update `README.md` capability and execution summaries**

Make these exact semantic changes:

```markdown
- Keep the top-level status as research-ready, attested paper-serving-ready, and direct exchange routing NO-GO.
- Replace “Action and environment v3” with a neutral heading such as “Action, observation, and execution contracts”.
- State that the flat observation identity is `baseline_residual_observation_v5` and includes seven pending-order coordinates per symbol.
- State that execution uses explicit persistent orders with latency, eligibility, trigger, shared processing-bar capacity, partial-fill carry, time in force, cancellation/replacement, rejection, expiry, and deterministic audit events.
- State that conservative execution promotion requires execution-policy-bound evidence and that optimistic/neutral paths are diagnostics only.
- Keep candidate bundle v5 and detached release attestation language.
- Add PostgreSQL catalog wording that explicitly keeps dataset arrays, checkpoints, policies, run payloads, and evidence as immutable filesystem artifacts.
- Expand the Live Training warning to say it is not exchange activity, model selection, sealed evaluation, profitability evidence, or release authorization.
- Link the dated architecture audit under Verification after the audit report exists; until Task 7, link only to `docs/ARCHITECTURE.md` and `docs/RESEARCH_STATUS.md`.
```

- [ ] **Step 2: Update `README.ja.md` to the current Japanese contract**

Apply these exact corrections:

```markdown
- Replace candidate bundle v4 with bundle v5.
- Add `learning`, `serving`, `release`, `studio`, `catalog`, and `telemetry` responsibility descriptions without claiming telemetry has an enforced layer.
- Describe observation v5 and the seven pending-order fields.
- Describe stateful execution and conservative promotion evidence.
- State that PostgreSQL stores searchable metadata/provenance/cache/dependency/lifecycle records, not arrays, datasets, checkpoints, models, or evidence payloads.
- State that Live Training BUY/SELL is exploratory weight-change visualization and is not exchange activity, selection, sealed evaluation, profitability evidence, or production authorization.
- Keep all production and profitability claims at NO-GO.
```

- [ ] **Step 3: Run focused documentation contracts**

```bash
uv run pytest -q tests/test_current_documentation_contract.py -k "schema or postgres or live_training"
```

Expected: schema, PostgreSQL, and Live Training tests pass. Layer and obsolete-capacity tests may still fail until later tasks.

- [ ] **Step 4: Commit entry-document synchronization**

```bash
git add README.md README.ja.md
git commit -m "docs: synchronize project capability contracts"
```

---

### Task 3: Synchronize Quickstart and Binance Data Documentation

**Files:**
- Modify: `START.md`
- Modify: `docs/BINANCE.md`

**Interfaces:**
- Consumes: public CLI commands, maintained dataset artifact format, `ExecutionCostConfig`, metadata modes, and stateful execution evidence.
- Produces: reproducible instructions that do not imply old stateless or previous-bar-only execution semantics.

- [ ] **Step 1: Update `START.md`**

Make these exact changes:

```markdown
- Change the Python requirement text from “3.12以上” to “Python 3.12.x” because `pyproject.toml` requires `>=3.12,<3.13`.
- Use `uv sync --extra dev --extra train-sb3` before the PPO quickstart command.
- Retain the deterministic demo-data warning and NO-GO status.
- In the artifact explanation, state that `environment.json` binds action, observation v5, pending-order, reward, risk, and execution-policy identities.
- In the causal contract, explain that target quantity is fixed from decision-time known state, orders become eligible only under configured latency, and realized processing-bar OHLCV is execution-only transition information that is never exposed to the policy before action.
- Explain that unfilled quantity can remain pending according to time in force and that later target changes reconcile/cancel/replace explicit residual orders rather than silently resizing them.
- Add a short “Reading results correctly” subsection separating pipeline success, candidate selection, sealed evaluation, execution promotion, and production release.
```

- [ ] **Step 2: Update `docs/BINANCE.md`**

Make these exact changes:

```markdown
- Keep Spot and USDⓈ-M support and COIN-M rejection.
- Keep `historical_signed`, `frozen_snapshot`, and `conservative_static` limitations.
- Add the distinction between base-quantity, contract-quantity, and quote-notional volume and state that quote-notional volume must not be multiplied by price again.
- Replace any generic post-selection cost-only interpretation with stateful closed-loop execution replay: order lifecycle, shared capacity, partial fills, carry, and later observations all change under stressed rules.
- State that conservative path mode is required for promotion evidence; neutral and optimistic are sensitivity-only.
- State that fixed smoke and metadata success are integrity evidence only, not profitability or release evidence.
```

- [ ] **Step 3: Run quickstart and Binance-focused tests**

```bash
uv run pytest -q \
  tests/examples/test_quickstart.py \
  tests/data/test_binance_public.py \
  tests/data/test_binance_derivatives.py \
  tests/workflows/test_execution_sensitivity.py \
  tests/simulation/test_stateful_execution.py
```

Expected: all selected tests pass. If a listed test path does not exist on the branch, replace it only with the exact existing test file discovered by `find tests -type f | sort | grep -E 'quickstart|binance|execution_sensitivity|stateful_execution'`, and record the resolved paths in the audit report.

- [ ] **Step 4: Run the obsolete-capacity contract**

```bash
uv run pytest -q tests/test_current_documentation_contract.py -k obsolete_universal_capacity
```

Expected: PASS.

- [ ] **Step 5: Commit quickstart and Binance synchronization**

```bash
git add START.md docs/BINANCE.md
git commit -m "docs: update causal data and execution guidance"
```

---

### Task 4: Rewrite the Authoritative Architecture and Research Status

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: `.importlinter`, current schema constants, order lifecycle modules, selection/promotion/replay modules, artifact/catalog/release code, and Studio boundaries.
- Produces: the authoritative present-state architecture contract and empirical interpretation contract.

- [ ] **Step 1: Replace the Architecture responsibility map with the exact enforced order**

Use this exact order and include one sentence per responsibility:

```text
trade_rl.cli
trade_rl.studio
trade_rl.workflows
trade_rl.integrations
trade_rl.serving
trade_rl.learning
trade_rl.rl
trade_rl.risk
trade_rl.simulation
trade_rl.strategies
trade_rl.data
trade_rl.catalog
trade_rl.evaluation
trade_rl.release
trade_rl.artifacts
trade_rl.telemetry
trade_rl.domain
```

Immediately after the list, document that `trade_rl.telemetry` is an enforced standard-library-only diagnostic layer below artifacts and above domain. Record the package-initializer replacement pattern separately if confirmed by the post-remediation audit.

- [ ] **Step 2: Update the Architecture market/execution and observation contracts**

The updated sections must include all of the following:

```markdown
- decision-time fixed order quantity;
- explicit order intent and persistent order-book state;
- latency and eligibility;
- deterministic conservative OHLC path handling;
- market, limit, and stop-market semantics;
- shared per-symbol processing-bar capacity;
- correct base/quote/contract volume conversion;
- partial-fill carry and time in force;
- cancel/replace reconciliation and no self-cross;
- reason-coded rejection/expiry/cancellation;
- deterministic execution event evidence and replay evidence;
- conservative execution promotion evidence;
- `baseline_residual_observation_v5` and seven pending-order fields;
- serving snapshot binding to pending-order state and execution-policy digest.
```

- [ ] **Step 3: Add Catalog, Telemetry, and Studio architecture sections**

Add three focused sections:

```markdown
## Artifact catalog boundary
Filesystem artifacts are canonical. PostgreSQL stores searchable metadata, provenance, dependency edges, cache identities, lifecycle state, and locations. Catalog failure cannot rewrite artifact identity.

## Training telemetry boundary
Telemetry is append-only exploratory visualization data. It is excluded from dataset identity, model selection, sealed evaluation, promotion, bundle approval, and execution. It is an enforced standard-library-only layer; the package-initializer replacement pattern and stream identity remain separate audit questions.

## Studio boundary
Studio invokes maintained workflows and reads validated artifacts. It does not independently rank candidates, open sealed ranges, sign approvals, activate bundles, route exchange orders, or handle private keys.
```

- [ ] **Step 4: Update `docs/RESEARCH_STATUS.md`**

Apply these exact corrections:

```markdown
- Remove the universal previous-completed-bar volume capacity statement.
- Describe stateful order capacity and execution-only processing-bar data causally.
- State that failed promotion evidence forces baseline fallback or blocks selected-final publication.
- Add the latest conservative-order-simulator smoke interpretation: deterministic replay and 3-seed pipeline completion are software/reproducibility evidence; the candidate was not promoted and baseline fallback remained selected.
- Keep archived historical result sections clearly dated and separate from maintained current contracts.
- Separate software verification, research validity, profitability, release eligibility, and direct-exchange capability into distinct status lines.
```

- [ ] **Step 5: Run architecture contracts and import-linter**

```bash
uv run pytest -q tests/test_current_documentation_contract.py -k "architecture or obsolete"
uv run lint-imports
```

Expected: documentation architecture tests pass; Import Linter passes with the existing configured contracts while the telemetry omission remains documented rather than silently modified.

- [ ] **Step 6: Commit authoritative architecture synchronization**

```bash
git add docs/ARCHITECTURE.md docs/RESEARCH_STATUS.md
git commit -m "docs: synchronize architecture and research status"
```

---

### Task 5: Synchronize Docker Operations and Studio Documentation

**Files:**
- Modify: `docs/operations/docker-gpu-full-training.md`
- Modify: `studio/README.md`

**Interfaces:**
- Consumes: `compose.training.yaml`, root Compose services, Studio settings/routes, training telemetry API, checkpoint evidence reader, and current GitHub workflow restrictions.
- Produces: operational guidance consistent with actual containers, artifact boundaries, and Studio evidence semantics.

- [ ] **Step 1: Audit commands in the Docker operations guide before editing**

Run:

```bash
docker compose config >/tmp/trade-rl-compose.txt
docker compose -f compose.training.yaml config >/tmp/trade-rl-training-compose.txt
grep -nE "trainer|postgres|volume|user|entrypoint|command" /tmp/trade-rl-compose.txt /tmp/trade-rl-training-compose.txt
```

Expected: both Compose configurations render successfully. Use the rendered service, volume, user, and command names exactly in the guide.

- [ ] **Step 2: Update `docs/operations/docker-gpu-full-training.md`**

Ensure the guide explicitly documents:

```markdown
- CUDA and non-root preflight;
- develop, train-selected, and finalize phase boundaries;
- persisted waiting states for external selection authorization or fresh confirmation;
- immutable filesystem artifact extraction;
- PostgreSQL catalog as optional metadata indexing, not model/data payload storage;
- exact log/evidence retention before cleanup;
- fresh retry versus resume semantics;
- source commit, lock digest, image identity, and runtime identity capture;
- success as research/software evidence only, never profitability or production approval.
```

- [ ] **Step 3: Update `studio/README.md`**

Add or correct these contracts:

```markdown
- Live Training telemetry is not exchange activity, selection evidence, sealed evaluation, profitability evidence, or production authorization.
- `environment_id` and episode identity are part of the stream semantics; the current UI must not be described as proving a single continuous portfolio when multiple environments or resets are present.
- Checkpoint evidence is read-only and separately validated; Studio never computes ranking or opens sealed ranges.
- Strict indexed JSONL polling and duplicate-stream rejection are documented as remediated; vector-environment and episode isolation are listed as audit follow-ups only when confirmed.
- Serving Monitor remains read-only and has no activation/private-key/order-routing capability.
```

- [ ] **Step 4: Run Studio-focused backend and frontend validation**

```bash
uv run pytest -q tests/telemetry tests/rl/test_training_telemetry.py tests/studio
npm ci --prefix studio --no-audit --no-fund
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Expected: all Python and frontend checks pass. The documentation must not claim stream-isolation behavior that the current tests do not prove.

- [ ] **Step 5: Run internal-link validation**

```bash
uv run pytest -q tests/test_current_documentation_contract.py -k internal_markdown_links
```

Expected: PASS.

- [ ] **Step 6: Commit operations and Studio synchronization**

```bash
git add docs/operations/docker-gpu-full-training.md studio/README.md
git commit -m "docs: align operations and Studio evidence boundaries"
```

---

### Task 6: Perform the Seven-Path Architecture Audit

**Files:**
- Create: `docs/verification/2026-07-22-post-merge-architecture-audit.md`
- Inspect: all code and tests relevant to the seven audit paths defined in the approved design.

**Interfaces:**
- Consumes: exact branch head, source code, tests, configuration, Compose files, CI workflow definitions, and validation output.
- Produces: evidence-backed findings with IDs, status, priority, impact, reproduction, boundary, and independent remediation PR.

- [ ] **Step 1: Create the audit report with the complete fixed structure**

Create the file with these exact headings:

```markdown
# Post-Merge Architecture Audit — 2026-07-22

## 1. Audit target and environment
## 2. Capability boundary
## 3. Responsibility and dependency reality
## 4. Market data and causality
## 5. Orders, execution, and accounting
## 6. Training, selection, and sealed evaluation
## 7. Training-serving parity
## 8. Artifacts, PostgreSQL, and release
## 9. Studio and Live Training
## 10. CI, Docker, and privileged execution
## 11. Findings
## 12. Remediation PR roadmap
## 13. Final judgment
```

Do not insert empty placeholder sections. Populate each section during the corresponding audit steps below before committing.

- [ ] **Step 2: Audit market data and causality**

Inspect and cite exact repository paths for:

```text
trade_rl/data/
trade_rl/rl/market_inputs.py
trade_rl/rl/observations.py
trade_rl/rl/sequence_observations.py
trade_rl/rl/normalization.py
trade_rl/rl/sequence_normalization.py
trade_rl/workflows/market_walk_forward.py
relevant tests under tests/data, tests/rl, tests/workflows
```

Run:

```bash
uv run pytest -q tests/data tests/rl/test_observation_v2.py tests/rl/test_pending_order_observation.py tests/rl/test_sequence_observations.py tests/rl/test_sequence_normalization.py
```

Record whether future values, future tradability, evaluation statistics, or unavailable metadata can reach policy input. Create findings only from reproducible facts.

- [ ] **Step 3: Audit orders, execution, and accounting**

Inspect and cite:

```text
trade_rl/simulation/orders.py
trade_rl/simulation/stateful_execution.py
trade_rl/simulation/bar_path.py
trade_rl/simulation/liquidity.py
trade_rl/simulation/execution.py
trade_rl/simulation/accounting.py
trade_rl/simulation/execution_promotion.py
trade_rl/simulation/replay.py
relevant tests under tests/simulation
```

Run:

```bash
uv run pytest -q tests/simulation
```

Record evidence for fixed decision-time quantity, no overfill, no self-cross, shared capacity, volume conversion, partial carry, terminal behavior, deterministic replay, and promotion evidence.

- [ ] **Step 4: Audit training and evaluation separation**

Inspect and cite:

```text
trade_rl/learning/
trade_rl/rl/training.py
trade_rl/integrations/sb3_training.py
trade_rl/workflows/training_run.py
trade_rl/workflows/market_walk_forward.py
trade_rl/evaluation/
relevant tests under tests/learning, tests/integrations, tests/workflows, tests/evaluation
```

Run:

```bash
uv run pytest -q tests/learning tests/evaluation tests/workflows tests/integrations
```

Record evidence for fold capabilities, fixed seeds, checkpoint selection, baseline fallback, one-shot sealed access, selected-final authorization, execution sensitivity exclusion from selection, and telemetry exclusion.

- [ ] **Step 5: Audit training-serving parity and release**

Inspect and cite:

```text
trade_rl/serving/
trade_rl/release/
trade_rl/integrations/sb3_policy_loader.py
trade_rl/rl/observations.py
trade_rl/rl/sequence_observations.py
relevant tests under tests/serving, tests/release, tests/integrations
```

Run:

```bash
uv run pytest -q tests/serving tests/release tests/integrations/test_sb3_policy_loader.py tests/rl/test_pending_order_observation.py
```

Record exact fail-closed checks for schema, symbol order, normalizer, pending state, execution-policy digest, state monotonicity, action shape/finite/bounds, public key purpose, expiry, and private-key isolation.

- [ ] **Step 6: Audit artifacts, catalog, Studio, CI, and Docker**

Inspect and cite:

```text
trade_rl/artifacts/
trade_rl/catalog/
trade_rl/telemetry/
trade_rl/studio/
studio/src/
.github/workflows/
compose.yaml or docker-compose.yaml present at repository root
compose.training.yaml
Dockerfile and training image files
.importlinter
pyproject.toml
```

Run:

```bash
uv run pytest -q tests/artifacts tests/catalog tests/telemetry tests/studio
uv run lint-imports
uv run python scripts/check_critical_coverage.py coverage.json || true
```

If `scripts/check_critical_coverage.py` requires a fresh coverage file, defer that command to Task 8 after the full pytest run and record that fact. Do not report an absent/stale coverage file as a product defect.

- [ ] **Step 7: Confirm or reject the initial telemetry findings**

Use code plus focused tests to classify these hypotheses:

```text
AUD-TEL-001: records from multiple `environment_id` values or reset episodes can be combined into one misleading Live Training market/equity series.
AUD-TEL-002: strict/indexed telemetry behavior is installed through package-initializer `setattr` replacement instead of direct module exports.
AUD-TEL-003: records from distinct reset episodes lack an explicit episode identity and can be presented as one continuous series.
```

For each item, write one complete finding block:

```markdown
### AUD-TEL-00X — <title>
- Status: CONFIRMED | RISK | NOT_FOUND
- Priority: P0 | P1 | P2 | P3
- Affected responsibilities and files: ...
- Observed fact: ...
- Invariant: ...
- Concrete impact: ...
- Reproduction or missing test: ...
- Recommended boundary: ...
- Independent remediation PR: ...
```

Do not label `AUD-TEL-001` or `AUD-TEL-003` confirmed solely from visual code review; add a minimal reproduction or a source-level invariant proof plus a missing-test statement. `AUD-TEL-002` may be confirmed from the explicit `setattr` replacement in package initializers.

- [ ] **Step 8: Record important negative findings**

Add `NOT_FOUND` entries when evidence supports them for at least:

```text
future policy observation leakage;
decision-time quantity resizing from future eligible open;
quote-notional double price multiplication;
sealed outer data entering selection;
private signing key import into runtime/training paths;
PostgreSQL becoming canonical payload storage;
Studio opening sealed ranges or activating bundles.
```

- [ ] **Step 9: Commit the evidence-backed audit report**

Before commit, verify no empty section and no unsupported “confirmed” statement:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/verification/2026-07-22-post-merge-architecture-audit.md")
text = path.read_text(encoding="utf-8")
for heading in (
    "## 1. Audit target and environment",
    "## 2. Capability boundary",
    "## 3. Responsibility and dependency reality",
    "## 4. Market data and causality",
    "## 5. Orders, execution, and accounting",
    "## 6. Training, selection, and sealed evaluation",
    "## 7. Training-serving parity",
    "## 8. Artifacts, PostgreSQL, and release",
    "## 9. Studio and Live Training",
    "## 10. CI, Docker, and privileged execution",
    "## 11. Findings",
    "## 12. Remediation PR roadmap",
    "## 13. Final judgment",
):
    assert heading in text
assert "TBD" not in text
assert "TODO" not in text
PY

git add docs/verification/2026-07-22-post-merge-architecture-audit.md
git commit -m "docs: record post-merge architecture audit"
```

---

### Task 7: Add Audit Navigation and Close Documentation Contracts

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `studio/README.md`
- Test: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: completed audit report and stable finding IDs.
- Produces: discoverable audit navigation and green maintained-document contracts.

- [ ] **Step 1: Link the dated audit from maintained documents**

Add relative links from:

```text
README.md
README.ja.md
docs/ARCHITECTURE.md
docs/RESEARCH_STATUS.md
studio/README.md
```

Use link text that makes clear the report is dated evidence, not a continuously current status page.

- [ ] **Step 2: Add the remediation roadmap summary**

Where the audit confirmed the initial telemetry items, link their IDs to these independent follow-ups:

```text
Live Training environment/episode stream isolation;
direct telemetry exports that remove package-initializer replacement, if retained as a finding.
```

Do not claim those fixes are implemented in this documentation PR.

- [ ] **Step 3: Run the complete documentation contract**

```bash
uv run pytest -q tests/test_current_documentation_contract.py
```

Expected: all tests pass.

- [ ] **Step 4: Commit audit navigation**

```bash
git add README.md README.ja.md docs/ARCHITECTURE.md docs/RESEARCH_STATUS.md studio/README.md
git commit -m "docs: link current architecture audit"
```

---

### Task 8: Run Full Exact-Head Verification and Finalize the Audit Record

**Files:**
- Modify: `docs/verification/2026-07-22-post-merge-architecture-audit.md`
- Inspect: GitHub Actions results for the exact final documentation head.

**Interfaces:**
- Consumes: all documentation changes, audit findings, repository tests, Studio build, Compose files, and CI artifacts.
- Produces: final exact-head evidence and a reviewable draft PR.

- [ ] **Step 1: Run Python static analysis and complete tests**

```bash
uv sync --extra dev --extra train-sb3 --extra studio --extra postgres
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python scripts/check_critical_coverage.py coverage.json
```

Expected: all commands exit zero and coverage meets both global and critical branch thresholds in `pyproject.toml`.

- [ ] **Step 2: Run complete Studio validation**

```bash
npm ci --prefix studio --no-audit --no-fund
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Expected: all commands exit zero.

- [ ] **Step 3: Validate Compose configuration and PostgreSQL integration**

```bash
docker compose config
docker compose -f compose.training.yaml config
docker compose up -d postgres
uv run trade-rl catalog migrate
uv run trade-rl catalog health
uv run pytest -q -m postgres
docker compose down
```

Expected: both Compose files render, PostgreSQL becomes healthy, migrations succeed, catalog health succeeds, PostgreSQL-marked tests pass, and `docker compose down` preserves the named volume.

- [ ] **Step 4: Record local exact-head evidence in the audit report**

Append a table containing:

```text
commit SHA
Python version
uv lock digest
Ruff result
Format result
Mypy result
Import Linter result
pytest passed/skipped/warnings
branch coverage
critical coverage
Studio test/typecheck/build/layout results
Compose validation
PostgreSQL migration/integration result
unresolved environmental limitations
```

Use actual command output only. Do not predict counts or IDs.

- [ ] **Step 5: Commit the local verification evidence**

```bash
git add docs/verification/2026-07-22-post-merge-architecture-audit.md
git commit -m "docs: record architecture audit verification"
```

- [ ] **Step 6: Push and open a draft documentation/audit PR**

```bash
git push -u origin docs/current-architecture-sync-20260722
gh pr create \
  --base main \
  --head docs/current-architecture-sync-20260722 \
  --draft \
  --title "docs: synchronize current architecture and audit boundaries" \
  --body-file /tmp/trade-rl-architecture-audit-pr.md
```

The PR body file must contain:

```markdown
## Summary
- synchronize maintained documentation with observation v5, bundle v5, stateful execution, execution promotion, replay evidence, Studio, and PostgreSQL catalog boundaries
- add executable documentation contracts
- publish a dated post-merge architecture audit with prioritized independent remediation PRs

## Non-goals
- no model, execution, selection, release, or profitability behavior changes
- no direct exchange routing

## Validation
- exact commands and results are recorded in `docs/verification/2026-07-22-post-merge-architecture-audit.md`
```

- [ ] **Step 7: Verify GitHub Actions on the exact PR head**

For every workflow associated with the exact head, record:

```text
workflow name
run ID
job conclusion
artifact ID and digest when present
```

All required CI, PostgreSQL, Studio, Ubuntu/Windows, and training-image integrity checks must pass. Do not mark the PR ready while a required run is queued, in progress, skipped unexpectedly, cancelled, or failed.

- [ ] **Step 8: Update the audit report with CI evidence and re-run docs contract**

```bash
uv run pytest -q tests/test_current_documentation_contract.py
git add docs/verification/2026-07-22-post-merge-architecture-audit.md
git commit -m "docs: bind audit to exact CI evidence"
git push
```

After the docs-only evidence commit, verify the new exact head again. If the repository reruns all required jobs for docs changes, wait for those exact-head jobs; do not reuse earlier-head results.

- [ ] **Step 9: Self-review the final diff**

```bash
git diff --check main...HEAD
git diff --stat main...HEAD
git log --oneline --decorate main..HEAD
```

Review for:

```text
contradictory capability claims;
stale v3/v4 schema references in maintained documents;
unsupported CONFIRMED findings;
broken links;
volatile CI values outside the dated audit;
behavioral code changes;
historical spec/plan rewrites.
```

Expected: no unwanted behavioral code changes and no unsupported claims.

- [ ] **Step 10: Mark the PR ready for review**

Only after exact-head verification succeeds:

```bash
gh pr ready
```

Do not merge this documentation/audit PR in the same step. Review findings first, then begin each confirmed behavioral remediation as an independent branch and PR.
