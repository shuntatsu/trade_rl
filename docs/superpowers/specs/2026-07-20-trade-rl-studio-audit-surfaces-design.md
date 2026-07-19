# Trade RL Studio Audit Surfaces Design

## Goal

Complete the three remaining read-only research surfaces in Trade RL Studio: run comparison, evidence inspection, and paper-serving monitoring. The implementation must not add exchange credentials, order routing, order submission, or live-capital mutation.

## Product behavior

### Compare

The user selects two validated run artifacts. The page compares headline metrics, configuration values, fold-level outcomes, cost and turnover diagnostics, and cumulative candidate-versus-baseline wealth. The comparison is derived only from canonical files already declared by each run manifest. Missing metrics are shown as unavailable rather than synthesized.

### Evidence Explorer

The user selects one run and sees an ordered evidence chain: dataset reference, run manifest, training configuration, policy ensemble, selection proposal, selection authorization, walk-forward evidence, gate evidence, confirmation evidence, and serving bundle linkage. Each node reports present, verified, absent, or invalid. File closure and SHA-256 validation remain delegated to the existing artifact validators.

Exploratory runs are not incorrectly marked broken merely because selected-final authorization files are absent. The page distinguishes optional evidence from evidence required by the run kind.

### Serving Monitor

The page inspects the local serving registry and active bundle without activating, replacing, or invoking order-routing components. It validates the active pointer, bundle file closure, bundle identities, release-attestation presence, observation/action schemas, policy identity, and selected-final evidence chain. It displays an optional latest paper inference snapshot from a Studio-owned JSON file when present; absence is an idle state, not an error.

## Architecture

Add focused backend readers under `trade_rl/studio/`:

- `comparison.py` derives typed comparisons from validated run directories.
- `evidence.py` validates and maps evidence nodes without duplicating cryptographic validation.
- `serving_monitor.py` reads the serving registry and optional paper telemetry.

`StudioCatalog` remains responsible for locating safe project-root-relative artifacts. FastAPI exposes read-only endpoints. React consumes typed responses through `studioApi.ts` and renders each surface in a bounded fixed-height workspace.

## API

- `GET /api/studio/compare?left_run_id=...&right_run_id=...`
- `GET /api/studio/runs/{run_id}/evidence`
- `GET /api/studio/serving`

Unknown IDs return 404. Invalid artifacts return structured status records where possible; unsafe paths and malformed requests fail with 400. No endpoint accepts credentials or order instructions.

## Data and validation rules

- Resolve runs only from `StudioSettings.run_roots` and exact catalog IDs.
- Validate run directories with the existing training or walk-forward manifest validators before deriving results.
- Use canonical `walk-forward.json` folds and selected metrics as the authoritative source for comparative performance.
- Use manifest-declared files and existing loaders for digest and closure checks.
- Validate serving bundles with `load_serving_bundle`; do not bypass release requirements to claim an active released state.
- Keep `productionStatus` as `NO-GO` throughout the Studio contracts.

## UI layout

All three pages use the existing no-document-scroll shell.

- Compare: compact run selectors at the top, metric-delta cards, configuration-diff table, and a two-column fold/wealth view.
- Evidence: run selector, bounded evidence rail, detail pane inside the workspace, and file-integrity summary.
- Serving: active-bundle identity cards, validation checklist, and the latest optional paper snapshot panel.

Panels may scroll internally when necessary, but `html`, `body`, and `#root` must remain locked to viewport height.

## Error handling

- Empty repositories render an explicit empty state.
- Invalid runs remain selectable only for diagnosis and show the validator error.
- Metric omissions display `—` and do not become zero.
- A missing serving registry displays `IDLE`.
- A malformed active pointer or bundle displays `INVALID` with the exact validation error.
- Telemetry parsing ignores no data silently; malformed telemetry marks the telemetry section invalid without changing bundle validation.

## Testing

Backend tests cover run resolution, training/walk-forward evidence requirements, comparison math, unsafe identifiers, serving pointer validation, bundle closure, and absent telemetry. Frontend tests cover loading, empty, valid, and invalid states plus run selection. Existing typecheck, build, Python test, coverage, Windows/Ubuntu compatibility, and no-page-scroll checks remain required.
