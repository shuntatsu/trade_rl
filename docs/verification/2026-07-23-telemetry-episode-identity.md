# Telemetry Episode Identity Verification

Date: 2026-07-23

## Scope

This verification covers Draft PR #103, which adds producer-issued episode identity to the existing Live Training environment/current-episode isolation boundary.

Design:

- `docs/superpowers/specs/2026-07-23-telemetry-episode-identity-design.md`

Implementation plan:

- `docs/superpowers/plans/2026-07-23-telemetry-episode-identity.md`

The implementation intentionally does not import the alternate Live Training page or track architecture from stacked Draft PR #84. The existing environment selector and current-episode presentation merged through PR #85 remain in place.

## Implemented contract

- Primary JSONL evidence remains `training_telemetry_v1`.
- `episode_id` is additive and nullable, so historical records remain readable.
- The telemetry sampler owns one active non-negative episode ID per vector environment.
- The same environment episode retains one ID across retained records.
- A terminal or truncated record is emitted with the current ID.
- The next retained record for that environment receives a new ID.
- Reopening an existing telemetry stream starts allocation above the existing sequence/episode range and does not reuse an earlier producer-issued ID.
- Producer fallback state for previous close and previous weights is cleared when the episode ends.
- Studio normalizes the field to `episodeId`.
- Python, Studio, and browser guards reject boolean, negative, and non-integer explicit values.
- Live Training prefers the latest explicit episode ID for the selected environment.
- Historical records with `null` identity continue to use the existing terminal and counter-rollback boundary.
- Stream generation, sequence, sparse-index, process-lock, cursor-reset, and JSONL file-identity contracts are unchanged.

## TDD RED evidence

RED head:

- `75204910275e7034eca0f66325645a6f9699db21`

GitHub Actions CI run:

- `29957910685`
- expected failure in `Verify Studio frontend`

The regression used one environment with explicit episode IDs `41` and `42` while `environmentStep` and `marketIndex` remained monotonic and no terminal marker was retained. The pre-change heuristic returned all four records instead of only the two records from episode `42`.

Ubuntu and Windows compatibility jobs passed at the RED head, confirming that the failure was isolated to the newly introduced frontend contract.

## GREEN implementation

The implementation adds:

- nullable record serialization and strict parsing;
- per-environment producer allocation and post-terminal rotation;
- restart-safe stream-local allocation which remains above existing IDs;
- Studio API normalization;
- TypeScript type and runtime guard support;
- explicit-ID-first current-episode selection;
- legacy-null fallback behavior;
- focused Python, Studio, integration, and frontend tests;
- a measured critical branch-coverage threshold for `trade_rl/rl/training_telemetry.py`.

No production training, checkpoint selection, sealed evaluation, promotion, release, Serving, or execution logic changes.

## Exact-head verification

Final verified implementation-and-test head before this documentation-only commit:

- `9e806f5fd5ceac751fa77e25a1b72b140de1e6a7`

GitHub Actions CI run `29958798560`: success.

- exact-head checkout: passed;
- complete Studio Vitest suite: passed;
- TypeScript type checking: passed;
- Studio production build: passed;
- fixed viewport verification: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1214 passed, 2 skipped, 11 warnings`;
- total coverage: `83.47%`;
- total branch coverage: `4868 / 6916 = 70.39%`;
- sampler branch coverage: `55 / 70 = 78.57% >= 78.5%`;
- indexed telemetry branch coverage remains `76 / 110 = 69.09% >= 69.0%`;
- all critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

PostgreSQL Catalog run `29958798616`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

## Exact-head artifacts

- Pytest diagnostics: ID `8545202616`, digest `sha256:208ccbe2158aed8d08f60c36f92af1e3bcc0e96a5da3728c859211bdb2157db2`;
- architecture diagnostics: ID `8545161810`, digest `sha256:ea475021bd459ac56700141adfe4f5ddec2436309ce8fa3da939fa0ddf5391bc`;
- static diagnostics: ID `8545161251`, digest `sha256:67442ae066d7a71180a7ac3a04c7797c612a0b471b4890ecefc7b9d7d9bf5eeb`;
- training-image evidence: ID `8545154614`, digest `sha256:cb13c90ecaba02860ceffaea829b2fb762afc69f39a875a19649c67c905ba474`;
- Studio layout diagnostics: ID `8545151953`, digest `sha256:6b580c8a33aee11623e1938d85899e172a0ad5b2963e411bb515014134ed6291`;
- Windows compatibility: ID `8545146264`, digest `sha256:4ec32c60a6b5fdbd188cf2d65336b9f2816907d748ad3b4735c5199ffbf2d6da`;
- Ubuntu compatibility: ID `8545143616`, digest `sha256:1756aefbb47f6b54f74fa033aec117708964a597cd3c303154128716f954d520`.

This verification file is documentation-only. The final PR head must pass exact-head CI again before merge.

## Review result

The effective comparison from current `main` contains exactly 16 implementation/test/design files before this verification note. It does not include the alternate page and track implementation from old PR #84. The current Live Training environment and episode-isolation tests remain intact.

The restart regression closes the remaining stream-local allocation ambiguity: a newly constructed sampler reading an existing JSONL continues both sequence and episode identity monotonically instead of reusing the first episode ID.

No critical or important review issue remains.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- No profitability or exchange-equivalent fill claim is introduced.
- Historical telemetry is not rewritten or repaired.
