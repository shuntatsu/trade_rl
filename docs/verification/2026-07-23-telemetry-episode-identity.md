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
- Studio API normalization;
- TypeScript type and runtime guard support;
- explicit-ID-first current-episode selection;
- legacy-null fallback behavior;
- focused Python, Studio, integration, and frontend tests;
- a measured critical branch-coverage threshold for `trade_rl/rl/training_telemetry.py`.

No production training, checkpoint selection, sealed evaluation, promotion, release, Serving, or execution logic changes.

## Exact-head verification

Final verified head before this documentation-only commit:

- `91423b49724bf740fb9c9e60d354d82b7def06a3`

GitHub Actions CI run `29958406026`: success.

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
- full Pytest: `1213 passed, 2 skipped, 11 warnings`;
- total coverage: `83.47%`;
- total branch coverage: `4868 / 6916 = 70.39%`;
- sampler branch coverage: `55 / 70 = 78.57% >= 78.5%`;
- indexed telemetry branch coverage remains `76 / 110 = 69.09% >= 69.0%`;
- all critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

PostgreSQL Catalog run `29958406465`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup and readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

## Exact-head artifacts

- Pytest diagnostics: ID `8545059144`, digest `sha256:8813e4d595ae24db8251b779c94230d6056ebd1b33c732af4d00c7204cf4033d`;
- architecture diagnostics: ID `8545012315`, digest `sha256:1435db92243ebbd517506ac0c2cc7c6ebcba52019f7881888ffc9e0da4416c5e`;
- static diagnostics: ID `8545011652`, digest `sha256:ec2e4f1c6471803bcebe0784c0054017dfd38ad371a24c1445c388b57f7908f1`;
- training-image evidence: ID `8545004825`, digest `sha256:fbf8fb80597d8ece373f4dce4039f3f0b58cc98c385fd039dcda236f61666606`;
- Studio layout diagnostics: ID `8545000826`, digest `sha256:2ccaae7ec835fb3e714f96a3406969a3a68e488759c3bed38909419fc44ef292`;
- Windows compatibility: ID `8544993849`, digest `sha256:dad4657b1d505fa579aceda304775f8c230ffd43a1730d198760b6a40bcd0432`;
- Ubuntu compatibility: ID `8544991093`, digest `sha256:9185352e617f3721bdb9f7a37aaf6cf02dc6dd173a3cc8d178db5f2a1c3f20fb`.

This verification file is documentation-only. The final PR head must pass exact-head CI again before merge.

## Review result

The effective comparison from current `main` contains exactly 16 implementation/test/design files before this verification note. It does not include the alternate page and track implementation from old PR #84. The current Live Training environment and episode-isolation tests remain intact.

No critical or important review issue remains.

## Safety boundary

- Production remains `NO-GO`.
- Direct exchange routing is not implemented.
- No profitability or exchange-equivalent fill claim is introduced.
- Historical telemetry is not rewritten or repaired.
