# Generation-Bound Telemetry Polling Verification

Date: 2026-07-23

Merged pull request: #99

Merge commit: `7bf93eaa7775903fa9d08b65dd3b77c052313404`

Verified implementation head: `1c96d0c4e85f7bc84027e586db2bbae8bbff2849`

Replacement for stacked Draft PR #83, rebuilt directly on current `main` after the architecture, environment-runtime, Live Training, and process-concurrency remediations were merged independently.

## Scope

This change binds telemetry sequence cursors to an opaque stream generation so Studio cannot silently reuse an old cursor after the JSONL path is replaced, truncated, or re-indexed.

Implemented contracts:

- the rebuildable sidecar index schema is `training_telemetry_index_v2`;
- primary telemetry JSONL evidence remains `training_telemetry_v1`;
- every valid sidecar index contains a canonical lower-case UUID generation;
- normal append and no-growth polling preserve the generation;
- path replacement, truncation, invalid index state, or index loss creates a new generation;
- a stale expected generation returns no records, `next_sequence=0`, and `reset_required=true`;
- unchanged status and event polls do not rewrite or `fsync` the sidecar index;
- the reader captures an identity-verified, size-bounded file snapshot under the per-stream process lock;
- page-body parsing occurs after the append serialization lock is released;
- Studio API responses expose `streamGeneration` and `resetRequired`;
- the endpoint accepts only canonical lower-case UUID generation queries;
- the frontend clears its buffer and sequence cursor before replaying a replacement generation;
- Status and Events responses from different generations are discarded and retried once before records are published;
- the existing connection-state contract, 2,048-record cap, stale-request cancellation, OS file locking, append-only evidence, and fail-closed corruption behavior remain intact.

## Architecture boundary

The generation is cursor/cache identity, not business or evidentiary identity.

- JSONL remains the primary append-only telemetry evidence.
- The sidecar index may be rebuilt and may rotate generation without rewriting JSONL.
- An open snapshot may finish reading its bounded old-generation inode after replacement; the next Status/Events comparison detects the new generation before mixed records are published.
- No generation field is used by training, checkpoint selection, sealed evaluation, promotion, release, Serving, or execution.

## TDD RED evidence

### Python telemetry

RED head: `6407c5496e402dce36c71375a7f64de05e379771`

Workflow run `29915043603`: expected failure.

- artifact: `8527504146`
- digest: `sha256:c6c97b601316a984bfad1092a0f2f2a38a2628e550515d8622ecd60c35871dc6`

The run reproduced missing generation/reset contracts, unchanged-poll index rewrites, and page parsing that held the append process lock. The near-tail bounded-parse contract already passed and was retained.

### Studio backend

Workflow run `29915513524`: expected failure.

- artifact: `8527690750`
- digest: `sha256:1e08ab17ac5482b763f1dc24777d20058f7a5c9c9f5a2670e13b7cc1a8372896`

The endpoint did not expose generation/reset fields, ignored stale generation queries, and did not reject invalid generation syntax.

### Frontend

Workflow run `29916129060`: expected failure.

- artifact: `8527945463`
- digest: `sha256:78b78936309483c08bea9bc37c6b48210860f5456d83a1dd6085495072dd6987`

The hook did not reset and replay a replacement generation, could publish mixed Status/Events generations, and accepted malformed generation/reset values.

## Focused GREEN evidence

Python implementation head `c04f4084375e6e56b91ffe415ed69026fa4d6389` passed workflow run `29915319097`, including Ruff, format, repository-wide Mypy, telemetry tests, spawn process-concurrency tests, and training integration tests.

Studio backend implementation head `ec6f7cdc76bdb355dadcaf6d717ab7d113887ce7` passed workflow run `29915833168`.

Frontend implementation head `69ff12afa4672b810bc4a74b775b49596e753967` passed workflow run `29917535977`, including focused hook tests, complete Vitest, TypeScript, and production build.

Cross-platform workflow run `29917703239` passed on Linux and Windows:

- Linux artifact `8528622708`, digest `sha256:95065be9c178d7942bff925ea464033e33e645729f21505f1fdbad727b695575`;
- Windows artifact `8528621179`, digest `sha256:9d36bfb92bc44ae85a20bedc164450e266566702a9d20375af05ce1dbca844bc`.

Windows exercised native `msvcrt.locking`.

## Original stacked-head verification

Original PR #83 final head: `b2ac0df43c2b254653bdcd8089d23f37a28c70d9`.

- CI run `29918835669`: success;
- PostgreSQL Catalog run `29918835603`: success;
- final Pytest artifact `8529103393`;
- digest `sha256:219adb2987dcd4a6b4caf310113099269bd2c14953a2584e4565311e1cae9873`.

The original branch was not merged because it contained the full unsquashed histories of its dependency PRs.

## Clean current-main reconstruction

PR #99 was created directly from current `main` at `c449f424d556bf7e7d4fe1f43625c786c0243dc0`. During verification, main advanced by documentation-only PR #98 at `b51cd9e840da28d987c1056bbe2f7d7532ca932a`; that change did not overlap the 16-file implementation scope.

The effective pull-request scope contained exactly:

- three design/plan/verification documents;
- one measured coverage-ratchet change;
- five Studio frontend API/type/guard/hook/test changes;
- two Studio backend API/reader changes;
- two Python telemetry record/index changes;
- two Python regression suites;
- one existing Live Training test fixture update.

No temporary workflow or patch file remained. No PR #79, #92, #95, or #98 implementation was repeated.

## Clean exact-head verification

GitHub Actions CI run `29957142356`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript, production build, and fixed viewport: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter architecture contracts: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1206 passed, 2 skipped, 11 warnings`;
- total coverage: `83.45%`;
- total branch coverage: `70.35%`;
- indexed telemetry branch coverage: `76 / 110 = 69.09%`;
- critical threshold: `69.09% >= 69.0%`;
- CLI smoke: passed;
- Ubuntu compatibility and generation/process-concurrency regressions: passed;
- Windows compatibility, native locking, and generation regressions: passed;
- complete training-image build and packaged non-root runtime probe: passed.

Exact-head artifacts:

- Pytest diagnostics `8544561261`, digest `sha256:85df4d375e4c409e709df914e497385c14ffd9428ea20d2d63b8a3be8971fe83`;
- architecture diagnostics `8544516089`, digest `sha256:fa87b745a44a7615edca253764b3054b017650127cf444c5e05dfb23d2b6d975`;
- static diagnostics `8544515628`, digest `sha256:a98315d2d8c74b6c3c3180e59b3a7d40df4e97f2e42c6bb04015b4a56f344d1e`;
- training-image evidence `8544510019`, digest `sha256:62cefb4020eecad1b2d1311dc24c34f4486eb124828696a077c3cc3ddc016596`;
- Studio layout diagnostics `8544504573`, digest `sha256:dcbfc3f54835bb4ea98124a5022f9815589d1007b9d138cc6131edd7e5e6826d`;
- Windows compatibility `8544500680`, digest `sha256:a05554f464e201a4ae35a3e6706b09ff49ec005da86b5307f847d4f580a94b9d`;
- Ubuntu compatibility `8544494654`, digest `sha256:839bec87feacf897364d63f7b69257ec456a5fa7ec2256a431f7aa111374ccc8`.

PostgreSQL Catalog run `29957142448`: success.

- exact-head checkout: passed;
- Compose validation: passed;
- PostgreSQL startup/readiness: passed;
- installation and migration: passed;
- unit and integration tests: passed;
- shutdown and cleanup: passed.

## Review result

- the sidecar generation is validated canonically in Python and at the HTTP boundary;
- replacement, truncation, invalid/lost index, stale cursor, and mixed-generation response paths are covered;
- no-growth polling avoids unnecessary durable index writes;
- page parsing no longer serializes append work after a bounded snapshot is captured;
- PR #95 process locking and obsolete-inode safeguards remain present;
- PR #85 Live Training environment/episode isolation tests remain present;
- public telemetry record schema identifier remains v1;
- no unresolved critical or important review issue remained before merge.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- malformed or truncated telemetry evidence is not automatically repaired;
- no replacement-generation record is returned against an old-generation cursor;
- the index remains rebuildable cache state rather than primary evidence;
- this documentation-only head must pass normal CI before merge.
