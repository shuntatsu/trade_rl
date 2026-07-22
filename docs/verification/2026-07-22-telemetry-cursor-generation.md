# Generation-Bound Telemetry Polling Verification

Date: 2026-07-23

Pull request: #100

Branch: `fix/bind-telemetry-cursor-generation-main-20260723`

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

PR #100 was created from current `main` at `c449f424d556bf7e7d4fe1f43625c786c0243dc0`, then synchronized with documentation merge `b51cd9e840da28d987c1056bbe2f7d7532ca932a` before verification.

A one-shot workflow applied only the audited 16-file delta from PR #81 head `2e56fd7b42f2677dc0440ff9ec3cc03a55e5c786` to PR #83 head `b2ac0df43c2b254653bdcd8089d23f37a28c70d9`, deleted itself, and pushed the resulting source commit.

The final pull-request scope contains exactly:

- three design/plan/verification documents;
- one measured coverage-ratchet change;
- five Studio frontend API/type/guard/hook/test changes;
- two Studio backend API/reader changes;
- two Python telemetry record/index changes;
- two Python regression suites;
- one existing Live Training test fixture update.

No temporary workflow or patch file remains. No PR #79, #92, #95, or #98 implementation is repeated.

## Current-main code-head verification

Code head: `962579565f794cdc8bf1cd765a1b7dae0c7147b5`.

GitHub Actions CI run `29957302097`: success.

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
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build and packaged non-root runtime probe: passed.

Exact code-head artifacts:

- Pytest diagnostics `8544641992`, digest `sha256:86904dd68a0043a8056635492fe1044114382dfe29043eb69897b9cc905a8ba0`;
- architecture diagnostics `8544592314`, digest `sha256:ee41e2e5f6c2d1357f49455d22762bd85abae614ff03531ae71f46de727b4ade`;
- static diagnostics `8544591862`, digest `sha256:7e33b8548e5c084c832ccd543edfa0c866fe624bc5d7f5f6e2f7659077412ee4`;
- training-image evidence `8544584480`, digest `sha256:1d312d6009e7085f79a80a9687f4e4ee3f8e7fdd3caea977b1e8f2ff12306b5a`;
- Studio layout diagnostics `8544580962`, digest `sha256:f82205f528ed878c89962117a882cd931d5ef209743d7a4eb94ad4245e655a7d`;
- Windows compatibility `8544571124`, digest `sha256:f45b550583ee7365ad60041c5d93302b661865ce4334bb1e2b652696c80f9063`;
- Ubuntu compatibility `8544565215`, digest `sha256:4d5cd78cd41a62ce18e131fa00859445af53a0415e895a8b2ecfbe25f7808f51`.

PostgreSQL Catalog run `29957301652`: success.

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
- current PR #95 process locking and obsolete-inode safeguards remain present;
- current PR #85 Live Training environment/episode isolation tests remain present;
- public telemetry record schema identifier remains v1;
- no unresolved critical or important review thread remains.

## Safety boundary

- production remains `NO-GO`;
- direct exchange routing is not implemented;
- no profitability or exchange-equivalent fill claim is introduced;
- malformed or truncated telemetry evidence is not automatically repaired;
- no replacement-generation record is returned against an old-generation cursor;
- the index remains rebuildable cache state rather than primary evidence;
- this document commit creates a new exact head which must pass normal CI and PostgreSQL Catalog before merge.
