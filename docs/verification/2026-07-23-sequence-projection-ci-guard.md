# Sequence Projection CI Guard Verification — 2026-07-23

## Baseline

Current `main` commit `16cccd6c3fe12f38435e58ab652c735b28a7aa72` already contains the `AUD-CI-002` numerical remediation through PR #88 and all later architecture/telemetry changes through PR #103:

- strict float64 historical output/input/parameter-gradient equivalence;
- float32 output, cosine-similarity, and relative-L2 semantic bounds;
- exact zero output and input gradient for a fully unavailable row.

No production change is required by this follow-up.

## Preserved qualification evidence

Expected RED source contract:

```text
Run: 29953598828
Artifact: 8543101802
Digest: sha256:f6af912a2e684196c48171e368ae9739045a0d373cb69f7e4b2dfe0c3adf96b8
```

Focused GREEN:

```text
Run: 29953725692
Artifact: 8543175208
Digest: sha256:be41b7b478e77d765c9b35c64790fc86403b2e1785f7a8b0a2c3351bce33b7ff
```

One-time repeated qualification:

```text
Run: 29953836687
Ubuntu: 100/100, artifact 8543322863
Ubuntu digest: sha256:0169c979f77ba1dd366d19d34896b24b1bb475d240d022e7ca68cad71bfb6af4
Windows: 100/100, artifact 8543351465
Windows digest: sha256:8f8df03dce221201c922f96138acf9856d5b4dd5b0997de4ce97cc2a7b781019
```

## Permanent guard

`tests/architecture/test_sequence_projection_stability.py` requires both stable test functions and forbids the former backend-sensitive test name.

`.github/workflows/sequence-projection-stability.yml`:

- triggers only for the sequence encoder, focused numerical test, AST contract, or workflow itself;
- runs on Ubuntu and Windows;
- repeats the three contracts 10 times;
- uses read-only permissions and pinned Actions;
- checks out the exact pull-request head;
- uploads per-platform logs;
- cancels stale runs for the same ref;
- does not run for unrelated changes.

## Current-main exact-head verification

Implementation-and-documentation head before this final verification-note update:

`5f946599ec121eeb11ffbd98f876ba76ff37bf00`

### Sequence Projection Stability

Run `29959655894`: success.

Ubuntu:

- 10/10 repetitions succeeded;
- artifact `8545470549`;
- digest `sha256:f9c32e49a06f8c2c4dc623c2cd13a40f3c760b5c77b477d21c11e19f1a1d7586`.

Windows:

- 10/10 repetitions succeeded;
- artifact `8545474575`;
- digest `sha256:ba3514e72958d622e0013e7808e4098647fe4aecb3abdf816af6075ad71b6e36`.

### Normal CI

Run `29959655770`: success.

- exact-head checkout: passed;
- Studio Vitest, TypeScript typecheck, production build, and fixed viewport: passed;
- workflow security: passed;
- Ruff and format: passed;
- Mypy: passed;
- Import Linter: passed;
- dead-code report: passed;
- recovery and structured Serving smoke: passed;
- full Pytest: `1215 passed, 2 skipped, 11 warnings`;
- total coverage: `83.47%`;
- total branch coverage: `4868 / 6916 = 70.39%`;
- critical branch-coverage ratchets: passed;
- CLI smoke: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- training-image build and packaged non-root runtime probe: passed.

Normal CI artifacts:

- Pytest diagnostics `8545526302`, digest `sha256:5b8703f9866667d23aa9c7f0f8f8865abc4b25cfe02d4f4f85db5c971cfa9ddf`;
- architecture diagnostics `8545484893`, digest `sha256:96e05340e5a5bb8e69de6c340466d3e1ce8ce2eb25be0790ca1e26f1dfd673b1`;
- static diagnostics `8545484335`, digest `sha256:b7f8f3c442ccb47e540a3fef82e2b81172eb4fedd1f00a1e7c64b8a0b2ef0234`;
- training-image evidence `8545485023`, digest `sha256:1aa97fb8a5e8a86a446b46ce5d855d4e04c67ae1c8ca9fdfe9f2c6cae49f9f35`;
- Studio layout diagnostics `8545475093`, digest `sha256:e53c33bdca0910508b4928cd48f2e177490a9760aa3766a1d3e01c295c5e26f1`;
- Windows compatibility `8545468162`, digest `sha256:9dcc8beac1823a93b60d2eb68aef501b4fb902d10668c6358b2cdad86be2b232`;
- Ubuntu compatibility `8545466411`, digest `sha256:1412e62d4f45b209bd12188d45b48c9496ab4ea6885d56bbfe17421ce55b80ff`.

The PostgreSQL workflow is intentionally not triggered because this five-file CI/test/documentation change does not match its catalog/runtime path filters.

## Review boundary

The effective PR contains only:

```text
.github/workflows/sequence-projection-stability.yml
docs/superpowers/specs/2026-07-23-sequence-projection-ci-guard-design.md
docs/superpowers/plans/2026-07-23-sequence-projection-ci-guard.md
docs/verification/2026-07-23-sequence-projection-ci-guard.md
tests/architecture/test_sequence_projection_stability.py
```

No file under `trade_rl/` changes. No test tolerance, model behavior, dataset behavior, execution behavior, or production threshold changes.

This documentation update creates a new final head. Both normal CI and the path-filtered sequence-projection workflow must pass on that exact head before merge.

## Safety boundary

Production remains `NO-GO`. No model, training, Serving, execution, selection, promotion, release, artifact, or direct exchange behavior changes.
