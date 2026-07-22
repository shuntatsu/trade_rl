# Sequence Projection CI Guard Verification — 2026-07-23

## Baseline

Current `main` commit `464c14669bd2355b6922e6813870030bcf6cc745` already contains the `AUD-CI-002` core remediation:

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
- uploads per-platform logs;
- does not run for unrelated changes.

## Review boundary

The effective PR must contain only:

```text
.github/workflows/sequence-projection-stability.yml
docs/superpowers/specs/2026-07-23-sequence-projection-ci-guard-design.md
docs/superpowers/plans/2026-07-23-sequence-projection-ci-guard.md
docs/verification/2026-07-23-sequence-projection-ci-guard.md
tests/architecture/test_sequence_projection_stability.py
```

No file under `trade_rl/` changes.

## Safety boundary

Production remains `NO-GO`. No model, training, serving, execution, selection, promotion, release, artifact, or direct exchange behavior changes. The PR remains Draft and unmerged.
