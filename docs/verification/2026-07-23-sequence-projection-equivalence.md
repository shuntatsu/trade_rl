# Sequence Projection Equivalence Stability Verification — 2026-07-23

## Scope

This verification closes `AUD-CI-002`, the backend-sensitive numerical flake in:

```text
tests/rl/test_sequence_policy_core.py::
test_projection_after_selection_matches_legacy_outputs_and_gradients
```

The branch starts from PR #84 exact head:

```text
703427cb162694a8b4990fe4e2ef17ea59a77f7a
```

No production module is changed. The maintained `CausalTimeframeEncoder` continues selecting the final available causal state before running the projection network.

## Original failure

The historical test compared two mathematically equivalent float32 graphs element by element:

1. historical path: project every timestep, then select the last available timestep;
2. maintained path: select the last available encoded timestep, then project once.

One prior CI attempt produced a maximum parameter-gradient difference of:

```text
4.112720489501953e-06
```

against an absolute tolerance of `1e-6`. A code-identical rerun passed. The graphs use different matrix shapes, so their float32 reduction order can differ by a few ULPs even though their selection, masking, forward result, and gradient semantics remain equivalent.

## Remediation design

The single assertion was replaced by two complementary contracts.

### Float64 historical equivalence

`test_projection_after_selection_matches_legacy_in_float64` compares:

- output tensors;
- input gradients;
- every named parameter gradient;
- exact zero output for a fully unavailable row;
- exact zero input gradient for a fully unavailable row.

Elementwise tolerance:

```text
rtol = 1e-9
atol = 1e-10
```

This retains a direct historical graph comparison at a precision where backend reduction-order noise is negligible.

### Float32 semantic gradient equivalence

`test_projection_after_selection_preserves_float32_gradient_semantics` verifies:

- output closeness with `rtol=1e-5`, `atol=2e-6`;
- matching gradient shapes, dtypes, and named parameter sets;
- finite gradients;
- cosine similarity of at least `0.999999` for every nonzero gradient pair;
- relative L2 error no greater than `2e-5`;
- exact zero output and input gradient for the fully unavailable row;
- finite nonzero input-gradient flow for selected rows.

The result distinguishes meaningful selection, masking, or autograd drift from backend-level accumulation noise. It does not introduce a global Torch tolerance or modify production behavior.

## TDD evidence

### RED source contract

Exact head:

```text
0d5852b1068e7c0a284b9e8bf11132942b0d20e5
```

GitHub Actions run:

```text
29953598828
```

The contract failed as intended because both stable test names were absent and the old backend-sensitive test still existed.

Artifact:

```text
ID: 8543101802
Digest: sha256:f6af912a2e684196c48171e368ae9739045a0d373cb69f7e4b2dfe0c3adf96b8
```

### Focused GREEN

GitHub Actions run:

```text
29953725692
```

Successful checks:

- Ruff;
- repository-wide Mypy;
- both focused projection-equivalence tests;
- stable source contract.

Verified implementation artifact:

```text
ID: 8543175208
Digest: sha256:be41b7b478e77d765c9b35c64790fc86403b2e1785f7a8b0a2c3351bce33b7ff
```

The verified test commit was:

```text
c355519e9fc46838218d9eef661e7aace38f5bde
```

## Repeated cross-platform stability

GitHub Actions run:

```text
29953836687
```

Each operating system executed both focused tests 100 times and stopped on the first failure.

### Ubuntu

```text
Result: 100/100 successful repetitions
Artifact ID: 8543322863
Digest: sha256:0169c979f77ba1dd366d19d34896b24b1bb475d240d022e7ca68cad71bfb6af4
```

### Windows

```text
Result: 100/100 successful repetitions
Artifact ID: 8543351465
Digest: sha256:8f8df03dce221201c922f96138acf9856d5b4dd5b0997de4ce97cc2a7b781019
```

No automatic rerun was used in this stability matrix.

## Cleanup-head exact verification

Cleanup head:

```text
67c947ea4150c4b2564a4471cff89ef2ac82f136
```

GitHub Actions:

```text
CI run: 29954450099 — success
PostgreSQL Catalog run: 29954450146 — success
```

Core results:

```text
1205 passed, 2 skipped, 11 warnings
Total coverage: 83.47%
Total branch coverage: 70.39%
```

Successful checks:

- Studio Vitest, TypeScript typecheck, production build, and fixed viewport;
- workflow security;
- Ruff and Ruff format;
- Mypy;
- Import Linter;
- dead-code report;
- recovery and structured Serving smoke;
- full Pytest and coverage;
- critical branch-coverage ratchets;
- CLI smoke;
- Ubuntu compatibility;
- Windows compatibility;
- training-image build and packaged non-root runtime probe;
- PostgreSQL Compose validation, startup/readiness, migration, unit/integration tests, and shutdown.

Pytest artifact:

```text
ID: 8543517263
Digest: sha256:6f35d5a2ef41e25d6ba9ea79d8c2c48e8850d5f08112f376055ca6fac3a5c955
```

## Effective diff review

Comparison from PR #84 head `703427cb162694a8b4990fe4e2ef17ea59a77f7a` to cleanup head contains only:

```text
docs/superpowers/specs/2026-07-23-sequence-projection-equivalence-design.md
docs/superpowers/plans/2026-07-23-sequence-projection-equivalence.md
tests/rl/test_sequence_policy_core.py
```

This verification record is the only change after the cleanup head. No temporary workflow, test-contract script, or patch script remains. No file under `trade_rl/` changed.

## Safety boundary

- Production remains `NO-GO`.
- No model, policy, observation, reward, environment, execution, selection, serving, promotion, release, or artifact behavior changed.
- No direct exchange routing was added.
- The change stabilizes test evidence without weakening the direct historical-equivalence contract.
- The pull request remains Draft and unmerged.
