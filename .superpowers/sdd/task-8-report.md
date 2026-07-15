# Task 8 Report: Preserve Packaged-Source Git Provenance

## Outcome

Packaged training now carries the host checkout's exact Git commit and dirty
state through Docker build arguments, image environment variables, maintained
runner-generated JSON configuration, training configuration identity, and
runtime provenance capture. The image and both runners fail closed for missing
or invalid provenance.

`TrainingRunConfig.git_dirty` is appended after the existing defaulted public
fields, preserving the positional meaning of every pre-existing constructor
argument. Runner JSON contains native booleans, not string-encoded booleans.

## TDD Evidence

Focused RED command:

```text
uv run pytest tests/workflows/test_training_run_config.py tests/examples/test_run_gpu_training_smoke.py tests/examples/test_binance_multitimeframe_full_assets.py tests/examples/test_docker_training_assets.py -q
```

RED result: 10 expected provenance failures and 23 passes. Failures covered the
absent config field/digest input, missing runner injection and validation, and
missing Docker/Compose build provenance.

Focused GREEN result after implementation and formatting:

```text
33 passed in 0.81s
```

## Validation Evidence

- Focused Ruff check: passed.
- Focused Ruff format check: 7 files already formatted.
- Configured MyPy: `Success: no issues found in 99 source files`.
- Explicit smoke-runner MyPy: `Success: no issues found in 1 source file`.
- Compose config with explicit 40-character commit and `false` dirty state:
  passed and rendered both build arguments.
- Architecture contract: 11 passed.
- Full suite: 611 passed, 1 skipped, 1 warning in 26.20s.
- `git diff --check`: passed before the report was written.

## Concerns

The full suite retains the existing Torch `TracerWarning` from policy export;
it is unrelated to this task and does not fail validation. A widened MyPy call
over `run_full_research.py` reports the three pre-existing `object`-to-`float`
metadata conversions already documented by Task 7; the repository-configured
MyPy scope and the newly changed packaged provenance paths pass.

No GPU training or complete valid Docker image build was run for this code-only
task. The focused asset tests cover fail-closed Dockerfile validation and
Compose wiring, while Compose interpolation itself was validated with explicit
provenance values.

## Independent Review Follow-up

The P2 execution-coverage finding is closed by
`test_execute_training_run_uses_explicit_provenance_without_git_lookup`. The
test runs the smallest maintained CPU training fixture with an explicit
40-character commit and `git_dirty: false`, replaces Git discovery with a
function that raises if called, and asserts the published `provenance.json`
retains the exact commit and native boolean `false`.

Regression RED was demonstrated by removing the two forwarding arguments from
`execute_training_run`: the test failed at the forbidden `git rev-parse`
lookup. Restoring `git_commit=config.git_commit` and
`git_dirty=config.git_dirty` produced GREEN: 1 passed in 5.11s.
