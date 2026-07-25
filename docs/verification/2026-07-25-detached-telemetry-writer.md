# Detached Telemetry Writer Verification

Date: 2026-07-25

## Failure mode

On POSIX, an open telemetry writer can continue to reference its original inode
after the path is atomically replaced. Append already compared the descriptor and
path identities and failed closed, but `flush()` and `close()` refreshed the sparse
index through the replacement path. This could raise a secondary index mismatch
and mutate index metadata for a stream the writer did not open.

## Resolution

The internal flush operation now compares the open descriptor identity with the
current path before refreshing the index.

- Strict internal flush and append paths reject identity drift.
- Public `flush()` and `close()` still fsync the original descriptor, then return
  without indexing the replacement path.
- No telemetry JSONL or sparse-index schema is changed.
- No automatic evidence truncation or repair is introduced.

## Verification boundary

The dependency-ordered checkout passed Ruff, repository-wide Mypy, all telemetry
tests, and training-telemetry integration tests. The PR now targets `main`, so its
exact head must additionally pass the standard CI, Windows/Ubuntu compatibility,
training-image build, complete suite, CLI smoke, and critical branch-coverage
ratchets before integration.
