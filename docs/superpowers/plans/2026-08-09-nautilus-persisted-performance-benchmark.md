# Nautilus Persisted Performance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing isolated legacy-versus-Nautilus performance benchmark so a reviewed canonical single-BTCUSDT market artifact can be measured without changing the deterministic synthetic CI benchmark contract.

**Architecture:** Keep the current synthetic benchmark as the default and preserve its source digest. Add an explicit persisted-artifact source contract in `tools/nautilus_training_throughput_benchmark.py`: the parent validates a canonical market artifact and binds its artifact digest into benchmark evidence, then passes the same artifact root to each isolated worker, which reloads and revalidates it before training. This creates a fail-closed path for representative local/catalog evidence without making CI depend on local runtime data or PostgreSQL.

**Tech Stack:** Python 3.12, `MarketDataset`, canonical dataset artifacts, Stable-Baselines3 PPO, NautilusTrader 1.230.0, pytest, mypy, Ruff, GitHub Actions.

## Global Constraints

- Production remains **NO-GO**.
- Maintained benchmark instrument is exactly `BTCUSDT` / `BTCUSDT-PERP.BINANCE`.
- `nautilus_trader` remains pinned to `1.230.0`.
- Existing synthetic benchmark invocation and synthetic source digest remain backward compatible.
- Persisted evidence must derive dataset identity from a validated canonical artifact; callers may not claim a persisted identity by supplying an arbitrary digest alone.
- Representative persisted runs remain `performance_approved=false` until an independently reviewed approval policy passes and approved evidence is materialized through the existing fail-closed approval path.
- Repository-local generated market data remains excluded from Git.

---

### Task 1: Bind Benchmark Source Digest to Dataset Identity

**Files:**
- Modify: `tools/nautilus_training_throughput_benchmark.py`
- Test: `tests/test_nautilus_training_throughput_benchmark.py`

**Interfaces:**
- Produces: `_benchmark_source_digest(workloads: tuple[int, ...], *, dataset_source_digest: str | None = None) -> str`
- Contract: omit `dataset_source_digest` when `None`, preserving the existing synthetic digest; validate a provided digest as lowercase SHA-256 and expose the benchmark-specific error contract.

- [x] **Step 1: Write failing identity-binding tests.**
- [x] **Step 2: Verify RED with `TypeError: unexpected keyword argument 'dataset_source_digest'`.**
- [x] **Step 3: Implement the optional validated digest binding using `require_sha256`.**
- [x] **Step 4: Verify focused pytest, targeted mypy, Ruff, and format checks.**
- [x] **Step 5: Commit as `feat: bind Nautilus benchmark to dataset source`.**

### Task 2: Resolve a Canonical Persisted Benchmark Source

**Files:**
- Modify: `tools/nautilus_training_throughput_benchmark.py`
- Modify: `tests/test_nautilus_training_throughput_benchmark.py`

**Interfaces:**
- Produce an internal immutable source record with:
  - `dataset_kind: Literal["deterministic_synthetic_btcusdt", "persisted_market_dataset_artifact"]`
  - `artifact_root: Path | None`
  - `dataset_source_digest: str | None`
- Produce `_resolve_benchmark_dataset_source(dataset_artifact: Path | None, *, workloads: tuple[int, ...])`.
- Persisted source resolution must call `inspect_published_market_dataset_artifact` and `load_market_dataset_artifact`, require `dataset.symbols == ("BTCUSDT",)`, and require at least `max(80, max(workloads) + 32)` bars.

- [x] **Step 1: Write a failing test using a real canonical temporary market artifact.**

```python
source = _resolve_benchmark_dataset_source(root, workloads=(8, 32))
assert source.dataset_kind == "persisted_market_dataset_artifact"
assert source.artifact_root == root.resolve()
assert source.dataset_source_digest == published.artifact_digest
```

Also assert that a non-canonical/missing artifact, a non-BTCUSDT artifact, and an artifact shorter than the workload requirement fail closed.

- [x] **Step 2: Run the focused test and verify it fails because the resolver does not exist.**

Run:

```bash
uv run pytest -q tests/test_nautilus_training_throughput_benchmark.py
```

- [x] **Step 3: Implement the source resolver with no PostgreSQL or catalog dependency.**

```python
@dataclass(frozen=True, slots=True)
class _BenchmarkDatasetSource:
    dataset_kind: Literal[
        "deterministic_synthetic_btcusdt",
        "persisted_market_dataset_artifact",
    ]
    artifact_root: Path | None
    dataset_source_digest: str | None
```

The synthetic branch returns the existing kind with both optional fields `None`. The persisted branch derives its digest only from the validated `PublishedDatasetArtifact`.

- [x] **Step 4: Run focused pytest, mypy, Ruff and format checks.**

```bash
uv run pytest -q tests/test_nautilus_training_throughput_benchmark.py tests/data/test_market_artifact.py
uv run mypy tools/nautilus_training_throughput_benchmark.py
uv run ruff check tools/nautilus_training_throughput_benchmark.py tests/test_nautilus_training_throughput_benchmark.py
uv run ruff format --check --diff tools/nautilus_training_throughput_benchmark.py tests/test_nautilus_training_throughput_benchmark.py
```

- [x] **Step 5: Commit the resolver and tests.**

```bash
git add tools/nautilus_training_throughput_benchmark.py tests/test_nautilus_training_throughput_benchmark.py
git commit -m "feat: resolve persisted Nautilus benchmark dataset"
```

### Task 3: Feed the Same Persisted Artifact Through Isolated Workers

**Files:**
- Modify: `tools/nautilus_training_throughput_benchmark.py`
- Modify: `tests/test_nautilus_training_throughput_benchmark.py`

**Interfaces:**
- Extend `run_benchmark(*, timesteps: int | Sequence[int], dataset_artifact: Path | None = None) -> dict[str, Any]`.
- Extend `_run_worker_subprocess(..., dataset_artifact: Path | None = None)` and worker CLI with `--worker-dataset-artifact`.
- Extend `_worker_training_measurement(..., dataset_artifact: Path | None = None)`.
- Workers must reload the canonical artifact themselves; the parent must not serialize a `MarketDataset` through the process boundary.

- [x] **Step 1: Write failing command-propagation and worker-dataset tests.**

Test that an artifact root appears exactly once in the worker command when configured and is absent for the synthetic default. Test that the worker loader returns the persisted dataset identity instead of rebuilding the synthetic fixture.

- [x] **Step 2: Verify RED for the missing `dataset_artifact` parameters.**

```bash
uv run pytest -q tests/test_nautilus_training_throughput_benchmark.py
```

- [x] **Step 3: Refactor synthetic dataset construction into a helper and add persisted loading.**

Use one helper for dataset selection so legacy and streaming workers receive identical data. Revalidate the canonical artifact in the child and preserve the existing synthetic dataset byte-for-byte when no artifact is provided.

- [x] **Step 4: Bind the resolved source into `RuntimePerformanceEvidence`.**

For persisted runs:

```python
dataset_kind=source.dataset_kind
source_digest=_benchmark_source_digest(
    workloads,
    dataset_source_digest=source.dataset_source_digest,
)
```

Keep `performance_approved=False` and do not attach an approval policy automatically.

- [x] **Step 5: Run focused runtime tests and static analysis.**

```bash
uv run pytest -q tests/test_nautilus_training_throughput_benchmark.py tests/simulation/test_runtime_performance.py tests/simulation/test_runtime_performance_artifact_identity.py
uv run mypy tools/nautilus_training_throughput_benchmark.py
uv run ruff check tools/nautilus_training_throughput_benchmark.py tests/test_nautilus_training_throughput_benchmark.py
uv run ruff format --check --diff tools/nautilus_training_throughput_benchmark.py tests/test_nautilus_training_throughput_benchmark.py
```

- [x] **Step 6: Commit the worker propagation.**

```bash
git add tools/nautilus_training_throughput_benchmark.py tests/test_nautilus_training_throughput_benchmark.py
git commit -m "feat: benchmark persisted dataset with Nautilus workers"
```

### Task 4: Expose the Representative Artifact Path and Verify the Migration Gate

**Files:**
- Modify: `tools/nautilus_training_throughput_benchmark.py`
- Modify: `docs/NAUTILUS_MIGRATION.md`
- Modify: `.github/workflows/nautilus-capability.yml` only if the synthetic CI invocation needs explicit regression coverage; do not add local/catalog data to CI.

**Interfaces:**
- Add CLI option `--dataset-artifact PATH` for parent benchmark mode only.
- `--dataset-artifact` must identify an already-published canonical artifact directory; it must never accept raw NPZ/JSON or an arbitrary source digest.

- [x] **Step 1: Add a CLI parsing test that preserves the existing no-argument synthetic behavior and accepts `--dataset-artifact`.**
- [x] **Step 2: Implement the CLI plumbing to `run_benchmark` and document the representative-run command.**
- [x] **Step 3: Update migration status to distinguish “persisted benchmark path implemented” from “representative reviewed run retained”.**
- [ ] **Step 4: Run focused tests, full mypy/Ruff/format, architecture checks, and the complete test suite on the same head.**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
```

Also require the dedicated `Nautilus Capability`, `PostgreSQL Catalog`, and main `CI` GitHub workflows to succeed on the final head.

- [ ] **Step 5: Self-review the complete diff for authority escalation, synthetic digest drift, accidental generated artifacts, temporary workflows, and unrelated refactors.**

No code in this plan may set `performance_approved=true`, change the default authority from `legacy_authoritative`, or claim that a representative run has occurred unless a retained artifact proves it.
