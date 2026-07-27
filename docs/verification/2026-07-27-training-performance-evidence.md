# Training Performance Evidence Verification

Date: 2026-07-27  
Pull request: #202  
Implementation head: `1f597caf85fe5200fe7abc34461236b65ebb8b1d`  
Exact-head CI run: `30247326386` / run number `3574`

## Verified scope

H1 adds deterministic, per-member training performance evidence without changing the reward, action path, policy architecture, optimizer configuration, rollout semantics, environment execution, selection, Serving, release, or production status.

Each member that executes new learning work now publishes `training-performance.json` with:

- requested and observed environment steps;
- total synchronized learn wall time and environment steps per second;
- host-observed rollout collection and optimizer-update duration and call counts;
- vector-environment step duration and call count;
- policy feature-extraction host duration and call count;
- sequence reconstruction duration and call count;
- NumPy-to-Torch sequence materialization duration and call count;
- peak PyTorch CUDA allocated and reserved bytes when the resolved device is CUDA;
- an explicit declaration that component timers overlap;
- canonical schema and content digest validation.

The maintained GPU training smoke loads and validates this artifact for both the original run and checkpoint-resumed run. The self-hosted GPU nightly contract now requires schema version 6 and validates CUDA allocator evidence for both runs.

## Safety and lifecycle evidence

The recorder is activated only around `model.learn()`. It temporarily wraps the maintained SB3 model, policy, and vector-environment call boundaries, then restores the exact pre-existing instance attribute layout in `finally` blocks. Recorder objects and wrapper callables are not part of checkpoint state.

CUDA synchronization is intentionally limited to measurement start and final allocator capture. There is no per-step or per-minibatch synchronization. Component durations are observational and may be nested; they must not be summed as though they were disjoint phases.

A resume with no remaining learning work does not publish a fabricated new performance artifact.

## Test and static evidence

The complete CI workflow passed on the implementation head:

- Python tests: `1,912 passed`, `2 skipped`, `11 warnings`;
- total coverage: `85.93%`;
- branch coverage: `74.27%`;
- critical branch coverage: passed;
- Ruff: passed;
- Ruff format: passed;
- Mypy: passed;
- import architecture: passed;
- dead-code report: passed;
- recovery and structured-serving smoke: passed;
- CLI smoke: passed;
- Studio tests, typecheck, production build, and fixed-viewport audit: passed;
- Ubuntu compatibility: passed;
- Windows compatibility: passed;
- complete training-image build, identity capture, and packaged non-root runtime probe: passed.

Focused tests additionally verify exact timer accumulation with an injected monotonic clock, canonical digest persistence, invalid lifecycle rejection, wrapper restoration after exceptions, one sequence materialization per rollout cache, SB3 artifact publication, GPU smoke schema validation, and GPU-nightly workflow closure.

## Explicitly not claimed

GitHub-hosted CI does not provide the target RTX 4070 Ti SUPER. This verification therefore does not claim representative CUDA throughput, allocator peaks, GPU utilization, or a measured optimization speedup. H1 establishes the evidence contract needed to measure those values on the target GPU. H2 and later optimizations must compare their exact software heads against this evidence under identical data, seeds, model, and training configuration.

Production remains `NO-GO`.
