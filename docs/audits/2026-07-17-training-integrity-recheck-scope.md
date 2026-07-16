# Training Integrity Recheck Scope — 2026-07-17

This branch exists only to independently re-run the current `main` training integrity checks. It does not alter product behavior.

The recheck requires, on the same source tree:

- Ruff, formatting, MyPy, import-boundary, and dead-code checks
- the complete pytest suite with branch coverage and critical coverage enforcement
- real PPO, SAC, TD3, and TQC optimization with checkpoints
- behavior cloning followed by PPO
- PPO checkpoint resume and off-policy replay-buffer resume
- ONNX and TorchScript export/reload parity
- alpha, factor, risk-tilt, asset-set, and native multi-timeframe sequence paths
- three-seed walk-forward training, checkpoint evidence, selection, sealed outer-test control, and publication behavior
- configuration rejection for inactive or unsupported fields
- signal/dataset identity, train-only normalization, causality, purge, sealed-test, publication, and serving fail-closed tests
- Windows compatibility and the non-root training Docker image

The pull request will be closed without merging if the current `main` passes unchanged. Product fixes will be separated and reviewed only if the recheck finds a reproducible defect.
