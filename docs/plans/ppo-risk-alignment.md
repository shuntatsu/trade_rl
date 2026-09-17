# PPO risk alignment implementation

Status: Active

Specification: [ppo-risk-alignment.md](../specs/ppo-risk-alignment.md).

- [x] Audit training/evaluation differences and existing concurrent work.
- [x] RED: explicit risk/reset/fit propagation contracts with synthetic data.
- [x] Add optional configuration preserving legacy defaults.
- [ ] Review and run focused plus full permanent verification.
- [ ] Finish all original baseline arms without changing frozen source.
- [ ] Freeze paired five-seed risk-only comparison before candidate training.
- [ ] Execute, independently audit, and report relative and absolute results.

Implement in the isolated validation worktree of the same durable PR branch.
The directional experiment worktree stays frozen for its active processes.
