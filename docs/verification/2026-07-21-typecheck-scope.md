# Repository Type-Check Scope

The required command is `mypy .`.

The project configuration explicitly applies strict Mypy checks to production Python under `trade_rl/` and operational Python under `tools/`. Test fixtures, example entrypoints, and the TypeScript Studio workspace are excluded from recursive Mypy discovery; they remain covered by Ruff, pytest, frontend typecheck, build, fixed-viewport checks, and platform compatibility suites.

This preserves the repository's existing production type-check boundary while allowing the exact repository-root command to be used in CI. The training capability audit tool is included and its PPO, SAC, TD3, and TQC model variables and report mapping are explicitly typed.
