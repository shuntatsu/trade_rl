# Universal Episode Router RED Evidence

The contract-test commit is:

```text
3f6d64c37edcf870af7bd49aeec61a5928497459
```

This head adds the complete U2 contract tests before the implementation modules exist. The expected repository failure is collection-time `ModuleNotFoundError` for the new `trade_rl.rl.universal_*` modules. Any unrelated failure must be investigated before implementation begins.

The test-format correction is isolated in:

```text
a0fdd42779a584a7ac1bdbc1f8d7ed50541e440b
```

Formatted RED verification head:

```text
b7c89cd3f9369a364e9005c5d1c14f9416152864
```

At that exact head, frontend, workflow security, Ruff, Ruff format, MyPy, Import Linter, Vulture, serving smoke, Windows/Ubuntu compatibility, PostgreSQL specialist checks, and the complete training image passed. Full pytest stopped only on the four expected collection-time `ModuleNotFoundError` failures for the new U2 modules.

Initial GREEN implementation commits:

```text
b8642a644c9f93e6572fecb6713e8ba8245d5ad9  binding and balanced router
e84d764a96f0982f3d36187154d482418a7f7a4f  episode-routed Gymnasium facade
0ab717c8ea42fa0dbf14fb5d280e791beb81b229  policy/deployment identity separation
```

The verification PR was temporarily retargeted to `main` only to make GitHub Actions evaluate this exact branch head; it will return to its stacked U1 base after verification.

The active BTC training generation and existing runtime artifacts are outside this stacked PR and remain untouched.
