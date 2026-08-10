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

The verification PR was temporarily retargeted to `main` only to make GitHub Actions evaluate this exact branch head; it will return to its stacked U1 base after verification.

The active BTC training generation and existing runtime artifacts are outside this stacked PR and remain untouched.
