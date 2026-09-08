# Documentation

This directory describes the maintained system as it exists now. Git history stores completed design and implementation chronology; current docs do not duplicate that archive.

## Start here

| Need | Current authority |
| --- | --- |
| Understand the lean runtime/research flow | [Lean core](architecture/lean-core.md) |
| Understand package ownership, dependency direction, and public/private boundaries | [Package boundaries](architecture/package-boundaries.md) |
| See what has actually been validated and what remains unproven | [Current research status](research/current-status.md) |
| Modify docs or architecture as an agent | [Agent and documentation rules](AGENTS.md) |
| Understand licensing obligations | [Licensing](../LICENSES/LICENSING.md) |
| Trace source/provenance obligations | [Provenance](../LICENSES/PROVENANCE.md) |

## Reading rule

Use source code and executable tests as the observed implementation evidence, then use the current authority document for the maintained contract and explanation. If they disagree, do not silently choose one: investigate the mismatch and update the appropriate contract or implementation as part of the same scoped change.

Completed specs and implementation plans do not remain in the working tree merely for chronology. They are recoverable from Git history. Compliance material under `LICENSES/` is different: it is permanent and must not be treated as disposable history.
