# Documentation Site Build and Pages Completion Plan

1. Preserve `tests/docs/test_site_configuration.py` as the RED oracle.
2. Add `mkdocs.yml` exactly matching the pinned Zensical/current-section contract.
3. Expand `.github/workflows/docs.yml` to build on PR/main and deploy only on `main` push.
4. Run workflow-security and docs tests; fix implementation rather than weakening the site contract.
5. Run strict Zensical build, static checks, current-doc contract tests, full pytest, and same-head CI.
6. Perform final diff/falsification review and report Pages environment/publication state separately from site build success.
