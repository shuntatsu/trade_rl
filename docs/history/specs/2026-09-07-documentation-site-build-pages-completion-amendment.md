# Documentation Site Build and Pages Completion Amendment

## Purpose

Complete the already approved documentation-governance design by making the existing site/Pages RED contract executable and verifiable.

## Required implementation

1. Add root `mkdocs.yml` for Zensical 0.0.59 with `.docs-build/site-src` as `docs_dir`, `site/` as output, Japanese UI/search-oriented features, current-section navigation, generated topic/authority indexes, and history catalog.
2. Extend `.github/workflows/docs.yml` from tests-only to a build job that runs docs tests, validation, generation, and strict Zensical build on pull requests and `main` pushes.
3. Upload the built `site/` as a GitHub Pages artifact from the build job.
4. Add a deploy job that runs only on a `main` push, needs the build job, uses the `github-pages` environment, and grants only `contents: read`, `pages: write`, and `id-token: write`.
5. Keep every external Action pinned to the already verified immutable SHAs in `tests/docs/test_site_configuration.py`.
6. Update workflow-security policy only if needed to explicitly allow this narrowly-scoped hosted-runner GitHub Pages deployment; do not weaken privileged/self-hosted protections.
7. Preserve the site as a documentation-only presentation boundary. Build/deploy success cannot alter runtime, research, profitability, or Production status.

## Test oracle

- Existing `tests/docs/test_site_configuration.py` turns GREEN without changing its requirements to fit the implementation.
- Workflow security remains GREEN.
- `uv run python scripts/docs/generate.py` followed by `uvx --from zensical==0.0.59 zensical build --strict` succeeds on the exact PR HEAD.
- The deploy condition excludes pull requests and targets only `refs/heads/main` push events.
