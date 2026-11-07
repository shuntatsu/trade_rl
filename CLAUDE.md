@AGENTS.md

# Claude Code entry point

The imported `AGENTS.md` is the shared instruction authority. Read
`docs/AGENTS.md`, the current documentation entry points, relevant source and
tests before changing behavior; do not duplicate their rules in this file.

The project `SessionStart` hook reports **local** Git context only. It does not
fetch GitHub main, check open PRs or CI, grant an execution lease, or authorize
a merge or economic study. Independently verify those boundaries as prescribed
by `AGENTS.md`.

For an explicit, **author-side** adversarial challenge of a new research plan,
invoke `/research-challenge` manually. Its output is not an independent review
receipt or an authorization for economic execution.
