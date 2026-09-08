# Phase 1 CI topology note

Repository CI currently runs `pull_request` only when the PR base is `main`. The implementation PR is therefore opened against `main` for exact-head RED/GREEN evidence even though its logical dependency stack is PR #436 -> PR #437 -> design PR #438.

This validation topology does not authorize merging the implementation before its prerequisites or merging any PR into `main` without explicit user approval.
