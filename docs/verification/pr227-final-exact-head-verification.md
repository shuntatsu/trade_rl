# PR #227 final exact-head verification

This marker triggers the maintained pull-request workflows after the verified checkpoint runtime-state correction was published by GitHub Actions.

Required gates remain fail-closed: full pytest with branch coverage, critical coverage, real CPU training capability audit including hierarchical BC to PPO and checkpoint resume, sequence stability, PostgreSQL catalog, compatibility, and production training image.
