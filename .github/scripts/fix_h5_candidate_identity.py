from pathlib import Path

path = Path("examples/binance-multitimeframe/compare_gpu_training_smoke.py")
source = path.read_text(encoding="utf-8")

old_signature = (
    "def _load_sample(path: Path, *, legacy_profile: str) -> dict[str, object]:\n"
)
new_signature = (
    "def _load_sample(\n"
    "    path: Path, *, legacy_profile: str | None\n"
    ") -> dict[str, object]:\n"
)
if source.count(old_signature) != 1:
    raise SystemExit("H5 sample signature seam changed")
source = source.replace(old_signature, new_signature)

old_identity = '''    profile = payload.get("runtime_profile", legacy_profile)
    if profile not in _RUNTIME_PROFILES:
        raise ValueError("GPU smoke runtime profile is unsupported")
    commit = payload.get("git_commit")
    if commit is not None and (
        not isinstance(commit, str) or _GIT_COMMIT_PATTERN.fullmatch(commit) is None
    ):
        raise ValueError("GPU smoke git commit is invalid")
'''
new_identity = '''    if schema == "gpu_sequence_target_oracle_bc_training_smoke_v6":
        if legacy_profile is None:
            raise ValueError("accelerated candidate requires schema v7 evidence")
        profile = legacy_profile
        commit: str | None = None
    else:
        profile = payload.get("runtime_profile")
        if profile not in _RUNTIME_PROFILES:
            raise ValueError("GPU smoke runtime profile is unsupported")
        raw_commit = payload.get("git_commit")
        if (
            not isinstance(raw_commit, str)
            or _GIT_COMMIT_PATTERN.fullmatch(raw_commit) is None
        ):
            raise ValueError("GPU smoke git commit is invalid")
        commit = raw_commit
'''
if source.count(old_identity) != 1:
    raise SystemExit("H5 sample identity seam changed")
source = source.replace(old_identity, new_identity)

old_aggregate = "    legacy_profile: str,\n) -> dict[str, object]:\n"
new_aggregate = "    legacy_profile: str | None,\n) -> dict[str, object]:\n"
if source.count(old_aggregate) != 1:
    raise SystemExit("H5 aggregate signature seam changed")
source = source.replace(old_aggregate, new_aggregate)

old_candidate = '''    candidate = _aggregate(
        candidate_paths,
        ref=candidate_ref,
        expected_profile="accelerated",
        legacy_profile="accelerated",
    )
'''
new_candidate = '''    candidate = _aggregate(
        candidate_paths,
        ref=candidate_ref,
        expected_profile="accelerated",
        legacy_profile=None,
    )
'''
if source.count(old_candidate) != 1:
    raise SystemExit("H5 candidate aggregate seam changed")
source = source.replace(old_candidate, new_candidate)

old_compare = '''    """Validate repeated evidence and return one digest-bound median comparison."""

    baseline = _aggregate(
'''
new_compare = '''    """Validate repeated evidence and return one digest-bound median comparison."""

    for field, ref in (
        ("baseline_ref", baseline_ref),
        ("candidate_ref", candidate_ref),
    ):
        if _GIT_COMMIT_PATTERN.fullmatch(ref) is None:
            raise ValueError(f"{field} must be a lowercase 40-character commit")
    baseline = _aggregate(
'''
if source.count(old_compare) != 1:
    raise SystemExit("H5 comparison ref seam changed")

path.write_text(source.replace(old_compare, new_compare), encoding="utf-8")
