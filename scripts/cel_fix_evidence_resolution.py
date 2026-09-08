from pathlib import Path

path = Path("trade_rl/evaluation/experiments/evidence.py")
text = path.read_text(encoding="utf-8")

old = '''    normalized_config = replace(config, ppo_seed=plan.ppo_seeds[0])
    normalized_spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=normalized_config,
    )
    normalized_contract = _resolved_run_config(normalized_spec)
    _check_study_fixed_config(plan, normalized_contract)
    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        loaded_runs: dict[int, LoadedCandidateRun] = {}
        run_digests: list[tuple[int, str]] = []
        seedless_contract = _without_seed(normalized_contract)

        for seed in plan.ppo_seeds:
'''
new = '''    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        loaded_runs: dict[int, LoadedCandidateRun] = {}
        run_digests: list[tuple[int, str]] = []
        normalized_contract: ResolvedRunConfig | None = None
        seedless_contract: dict[str, object] | None = None

        for seed in plan.ppo_seeds:
'''
if old not in text:
    raise SystemExit("pre-loop resolution block not found exactly")
text = text.replace(old, new, 1)

old = '''            resolved = _resolved_run_config(spec)
            _check_study_fixed_config(plan, resolved)
            if _without_seed(resolved) != seedless_contract:
                raise ArtifactIntegrityError(
                    "EvidenceSet resolved config changed beyond ppo_seed"
                )
'''
new = '''            resolved = _resolved_run_config(spec)
            _check_study_fixed_config(plan, resolved)
            if normalized_contract is None:
                normalized_contract = resolved
                seedless_contract = _without_seed(resolved)
            elif _without_seed(resolved) != seedless_contract:
                raise ArtifactIntegrityError(
                    "EvidenceSet resolved config changed beyond ppo_seed"
                )
'''
if old not in text:
    raise SystemExit("loop resolution block not found exactly")
text = text.replace(old, new, 1)

old = '''        _verify_seed_invariance(
            loaded_runs,
'''
new = '''        if normalized_contract is None:
            raise ArtifactIntegrityError("EvidenceSet contains no seed Runs")
        _verify_seed_invariance(
            loaded_runs,
'''
if old not in text:
    raise SystemExit("post-loop block not found exactly")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
