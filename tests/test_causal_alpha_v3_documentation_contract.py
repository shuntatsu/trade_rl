from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_causal_alpha_v3_is_documented_as_non_promotable_research_lane() -> None:
    universal = _text("docs/UNIVERSAL_TRAINING.md")
    research = _text("docs/RESEARCH_STATUS.md")
    combined = f"{universal}\n{research}".lower()

    for phrase in (
        "causal alpha v3",
        "research-only",
        "promotion_eligible=false",
        "overlap",
        "uncertainty",
        "anchored_target_residual",
        "dagger",
        "teacher admission",
    ):
        assert phrase in combined

    assert 'behavior_cloning_teacher: "causal_alpha_ridge"' in universal
    assert "canonical u6" in universal.lower()
    assert "target_weight" in universal
    assert "production statusは**no-go**" in universal.lower()


def test_v3_docs_preserve_reward_and_holdout_invariants() -> None:
    universal = _text("docs/UNIVERSAL_TRAINING.md").lower()

    for phrase in (
        "pure net-log-growth",
        "teacher-admission holdout",
        "reward unchanged",
        "does not bypass teacher admission",
    ):
        assert phrase in universal


def test_v3_runner_docs_close_the_deterministic_research_workflow() -> None:
    universal = _text("docs/UNIVERSAL_TRAINING.md").lower()
    research = _text("docs/RESEARCH_STATUS.md").lower()
    combined = f"{universal}\n{research}"

    for phrase in (
        "run_universal_causal_alpha_v3_research.py",
        "signal gate",
        "candidate freeze",
        "resumable",
        "production replay",
        "research-only teacher package",
        "exit code 2",
        "exit code 3",
        "exit code 4",
    ):
        assert phrase in combined

    for phrase in (
        "dagger -> bc",
        "anchored ppo",
        "only after teacher admission",
    ):
        assert phrase in combined


def test_v3_docs_distinguish_authoritative_records_from_diagnostics_and_legacy_jsonl() -> (
    None
):
    universal = _text("docs/UNIVERSAL_TRAINING.md").lower()
    research = _text("docs/RESEARCH_STATUS.md").lower()
    combined = f"{universal}\n{research}"

    for phrase in (
        "selection/records/",
        "selection/diagnostics/",
        "selection/progress.json",
        "legacy jsonl",
        "diagnostic-only",
        "gross-negative",
        "net-negative",
        "asymmetric threshold",
        "rolling-window",
    ):
        assert phrase in combined


def test_v3_docs_define_deterministic_machine_run_report_contract() -> None:
    reporting = _text("docs/RUN_REPORTING.md").lower()
    index = _text("docs/README.md").lower()

    assert "run_reporting.md" in index
    for phrase in (
        "build_run_report.py",
        "machine run report",
        "--profile chat",
        "--profile json",
        "reporting/stages/",
        "read-only",
        "llm",
        "pass / reject / in_progress / not_run / missing / invalid",
        "output must stay outside",
    ):
        assert phrase in reporting
