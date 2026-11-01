"""Risk-based verification routing for repository-agent development loops."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tools.agent_repo.git_state import read_git_state
from tools.agent_repo.path_safety import checked_repo_directory, checked_repo_file
from tools.agent_repo.semantic_diff import SemanticSignal, semantic_diff
from tools.agent_repo.source_index import SourceIndex


@dataclass(frozen=True, slots=True, order=True)
class VerificationStep:
    """One suggested verification action grouped by execution tier."""

    tier: str
    command: str
    reason: str


_FINAL_STEPS = (
    VerificationStep(
        "final",
        "uv run ruff check trade_rl tests tools",
        "permanent lint gate",
    ),
    VerificationStep(
        "final",
        "uv run ruff format --check trade_rl tests tools",
        "permanent formatting gate",
    ),
    VerificationStep("final", "uv run mypy trade_rl", "production type gate"),
    VerificationStep(
        "final",
        "uv run mypy tools/agent_repo tests/architecture/distribution.py",
        "repository-tooling type gate",
    ),
    VerificationStep(
        "final", "uv run pytest -q tests", "full deterministic test suite"
    ),
    VerificationStep("final", "uv build", "distribution build gate"),
    VerificationStep(
        "final",
        "uv run python -m tests.architecture.distribution dist/*.tar.gz dist/*.whl",
        "tracked source / sdist / direct wheel closure",
    ),
    VerificationStep(
        "final",
        "CI composite: rebuild wheel from sdist and verify source closure",
        "sdist must reproduce the maintained Python-source wheel closure",
    ),
    VerificationStep(
        "final",
        "CI composite: clean installed core smoke",
        "installed package imports and maintained CLIs must work outside checkout",
    ),
    VerificationStep(
        "final",
        "CI composite: package identity",
        "distribution and runtime versions must agree",
    ),
)

_TIER_ORDER = {"fast": 0, "final": 1, "extended": 2, "signal": 3}


def _existing(repository: Path, relative: str) -> bool:
    candidate = repository / relative
    if not candidate.exists() and not candidate.is_symlink():
        return False
    checked_repo_file(repository, candidate)
    return True


def _pytest_command(paths: set[str]) -> str | None:
    if not paths:
        return None
    return "uv run pytest -q " + " ".join(sorted(paths))


def _named_project_surfaces(signals: tuple[SemanticSignal, ...], kind: str) -> str:
    values = sorted(
        {
            f"{signal.name} ({signal.change})"
            for signal in signals
            if signal.kind == kind
        }
    )
    return ", ".join(values)


def _add_existing_tests(root: Path, selected: set[str], paths: tuple[str, ...]) -> None:
    """Only select maintained tests verified to exist as ordinary files."""

    for path in paths:
        if _existing(root, path):
            selected.add(path)


def _has_scope_symbols(root: Path, source: str) -> bool:
    """Detect real names, not comments mentioning dataset or symbol scope."""

    path = checked_repo_file(root, root / source)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=source)
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    names.update(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return any(
        any(piece in name.lower() for piece in ("feature", "symbol", "dataset_scope"))
        for name in names
    )


def _contract_routes(
    root: Path,
    source: str,
    *,
    validation_hints: tuple[str, ...],
    tests: set[str],
) -> set[str]:
    """Supplement static imports with existing cross-boundary contract oracles.

    These are regression-test routes, not authorization or task answer keys.
    """

    areas: set[str] = set()
    if source == "trade_rl/evaluation/runs/config.py":
        _add_existing_tests(
            root,
            tests,
            (
                "tests/evaluation/test_candidate_run_config.py",
                "tests/evaluation/test_candidate_run_cli.py",
                "tests/evaluation/test_candidate_run_artifact.py",
                "tests/evaluation/test_candidate_run_provenance.py",
            ),
        )
        areas.add("run_config")
    if source in (
        "trade_rl/evaluation/runs/artifact.py",
        "trade_rl/evaluation/runs/provenance.py",
    ):
        _add_existing_tests(
            root,
            tests,
            (
                "tests/evaluation/test_candidate_run_artifact.py",
                "tests/evaluation/test_candidate_run_artifact_identity.py",
                "tests/evaluation/test_candidate_run_malformed_archive_errors.py",
                "tests/evaluation/test_candidate_run_provenance.py",
            ),
        )
        areas.add("run_artifact")
    if source == "trade_rl/_validation.py" or validation_hints:
        _add_existing_tests(root, tests, ("tests/test_validation.py",))
        areas.add("validation")
    if source.startswith(
        (
            "trade_rl/data/",
            "trade_rl/strategies/",
            "trade_rl/evaluation/runs/",
        )
    ) and _has_scope_symbols(root, source):
        _add_existing_tests(
            root, tests, ("tests/strategies/test_dataset_scope_contract.py",)
        )
        areas.add("dataset_scope")
    if source in (
        "trade_rl/integrations/binance/transport.py",
        "trade_rl/integrations/binance/cache.py",
    ):
        _add_existing_tests(
            root,
            tests,
            (
                "tests/integrations/test_binance_exchange_info_fallback.py",
                "tests/integrations/test_binance_response_bounds.py",
                "tests/integrations/test_binance_cache_integrity_contract.py",
                "tests/integrations/test_binance_cache_only.py",
            ),
        )
        areas.add("binance")
    return areas


def plan_verification(
    repository: Path,
    *,
    base_ref: str,
) -> tuple[VerificationStep, ...]:
    """Plan targeted iteration plus the unchanged final repository gate."""

    root = repository.resolve()
    state = read_git_state(root, base_ref=base_ref)
    signals = semantic_diff(root, base_ref=base_ref)
    changed_paths = set(state.changed_paths) | set(state.untracked_paths)
    production_paths = {
        path
        for path in changed_paths
        if path.startswith("trade_rl/") and path.endswith(".py")
    }

    fast_tests: set[str] = set()
    contract_areas: set[str] = set()
    index = SourceIndex.build(root)
    for path in sorted(production_paths):
        candidate = root / path
        if not candidate.exists() and not candidate.is_symlink():
            continue
        checked_repo_file(root, candidate)
        context = index.context(path)
        fast_tests.update(context.test_candidates)
        contract_areas.update(
            _contract_routes(
                root,
                path,
                validation_hints=context.validation_hints,
                tests=fast_tests,
            )
        )
        if path.startswith("trade_rl/evaluation/runs/") and _existing(
            root, "tests/architecture/test_runs_capability_facade.py"
        ):
            fast_tests.add("tests/architecture/test_runs_capability_facade.py")

    # A changed test is itself a valid targeted test; no import is required.
    for path in changed_paths:
        if path.startswith("tests/") and path.endswith(".py"):
            _add_existing_tests(root, fast_tests, (path,))

    signal_kinds = {signal.kind for signal in signals}
    if "DEPENDENCY" in signal_kinds:
        for path in (
            "tests/architecture/test_lean_dependency_boundaries.py",
            "tests/architecture/test_import_collector_contract.py",
        ):
            if _existing(root, path):
                fast_tests.add(path)
    if "PUBLIC_EXPORT" in signal_kinds and _existing(
        root, "tests/architecture/test_runs_capability_facade.py"
    ):
        if any(
            path.startswith("trade_rl/evaluation/runs/") for path in production_paths
        ):
            fast_tests.add("tests/architecture/test_runs_capability_facade.py")

    if any(path.startswith("docs/") for path in changed_paths) and _existing(
        root, "tests/architecture/test_current_docs_layout.py"
    ):
        fast_tests.add("tests/architecture/test_current_docs_layout.py")

    steps: list[VerificationStep] = []
    command = _pytest_command(fast_tests)
    if command is not None:
        steps.append(
            VerificationStep(
                "fast",
                command,
                "closest contract/architecture tests for the observed change surface",
            )
        )

    steps.extend(_FINAL_STEPS)

    if "NETWORK_EFFECT" in signal_kinds:
        integration_tests = root / "tests" / "integrations"
        if not integration_tests.exists() and not integration_tests.is_symlink():
            steps.append(
                VerificationStep(
                    "extended",
                    "manual: inspect network/retry/fallback contract",
                    "network effect changed but no dedicated integration test directory was found",
                )
            )
        else:
            checked_repo_directory(root, integration_tests)
            steps.append(
                VerificationStep(
                    "extended",
                    "uv run pytest -q tests/integrations",
                    "network effect changed; verify transport, retry, fallback, and offline boundaries",
                )
            )
    if "run_artifact" in contract_areas:
        steps.append(
            VerificationStep(
                "extended",
                "manual: compare candidate artifact v1/v2 readers and tamper oracles",
                "preserve historical artifact compatibility, digest and malformed input rejection",
            )
        )
    if "binance" in contract_areas:
        steps.append(
            VerificationStep(
                "extended",
                "manual: inspect Binance fallback and invalid cache/response oracles",
                "endpoint fallback must not accept integrity-invalid cached or malformed data",
            )
        )
    if "DATA_FIELD" in signal_kinds:
        steps.append(
            VerificationStep(
                "extended",
                "manual: validate dataclass field serialization and consumer compatibility",
                "added or changed typed state requires matching persisted and public contracts",
            )
        )
    if "SCHEMA" in signal_kinds:
        steps.append(
            VerificationStep(
                "extended",
                "manual: run relevant artifact compatibility and tamper tests",
                "schema surface changed; verify historical compatibility and tamper rejection",
            )
        )
    if "FILESYSTEM_EFFECT" in signal_kinds:
        steps.append(
            VerificationStep(
                "extended",
                "manual: run relevant atomicity and failure-injection tests",
                "filesystem mutation surface changed; verify cleanup, retry, and no-partial-publication behavior",
            )
        )
    if "PROJECT_EXTRA" in signal_kinds:
        names = _named_project_surfaces(signals, "PROJECT_EXTRA")
        steps.append(
            VerificationStep(
                "extended",
                f"manual: verify affected optional capability extras: {names}",
                "optional dependency surface changed; verify only affected capability environments",
            )
        )
    if "PROJECT_SCRIPT" in signal_kinds:
        names = _named_project_surfaces(signals, "PROJECT_SCRIPT")
        steps.append(
            VerificationStep(
                "extended",
                f"manual: verify affected installed project scripts: {names}",
                "installed command surface changed; verify only affected project script entry points",
            )
        )

    if production_paths:
        steps.append(
            VerificationStep(
                "signal",
                "uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-fail-under=0 -q tests",
                "target=80%; signal only; investigate important uncovered paths before adding tests",
            )
        )

    deduplicated = {(step.tier, step.command, step.reason): step for step in steps}
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda step: (
                _TIER_ORDER.get(step.tier, 99),
                step.command,
                step.reason,
            ),
        )
    )


__all__ = ["VerificationStep", "plan_verification"]
