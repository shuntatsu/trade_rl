"""The maintained Tier D driver composes native APIs without a second ledger."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_tier_d_source_bytes_have_portable_lf_checkout():
    # Market admission compares working bytes with committed blobs. A Windows
    # autocrlf checkout must preserve the reviewed source bytes for that check.
    attributes = (ROOT / ".gitattributes").read_text("utf-8")
    assert "tools/tier_d_nonrl/*.py text eol=lf" in attributes
    for path in (ROOT / "tools/tier_d_nonrl").glob("*.py"):
        assert b"\r\n" not in path.read_bytes(), path.name


def test_tier_d_source_has_a_maintained_cli_and_normal_verification_routing():
    owner = ROOT / "tools/tier_d_nonrl"
    assert (owner / "__init__.py").is_file()
    assert (owner / "__main__.py").is_file()
    assert (ROOT / "tests/tools/tier_d_nonrl/test_runner.py").is_file()
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    research_type_routes = [
        line
        for line in workflow.splitlines()
        if "uv run mypy" in line and "tools/tier_d_nonrl" in line
    ]
    assert len(research_type_routes) == 2
    assert not any(
        (ROOT / "output/issue810-tier-d-nonrl-preparation" / name).exists()
        for name in ("tier_d_runner.py", "tier_d_supervisor.py", "test_runner.py")
    )


def test_windows_process_ownership_integration_is_a_ci_gate():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    assert "name: Windows process ownership tests" in workflow
    assert "if: matrix.os == 'windows-latest'" in workflow
    preflight_name = "name: Preflight Windows CIM readiness"
    ownership_name = "name: Windows process ownership tests"
    assert preflight_name in workflow
    preflight_start = workflow.index(preflight_name)
    ownership_start = workflow.index(ownership_name)
    assert preflight_start < ownership_start
    preflight_step = workflow[preflight_start:ownership_start]
    assert "if: matrix.os == 'windows-latest'" in preflight_step
    assert (
        "uv run --no-sync python -m tools.windows_process_ci_readiness"
        in preflight_step
    )
    tooling_type_routes = [
        line for line in workflow.splitlines() if "uv run mypy tools/agent_repo" in line
    ]
    assert len(tooling_type_routes) == 2
    assert all(
        "tools/windows_process_ci_readiness.py" in line for line in tooling_type_routes
    )
    boundaries = (ROOT / "docs/architecture/package-boundaries.md").read_text("utf-8")
    assert "child-free CIM readiness preflight" in boundaries
    assert "retries only the first timeout once" in boundaries
    for selector in (
        "test_actual_windows_redirector_memory_and_owned_leaf_termination",
        "test_actual_supervisor_cim_failure_stops_owned_leaf_keeps_unrelated",
        "test_actual_supervisor_samples_and_reaps_real_pool_workers",
        "test_actual_stalled_journal_stops_owned_child_and_retains_verified_prefix",
        "test_real_timeout_stops_sleeping_child_and_descendant_and_keeps_failure",
    ):
        assert selector in workflow
    assert "tests/tools/tier_d_nonrl/test_stopped_supervision.py" in workflow


def test_tier_d_tooling_does_not_implement_a_ledger_or_import_live_final_or_ppo():
    owner = ROOT / "tools/tier_d_nonrl"
    paths = tuple(owner.glob("*.py"))
    assert paths
    forbidden = (
        "torch",
        "stable_baselines3",
        "ccxt",
        "requests",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.simulation.accounting",
    )
    for path in paths:
        tree = ast.parse(path.read_text("utf-8"))
        imports = {
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
        }
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in forbidden
        ), path.name
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not calls & {"BookState", "env.step", "env.reset"}, path.name


def test_progress_owners_are_repository_diagnostics_without_domain_execution():
    owner = ROOT / "tools/tier_d_nonrl"
    for name in ("progress.py", "progress_io.py"):
        tree = ast.parse((owner / name).read_text("utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(
            name == "trade_rl" or name.startswith("trade_rl.") for name in imports
        )
        assert not imports & {"subprocess", "concurrent.futures", "multiprocessing"}
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not calls & {
            "run_terminal_allocation_execution",
            "carrier",
            "env.step",
            "env.reset",
        }
    runner = ast.parse((owner / "tier_d_runner.py").read_text("utf-8"))
    observed_calls = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "run_terminal_allocation_execution"
        and any(keyword.arg == "on_advance" for keyword in node.keywords)
    ]
    assert len(observed_calls) == 1
    callback = next(
        keyword.value
        for keyword in observed_calls[0].keywords
        if keyword.arg == "on_advance"
    )
    assert ast.unparse(callback) == "observer.advance"


def _calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.Call, ...]:
    return tuple(node for node in ast.walk(function) if isinstance(node, ast.Call))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_stopped_receipt_reconciliation_is_read_only_and_uses_financial_reader():
    owner = ROOT / "tools/tier_d_nonrl"
    resume_path = owner / "resume.py"
    resume = ast.parse(resume_path.read_text("utf-8"))
    reconcile = _function(resume, "reconcile_stopped_receipts")
    calls = _calls(reconcile)
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in calls
        if isinstance(node.func, (ast.Attribute, ast.Name))
    }

    # The candidate reader shares the existing identity and receipt oracles; it
    # does not execute work or mutate attempt files while it validates them.
    assert {"_restore_frozen_inputs", "_completed_receipt_files"} <= called_names
    assert not called_names & {
        "write_json",
        "save_gzip",
        "write_text",
        "write_bytes",
        "unlink",
        "remove",
        "rmdir",
        "mkdir",
        "rename",
        "replace",
        "run_terminal_allocation_execution",
        "carrier",
        "cost_rows",
        "restore_costs",
        "prepare_inputs",
        "declare_matrix",
        "fit",
    }
    readers = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_payload"
    ]
    assert len(readers) == 1
    assert any(keyword.arg == "expected_plan" for keyword in readers[0].keywords)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "trade_rl.evaluation.allocation_terminal_evidence"
        and any(
            alias.name == "TerminalAllocationExecutionReceipt" for alias in node.names
        )
        for node in resume.body
    )


def test_stopped_reconciliation_stays_separate_from_strict_public_resume():
    owner = ROOT / "tools/tier_d_nonrl"
    runner = ast.parse((owner / "tier_d_runner.py").read_text("utf-8"))
    execute = _function(runner, "execute")
    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "preflight_resume"
        for node in _calls(execute)
    )
    assert not any(
        isinstance(node.func, ast.Name) and node.func.id == "reconcile_stopped_receipts"
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
    )

    supervisor = ast.parse((owner / "tier_d_supervisor.py").read_text("utf-8"))
    publisher = _function(supervisor, "_publish_stopped_receipts")
    publish_calls = [
        node
        for node in _calls(publisher)
        if isinstance(node.func, ast.Name)
        and node.func.id == "reconcile_stopped_receipts"
    ]
    assert len(publish_calls) == 1
    guard = next(
        node
        for node in publisher.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    )
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "not owned_tree_stopped"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_publish_stopped_receipts"
        and any(
            keyword.arg == "owned_tree_stopped"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        for node in ast.walk(_function(supervisor, "supervised_run"))
    )


def test_sampled_rss_attribution_is_separate_from_kill_and_reconciliation_authority():
    owner = ROOT / "tools/tier_d_nonrl"
    process_tree_source = (owner / "tier_d_process_tree.py").read_text("utf-8")
    process_tree = ast.parse(process_tree_source)
    assert "def from_popen(" in process_tree_source
    assert "_process_identity_from_popen(process)" in process_tree_source
    assert "_CIM_CREATION_TOLERANCE_TICKS" in process_tree_source
    sample = next(
        node
        for node in process_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProcessTreeSample"
    )
    sample_fields = {
        node.target.id
        for node in sample.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {"members", "owned_members", "ambiguous_members"} <= sample_fields

    terminate = _function(process_tree, "terminate_owned_tree")
    kill_list = next(
        node.value
        for node in terminate.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "members"
            for target in node.targets
        )
    )
    assert ast.unparse(kill_list) == (
        "[asdict(member) for member in reversed(sample.owned_members)]"
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "members"
        for node in ast.walk(terminate)
    )
    assert "final_sample.ambiguous_members" in ast.unparse(terminate)

    supervisor_source = (owner / "tier_d_supervisor.py").read_text("utf-8")
    assert "owned_process_count=len(initial_sample.owned_members)" in supervisor_source
    assert (
        "rss_attributed_process_count=len(initial_sample.members)" in supervisor_source
    )
    assert "ambiguous_rss_process_count=len(" in supervisor_source
    assert (
        "ambiguous RSS-attributed processes remained after worker exit"
        in supervisor_source
    )
    assert "worker root process birth identity was never observed" in supervisor_source
    assert "ProcessTree.from_popen(process)" in supervisor_source

    scaling_source = (ROOT / "tools/tier_d_scaling/supervision.py").read_text("utf-8")
    assert '"owned_process_count": len(observed.owned_members)' in scaling_source
    assert '"rss_attributed_process_count": len(observed.members)' in scaling_source
    assert (
        '"ambiguous_rss_process_count": len(observed.ambiguous_members)'
        in scaling_source
    )
    assert '"ambiguous_rss_members_remaining": ambiguous_remaining' in scaling_source
    assert "worker root process birth identity was never observed" in scaling_source
    assert "ProcessTree.from_popen(process)" in scaling_source

    architecture = (ROOT / "docs/architecture/package-boundaries.md").read_text("utf-8")
    assert (
        "Historical-only rows and their descendants contribute to the" in architecture
    )
    assert "blocks stopped-receipt reconciliation" in architecture
    normalized_architecture = " ".join(architecture.split())
    assert (
        "unless a snapshot captured the worker root's PID and birth identity"
        in normalized_architecture
    )
    assert (
        "reads its creation FILETIME through the process handle"
        in normalized_architecture
    )
    assert (
        "not a claim that CIM supplied an exact 100-nanosecond timestamp"
        in normalized_architecture
    )
    assert (
        "A visible root row with that missing or mismatched launch identity also contributes RSS as ambiguous evidence, but is never kill-owned."
        in normalized_architecture
    )
