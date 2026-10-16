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
    assert workflow.count("tools/tier_d_nonrl") == 2
    assert not any(
        (ROOT / "output/issue810-tier-d-nonrl-preparation" / name).exists()
        for name in ("tier_d_runner.py", "tier_d_supervisor.py", "test_runner.py")
    )


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
