"""Scheduled evidence observes upper native seams without another learner/ledger."""

import ast
import re
import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
OWNERS = (
    "scheduled_transition_trace",
    "scheduled_transition_validation",
    "scheduled_transition_trace_io",
)


def test_scheduled_trace_owners_do_not_resample_or_import_optional_backends():
    for name in OWNERS:
        path = PACKAGE / "evaluation/rl_allocation" / f"{name}.py"
        imports = ImportCollector(PACKAGE).collect(path)
        assert not any(
            within_module(module, prefix)
            for module in imports
            for prefix in ("torch", "stable_baselines3")
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        attributes = [
            node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        ]
        assert not any(
            node.attr
            in (
                "evaluate_actions",
                "predict",
                "forward",
                "apply_fill",
                "execute_interval",
                "execute_allocation_action",
            )
            for node in attributes
        ), path
        assert not any(
            node.attr == "get" and "buffer" in ast.unparse(node.value)
            for node in attributes
        ), path
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','stable_baselines3'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from trade_rl.evaluation.rl_allocation.scheduled_training import fit_allocation_ppo_schedule
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace import ScheduledAllocationTransitionRecorder
from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import validate_scheduled_transition_events
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace_io import read_scheduled_allocation_transition_trace
assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_scheduled_trace_default_and_lower_receipts_remain_separate():
    tree = ast.parse(
        (PACKAGE / "evaluation/rl_allocation/scheduled_training.py").read_text(
            encoding="utf-8"
        )
    )
    fit = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "fit_allocation_ppo_schedule"
    )
    index = next(
        i
        for i, node in enumerate(fit.args.kwonlyargs)
        if node.arg == "transition_trace"
    )
    assert (
        isinstance(fit.args.kw_defaults[index], ast.Constant)
        and fit.args.kw_defaults[index].value is None
    )
    for name in (
        "allocation_manifest",
        "allocation_schedule_receipt",
        "allocation_training_schedule",
        "allocation_protocol_receipt",
    ):
        path = PACKAGE / "strategies/rl" / f"{name}.py"
        imports = ImportCollector(PACKAGE).collect(path)
        assert not any(
            within_module(module, "trade_rl.evaluation.rl_allocation")
            for module in imports
        ), path
        assert "scheduled_transition_trace" not in path.read_text(encoding="utf-8")
    env = ast.parse(
        (PACKAGE / "evaluation/rl_allocation/env.py").read_text(encoding="utf-8")
    )
    protocol = next(
        node
        for node in env.body
        if isinstance(node, ast.ClassDef) and node.name == "AllocationExecutionObserver"
    )
    assert {
        node.name for node in protocol.body if isinstance(node, ast.FunctionDef)
    } == {"freeze_execution"}


def test_both_actual_trace_and_delayed_actor_modules_are_in_real_runtime_job():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = re.search(r"^  ppo-runtime:.*?(?=^  [\w-]+:|\Z)", workflow, re.M | re.S)
    assert job is not None
    commands = re.findall(
        r"^      - name: Real SB3 PPO integration\n        run: ([^\n]+)",
        job.group(),
        re.M,
    )
    assert len(commands) == 1
    for name in (
        "test_allocation_scheduled_transition_runtime.py",
        "test_allocation_scheduled_delayed_credit_runtime.py",
    ):
        assert "tests/integrations/" + name in commands[0]
    for name, heading in (
        ("architecture/lean-core.md", "Scheduled actor and native learner sidecar"),
        (
            "architecture/package-boundaries.md",
            "Scheduled allocation transition evidence ownership",
        ),
        (
            "architecture/research-assurance.md",
            "Scheduled actor trace and finite delayed credit G0-G2",
        ),
        (
            "research/current-status.md",
            "scheduled actor trace and fixed delayed-credit candidate",
        ),
    ):
        assert heading in (ROOT / "docs" / name).read_text(encoding="utf-8")
