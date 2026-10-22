"""Keep bounded B4 receipt parsing outside production and execution authority."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "tools.market_b4_oracle"
INTERNAL_SEMANTIC_COMPONENTS = frozenset(
    {
        "binary64_account.py",
        "artificial_component.py",
        "artificial_spec.py",
        "decision_identity.py",
        "expected_intent.py",
        "frozen_decision.py",
        "frozen_forecast.py",
        "frozen_forecast_index.py",
        "frozen_market_replay.py",
        "offline_replay_consumer.py",
        "outer_witness.py",
    }
)
PUBLIC_FACADES = {
    "tools.market_account_oracle",
    "tools.market_replay_oracle",
}
FORBIDDEN_STDLIB_ROOTS = {
    "asyncio",
    "concurrent",
    "ftplib",
    "http",
    "multiprocessing",
    "socket",
    "smtplib",
    "ssl",
    "subprocess",
    "threading",
    "urllib",
}


def _public_exports(module: str) -> set[str]:
    facade = ROOT.joinpath(*module.split("/"), "__init__.py")
    tree = ast.parse(facade.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list) and all(
                isinstance(item, str) for item in value
            )
            return set(value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                value = ast.literal_eval(node.value)
                assert isinstance(value, list) and all(
                    isinstance(item, str) for item in value
                )
                return set(value)
    return set()


def _resolved_imports(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    module_path = path.relative_to(ROOT).with_suffix("")
    package = list(module_path.parts[:-1])
    imports: list[tuple[str, tuple[str, ...]]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imports.append(
                        (node.module, tuple(alias.name for alias in node.names))
                    )
                continue
            base = package[: len(package) - node.level + 1]
            if node.module:
                base.extend(node.module.split("."))
                imports.append(
                    (".".join(base), tuple(alias.name for alias in node.names))
                )
            else:
                imports.extend(
                    (".".join((*base, alias.name)), ()) for alias in node.names
                )
    return imports


def _is_allowed_import(module: str) -> bool:
    root = module.split(".", 1)[0]
    return root not in FORBIDDEN_STDLIB_ROOTS and (
        root in sys.stdlib_module_names
        or module == "numpy"
        or module.startswith("numpy.")
        or module == OWNER
        or module.startswith(OWNER + ".")
        or module in PUBLIC_FACADES
    )


def test_composition_imports_are_limited_to_stdlibrary_numpy_and_public_facades() -> (
    None
):
    sources = tuple(sorted((ROOT / OWNER.replace(".", "/")).rglob("*.py")))
    assert sources
    public_exports = {
        facade: _public_exports(facade.replace(".", "/")) for facade in PUBLIC_FACADES
    }
    for path in sources:
        for module, imported_names in _resolved_imports(path):
            assert _is_allowed_import(module), (path.relative_to(ROOT), module)
            if module in PUBLIC_FACADES:
                assert set(imported_names) <= public_exports[module], (
                    path.relative_to(ROOT),
                    module,
                    imported_names,
                )
            assert not module.startswith("trade_rl"), (path.relative_to(ROOT), module)


def test_replay_identity_primitives_are_exported_by_the_public_facade() -> None:
    exports = _public_exports("tools/market_replay_oracle")
    assert {"ARRAY_FIELDS", "canonical", "content_id"} <= exports


def test_production_modules_do_not_import_the_composition_package() -> None:
    for path in (ROOT / "trade_rl").rglob("*.py"):
        for module, _ in _resolved_imports(path):
            assert module != OWNER and not module.startswith(OWNER + "."), (
                path.relative_to(ROOT),
                module,
            )


def test_unfinished_composition_facade_grants_no_public_result_api() -> None:
    assert _public_exports(OWNER.replace(".", "/")) == set()
    tree = ast.parse((ROOT / "tools/market_b4_oracle/__init__.py").read_text())
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)


def test_composition_package_is_in_both_research_type_routes_and_lf_scoped() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    routes = [
        line
        for line in workflow.splitlines()
        if "uv run mypy" in line and "tools/market_replay_oracle" in line
    ]
    assert len(routes) == 2
    assert all("tools/market_b4_oracle" in line for line in routes)
    assert all("tools/market_b4_outer_runner.py" in line for line in routes)
    assert all(
        "tools/market_b4_artificial_component_operator.py" in line for line in routes
    )
    assert workflow.count("tools/market_b4_oracle") == 2
    assert workflow.count("tools/market_b4_artificial_component_operator.py") == 2

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "tools/market_b4_oracle/*.py text eol=lf" in attributes
    assert "tools/market_b4_outer_runner.py text eol=lf" in attributes
    assert "tools/market_b4_artificial_component_operator.py text eol=lf" in attributes


def test_artificial_h_operator_has_no_q_comparator_or_native_market_cli_imports() -> (
    None
):
    path = ROOT / "tools/market_b4_artificial_component_operator.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for module, _ in _resolved_imports(path):
        root = module.split(".", 1)[0]
        assert (
            root in sys.stdlib_module_names
            or root == "numpy"
            or module == "trade_rl"
            or module.startswith(("trade_rl.", "tools.tier_d_nonrl"))
        ), (module,)
        assert not module.startswith(
            (
                "tools.market_b4_oracle",
                "tools.market_b4_outer_runner",
                "tools.market_replay_oracle",
                "tools.market_account_oracle",
            )
        ), (module,)
    forbidden_calls = {"fit", "fit_predict", "run_market", "prepare_inputs"}
    assert not any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in forbidden_calls)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden_calls
            )
        )
        for node in ast.walk(tree)
    )


def test_artificial_h_operator_boundary_is_documented() -> None:
    boundaries = (ROOT / "docs/architecture/package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert "`tools/market_b4_artificial_component_operator.py`" in boundaries
    assert "ARTIFICIAL_COMPARISON_ONLY" in boundaries
    assert "do not authenticate its producer" in boundaries
    assert "`operator-failure.json`" in boundaries
    assert "re-raises the original exception" in boundaries
    assert "Failure cleanup removes partial" in boundaries
    assert "existing five-file" in boundaries
    assert "Q's shared pair" in boundaries
    assert "Dataset before consuming either role receipt" in boundaries
    assert "Windows alternate data streams" in boundaries
    assert "before checking target absence or opening the sidecar" in boundaries


def test_outer_runner_default_cli_is_offline_and_pins_observer_dependency() -> None:
    runner = (ROOT / "tools/market_b4_outer_runner.py").read_text(encoding="utf-8")
    runner_tree = ast.parse(runner)
    main = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    observer = (ROOT / "tools/market_b4_oracle/outer_witness.py").read_text(
        encoding="utf-8"
    )
    assert '"native execution requires a frozen H source-root' in runner
    assert '"process_started": False' in runner
    assert "def run_and_consume_frozen_h_market_matrix(" in runner
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "Popen"
        for node in ast.walk(main)
    )
    assert '"tools" / "tier_d_nonrl" / "tier_d_process_tree.py"' in observer


def test_frozen_h_plan_and_matrix_consumer_remain_unlaunched_comparison_scope() -> None:
    runner_path = ROOT / "tools/market_b4_outer_runner.py"
    runner = runner_path.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner)
    plan_builder = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_frozen_h_launch_plan"
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "Popen"
        for node in ast.walk(plan_builder)
    )
    assert '"tools.tier_d_nonrl"' in runner
    assert '"--mode"' in runner and '"market"' in runner
    assert '"--worker"' not in runner
    assert '"--resume-from"' not in runner

    consumer_path = ROOT / "tools/market_b4_oracle/offline_replay_consumer.py"
    consumer = consumer_path.read_text(encoding="utf-8")
    consumer_tree = ast.parse(consumer)
    matrix_consumer = next(
        node
        for node in consumer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "consume_indexed_market_matrix"
    )
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "recheck"
            for node in ast.walk(matrix_consumer)
        )
        == 2
    )
    assert "output_index_rechecks=2" in consumer
    assert "full_b4_authorized: Literal[False] = False" in consumer
    assert "def derive_frozen_native_market_roster(" in consumer


def test_native_arm_checks_pinned_v4_event_rows_before_q_replay() -> None:
    consumer_path = ROOT / "tools/market_b4_oracle/offline_replay_consumer.py"
    tree = ast.parse(consumer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(function: ast.FunctionDef) -> list[tuple[str, ast.Call]]:
        found = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            found.append((name, node))
        return sorted(found, key=lambda item: item[1].lineno)

    arm = functions["_consume_indexed_market_arm"]
    arm_calls = calls(arm)
    validation_line = next(
        node.lineno for name, node in arm_calls if name == "_validate_native_event_rows"
    )
    forecast_line = next(
        node.lineno for name, node in arm_calls if name == "FrozenForecastIndex"
    )
    replay_line = next(node.lineno for name, node in arm_calls if name == "advance")
    assert validation_line < forecast_line < replay_line

    root_arm_calls = calls(functions["consume_root_observed_market_arm"])
    gate_line = next(
        node.lineno
        for name, node in root_arm_calls
        if name == "_validate_root_observed_attempt"
    )
    consumer_line = next(
        node.lineno
        for name, node in root_arm_calls
        if name == "consume_indexed_market_arm"
    )
    assert gate_line < consumer_line


def test_identity_composer_stays_internal_and_reuses_expected_intent():
    package = ROOT / "tools" / "market_b4_oracle"
    existing = {path.name for path in package.glob("*.py")}
    assert INTERNAL_SEMANTIC_COMPONENTS <= existing
    assert not (
        {"ActorIdentity", "OwnedAccountComponents", "build_actor_identity"}
        & _public_exports(OWNER.replace(".", "/"))
    )

    imports = _resolved_imports(package / "decision_identity.py")
    assert any(
        module == "tools.market_b4_oracle.expected_intent" and "ExpectedIntent" in names
        for module, names in imports
    )

    identity_tree = ast.parse((package / "decision_identity.py").read_text())
    public_builder = next(
        node
        for node in identity_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_actor_identity"
    )
    assert not public_builder.args.posonlyargs
    assert tuple(argument.arg for argument in public_builder.args.args) == (
        "account",
        "allocation_inputs",
        "allocator",
    )
    assert public_builder.args.vararg is None
    assert tuple(argument.arg for argument in public_builder.args.kwonlyargs) == (
        "independent_target_weight",
        "independent_objective_value",
        "decision_fields",
        "action_contract",
        "action_code",
        "final_risk_weights",
    )
    assert public_builder.args.kwarg is None
    assert not public_builder.args.defaults
    assert public_builder.args.kw_defaults == [None] * 6

    public_mapper = next(
        node
        for node in identity_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "owned_order_mapping"
    )
    assert not public_mapper.args.posonlyargs
    assert tuple(argument.arg for argument in public_mapper.args.args) == (
        "order",
        "account",
    )
    assert public_mapper.args.vararg is None
    assert not public_mapper.args.kwonlyargs
    assert public_mapper.args.kwarg is None
    assert not public_mapper.args.defaults
    assert not public_mapper.args.kw_defaults

    exported_names = next(
        ast.literal_eval(node.value)
        for node in identity_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert exported_names == [
        "ActorIdentity",
        "OwnedAccountComponents",
        "OwnedIntentComponents",
        "OwnedPendingOrder",
        "build_actor_identity",
        "canonical_json_bytes",
        "owned_order_mapping",
    ]
