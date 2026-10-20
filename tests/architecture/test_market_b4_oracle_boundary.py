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
        "decision_identity.py",
        "expected_intent.py",
        "frozen_decision.py",
        "frozen_forecast.py",
        "frozen_forecast_index.py",
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
    assert workflow.count("tools/market_b4_oracle") == 2

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "tools/market_b4_oracle/*.py text eol=lf" in attributes


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
