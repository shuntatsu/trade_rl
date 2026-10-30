from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
STORE = PACKAGE / "integrations" / "market_store"
DEPLOY = ROOT / "deploy" / "market-store"

MODULES = {
    "__init__.py",
    "__main__.py",
    "checksum.py",
    "cli.py",
    "contracts.py",
    "embargo.py",
    "export.py",
    "fetch.py",
    "ingest.py",
    "parse.py",
    "plan.py",
    "rules.py",
    "store.py",
}
NETWORK = {"socket", "http", "urllib", "requests", "ssl", "ftplib", "httpx", "aiohttp"}


def _module_scope_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _all_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_store_layout_is_the_documented_roster() -> None:
    assert {path.name for path in STORE.glob("*.py")} == MODULES
    names = sorted(path.name for path in (STORE / "migrations").glob("*.sql"))
    assert names == ["0000_schema_version.sql", "0001_market_store.sql"]
    assert all(re.fullmatch(r"\d{4}_[a-z0-9_]+\.sql", name) for name in names)


def test_store_depends_only_on_lower_layers() -> None:
    collector = ImportCollector(PACKAGE)
    forbidden = (
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.risk",
    )
    for path in sorted(STORE.glob("*.py")):
        imports = collector.collect(path)
        assert not any(
            within_module(name, prefix) for name in imports for prefix in forbidden
        ), path.name


def test_only_store_imports_psycopg_and_only_fetch_imports_a_network_client() -> None:
    for path in sorted(STORE.glob("*.py")):
        top = {name.split(".", 1)[0] for name in _all_imports(path)}
        assert ("psycopg" in top) == (path.name == "store.py"), path.name
        assert bool(top & NETWORK) == (path.name == "fetch.py"), path.name


def test_cli_loads_database_and_network_modules_lazily() -> None:
    imported = _module_scope_imports(STORE / "cli.py")
    heavy = {
        "trade_rl.integrations.market_store.store",
        "trade_rl.integrations.market_store.fetch",
        "trade_rl.integrations.market_store.export",
        "psycopg",
    }
    assert not imported & heavy
    assert _module_scope_imports(STORE / "__init__.py") == set()


def test_the_binance_facade_and_integrations_facade_are_unchanged() -> None:
    import trade_rl.integrations as integrations

    assert "market_store" not in integrations.__all__
    text = (PACKAGE / "integrations" / "__init__.py").read_text(encoding="utf-8")
    assert "market_store" not in text


def test_pyproject_pins_psycopg_for_the_store_extra_and_ships_migrations() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    store = [item for item in extras["store"] if item.startswith("psycopg")]
    assert len(store) == 1 and re.fullmatch(
        r"psycopg\[binary\]==\d+\.\d+\.\d+", store[0]
    )
    assert store[0] in extras["dev"]
    assert not any("psycopg" in item for item in project["project"]["dependencies"])
    package_data = project["tool"]["setuptools"]["package-data"]
    assert package_data["trade_rl.integrations.market_store"] == ["migrations/*.sql"]


def test_compose_binds_loopback_pins_the_image_and_keeps_secrets_out() -> None:
    compose = (DEPLOY / "compose.yml").read_text(encoding="utf-8")
    assert re.search(r"image: postgres:\d+\.\d+\s", compose)
    assert "latest" not in compose
    assert '"127.0.0.1:${MARKET_STORE_PORT' in compose
    assert "healthcheck:" in compose and "pg_isready" in compose
    assert "market_store_data:/var/lib/postgresql/data" in compose
    assert "MARKET_STORE_PASSWORD:?" in compose
    example = (DEPLOY / ".env.example").read_text(encoding="utf-8")
    assert "MARKET_STORE_PASSWORD=CHANGE_ME" in example
    ignored = subprocess.run(
        ["git", "check-ignore", "deploy/market-store/.env"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0
    not_ignored = subprocess.run(
        ["git", "check-ignore", "deploy/market-store/.env.example"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert not_ignored.returncode == 1
    assert not (DEPLOY / ".env").exists() or ignored.returncode == 0


def test_ci_runs_the_postgres_integration_tests_in_a_service_container() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = workflow.split("  market-store:\n", 1)[1].split("  independent-review:\n", 1)[
        0
    ]
    assert "image: postgres:16.4" in job
    assert "TRADE_RL_TEST_POSTGRES_URL" in job
    assert "--extra store" in job
    assert "tests/integrations/market_store" in job
    assert 'skipped="0"' in job
    assert "pull_request" in job


def test_runbook_documents_embargo_release_and_the_non_goals() -> None:
    text = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert "## Market-data store runbook" in text
    for required in (
        "integrations/market_store/",
        "--allow-network",
        "--release-sealed-window",
        "pg_dump",
        "TRADE_RL_MARKET_STORE_URL",
        "TRADE_RL_TEST_POSTGRES_URL",
    ):
        assert required in text, required
