from __future__ import annotations

import io
import tomllib
from importlib.metadata import version
from pathlib import Path

import trade_rl
from trade_rl.artifacts.provenance import source_tree_digest
from trade_rl.cli import main as cli_main

_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_version_matches_installed_package_metadata() -> None:
    assert trade_rl.__version__ == version("trade-rl")


def test_cli_version_matches_installed_package_metadata() -> None:
    output = io.StringIO()

    assert cli_main(["--version"], stdout=output) == 0
    assert output.getvalue() == f"trade-rl {version('trade-rl')}\n"


def test_package_metadata_uses_canonical_version_module() -> None:
    payload = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert "version" not in project
    assert "version" in project["dynamic"]
    assert payload["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "trade_rl._version.__version__"
    }


def test_uv_toolchain_has_one_required_version_source() -> None:
    uv_config = _ROOT / "uv.toml"
    assert uv_config.is_file()
    payload = tomllib.loads(uv_config.read_text(encoding="utf-8"))
    required = payload.get("required-version")
    assert required == "==0.10.0"

    dockerfile = (_ROOT / "Dockerfile.training").read_text(encoding="utf-8")
    assert "COPY uv.toml /tmp/trade-rl-uv.toml" in dockerfile
    assert 'tomllib.load(open("/tmp/trade-rl-uv.toml", "rb"))' in dockerfile
    assert "pip install --no-cache-dir uv==0.10.0" not in dockerfile


def test_setup_uv_workflows_resolve_version_from_repository_config() -> None:
    workflows = sorted((_ROOT / ".github" / "workflows").glob("*.yml"))
    setup_workflows = tuple(
        path
        for path in workflows
        if "astral-sh/setup-uv@" in path.read_text(encoding="utf-8")
    )
    assert setup_workflows
    for path in setup_workflows:
        text = path.read_text(encoding="utf-8")
        assert text.index("actions/checkout@") < text.index("astral-sh/setup-uv@"), path
        assert "version: latest" not in text, path


def test_uv_toolchain_version_is_bound_to_source_identity(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package_root = root / "trade_rl"
    package_root.mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='trade-rl'\n")
    (root / "uv.lock").write_text("lock")
    (root / "uv.toml").write_text('required-version = "==0.10.0"\n')
    (package_root / "module.py").write_text("module")
    (root / "examples" / "runner.py").write_text("runner")
    before = source_tree_digest(root)

    (root / "uv.toml").write_text('required-version = "==0.10.1"\n')

    assert source_tree_digest(root) != before
