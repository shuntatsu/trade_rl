"""The W01 R2 archival tool is read-only and has no trading runtime authority."""

from __future__ import annotations

import ast
from pathlib import Path

from tools import issue810_r2_release_audit as audit

SOURCE = Path(__file__).resolve().parents[2] / "tools" / "issue810_r2_release_audit.py"
ALLOWED_IMPORTS = {
    "argparse",
    "hashlib",
    "io",
    "json",
    "re",
    "stat",
    "zipfile",
    "pathlib",
    "typing",
    "__future__",
}


def test_issue810_release_auditor_imports_no_market_or_transport_owner() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports == ALLOWED_IMPORTS


def test_issue810_release_auditor_pins_original_market_result_and_no_approval() -> None:
    assert audit.R2_ZIP_BYTES == 34_538_439
    assert audit.R2_ZIP_SHA256 == (
        "03bcbd613a516e9e0504d5c074297a896d3e680d87c29adcdbc2fad2d4dc1556"
    )
    assert audit.R2_MANIFEST_SHA256 == (
        "9e1c4ff1d0f30e9fcb98fc915edcf10348e8dfdce1a04a14b1b7acabe312935e"
    )
    assert audit.R2_HEAD == "30c5ab54fdb9db7f03b947c1687a09c8b929c461"
    assert len(audit.R2_SOURCE_FILES) == 8
    assert len(audit.R2_ROOT_FILES) == 13


DATASET_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "issue810_r2_dataset_source_preflight.py"
)
DATASET_ALLOWED_IMPORTS = {
    "argparse",
    "hashlib",
    "json",
    "os",
    "stat",
    "dataclasses",
    "pathlib",
    "typing",
    "__future__",
}


def test_r2_dataset_source_preflight_has_no_trading_or_network_owner() -> None:
    tree = ast.parse(DATASET_SOURCE.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports == DATASET_ALLOWED_IMPORTS


def test_r2_dataset_preflight_cannot_grant_source_pit_or_economic_authority() -> None:
    from tools import issue810_r2_dataset_source_preflight as preflight

    assert preflight.R2_DATASET_ID == (
        "e916ff716fb5e4b90d5805eaa79200824b09f103984fa845bbeaedb57bef194f"
    )
    assert preflight.ORIGINAL_PINS.manifest_sha256 == (
        "6ac5099b98a32bd33e4f98c5ac5fb2dbffd19b0b9e3c7c3a9b5a8f6860810ee6"
    )
    assert preflight.ORIGINAL_PINS.arrays_sha256 == (
        "25c97a4ebd1e81ac64f4bc7e78dce4ab102c522e548e30a2154fcfa0143842a3"
    )
    assert preflight.R2_ROWS == 9504
    assert len(preflight.R2_FEATURES) == 6
