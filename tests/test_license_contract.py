from __future__ import annotations

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPDX_ID = "LGPL-3.0-or-later"


def test_repository_license_contract_is_lgpl_3_or_later() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text

    lgpl_text = (ROOT / "LICENSES" / "LGPL-3.0-or-later.txt").read_text(
        encoding="utf-8"
    )
    gpl_text = (ROOT / "LICENSES" / "GPL-3.0-or-later.txt").read_text(
        encoding="utf-8"
    )
    mit_text = (ROOT / "LICENSES" / "MIT.txt").read_text(encoding="utf-8")

    assert lgpl_text == license_text
    assert "GNU GENERAL PUBLIC LICENSE" in gpl_text
    assert "MIT License" in mit_text


def test_package_metadata_uses_same_spdx_license() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["license"] == SPDX_ID
    assert set(project["license-files"]) >= {
        "LICENSE",
        "LICENSES/*",
        "THIRD_PARTY_NOTICES.md",
    }

    package_json = json.loads((ROOT / "studio" / "package.json").read_text(encoding="utf-8"))
    assert package_json["license"] == SPDX_ID


def test_license_transition_has_provenance_and_notice_documents() -> None:
    licensing = (ROOT / "docs" / "LICENSING.md").read_text(encoding="utf-8")
    provenance = (ROOT / "docs" / "LICENSING_PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert SPDX_ID in licensing
    assert "MIT" in licensing
    assert "historical" in licensing.lower()
    assert "provenance" in provenance.lower()
    assert "AI" in provenance
    assert "NautilusTrader" in notices
    assert "not affiliated" in notices.lower()
