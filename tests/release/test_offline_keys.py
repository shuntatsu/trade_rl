from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from trade_rl.release.offline_keys import load_offline_signing_key


def _write_key(path: Path, *, purpose: str = "selection-authorization") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ed25519_private_key_v1",
                "algorithm": "ed25519",
                "key_id": "offline-key",
                "purpose": purpose,
                "private_key": base64.b64encode(b"x" * 32).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_offline_key_loader_requires_exact_purpose_and_private_permissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "key.json"
    _write_key(path)

    key = load_offline_signing_key(path, required_purpose="selection-authorization")

    assert key.key_id == "offline-key"
    assert key.purpose == "selection-authorization"
    assert len(key.private_key.private_bytes_raw()) == 32

    with pytest.raises(ValueError, match="purpose"):
        load_offline_signing_key(path, required_purpose="release-verification")

    if os.name == "posix":
        path.chmod(0o644)
        with pytest.raises(PermissionError, match="permissions"):
            load_offline_signing_key(path, required_purpose="selection-authorization")


def test_offline_key_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "key.json"
    _write_key(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="fields"):
        load_offline_signing_key(path, required_purpose="selection-authorization")
