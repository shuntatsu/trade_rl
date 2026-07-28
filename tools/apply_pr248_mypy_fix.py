from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} anchor count was {count}, expected 1")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"applied: {label}")


def main() -> None:
    replace_once(
        Path("trade_rl/workflows/config_fields.py"),
        "    model: type[object],\n",
        "    model: Any,\n",
        label="dataclass model type",
    )

    policy_loader = Path("trade_rl/serving/policy_loader.py")
    replace_once(
        policy_loader,
        "from typing import Any, Final\n",
        "from typing import Any, Final, TypedDict\n",
        label="typed dict import",
    )
    replace_once(
        policy_loader,
        'STRUCTURED_POLICY_LOADER_SCHEMA: Final = "structured_policy_loader_v1"\n\n\n',
        '''STRUCTURED_POLICY_LOADER_SCHEMA: Final = "structured_policy_loader_v1"


class StructuredPolicyLoaderMember(TypedDict):
    manifest: str
    manifest_digest: str
    model: str
    model_digest: str


class StructuredPolicyLoaderManifest(TypedDict):
    action_size: int
    architecture_digest: str
    digest: str
    members: tuple[StructuredPolicyLoaderMember, ...]
    schema_version: str


''',
        label="loader typed dicts",
    )
    replace_once(
        policy_loader,
        "def load_structured_policy_loader_manifest(path: Path) -> dict[str, object]:\n",
        "def load_structured_policy_loader_manifest(\n"
        "    path: Path,\n"
        ") -> StructuredPolicyLoaderManifest:\n",
        label="loader return type",
    )
    replace_once(
        policy_loader,
        '    architecture_digest = payload.get("architecture_digest")\n'
        '    require_sha256(architecture_digest, field="architecture_digest")\n',
        '    architecture_digest = payload.get("architecture_digest")\n'
        '    if not isinstance(architecture_digest, str):\n'
        '        raise ValueError("structured policy loader architecture digest is invalid")\n'
        '    require_sha256(architecture_digest, field="architecture_digest")\n',
        label="architecture digest type",
    )
    replace_once(
        policy_loader,
        '    if not isinstance(raw_members, list) or not raw_members:\n'
        '        raise ValueError("structured policy loader members must be a non-empty list")\n'
        '    members: list[dict[str, object]] = []\n',
        '    if not isinstance(raw_members, list) or not raw_members:\n'
        '        raise ValueError("structured policy loader members must be a non-empty list")\n'
        '    members: list[StructuredPolicyLoaderMember] = []\n',
        label="member list type",
    )
    replace_once(
        policy_loader,
        '        manifest_digest = member.get("manifest_digest")\n'
        '        model_digest = member.get("model_digest")\n'
        '        require_sha256(manifest_digest, field="manifest_digest")\n'
        '        require_sha256(model_digest, field="model_digest")\n',
        '        manifest_digest = member.get("manifest_digest")\n'
        '        model_digest = member.get("model_digest")\n'
        '        if not isinstance(manifest_digest, str):\n'
        '            raise ValueError("structured member manifest digest is invalid")\n'
        '        if not isinstance(model_digest, str):\n'
        '            raise ValueError("structured member model digest is invalid")\n'
        '        require_sha256(manifest_digest, field="manifest_digest")\n'
        '        require_sha256(model_digest, field="model_digest")\n',
        label="member digest types",
    )
    replace_once(
        policy_loader,
        '    digest = payload.get("digest")\n'
        '    require_sha256(digest, field="structured_policy_loader.digest")\n',
        '    digest = payload.get("digest")\n'
        '    if not isinstance(digest, str):\n'
        '        raise ValueError("structured policy loader digest is invalid")\n'
        '    require_sha256(digest, field="structured_policy_loader.digest")\n',
        label="loader digest type",
    )
    replace_once(
        policy_loader,
        '    return {"digest": digest, **digest_payload}\n',
        '''    return {
        "action_size": action_size,
        "architecture_digest": architecture_digest,
        "digest": digest,
        "members": tuple(members),
        "schema_version": STRUCTURED_POLICY_LOADER_SCHEMA,
    }
''',
        label="typed loader return",
    )
    replace_once(
        policy_loader,
        "    fallback: object | None = None,\n) -> object:\n",
        "    fallback: object | None = None,\n) -> Any:\n",
        label="canonical loader return type",
    )


if __name__ == "__main__":
    main()
