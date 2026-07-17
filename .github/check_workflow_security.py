from __future__ import annotations

import re
import sys
from pathlib import Path

_ACTION_REFERENCE = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)\s*$", re.MULTILINE)
_IMMUTABLE_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _workflow_files(root: Path) -> tuple[Path, ...]:
    workflow_root = root / ".github" / "workflows"
    if not workflow_root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for pattern in ("*.yml", "*.yaml")
            for path in workflow_root.glob(pattern)
            if path.is_file()
        )
    )


def _privileged_errors(path: Path, content: str) -> list[str]:
    relative = path.as_posix()
    errors: list[str] = []
    if "pull_request" in content or "pull_request_target" in content:
        errors.append(f"{relative}: pull_request cannot target a self-hosted runner")
    if "environment: gpu-full-training" not in content:
        errors.append(
            f"{relative}: self-hosted workflow requires gpu-full-training environment"
        )
    if "github.actor == github.repository_owner" not in content:
        errors.append(
            f"{relative}: self-hosted workflow must restrict github.actor to repository_owner"
        )
    if "github.ref == 'refs/heads/main'" not in content and (
        'github.ref == "refs/heads/main"' not in content
    ):
        errors.append(
            f"{relative}: self-hosted workflow must restrict dispatch to refs/heads/main"
        )
    if "contents: write" in content:
        errors.append(f"{relative}: privileged workflow cannot grant contents: write")
    for reference in _ACTION_REFERENCE.findall(content):
        if reference.startswith("./"):
            continue
        if not _IMMUTABLE_ACTION.fullmatch(reference):
            errors.append(f"{relative}: mutable action reference: {reference}")
    return errors


def validate_workflow_security(root: Path) -> tuple[str, ...]:
    """Return deterministic policy violations for repository workflows."""

    errors: list[str] = []
    for path in _workflow_files(root):
        content = path.read_text(encoding="utf-8")
        if "self-hosted" not in content:
            continue
        errors.extend(_privileged_errors(path.relative_to(root), content))
    return tuple(sorted(errors))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(arguments[0]) if arguments else Path.cwd()
    errors = validate_workflow_security(root)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
