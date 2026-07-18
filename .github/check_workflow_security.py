from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

_IMMUTABLE_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_HOSTED_RUNNER = re.compile(r"^(ubuntu|windows|macos)-[A-Za-z0-9_.-]+$")


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


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _load(path: Path) -> Mapping[str, object]:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except yaml.YAMLError as error:
        raise ValueError(
            f"{path.as_posix()}: invalid workflow YAML: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.as_posix()}: workflow root must be an object")
    return _mapping(payload)


def _trigger_names(workflow: Mapping[str, object]) -> set[str]:
    raw = workflow.get("on")
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return {str(item) for item in raw}
    return set(_mapping(raw))


def _runner_labels(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in _sequence(value))


def _is_privileged_runner(value: object) -> bool:
    labels = _runner_labels(value)
    if not labels:
        return False
    if any(label == "self-hosted" for label in labels):
        return True
    for label in labels:
        if label.startswith("${{"):
            continue
        if not _HOSTED_RUNNER.fullmatch(label):
            return True
    return False


def _environment_name(value: object) -> str | None:
    if isinstance(value, str):
        return value
    name = _mapping(value).get("name")
    return name if isinstance(name, str) else None


def _permissions_write(value: object) -> bool:
    if isinstance(value, str):
        return value == "write-all"
    return any(str(permission) == "write" for permission in _mapping(value).values())


def _iter_steps(job: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [_mapping(item) for item in _sequence(job.get("steps"))]


def _action_errors(relative: str, jobs: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for job_name, raw_job in jobs.items():
        job = _mapping(raw_job)
        for index, step in enumerate(_iter_steps(job)):
            reference = step.get("uses")
            if not isinstance(reference, str) or reference.startswith("./"):
                continue
            if not _IMMUTABLE_ACTION.fullmatch(reference):
                errors.append(
                    f"{relative}: jobs.{job_name}.steps[{index}] mutable action reference: {reference}"
                )
    return errors


def _checkout_errors(
    relative: str,
    job_name: str,
    job: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    checkout_steps = [
        step
        for step in _iter_steps(job)
        if isinstance(step.get("uses"), str)
        and str(step["uses"]).startswith("actions/checkout@")
    ]
    if not checkout_steps:
        errors.append(
            f"{relative}: jobs.{job_name} requires an immutable checkout step"
        )
        return errors
    for step in checkout_steps:
        settings = _mapping(step.get("with"))
        if settings.get("ref") != "${{ github.sha }}":
            errors.append(
                f"{relative}: jobs.{job_name} checkout must use ref: ${{{{ github.sha }}}}"
            )
        if settings.get("persist-credentials") != "false":
            errors.append(
                f"{relative}: jobs.{job_name} checkout must disable persist-credentials"
            )
    return errors


def _contains_secret_reference(value: object) -> bool:
    if isinstance(value, str):
        return "secrets." in value
    if isinstance(value, Mapping):
        return any(_contains_secret_reference(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_secret_reference(item) for item in value)
    return False


def _is_read_only_monitor(
    workflow: Mapping[str, object],
    job_name: str,
    job: Mapping[str, object],
) -> bool:
    if "monitor" not in job_name.lower():
        return False
    triggers = _trigger_names(workflow)
    if "schedule" not in triggers or not triggers <= {"schedule", "workflow_dispatch"}:
        return False
    if _environment_name(job.get("environment")) is not None:
        return False
    condition = job.get("if")
    condition_text = condition if isinstance(condition, str) else ""
    if "github.ref == 'refs/heads/main'" not in condition_text and (
        'github.ref == "refs/heads/main"' not in condition_text
    ):
        return False
    if "workflow_dispatch" in triggers and (
        "github.actor == github.repository_owner" not in condition_text
    ):
        return False
    if _contains_secret_reference(job):
        return False
    run_scripts = "\n".join(
        str(step.get("run", "")) for step in _iter_steps(job) if step.get("run")
    )
    if "full_run_supervisor.py status" not in run_scripts:
        return False
    if re.search(r"full_run_supervisor\.py\s+(start|stop)", run_scripts):
        return False
    return True


def _privileged_errors(
    relative: str,
    workflow: Mapping[str, object],
    jobs: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    triggers = _trigger_names(workflow)
    workflow_permissions = workflow.get("permissions")
    for job_name, raw_job in jobs.items():
        job = _mapping(raw_job)
        if not _is_privileged_runner(job.get("runs-on")):
            continue
        if triggers & {"pull_request", "pull_request_target"}:
            errors.append(
                f"{relative}: pull_request cannot target a self-hosted or custom runner"
            )
        monitor = _is_read_only_monitor(workflow, job_name, job)
        if (
            not monitor
            and _environment_name(job.get("environment")) != "gpu-full-training"
        ):
            errors.append(
                f"{relative}: jobs.{job_name} requires gpu-full-training environment"
            )
        condition = job.get("if")
        condition_text = condition if isinstance(condition, str) else ""
        if (
            not monitor
            and "github.actor == github.repository_owner" not in condition_text
        ):
            errors.append(
                f"{relative}: jobs.{job_name} must restrict github.actor to repository_owner"
            )
        if "github.ref == 'refs/heads/main'" not in condition_text and (
            'github.ref == "refs/heads/main"' not in condition_text
        ):
            errors.append(
                f"{relative}: jobs.{job_name} must restrict dispatch to refs/heads/main"
            )
        if _permissions_write(workflow_permissions) or _permissions_write(
            job.get("permissions")
        ):
            errors.append(
                f"{relative}: jobs.{job_name} privileged workflow cannot grant write permissions"
            )
        errors.extend(_checkout_errors(relative, job_name, job))
    return errors


def validate_workflow_security(root: Path) -> tuple[str, ...]:
    """Return deterministic structural policy violations for workflows."""

    errors: list[str] = []
    for path in _workflow_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            workflow = _load(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        jobs = _mapping(workflow.get("jobs"))
        errors.extend(_action_errors(relative, jobs))
        errors.extend(_privileged_errors(relative, workflow, jobs))
    return tuple(sorted(set(errors)))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(arguments[0]) if arguments else Path.cwd()
    errors = validate_workflow_security(root)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
