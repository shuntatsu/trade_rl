from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "trade_rl/rl/structured_export.py"


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    old_build = '''        payload = {
            "action_size": action_size,
            "architecture_digest": architecture_digest,
            "inputs": inputs,
            "max_abs_error": max_abs_error,
            "model_digest": _file_digest(model_path),
            "model_path": STRUCTURED_EXPORT_MODEL_NAME,
            "model_size_bytes": model_path.stat().st_size,
            "policy_identity": policy_payload,
            "policy_identity_digest": content_digest(policy_payload),
            "schema_version": STRUCTURED_EXPORT_SCHEMA,
            "tolerance": tolerance,
        }
        return cls(digest=content_digest(payload), **payload)
'''
    new_build = '''        model_digest = _file_digest(model_path)
        model_size_bytes = model_path.stat().st_size
        policy_identity_digest = content_digest(policy_payload)
        payload = {
            "action_size": action_size,
            "architecture_digest": architecture_digest,
            "inputs": inputs,
            "max_abs_error": max_abs_error,
            "model_digest": model_digest,
            "model_path": STRUCTURED_EXPORT_MODEL_NAME,
            "model_size_bytes": model_size_bytes,
            "policy_identity": policy_payload,
            "policy_identity_digest": policy_identity_digest,
            "schema_version": STRUCTURED_EXPORT_SCHEMA,
            "tolerance": tolerance,
        }
        return cls(
            digest=content_digest(payload),
            model_path=STRUCTURED_EXPORT_MODEL_NAME,
            model_digest=model_digest,
            model_size_bytes=model_size_bytes,
            policy_identity=policy_payload,
            policy_identity_digest=policy_identity_digest,
            architecture_digest=architecture_digest,
            inputs=inputs,
            action_size=action_size,
            tolerance=tolerance,
            max_abs_error=max_abs_error,
            schema_version=STRUCTURED_EXPORT_SCHEMA,
        )
'''
    text = _replace_once(text, old_build, new_build, label="manifest build")
    text = _replace_once(
        text,
        "        prediction = self.policy._predict(observation, deterministic=True)  # type: ignore[attr-defined]\n",
        "        prediction = self.policy._predict(observation, deterministic=True)\n",
        label="unused type ignore",
    )

    old_helpers = '''def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_structured_export_manifest(path: Path) -> StructuredExportManifest:
'''
    new_helpers = '''def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def load_structured_export_manifest(path: Path) -> StructuredExportManifest:
'''
    text = _replace_once(text, old_helpers, new_helpers, label="typed helpers")

    old_input = '''            StructuredInputSpec(
                name=str(item["name"]),
                shape=tuple(int(value) for value in raw_shape),
                dtype=str(item["dtype"]),
            )
'''
    new_input = '''            StructuredInputSpec(
                name=_string(item["name"], field=f"inputs[{index}].name"),
                shape=tuple(
                    _integer(value, field=f"inputs[{index}].shape")
                    for value in raw_shape
                ),
                dtype=_string(item["dtype"], field=f"inputs[{index}].dtype"),
            )
'''
    text = _replace_once(text, old_input, new_input, label="input parser")

    old_manifest = '''    return StructuredExportManifest(
        digest=str(payload["digest"]),
        model_path=str(payload["model_path"]),
        model_digest=str(payload["model_digest"]),
        model_size_bytes=int(payload["model_size_bytes"]),
        policy_identity=dict(policy_identity),
        policy_identity_digest=str(payload["policy_identity_digest"]),
        architecture_digest=str(payload["architecture_digest"]),
        inputs=tuple(inputs),
        action_size=int(payload["action_size"]),
        tolerance=float(payload["tolerance"]),
        max_abs_error=float(payload["max_abs_error"]),
        schema_version=str(payload["schema_version"]),
    )
'''
    new_manifest = '''    return StructuredExportManifest(
        digest=_string(payload["digest"], field="digest"),
        model_path=_string(payload["model_path"], field="model_path"),
        model_digest=_string(payload["model_digest"], field="model_digest"),
        model_size_bytes=_integer(
            payload["model_size_bytes"], field="model_size_bytes"
        ),
        policy_identity=dict(policy_identity),
        policy_identity_digest=_string(
            payload["policy_identity_digest"], field="policy_identity_digest"
        ),
        architecture_digest=_string(
            payload["architecture_digest"], field="architecture_digest"
        ),
        inputs=tuple(inputs),
        action_size=_integer(payload["action_size"], field="action_size"),
        tolerance=_number(payload["tolerance"], field="tolerance"),
        max_abs_error=_number(payload["max_abs_error"], field="max_abs_error"),
        schema_version=_string(payload["schema_version"], field="schema_version"),
    )
'''
    text = _replace_once(text, old_manifest, new_manifest, label="manifest parser")

    ast.parse(text, filename=str(TARGET))
    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
