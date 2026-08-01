from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _remove_top_level_nodes(
    source: str,
    *,
    definitions: set[str],
    assignments: set[str],
) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    for node in tree.body:
        remove = False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            remove = node.name in definitions
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            remove = bool(names & assignments)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            remove = node.target.id in assignments
        if not remove:
            continue
        if node.end_lineno is None:
            raise RuntimeError("AST node lacks end line information")
        start = node.lineno - 1
        end = node.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        spans.append((start, end))
    for start, end in reversed(spans):
        del lines[start:end]
    return "".join(lines)


def _write_domain_config_fields() -> None:
    source_path = ROOT / "trade_rl/workflows/config_fields.py"
    source = source_path.read_text(encoding="utf-8")
    if "def require_exact_fields(" not in source:
        raise RuntimeError("workflow config field implementation marker is missing")
    domain_source = source.replace(
        '"""Field-closed mapping validation for public workflow configuration."""',
        '"""Field-closed mapping validation for authored configuration contracts."""',
        1,
    )
    (ROOT / "trade_rl/domain/config_fields.py").write_text(
        domain_source,
        encoding="utf-8",
    )
    source_path.write_text(
        textwrap.dedent(
            '''\
            """Compatibility exports for configuration field validation."""

            from trade_rl.domain.config_fields import (
                require_dataclass_fields,
                require_exact_fields,
            )

            __all__ = ["require_dataclass_fields", "require_exact_fields"]
            '''
        ),
        encoding="utf-8",
    )


def _move_training_run_config() -> None:
    workflow_path = ROOT / "trade_rl/workflows/training_run.py"
    source = workflow_path.read_text(encoding="utf-8")
    start_marker = "def _mapping("
    end_marker = "@dataclass(frozen=True, slots=True)\nclass TrainingRunResult:"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError("training config extraction markers are not unique")
    start = source.index(start_marker)
    end = source.index(end_marker)
    contract_body = source[start:end].rstrip() + "\n"
    contract_header = textwrap.dedent(
        '''\
        """Authored training-run configuration contract and stable identity."""

        from __future__ import annotations

        import json
        import math
        from dataclasses import asdict, dataclass, field, replace
        from dataclasses import fields as dataclass_fields
        from pathlib import Path
        from typing import Any

        from trade_rl.artifacts.hashing import content_digest
        from trade_rl.artifacts.signals import SignalKind, load_signal_artifact
        from trade_rl.domain.config_fields import (
            require_dataclass_fields,
            require_exact_fields,
        )
        from trade_rl.risk.emergency import EmergencyRiskConfig
        from trade_rl.risk.portfolio import PortfolioRiskConfig
        from trade_rl.risk.pretrade import PreTradeRiskConfig
        from trade_rl.rl.actions import ActionSpec, AlphaContract
        from trade_rl.rl.checkpointing import load_checkpoint_manifest
        from trade_rl.rl.environment_config import ResidualMarketEnvConfig
        from trade_rl.rl.rewards import RewardConfig
        from trade_rl.rl.training import ResidualTrainingConfig
        from trade_rl.simulation.execution import ExecutionCostConfig
        from trade_rl.strategies.trend import TrendConfig

        '''
    )
    contract_footer = textwrap.dedent(
        '''\

        __all__ = ["TRAINING_RUN_CONFIG_SCHEMA", "TrainingRunConfig"]
        '''
    )
    (ROOT / "trade_rl/rl/training_run_config.py").write_text(
        contract_header + contract_body + contract_footer,
        encoding="utf-8",
    )
    rewritten = source[:start] + source[end:]
    import_marker = (
        "from trade_rl.rl.training import ResidualTrainingConfig, "
        "train_residual_ensemble\n"
    )
    if rewritten.count(import_marker) != 1:
        raise RuntimeError("training workflow import marker is not unique")
    rewritten = rewritten.replace(
        import_marker,
        import_marker
        + "from trade_rl.rl.training_run_config import (\n"
        + "    TRAINING_RUN_CONFIG_SCHEMA,\n"
        + "    TrainingRunConfig,\n"
        + ")\n",
        1,
    )
    workflow_path.write_text(rewritten, encoding="utf-8")

    studio_path = ROOT / "trade_rl/studio/config_catalog.py"
    studio_source = studio_path.read_text(encoding="utf-8")
    old = "from trade_rl.workflows.training_run import TrainingRunConfig"
    new = "from trade_rl.rl.training_run_config import TrainingRunConfig"
    if studio_source.count(old) != 1:
        raise RuntimeError("Studio training config import marker is not unique")
    studio_path.write_text(studio_source.replace(old, new, 1), encoding="utf-8")


def _structured_contract_source() -> str:
    return textwrap.dedent(
        '''\
        """Neutral artifact contract for structured policy exports."""

        from __future__ import annotations

        import json
        import math
        from collections.abc import Mapping
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Final

        from trade_rl.artifacts.codec import canonical_json_bytes
        from trade_rl.artifacts.hashing import content_digest
        from trade_rl.artifacts.verified_file import (
            file_digest_and_size,
            open_regular_binary,
        )
        from trade_rl.domain.common import require_sha256

        STRUCTURED_EXPORT_SCHEMA: Final = "structured_policy_export_v2"
        STRUCTURED_EXPORT_MANIFEST_NAME: Final = "structured-export.json"
        STRUCTURED_EXPORT_MODEL_NAME: Final = "policy.structured.torchscript.pt"
        STRUCTURED_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")
        _POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v4"
        _BASE_KEYS: Final = (
            "current_snapshot",
            "asset_state",
            "global_state",
            "active",
            "current_weights",
        )
        _SEQUENCE_PLANES: Final = ("values", "available", "staleness")
        _SUPPORTED_DTYPES: Final = frozenset(
            {"float16", "float32", "float64", "int32", "int64", "uint8", "bool"}
        )


        def canonical_structured_observation_keys() -> tuple[str, ...]:
            keys = list(_BASE_KEYS)
            for timeframe in STRUCTURED_TIMEFRAMES:
                for plane in _SEQUENCE_PLANES:
                    keys.append(f"sequence_{timeframe}_{plane}")
            return tuple(keys)


        def _validated_policy_identity(value: object) -> dict[str, object]:
            if not isinstance(value, Mapping) or not value:
                raise ValueError("structured export requires policy identity")
            payload = dict(value)
            if payload.get("schema_version") != _POLICY_IDENTITY_SCHEMA:
                raise ValueError("structured export policy identity schema mismatch")
            if payload.get("observation_encoder") != "hierarchical_sequence_v2":
                raise ValueError("structured export requires hierarchical sequence policy")
            architecture_digest = payload.get("policy_architecture_digest")
            if not isinstance(architecture_digest, str):
                raise ValueError("structured export policy lacks architecture digest")
            require_sha256(
                architecture_digest,
                field="policy_architecture_digest",
            )
            return payload


        @dataclass(frozen=True, slots=True)
        class StructuredInputSpec:
            name: str
            shape: tuple[int, ...]
            dtype: str

            def __post_init__(self) -> None:
                if not self.name:
                    raise ValueError("structured input name must be non-empty")
                if not self.shape or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in self.shape
                ):
                    raise ValueError(
                        "structured input shape must contain positive integers"
                    )
                if self.dtype not in _SUPPORTED_DTYPES:
                    raise ValueError("structured input dtype is unsupported")


        @dataclass(frozen=True, slots=True)
        class StructuredExportManifest:
            digest: str
            model_path: str
            model_digest: str
            model_size_bytes: int
            policy_identity: Mapping[str, object]
            policy_identity_digest: str
            architecture_digest: str
            inputs: tuple[StructuredInputSpec, ...]
            action_size: int
            tolerance: float
            max_abs_error: float
            schema_version: str = STRUCTURED_EXPORT_SCHEMA

            def __post_init__(self) -> None:
                require_sha256(self.digest, field="structured_export.digest")
                require_sha256(
                    self.model_digest,
                    field="structured_export.model_digest",
                )
                require_sha256(
                    self.policy_identity_digest,
                    field="structured_export.policy_identity_digest",
                )
                require_sha256(
                    self.architecture_digest,
                    field="structured_export.architecture_digest",
                )
                if self.schema_version != STRUCTURED_EXPORT_SCHEMA:
                    raise ValueError("unsupported structured export schema")
                if self.model_path != STRUCTURED_EXPORT_MODEL_NAME:
                    raise ValueError("structured export model path is invalid")
                if (
                    isinstance(self.model_size_bytes, bool)
                    or not isinstance(self.model_size_bytes, int)
                    or self.model_size_bytes <= 0
                ):
                    raise ValueError("structured export model must be non-empty")
                policy_payload = _validated_policy_identity(self.policy_identity)
                object.__setattr__(self, "policy_identity", policy_payload)
                if content_digest(policy_payload) != self.policy_identity_digest:
                    raise ValueError(
                        "structured export policy identity digest mismatch"
                    )
                if (
                    policy_payload.get("policy_architecture_digest")
                    != self.architecture_digest
                ):
                    raise ValueError("structured export architecture digest mismatch")
                if (
                    tuple(item.name for item in self.inputs)
                    != canonical_structured_observation_keys()
                ):
                    raise ValueError("structured export input order is not canonical")
                if (
                    isinstance(self.action_size, bool)
                    or not isinstance(self.action_size, int)
                    or self.action_size <= 0
                ):
                    raise ValueError("structured export action_size must be positive")
                if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
                    raise ValueError(
                        "structured export tolerance must be finite and positive"
                    )
                if (
                    not math.isfinite(self.max_abs_error)
                    or self.max_abs_error < 0.0
                    or self.max_abs_error > self.tolerance
                ):
                    raise ValueError(
                        "structured export parity error exceeds tolerance"
                    )
                if self.digest != content_digest(self.digest_payload()):
                    raise ValueError("structured export manifest digest mismatch")

            def digest_payload(self) -> dict[str, object]:
                return {
                    "action_size": self.action_size,
                    "architecture_digest": self.architecture_digest,
                    "inputs": self.inputs,
                    "max_abs_error": self.max_abs_error,
                    "model_digest": self.model_digest,
                    "model_path": self.model_path,
                    "model_size_bytes": self.model_size_bytes,
                    "policy_identity": dict(self.policy_identity),
                    "policy_identity_digest": self.policy_identity_digest,
                    "schema_version": self.schema_version,
                    "tolerance": self.tolerance,
                }

            @classmethod
            def build(
                cls,
                *,
                model_path: Path,
                policy_identity: Mapping[str, object],
                inputs: tuple[StructuredInputSpec, ...],
                action_size: int,
                tolerance: float,
                max_abs_error: float,
            ) -> StructuredExportManifest:
                policy_payload = _validated_policy_identity(policy_identity)
                architecture_digest = policy_payload["policy_architecture_digest"]
                assert isinstance(architecture_digest, str)
                model_digest, model_size_bytes = file_digest_and_size(
                    model_path,
                    field="structured export model",
                )
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


        def _mapping(value: object, *, field: str) -> Mapping[str, object]:
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


        def load_structured_export_manifest_bytes(
            raw: bytes,
        ) -> StructuredExportManifest:
            """Parse one exact canonical structured-export manifest byte sequence."""

            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "structured export manifest must be valid JSON"
                ) from error
            payload = _mapping(value, field="structured export manifest")
            expected = {
                "action_size",
                "architecture_digest",
                "digest",
                "inputs",
                "max_abs_error",
                "model_digest",
                "model_path",
                "model_size_bytes",
                "policy_identity",
                "policy_identity_digest",
                "schema_version",
                "tolerance",
            }
            if set(payload) != expected:
                raise ValueError("structured export manifest fields are invalid")
            raw_inputs = payload["inputs"]
            if not isinstance(raw_inputs, list):
                raise ValueError("structured export inputs must be a list")
            inputs: list[StructuredInputSpec] = []
            for index, raw_input in enumerate(raw_inputs):
                item = _mapping(raw_input, field=f"inputs[{index}]")
                if set(item) != {"dtype", "name", "shape"}:
                    raise ValueError(
                        "structured input manifest fields are invalid"
                    )
                raw_shape = item["shape"]
                if not isinstance(raw_shape, list):
                    raise ValueError("structured input shape must be a list")
                inputs.append(
                    StructuredInputSpec(
                        name=_string(
                            item["name"],
                            field=f"inputs[{index}].name",
                        ),
                        shape=tuple(
                            _integer(
                                shape_value,
                                field=f"inputs[{index}].shape",
                            )
                            for shape_value in raw_shape
                        ),
                        dtype=_string(
                            item["dtype"],
                            field=f"inputs[{index}].dtype",
                        ),
                    )
                )
            policy_identity = _mapping(
                payload["policy_identity"],
                field="policy_identity",
            )
            manifest = StructuredExportManifest(
                digest=_string(payload["digest"], field="digest"),
                model_path=_string(
                    payload["model_path"],
                    field="model_path",
                ),
                model_digest=_string(
                    payload["model_digest"],
                    field="model_digest",
                ),
                model_size_bytes=_integer(
                    payload["model_size_bytes"],
                    field="model_size_bytes",
                ),
                policy_identity=dict(policy_identity),
                policy_identity_digest=_string(
                    payload["policy_identity_digest"],
                    field="policy_identity_digest",
                ),
                architecture_digest=_string(
                    payload["architecture_digest"],
                    field="architecture_digest",
                ),
                inputs=tuple(inputs),
                action_size=_integer(
                    payload["action_size"],
                    field="action_size",
                ),
                tolerance=_number(
                    payload["tolerance"],
                    field="tolerance",
                ),
                max_abs_error=_number(
                    payload["max_abs_error"],
                    field="max_abs_error",
                ),
                schema_version=_string(
                    payload["schema_version"],
                    field="schema_version",
                ),
            )
            if raw != canonical_json_bytes(manifest):
                raise ValueError(
                    "structured export manifest must use canonical encoding"
                )
            return manifest


        def load_structured_export_manifest(path: Path) -> StructuredExportManifest:
            with open_regular_binary(
                path,
                field="structured export manifest",
            ) as handle:
                return load_structured_export_manifest_bytes(handle.read())


        __all__ = [
            "STRUCTURED_EXPORT_MANIFEST_NAME",
            "STRUCTURED_EXPORT_MODEL_NAME",
            "STRUCTURED_EXPORT_SCHEMA",
            "STRUCTURED_TIMEFRAMES",
            "StructuredExportManifest",
            "StructuredInputSpec",
            "canonical_structured_observation_keys",
            "load_structured_export_manifest",
            "load_structured_export_manifest_bytes",
        ]
        '''
    )


def _move_structured_export_contract() -> None:
    contract_path = ROOT / "trade_rl/artifacts/structured_policy_contract.py"
    contract_path.write_text(_structured_contract_source(), encoding="utf-8")

    exporter_path = ROOT / "trade_rl/rl/structured_export.py"
    source = exporter_path.read_text(encoding="utf-8")
    source = _remove_top_level_nodes(
        source,
        definitions={
            "StructuredExportManifest",
            "StructuredInputSpec",
            "_integer",
            "_mapping",
            "_number",
            "_string",
            "canonical_structured_observation_keys",
            "load_structured_export_manifest",
            "load_structured_export_manifest_bytes",
        },
        assignments={
            "STRUCTURED_EXPORT_MANIFEST_NAME",
            "STRUCTURED_EXPORT_MODEL_NAME",
            "STRUCTURED_EXPORT_SCHEMA",
            "_BASE_KEYS",
            "_SEQUENCE_PLANES",
            "_SUPPORTED_DTYPES",
            "_TIMEFRAMES",
        },
    )
    import_marker = "from torch import nn\n\n"
    if source.count(import_marker) != 1:
        raise RuntimeError("structured exporter import marker is not unique")
    import_block = textwrap.dedent(
        '''\
        from trade_rl.artifacts.structured_policy_contract import (
            STRUCTURED_EXPORT_MANIFEST_NAME,
            STRUCTURED_EXPORT_MODEL_NAME,
            STRUCTURED_EXPORT_SCHEMA,
            STRUCTURED_TIMEFRAMES,
            StructuredExportManifest,
            StructuredInputSpec,
            canonical_structured_observation_keys,
            load_structured_export_manifest,
            load_structured_export_manifest_bytes,
        )

        '''
    )
    source = source.replace(import_marker, import_marker + import_block, 1)
    source = source.replace("_TIMEFRAMES", "STRUCTURED_TIMEFRAMES")
    exporter_path.write_text(source, encoding="utf-8")

    for relative in (
        "trade_rl/serving/policy_loader.py",
        "trade_rl/serving/structured_policy.py",
    ):
        path = ROOT / relative
        serving_source = path.read_text(encoding="utf-8")
        old = "from trade_rl.rl.structured_export import ("
        new = "from trade_rl.artifacts.structured_policy_contract import ("
        if serving_source.count(old) != 1:
            raise RuntimeError(f"Serving import marker is not unique: {relative}")
        path.write_text(serving_source.replace(old, new, 1), encoding="utf-8")


def _document_future_market_roles() -> None:
    architecture_path = ROOT / "docs/ARCHITECTURE.md"
    architecture = architecture_path.read_text(encoding="utf-8")
    section = textwrap.dedent(
        '''\

        ## Future asymmetric cross-market boundary

        Stage Bは現在未実装です。将来の市場役割は次の非対称Contractに固定します。

        ```text
        SpotLongBook: FUTURE_LONG_ONLY_ROLE
        USDSMShortBook: FUTURE_SHORT_ONLY_ROLE
        StageBSpotFuturesGeneralization: NOT_IMPLEMENTED
        ```

        Binance SpotはLong側Book、USDⓈ-M先物はShort側Bookとして扱います。将来のPortfolio coordinatorは両Bookを段階的に構成できますが、SpotをShort可能な市場として扱ったり、USDⓈ-M先物を無制約なLong/Short市場として扱ったりしません。
        '''
    )
    if "SpotLongBook: FUTURE_LONG_ONLY_ROLE" not in architecture:
        marker = "\n## Artifact store and PostgreSQL\n"
        if architecture.count(marker) != 1:
            raise RuntimeError("architecture documentation insertion marker is missing")
        architecture = architecture.replace(marker, section + marker, 1)
        architecture_path.write_text(architecture, encoding="utf-8")

    status_path = ROOT / "docs/RESEARCH_STATUS.md"
    status = status_path.read_text(encoding="utf-8")
    if "SpotLongBook: FUTURE_LONG_ONLY_ROLE" not in status:
        marker = "StageBSpotFuturesGeneralization: NOT_IMPLEMENTED\n"
        if status.count(marker) != 1:
            raise RuntimeError("research status Stage B marker is not unique")
        status = status.replace(
            marker,
            marker
            + "SpotLongBook: FUTURE_LONG_ONLY_ROLE\n"
            + "USDSMShortBook: FUTURE_SHORT_ONLY_ROLE\n",
            1,
        )
        status_path.write_text(status, encoding="utf-8")


def main() -> None:
    _write_domain_config_fields()
    _move_training_run_config()
    _move_structured_export_contract()
    _document_future_market_roles()
    Path(__file__).unlink()


if __name__ == "__main__":
    main()
