from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v4_pipeline import (
    CausalAlphaV4ResearchPackage,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_entry import (
    execute_causal_alpha_v4_prepared_entry,
)


def _digest(char: str) -> str:
    return char * 64


class _Evidence:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.passed = True
        self.digest = content_digest({"stage": stage, "passed": True})

    def to_payload(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": f"test_{self.stage}_v1",
            "stage": self.stage,
            "passed": True,
        }
        return {**body, "artifact_digest": content_digest(body)}


def test_prepared_v4_entry_uses_one_store_and_persists_identity_chain(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "v4.json"
    config_path.write_text(json.dumps({"authored": "v4"}), encoding="utf-8")
    config = SimpleNamespace(digest=_digest("c"))
    prepared = SimpleNamespace(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        run_manifest_digest=_digest("a"),
        base_runtime_manifest_digest=_digest("b"),
        v4_context_manifest_digest=_digest("d"),
        config_digest=_digest("c"),
        execution_identity_digest=_digest("e"),
        nested_partition_digest=_digest("f"),
        generator_code_digest=_digest("1"),
    )
    stores: list[object] = []

    def signal_stage(value: object, **kwargs: object) -> _Evidence:
        assert value is prepared
        stores.append(kwargs["store"])
        return _Evidence("signal")

    def selection_stage(value: object, signal: object, **kwargs: object) -> _Evidence:
        assert value is prepared
        assert isinstance(signal, _Evidence)
        stores.append(kwargs["store"])
        return _Evidence("selection")

    def admission_stage(
        value: object, signal: object, selection: object, **kwargs: object
    ) -> _Evidence:
        assert value is prepared
        assert isinstance(signal, _Evidence)
        assert isinstance(selection, _Evidence)
        stores.append(kwargs["store"])
        return _Evidence("admission")

    package = execute_causal_alpha_v4_prepared_entry(
        config=config,
        config_path=config_path,
        prepared=prepared,
        output_root=tmp_path / "output",
        signal_stage=signal_stage,
        selection_stage=selection_stage,
        admission_stage=admission_stage,
    )

    assert isinstance(package, CausalAlphaV4ResearchPackage)
    assert package.run_manifest_digest == _digest("a")
    assert len(stores) == 3
    assert stores[0] is stores[1] is stores[2]
    root = tmp_path / "output"
    for relative in (
        "run-manifest.json",
        "authored-config.json",
        "signal/evidence.json",
        "selection/evidence.json",
        "admission/evidence.json",
        "result.json",
    ):
        assert (root / relative).is_file()
