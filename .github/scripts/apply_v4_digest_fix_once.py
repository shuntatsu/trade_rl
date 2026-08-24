"""Apply the bounded V4/non-V4 observation contract compatibility correction."""

from __future__ import annotations

from pathlib import Path


def _replace_once(*, source: str, old: str, new: str, field: str) -> str:
    count = source.count(old)
    if count == 0 and new in source:
        return source
    if count != 1:
        raise SystemExit(f"unexpected {field} source count: {count}")
    return source.replace(old, new)


def main() -> int:
    environment_path = Path("trade_rl/rl/universal_single_instrument_env.py")
    environment = environment_path.read_text(encoding="utf-8")
    old_environment = """        self._observation_contract_digest = content_digest(
            {
                "concrete_observation_contract_digest": (
                    concrete_observation_digest
                    if training_contract_digest is None
                    else None
                ),
                "instrument_context_schema_digest": context_schema_digest,
                "schema_version": UNIVERSAL_OBSERVATION_SCHEMA,
                "training_contract_digest": training_contract_digest,
                "v4_context_schema_digest": v4_context_schema_digest,
            }
        )
"""
    new_environment = """        observation_contract_payload: dict[str, object] = {
            "concrete_observation_contract_digest": (
                concrete_observation_digest
                if training_contract_digest is None
                else None
            ),
            "instrument_context_schema_digest": context_schema_digest,
            "schema_version": UNIVERSAL_OBSERVATION_SCHEMA,
            "training_contract_digest": training_contract_digest,
        }
        if v4_context_schema_digest is not None:
            observation_contract_payload["v4_context_schema_digest"] = (
                v4_context_schema_digest
            )
        self._observation_contract_digest = content_digest(
            observation_contract_payload
        )
"""
    environment_path.write_text(
        _replace_once(
            source=environment,
            old=old_environment,
            new=new_environment,
            field="universal observation digest",
        ),
        encoding="utf-8",
    )

    test_path = Path("tests/rl/test_universal_v4_context.py")
    test = test_path.read_text(encoding="utf-8")
    test = _replace_once(
        source=test,
        old="""import pytest

from trade_rl.data.v4_context import (
""",
        new="""import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import (
""",
        field="V4 test hashing import",
    )
    test = _replace_once(
        source=test,
        old="""from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
""",
        new="""from trade_rl.rl.universal_single_instrument_env import (
    UNIVERSAL_OBSERVATION_SCHEMA,
    EpisodeRoutedSingleInstrumentEnv,
)
""",
        field="V4 environment test import",
    )
    test = _replace_once(
        source=test,
        old="""        assert plain.observation_contract_digest != v4.observation_contract_digest
        layout = v4.sequence_layout_metadata
""",
        new="""        assert plain.observation_contract_digest != v4.observation_contract_digest
        expected_v4_digest = content_digest(
            {
                "concrete_observation_contract_digest": _digest("a"),
                "instrument_context_schema_digest": None,
                "schema_version": UNIVERSAL_OBSERVATION_SCHEMA,
                "training_contract_digest": None,
                "v4_context_schema_digest": provider.schema_digest,
            }
        )
        assert v4.observation_contract_digest == expected_v4_digest
        layout = v4.sequence_layout_metadata
""",
        field="V4 digest assertion",
    )
    test_path.write_text(test, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
