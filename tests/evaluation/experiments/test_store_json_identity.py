"""A committed Study JSON object must have one unambiguous interpretation."""

from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.parametrize(
    "payload",
    [
        b'{"decision":"ACCEPT","decision":"KEEP_BASELINE"}',
        b'{"nested":[{"seed":2,"seed":2}]}',
        rb'{"seed":1,"\u0073eed":2}',
        b'{"value":NaN}',
        b'{"nested":[Infinity]}',
        b'{"value":-Infinity}',
        b'{"value":1e400}',
    ],
)
def test_study_read_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path, payload: bytes
) -> None:
    store = StudyStore(tmp_path / "study")
    published = store.publish_json_once("plan.json", {"value": 1})
    published.write_bytes(payload)
    with pytest.raises(ArtifactIntegrityError, match="duplicate JSON member|finite"):
        store.read_json("plan.json")


def test_study_read_accepts_valid_noncompact_json(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    published = store.publish_json_once("plan.json", {"value": 1})
    published.write_bytes(b'{ "value" : 1.0, "other" : [null, true] }\n')
    assert store.read_json("plan.json") == {"value": 1.0, "other": [None, True]}
