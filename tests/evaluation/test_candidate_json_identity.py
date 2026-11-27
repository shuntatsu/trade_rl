"""Semantic Run identity must not discard ambiguous or nonfinite JSON."""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.evaluation.test_candidate_run_artifact_identity import _write_root
from trade_rl.evaluation.runs.artifact import (
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
)


@pytest.mark.parametrize(
    "reader", [load_candidate_run_artifact, inspect_candidate_run_artifact]
)
@pytest.mark.parametrize("file_name", ["summary.json", "provenance.json"])
@pytest.mark.parametrize(
    "injection",
    ["duplicate-root", "duplicate-nested", "escaped-alias", "nan", "overflow"],
)
def test_run_readers_reject_ambiguous_or_nonfinite_json(
    tmp_path: Path, file_name: str, injection: str, reader: Callable[[Path], object]
) -> None:
    root = tmp_path / "run"
    _write_root(root, compressed=True)
    path = root / file_name
    original = path.read_text(encoding="utf-8")
    if injection == "duplicate-root":
        changed = original.replace("{", '{"schema_version":"wrong",', 1)
    elif injection == "duplicate-nested":
        changed = original.replace("{", '{"extra":{"a":1,"a":1},', 1)
    elif injection == "escaped-alias":
        changed = original.replace("{", r'{"\u0073chema_version":"wrong",', 1)
    else:
        raw = "NaN" if injection == "nan" else "1e400"
        changed = original.replace("{", f'{{"extra":{raw},', 1)
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON member|finite"):
        reader(root)
