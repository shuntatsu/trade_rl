from __future__ import annotations

import base64
import re
from pathlib import Path


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    assert count == 1, f"{label}: expected one match, found {count}"
    return source.replace(old, new, 1)


def _patch_core_test(source: str) -> str:
    pattern = re.compile(
        r"(?P<indent>\s*)required_adverse_passed=True,\n"
        r"(?P=indent)\)"
    )
    source, count = pattern.subn(
        lambda match: (
            f"{match.group('indent')}required_adverse_passed=True,\n"
            f"{match.group('indent')}required_adverse_evidence_digest=\"a\" * 64,\n"
            f"{match.group('indent')})"
        ),
        source,
    )
    assert count == 2, f"core test: expected two report fixtures, found {count}"
    return source


def _patch_integration_test(source: str) -> str:
    source = _replace_once(
        source,
        "from trade_rl.cli.extended import main\n",
        "from trade_rl.cli.extended import main\n"
        "from trade_rl.evaluation.causal_scenario_c3_adverse import (\n"
        "    C3AdverseFoldEvidence,\n"
        ")\n",
        label="integration import",
    )
    source = _replace_once(
        source,
        "            required_adverse_passed=True,\n"
        "        )\n",
        "            required_adverse_passed=True,\n"
        "            required_adverse_evidence_digest=\"a\" * 64,\n"
        "        )\n",
        label="integration report fixture",
    )
    source = _replace_once(
        source,
        "            adverse[fold_id] = True\n",
        "            adverse[fold_id] = C3AdverseFoldEvidence(\n"
        "                fold_index=fold_index,\n"
        "                source_artifact_digest=sha(\"a\"),\n"
        "                thresholds_digest=sha(\"b\"),\n"
        "                required_scenario=\"adverse_cost_2x\",\n"
        "                selected_return=0.02,\n"
        "                baseline_uplift=0.01,\n"
        "                cost_fraction=0.01,\n"
        "                turnover_per_day=0.5,\n"
        "                maximum_drawdown=0.1,\n"
        "                failed_conditions=(),\n"
        "            )\n",
        label="integration adverse fixture",
    )
    source = _replace_once(
        source,
        "        required_adverse_passed=adverse,\n",
        "        required_adverse_evidence=adverse,\n",
        label="integration batch argument",
    )
    return source


def _patch_evaluation(source: str) -> str:
    source = _replace_once(
        source,
        "from trade_rl.artifacts.hashing import content_digest\n",
        "from trade_rl.artifacts.codec import canonical_json_bytes\n"
        "from trade_rl.artifacts.hashing import content_digest\n",
        label="evaluation codec import",
    )
    source = _replace_once(
        source,
        "from trade_rl.evaluation.causal_scenario_c3_contracts import (\n",
        "from trade_rl.evaluation.causal_scenario_c3_adverse import (\n"
        "    C3AdverseFoldEvidence,\n"
        ")\n"
        "from trade_rl.evaluation.causal_scenario_c3_adverse_source import (\n"
        "    load_c3_source_adverse_evidence,\n"
        ")\n"
        "from trade_rl.evaluation.causal_scenario_c3_contracts import (\n",
        label="evaluation adverse imports",
    )
    source = _replace_once(
        source,
        "def _source_fold_map(walk_forward: dict[str, object]) -> dict[int, dict[str, object]]:\n",
        "def _walk_forward_config_payload(source_root: Path) -> dict[str, object]:\n"
        "    path = source_root / \"walk-forward-config.json\"\n"
        "    try:\n"
        "        raw = _object(\n"
        "            json.loads(path.read_text(encoding=\"utf-8\")),\n"
        "            field=\"walk-forward-config\",\n"
        "        )\n"
        "    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:\n"
        "        raise ValueError(\"walk-forward configuration artifact is invalid\") from error\n"
        "    if canonical_json_bytes(raw) != path.read_bytes():\n"
        "        raise ValueError(\"walk-forward configuration is not canonical JSON\")\n"
        "    return raw\n\n\n"
        "def _source_fold_map(walk_forward: dict[str, object]) -> dict[int, dict[str, object]]:\n",
        label="evaluation config helper",
    )
    source = _replace_once(
        source,
        "    source_folds = _source_fold_map(walk_forward)\n\n"
        "    destination = Path(output_root)\n"
        "    batch_queries: list[C3BatchQuery] = []\n"
        "    fold_selection_days: dict[str, int] = {}\n"
        "    required_adverse_passed: dict[str, bool] = {}\n",
        "    source_folds = _source_fold_map(walk_forward)\n"
        "    walk_forward_config = _walk_forward_config_payload(source_root)\n"
        "    source_adverse = load_c3_source_adverse_evidence(\n"
        "        source_root,\n"
        "        walk_forward_config=walk_forward_config,\n"
        "        source_folds=source_folds,\n"
        "        dataset_id=source_manifest.dataset_id,\n"
        "    )\n\n"
        "    destination = Path(output_root)\n"
        "    batch_queries: list[C3BatchQuery] = []\n"
        "    fold_selection_days: dict[str, int] = {}\n"
        "    required_adverse_evidence: dict[str, C3AdverseFoldEvidence] = {}\n",
        label="evaluation source evidence load",
    )
    source = _replace_once(
        source,
        "                \"queries\",\n"
        "                \"required_adverse_passed\",\n"
        "                \"selection_days\",\n",
        "                \"queries\",\n",
        label="evaluation request fold fields",
    )
    source = _replace_once(
        source,
        "        fold_selection_days[fold_id] = _integer(\n"
        "            fold[\"selection_days\"], field=f\"{field}.selection_days\", positive=True\n"
        "        )\n"
        "        required_adverse_passed[fold_id] = _boolean(\n"
        "            fold[\"required_adverse_passed\"],\n"
        "            field=f\"{field}.required_adverse_passed\",\n"
        "        )\n",
        "        adverse_evidence = source_adverse.by_fold_index.get(fold_index)\n"
        "        selection_days = source_adverse.selection_days_by_fold.get(fold_index)\n"
        "        if adverse_evidence is None or selection_days is None:\n"
        "            raise ValueError(\"source adverse evidence is missing for C3 fold\")\n"
        "        fold_selection_days[fold_id] = selection_days\n"
        "        required_adverse_evidence[fold_id] = adverse_evidence\n",
        label="evaluation derived fold evidence",
    )
    source = _replace_once(
        source,
        "        if required_adverse_passed[fold_id] and scenarios == {\"nominal\"}:\n"
        "            raise ValueError(\"required adverse evidence is missing from C3 request\")\n\n"
        "    if not batch_queries:\n",
        "        required_scenario = source_adverse.required_scenario\n"
        "        if required_scenario not in scenarios:\n"
        "            raise ValueError(\n"
        "                \"required adverse execution scenario is missing from C3 request\"\n"
        "            )\n\n"
        "    if seen_fold_indices != set(source_folds):\n"
        "        raise ValueError(\"C3 request folds do not match source walk-forward run\")\n"
        "    if not batch_queries:\n",
        label="evaluation adverse scenario check",
    )
    source = _replace_once(
        source,
        "        required_adverse_passed=required_adverse_passed,\n",
        "        required_adverse_evidence=required_adverse_evidence,\n",
        label="evaluation batch argument",
    )
    return source


def _patch_workflow_test(source: str) -> str:
    source = _replace_once(
        source,
        "                ],\n"
        "                \"required_adverse_passed\": True,\n"
        "                \"selection_days\": 30,\n"
        "            }\n",
        "                ],\n"
        "            }\n",
        label="workflow request fields",
    )
    source = _replace_once(
        source,
        "                    \"train_range\": [0, 90],\n"
        "                    \"test_range\": [90, 300],\n",
        "                    \"train_range\": [0, 90],\n"
        "                    \"selection_range\": [90, 2_970],\n"
        "                    \"test_range\": [90, 300],\n",
        label="workflow source selection range",
    )
    source = _replace_once(
        source,
        "    monkeypatch.setattr(\n"
        "        module,\n"
        "        \"load_causal_scenario_library_artifact\",\n",
        "    monkeypatch.setattr(\n"
        "        module,\n"
        "        \"_walk_forward_config_payload\",\n"
        "        lambda root: calls.append(f\"config:{root.name}\") or {},\n"
        "    )\n"
        "    adverse = SimpleNamespace(passed=True, digest=_sha(\"7\"))\n"
        "    source_adverse = SimpleNamespace(\n"
        "        required_scenario=\"cost_2x\",\n"
        "        selection_days_by_fold={0: 30},\n"
        "        by_fold_index={0: adverse},\n"
        "    )\n"
        "    monkeypatch.setattr(\n"
        "        module,\n"
        "        \"load_c3_source_adverse_evidence\",\n"
        "        lambda root, **kwargs: (\n"
        "            calls.append(f\"adverse:{root.name}\") or source_adverse\n"
        "        ),\n"
        "    )\n"
        "    monkeypatch.setattr(\n"
        "        module,\n"
        "        \"load_causal_scenario_library_artifact\",\n",
        label="workflow source evidence mocks",
    )
    source = _replace_once(
        source,
        "        \"run:source-run\",\n"
        "        \"library:library\",\n",
        "        \"run:source-run\",\n"
        "        \"config:source-run\",\n"
        "        \"adverse:source-run\",\n"
        "        \"library:library\",\n",
        label="workflow expected calls",
    )
    return source


def test_export_c3_patches() -> None:
    patches = {
        "tests/evaluation/test_causal_scenario_c3.py": _patch_core_test,
        "tests/evaluation/test_causal_scenario_c3_integration.py": _patch_integration_test,
        "trade_rl/workflows/causal_scenario/c3_evaluation.py": _patch_evaluation,
        "tests/workflows/test_causal_scenario_c3_walk_forward.py": _patch_workflow_test,
    }
    for relative, patcher in patches.items():
        source = Path(relative).read_text(encoding="utf-8")
        encoded = base64.b64encode(patcher(source).encode("utf-8")).decode("ascii")
        print(f"C3_EXPORT_BEGIN:{relative}")
        print(encoded)
        print(f"C3_EXPORT_END:{relative}")
    raise AssertionError("intentional C3 patch export")
