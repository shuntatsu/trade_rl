# ruff: noqa
# fmt: off
import base64,re
from pathlib import Path

def r(s,o,n,l):
 c=s.count(o);assert c==1,f"{l}:{c}";return s.replace(o,n,1)

def core(s):
 p=re.compile(r'(?P<i>\s*)required_adverse_passed=True,\n(?P=i)\)')
 s,c=p.subn(lambda m:f'{m.group("i")}required_adverse_passed=True,\n{m.group("i")}required_adverse_evidence_digest="a" * 64,\n{m.group("i")})',s)
 assert c==2,c
 return s

def integ(s):
 s=r(s,'from trade_rl.cli.extended import main\n','from trade_rl.cli.extended import main\nfrom trade_rl.evaluation.causal_scenario_c3_adverse import (\n    C3AdverseFoldEvidence,\n)\n','ii')
 s=r(s,'            required_adverse_passed=True,\n        )\n','            required_adverse_passed=True,\n            required_adverse_evidence_digest="a" * 64,\n        )\n','ir')
 s=r(s,'            adverse[fold_id] = True\n','''            adverse[fold_id] = C3AdverseFoldEvidence(
                fold_index=fold_index,
                source_artifact_digest=sha("a"),
                thresholds_digest=sha("b"),
                required_scenario="adverse_cost_2x",
                selected_return=0.02,
                baseline_uplift=0.01,
                cost_fraction=0.01,
                turnover_per_day=0.5,
                maximum_drawdown=0.1,
                failed_conditions=(),
            )
''','ia')
 return r(s,'        required_adverse_passed=adverse,\n','        required_adverse_evidence=adverse,\n','ib')

def evaluation(s):
 s=r(s,'from trade_rl.artifacts.hashing import content_digest\n','from trade_rl.artifacts.codec import canonical_json_bytes\nfrom trade_rl.artifacts.hashing import content_digest\n','ec')
 s=r(s,'from trade_rl.evaluation.causal_scenario_c3_contracts import (\n','from trade_rl.evaluation.causal_scenario_c3_adverse import (\n    C3AdverseFoldEvidence,\n)\nfrom trade_rl.evaluation.causal_scenario_c3_adverse_source import (\n    load_c3_source_adverse_evidence,\n)\nfrom trade_rl.evaluation.causal_scenario_c3_contracts import (\n','ei')
 s=r(s,'def _source_fold_map(walk_forward: dict[str, object]) -> dict[int, dict[str, object]]:\n','''def _walk_forward_config_payload(source_root: Path) -> dict[str, object]:
    path = source_root / "walk-forward-config.json"
    try:
        raw = _object(json.loads(path.read_text(encoding="utf-8")), field="walk-forward-config")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("walk-forward configuration artifact is invalid") from error
    if canonical_json_bytes(raw) != path.read_bytes():
        raise ValueError("walk-forward configuration is not canonical JSON")
    return raw


def _source_fold_map(walk_forward: dict[str, object]) -> dict[int, dict[str, object]]:
''','eh')
 s=r(s,'    source_folds = _source_fold_map(walk_forward)\n\n    destination = Path(output_root)\n    batch_queries: list[C3BatchQuery] = []\n    fold_selection_days: dict[str, int] = {}\n    required_adverse_passed: dict[str, bool] = {}\n','''    source_folds = _source_fold_map(walk_forward)
    walk_forward_config = _walk_forward_config_payload(source_root)
    source_adverse = load_c3_source_adverse_evidence(
        source_root,
        walk_forward_config=walk_forward_config,
        source_folds=source_folds,
        dataset_id=source_manifest.dataset_id,
    )

    destination = Path(output_root)
    batch_queries: list[C3BatchQuery] = []
    fold_selection_days: dict[str, int] = {}
    required_adverse_evidence: dict[str, C3AdverseFoldEvidence] = {}
''','el')
 s=r(s,'                "queries",\n                "required_adverse_passed",\n                "selection_days",\n','                "queries",\n','ef')
 s=r(s,'        fold_selection_days[fold_id] = _integer(\n            fold["selection_days"], field=f"{field}.selection_days", positive=True\n        )\n        required_adverse_passed[fold_id] = _boolean(\n            fold["required_adverse_passed"],\n            field=f"{field}.required_adverse_passed",\n        )\n','''        adverse_evidence = source_adverse.by_fold_index.get(fold_index)
        selection_days = source_adverse.selection_days_by_fold.get(fold_index)
        if adverse_evidence is None or selection_days is None:
            raise ValueError("source adverse evidence is missing for C3 fold")
        fold_selection_days[fold_id] = selection_days
        required_adverse_evidence[fold_id] = adverse_evidence
''','ed')
 s=r(s,'        if required_adverse_passed[fold_id] and scenarios == {"nominal"}:\n            raise ValueError("required adverse evidence is missing from C3 request")\n\n    if not batch_queries:\n','''        if source_adverse.required_scenario not in scenarios:
            raise ValueError(
                "required adverse execution scenario is missing from C3 request"
            )

    if seen_fold_indices != set(source_folds):
        raise ValueError("C3 request folds do not match source walk-forward run")
    if not batch_queries:
''','es')
 return r(s,'        required_adverse_passed=required_adverse_passed,\n','        required_adverse_evidence=required_adverse_evidence,\n','eb')

def wf(s):
 s=r(s,'                ],\n                "required_adverse_passed": True,\n                "selection_days": 30,\n            }\n','                ],\n            }\n','wf1')
 s=r(s,'                    "train_range": [0, 90],\n                    "test_range": [90, 300],\n','                    "train_range": [0, 90],\n                    "selection_range": [90, 2_970],\n                    "test_range": [90, 300],\n','wf2')
 s=r(s,'    monkeypatch.setattr(\n        module,\n        "load_causal_scenario_library_artifact",\n','''    monkeypatch.setattr(
        module,
        "_walk_forward_config_payload",
        lambda root: calls.append(f"config:{root.name}") or {},
    )
    adverse = SimpleNamespace(passed=True, digest=_sha("7"))
    source_adverse = SimpleNamespace(
        required_scenario="cost_2x",
        selection_days_by_fold={0: 30},
        by_fold_index={0: adverse},
    )
    monkeypatch.setattr(
        module,
        "load_c3_source_adverse_evidence",
        lambda root, **kwargs: calls.append(f"adverse:{root.name}") or source_adverse,
    )
    monkeypatch.setattr(
        module,
        "load_causal_scenario_library_artifact",
''','wf3')
 return r(s,'        "run:source-run",\n        "library:library",\n','        "run:source-run",\n        "config:source-run",\n        "adverse:source-run",\n        "library:library",\n','wf4')

def test_export():
 items={'tests/evaluation/test_causal_scenario_c3.py':core,'tests/evaluation/test_causal_scenario_c3_integration.py':integ,'trade_rl/workflows/causal_scenario/c3_evaluation.py':evaluation,'tests/workflows/test_causal_scenario_c3_walk_forward.py':wf}
 for p,f in items.items():
  x=base64.b64encode(f(Path(p).read_text()).encode()).decode();print('C3_EXPORT_BEGIN:'+p);print(x);print('C3_EXPORT_END:'+p)
 assert False,'export'
