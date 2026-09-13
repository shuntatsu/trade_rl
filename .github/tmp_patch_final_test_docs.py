from pathlib import Path

package_path = Path("docs/architecture/package-boundaries.md")
package = package_path.read_text(encoding="utf-8")
old_tree = "    ├── runs/{candidate.py,candidate_suite.py,config.py,execute.py,provenance.py,artifact.py}\n    └── experiments/"
new_tree = "    ├── runs/{candidate.py,candidate_suite.py,config.py,execute.py,provenance.py,artifact.py}\n    ├── final_test/{__init__.py,contracts.py,workflow.py}\n    └── experiments/"
if old_tree not in package:
    raise SystemExit("package tree marker missing")
package = package.replace(old_tree, new_tree, 1)
old_intro = "`evaluation/experiments/` はdevelopment-onlyのhigher-level Study lifecycleを所有し、`evaluation/runs/` のverified Run Coreを再利用する。`evaluation/experiments/bootstrap/` はそのStudyを実行する前のcanonical preparationだけを所有する。"
new_intro = old_intro + "\n\n`evaluation/final_test/` はterminal `WINNER` Studyをread-onlyで検証し、unused-future / final evaluationを開く資格だけを別rootへone-shotで封印する。Dataset取得、strategy実行、Replay/P&L、stress、Production認可は所有しない。詳細契約は `architecture/final-evaluation-authorization.md` を正本とする。"
if old_intro not in package:
    raise SystemExit("package intro marker missing")
package = package.replace(old_intro, new_intro, 1)
package_path.write_text(package, encoding="utf-8")

agents_path = Path("docs/AGENTS.md")
agents = agents_path.read_text(encoding="utf-8")
old_row = "| Study/Experiment/EvidenceSet、controlled factor、lineage、freeze | `architecture/controlled-experiment-loop.md`, experiment contract/workflow tests |\n| 候補strategy/control、fit scope、evaluation scope | `research/current-status.md`, candidate/strategy tests |"
new_row = "| Study/Experiment/EvidenceSet、controlled factor、lineage、freeze | `architecture/controlled-experiment-loop.md`, experiment contract/workflow tests |\n| final-test authorization、unused/final window opening gate | `architecture/final-evaluation-authorization.md`, `tests/evaluation/final_test/`, `tests/architecture/test_final_test_boundary.py` |\n| 候補strategy/control、fit scope、evaluation scope | `research/current-status.md`, candidate/strategy tests |"
if old_row not in agents:
    raise SystemExit("docs AGENTS routing marker missing")
agents = agents.replace(old_row, new_row, 1)
agents_path.write_text(agents, encoding="utf-8")

root_readme_path = Path("README.md")
root_readme = root_readme_path.read_text(encoding="utf-8")
old_bullet = "- `docs/architecture/controlled-experiment-loop.md` — append-only Study/Experiment/EvidenceSet lifecycle and freeze contract\n- `docs/research/current-status.md` — research status, candidate comparison, development/final protocol"
new_bullet = "- `docs/architecture/controlled-experiment-loop.md` — append-only Study/Experiment/EvidenceSet lifecycle and freeze contract\n- `docs/architecture/final-evaluation-authorization.md` — WINNER-only one-shot gate before any unused/final-data access\n- `docs/research/current-status.md` — research status, candidate comparison, development/final protocol"
if old_bullet not in root_readme:
    raise SystemExit("root README documentation marker missing")
root_readme = root_readme.replace(old_bullet, new_bullet, 1)
root_readme_path.write_text(root_readme, encoding="utf-8")
