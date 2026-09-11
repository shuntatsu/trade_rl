# Agent entry point

Trade RL の変更を行う Agent は、この入口を読んだら、詳細文書・実装へ進む前に `main/current HEAD`、作業branch、open PRのhead/base、recent commitsを確認する。重複・競合する実装線がないことを確かめてから次を読む。

1. `docs/README.md` — 現行ドキュメントの入口と正本の一覧
2. `docs/AGENTS.md` — 読む順序、更新先、保持・削除ルール、検証契約
3. 変更対象に応じて `docs/architecture/lean-core.md`、`docs/architecture/package-boundaries.md`、`docs/research/current-status.md`

Local repository tooling (`python -m tools.agent_repo`) は preflight / context / impact / semantic diff / verification routing をsource-derivedで要約するために利用してよい。ただし、その出力はsource review、GitHub open-PR/branch overlap確認、final full CIの代替ではない。生成reportはcommitしない。

このRepositoryの現treeは現行システムだけを表す。完了済み設計・migration経緯・旧世代を保存するための `docs/history` / `docs/archive` は作らず、過去の内容は Git history から参照する。

コード構造を変更した場合は対応するcurrent docsと `tests/architecture/` の契約を同じ変更で更新する。研究上の前提・比較対象・評価手順・研究状態を変更した場合は `docs/research/current-status.md` を更新する。

ライセンス・provenance・third-party noticeは通常docsとは別の永久保持契約であり、`LICENSE` と `LICENSES/` を正本とする。明示的な法務目的なしに移動・削除・要約置換しない。

完了を判断する前に、対象テストだけでなくRepositoryのquality gateを確認する。少なくとも現在のCIが要求する Ruff、Format、Mypy、repository-tooling type check、full `tests/`、build、tracked source / sdist / wheel closure、clean-installed package identity と、変更内容に応じたarchitecture/contract testを実行し、未検証事項を成功扱いしない。
