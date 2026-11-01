# Agent entry point

作業前に current `main` HEAD、現在branch/worktree、open PRのhead/base、recent commitsを確認し、同目的の作業を重複させない。仕様については現行source・public API・近傍testsを優先し、docsとの不一致は同じ変更で修正する。

読む順序: [docs/README.md](docs/README.md) → [docs/AGENTS.md](docs/AGENTS.md) → 対象に応じたarchitecture/research docs → 対象source・tests・CI。全docsや旧計画を一律に読まず、必要な正本を選ぶ。

## Development

- 最初にObjective、Acceptance Criteria、Invariants、Failure Modes、Test Oracleを定める。変更は最小限とし、過去の説明を追記するより現行記述を置換する。
- 実装は専用branch/worktreeで行う。原則1 write Taskにつきdurable remote branchを1本とし、RED・format・smoke用のremote branchを増やさない。
- Bugfixでは再現→RED→修正→GREENを基本とする。新function/classは可能ならproduction投入前に小さなfake/mockで主要な入出力・失敗を確認する。fakeだけで統合/E2Eを代替しない。
- local-firstのtargeted tests・Ruff・format・mypy、影響に応じたintegrationを確認し、final diff/status/HEADを点検してpushする。詳細の実行コマンドはdocs/AGENTS.mdとCIを正本とする。

## PR / integration safety

- 通常の統合経路はPR。Agentはmainへの直接変更、force-push、history rewriteを行わない。branch削除は、次の厳格な **merged-PR sourceの個別整理** に限り許可する。
- **tested PR head contains current `main`**。PR最終HEADに対応するFull CI・必要な独立レビューを確認し、merge直前にもmainの祖先関係を再確認する。HEADまたはmainが進めば旧Greenを使わず再検証する。Fast PushはFull CIの代わりではない。
- Independent Research Review / Generic reviewのreceipt・差戻し・承認条件はdocs/AGENTS.mdとdocs/architecture/research-assurance.mdに従う。ソフトウェアPRのレビューがGreenでも経済実験、unused/final data、live注文は認可されない。
- GitHubのbranch protection/rulesetはGit tree外の状態。必須チェック・force-push/削除禁止をread-back確認できない限り、設定済みと主張しない。

## Research Assurance Gate (#687)

新しい経済研究プロトコルと既存プロトコルの重要変更には、G0のcounter-hypothesis/cheapest falsifier/stop rule、G1のmechanism chain、G2の独立oracle、G3のevidence/claim boundaryを記述した版付きauthor packetを作成する。tools.agent_repo assurance の検証結果がUNREVIEWED/BLOCKEDならreviewと実験認可の成立を宣言しない。独立reviewは正確なpacket digestとprotocol/implementation HEADの外部確認が必要で、著者のPASS自己申告を受理しない。ソフトウェアPRのCI Greenと経済実験の認可を混同しない。正本は docs/architecture/research-assurance.md。

## Research / documentation / branch retention

- causal/point-in-time data、現実的執行費用、common accounting、hard risk、unused data、研究証拠の不変性を維持する。結果後の合格条件変更・leakage・過去の封印結果の再分類は禁止。研究変更はG0–G2のresult-blind反証と必要な独立reviewを先に行う。
- `tools.agent_repo` のsource-derived出力は便利な補助であり、GitHub PR/branch現況・実コード・Full CIの代替ではない。生成reportをcommitしない。
- `main`、open PR head/base、未完了workflow（requested/waiting/pending/queued/in_progress）、active Task、保護branch、研究・freeze・seal・run・evidence、`provenance/branch-retention`を保持する。未統合・曖昧なbranch、Task/leaseの完全性を必要とする一般cleanupは引き続き削除不可とする。
- 例外は **すでにmainに統合済みのローカルPRのexact final-head source branch** を1本ずつ明示して整理する場合のみ。`tools.verified_merged_branch_prune` が、closed/merged PRと最終HEADの一致、mainとretentionへのSHA祖先関係、retention commitのbranch名・exact SHA記録と追加parent、1時間以上の猶予、open PRとunfinished workflowからの非参照、前後main/retention/sourceの不変性を再検証し、Gitのexact-SHA leaseによるconditional deleteと削除後readbackを行う。条件未達・失敗・不明なら必ずfail-closedで削除しない。消してもcommitと復元記録は残す。Active Task等がなお使用中と判明したbranchはmerged済みでも保持する。
- `tools.branch_hygiene --apply`と直接`apply_cleanup()`は一括削除のまま**常時拒否**し、既存のBranch hygiene Actions workflowもread-only dry-runに維持する。別の `.github/workflows/prune-merged-branch.yml` は明示的な `workflow_dispatch` のみ（1件・確認フラグ・exact SHAとmerged PR/retention record必須）で削除可能とし、自動/scheduled削除は行わない。上記の個別pruneは本番CoordinatorのTask登録/`cleanup_authorized`をtrueにしない。
- `docs/` は現行正本のみ。完了済みplan/specは必要な契約を昇格してGit historyに残し、`docs/history`・`docs/archive`を増やさない。`LICENSE`・`LICENSES/`と、immutable研究artifactは通常のdocs整理から除外する。
