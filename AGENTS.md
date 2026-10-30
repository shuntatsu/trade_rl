# Agent entry point

作業前に current `main` HEAD、現在branch/worktree、open PRのhead/base、recent commitsを確認し、同目的の作業を重複させない。仕様については現行source・public API・近傍testsを優先し、docsとの不一致は同じ変更で修正する。

読む順序: [docs/README.md](docs/README.md) → [docs/AGENTS.md](docs/AGENTS.md) → 対象に応じたarchitecture/research docs → 対象source・tests・CI。全docsや旧計画を一律に読まず、必要な正本を選ぶ。

## Development

- 最初にObjective、Acceptance Criteria、Invariants、Failure Modes、Test Oracleを定める。変更は最小限とし、過去の説明を追記するより現行記述を置換する。
- 実装は専用branch/worktreeで行う。原則1 write Taskにつきdurable remote branchを1本とし、RED・format・smoke用のremote branchを増やさない。
- Bugfixでは再現→RED→修正→GREENを基本とする。新function/classは可能ならproduction投入前に小さなfake/mockで主要な入出力・失敗を確認する。fakeだけで統合/E2Eを代替しない。
- local-firstのtargeted tests・Ruff・format・mypy、影響に応じたintegrationを確認し、final diff/status/HEADを点検してpushする。詳細の実行コマンドはdocs/AGENTS.mdとCIを正本とする。

## PR / integration safety

- 通常の統合経路はPR。Agentはmainへの直接変更、force-push、history rewrite、branch削除を行わない。
- **tested PR head contains current `main`**。PR最終HEADに対応するFull CI・必要な独立レビューを確認し、merge直前にもmainの祖先関係を再確認する。HEADまたはmainが進めば旧Greenを使わず再検証する。Fast PushはFull CIの代わりではない。
- Independent Research Review / Generic reviewのreceipt・差戻し・承認条件はdocs/AGENTS.mdとdocs/architecture/research-assurance.mdに従う。ソフトウェアPRのレビューがGreenでも経済実験、unused/final data、live注文は認可されない。
- GitHubのbranch protection/rulesetはGit tree外の状態。必須チェック・force-push/削除禁止をread-back確認できない限り、設定済みと主張しない。

## Research / documentation / branch retention

- causal/point-in-time data、現実的執行費用、common accounting、hard risk、unused data、研究証拠の不変性を維持する。結果後の合格条件変更・leakage・過去の封印結果の再分類は禁止。研究変更はG0–G2のresult-blind反証と必要な独立reviewを先に行う。
- `tools.agent_repo` のsource-derived出力は便利な補助であり、GitHub PR/branch現況・実コード・Full CIの代替ではない。生成reportをcommitしない。
- `main`、open PR head/base、未完了workflow（requested/waiting/pending/queued/in_progress）、active Task、`provenance/branch-retention`等を保全する。現行のTask/lease同期が不完全な間、branch-hygieneのActions実行はread-only dry-run、手動CLIの`tools.branch_hygiene --apply`もfail-closedで拒否する。既存mainの削除workflowが安全化されるまでは有効化済みとみなさず、正本コードとGitHub設定を照合する。削除を再開する場合は完全なTask/PR/workflow保護、競合反証、retentionと復元を独立に検証する。
- `docs/` は現行正本のみ。完了済みplan/specは必要な契約を昇格してGit historyに残し、`docs/history`・`docs/archive`を増やさない。`LICENSE`・`LICENSES/`と、immutable研究artifactは通常のdocs整理から除外する。
