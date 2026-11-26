# Agent entry point

作業前に current `main` HEAD、現在branch/worktree、open PRのhead/base、recent commitsを確認し、同目的の作業を重複させない。仕様については現行source・public API・近傍testsを優先し、docsとの不一致は同じ変更で修正する。

読む順序: [docs/README.md](docs/README.md) → [docs/AGENTS.md](docs/AGENTS.md) の共通・対象節 → 関連architecture/research docs → 対象source・tests・CI。通常は `python -m tools.agent_docs_context --topic core` で原文の共通節を表示し、必要な専門節（`guide` / `claude` / `tooling` / `coordination` / `eval`）を追加する。CLIが使用できなければ正本を直接読む。全docsや旧計画を一律に読まず、必要な正本を選ぶ。

## Claude Code integration

- The repository root `CLAUDE.md` imports this shared `AGENTS.md` instead of maintaining a second instruction authority. Claude Code and other agents must follow the same source/test/docs and research-gate contracts.
- `.claude/settings.json` runs a read-only `SessionStart` hook implemented by `tools/claude_session_context.py`. Its branch/HEAD/worktree and cached origin/main output is orientation only: it does **not** refresh GitHub, establish PR ownership, validate independent review/CI, or authorize writing/merging/experiments.
- `python -m tools.claude_host_preflight` is a redacted, read-only local host prerequisite check for Issue #1158. A green preflight **does not prove Claude Code loaded the configuration/Skill** and does not authorize writes, economic studies, or merging; actual host startup/`/context`/`/hooks`/`/skills` still need operator evidence.
- Claude Code integration is advisory, not a hard authorization gate: the project hook must not execute economic tests or mutate Git index; on a trusted local Claude Code installation with Python on PATH, verify `/status`, `/hooks`, `/skills`, and `claude doctor` plus a real startup/fork before claiming activation. Synthetic/CI tests do not prove the actual Claude Code host loaded the hook or skill.
- `.claude/skills/research-challenge/SKILL.md` packages the recurring result-blind adversarial challenge. It is author-side preparation and never an independent research review receipt.
- Keep personal Claude memory, credentials, permissions and local settings out of tracked source. Never turn generic assistant prompts, memory, browser automation, or scheduled tasks into trading decisions, research approval, or unsupervised live execution.

## Development

- 新規の個別研究は [trade_rl_research](https://github.com/shuntatsu/trade_rl_research) で行う。基盤の修正・汎用機能はこのrepoのIssue/PRに集約し、移行中の既存実行経路は承認済みcutoverまで維持する。
- 変更規模に応じて目的・受入条件・検証を明確にする。小規模なdocs/test/意味保存refactorは短い記録でよく、独立した設計文書や新しい経済研究packetを要求しない。大きな変更ではObjective、Acceptance Criteria、Invariants、Failure Modes、Test Oracleを定める。詳細はdocs/AGENTS.md。
- 実装は専用branch/worktreeで行う。原則1 write Taskにつきdurable remote branchを1本とし、RED・format・smoke用のremote branchを増やさない。
- Bugfixでは再現→RED→修正→GREENを基本とする。新function/classは可能ならproduction投入前に小さなfake/mockで主要な入出力・失敗を確認する。fakeだけで統合/E2Eを代替しない。
- local-firstのtargeted tests・Ruff・format・mypy、影響に応じたintegrationを確認し、final diff/status/HEADを点検してpushする。詳細の実行コマンドはdocs/AGENTS.mdとCIを正本とする。
- `.md`を変更したときは `python -m tools.markdown_links --check` で全追跡Markdownの相対リンクを検証し、authority・Guide fingerprint・研究根拠は各専用検査で別途確認する。リンクが通ることは仕様や研究の正当性を意味しない。
- コード・仕様書の意味上の不一致は `python -m tools.docs_source_consistency --check` で登録済み契約を検査し、Fast/Full CIで拒否する。歴史的INVALID/HOLD、source/spec digest、as-of会計反例の照合であり、登録されていない自然文すべての正しさを保証しない。新しい研究規範には独立した合成テストとreviewも追加する。

## PR / integration safety

- 通常の統合経路はPR。Agentはmainへの直接変更、force-push、history rewriteを行わない。mainに統合済みのPRの**exact final HEADと一致する**不要な作業branchは、exact-SHA leaseとGitHub再照合によって定期的に削除する。
- **tested PR head contains current `main`**。PR最終HEADに対応するFull CI・必要な独立レビューを確認し、merge直前にもmainの祖先関係を再確認する。HEADまたはmainが進めば旧Greenを使わず再検証する。Fast PushはFull CIの代わりではない。
- Independent Research Review / Generic reviewのreceipt・差戻し・承認条件はdocs/AGENTS.mdとdocs/architecture/research-assurance.mdに従う。ソフトウェアPRのレビューがGreenでも経済実験、unused/final data、live注文は認可されない。
- GitHubのbranch protection/rulesetはGit tree外の状態。必須チェック・force-push/削除禁止をread-back確認できない限り、設定済みと主張しない。

## Research Assurance Gate (#687)

新しい経済研究プロトコルと既存プロトコルの重要変更には、G0のcounter-hypothesis/cheapest falsifier/stop rule、G1のmechanism chain、G2の独立oracle、G3のevidence/claim boundaryを記述した版付きauthor packetを作成する。tools.agent_repo assurance の検証結果がUNREVIEWED/BLOCKEDならreviewと実験認可の成立を宣言しない。独立reviewは正確なpacket digestとprotocol/implementation HEADの外部確認が必要で、著者のPASS自己申告を受理しない。ソフトウェアPRのCI Greenと経済実験の認可を混同しない。正本は docs/architecture/research-assurance.md。

## Research / documentation / branch retention

- causal/point-in-time data、現実的執行費用、common accounting、hard risk、unused data、研究証拠の不変性を維持する。結果後の合格条件変更・leakage・過去の封印結果の再分類は禁止。研究変更はG0–G2のresult-blind反証と必要な独立reviewを先に行う。
- `tools.agent_repo` のsource-derived出力は便利な補助であり、GitHub PR/branch現況・実コード・Full CIの代替ではない。生成reportをcommitしない。
- `main`、保護ref、open PR head/base、未完了workflow（requested/waiting/pending/queued/in_progress）、研究・freeze・seal・run・evidence、`provenance/branch-retention`を保持する。**mainに未統合のbranchも削除しない。** 元PRがclosedでも、別のmain向けmerged PRの実際のnormal mergeがそのexact source tipを初めてmainへ導入したことを確認できる場合に限り、集約済みsourceとして扱える。 現在使うTaskはopen PRまたは明示された実行中workflowで参照を残し、統合済み旧branchの恒久保存は原則要求しない（次のM1監査例外を除く）。
- **マージ済みPRのexact source tipがmainに到達可能であれば、保全はmainのGit historyで完了している。** 旧契約の「1本ごとにretention追加parent、1時間grace、手動1件」条件は、統合済みbranchの定常削除には適用しない。未統合履歴や封印済み研究成果の保全条件は変更しない。
- **M1監査の限定保全例外**：`tools.branch_hygiene.plan_merged_retirement` は `fix/m1-*`、`fix/windows-job-child-pid-marker-20261105`、`fix/issue1112-m1-audit-20261123`、`fix/issue1112-junit-nested-counts-20261124` を `m1-source-preservation` として通常のmerged-PR source削除から除外する。M1の検証引継ぎのためにマージ後も元branch refを保持する明示的な例外であり、他の統合済みbranchへの恒久保存の一般化やTask/研究/実売買の承認ではない。例外の変更・解除には別途source/tests/docsと安全な削除条件の再検証を要する。
- 旧手動`prune-merged-branch.yml`はソースから廃止し、GitHub上で`disabled_manually`だった旧`branch-hygiene.yml`もtracked treeから除去した。現行の自動退役workflowは`.github/workflows/retire-merged-pr-branches.yml`（workflow ID `409000620`）の単一経路に限定する。GitHub Actionsで`active`と確認済みだが状態は変わり得るため、実行・統合時には都度readbackし、二つの削除workflowを同時に有効化しない。
- 過去の一回限りのarchive退役例外は廃止し、現行の証拠付きsource退役経路だけを実行可能にする。固定archive ref・研究証拠は保持し、過去の対象、実装、承認・結果はGit historyとIssue #1057から参照する。旧コメントの再実行や、未統合refの一般削除は認めない。
- `tools.branch_hygiene --retire-merged --apply` を標準の削除経路とし、`.github/workflows/retire-merged-pr-branches.yml`がmain更新時と定期実行で処理する。現在のmain SHA、全branchのexact SHA、closed/merged PRのexact source HEAD、open PR、未完了workflowを取得し、**source自身のmerged PR証拠を持つmain祖先refのみ**をatomicなSHA leaseで限定削除、各バッチ後にGitHubを再照合する。情報不完全・更新競合・認証失敗時は停止し、成功とみなさない。集約済みsourceは自動workflowと独立した手動CLIの `--include-consolidated` で読み取り専用previewできるが、自動workflowのpreview/apply対象外。保全対象でないことを個別に確認し、ownerが承認したexact branch/SHAを `--apply --include-consolidated --consolidated-ref BRANCH=SHA`（繰り返し指定可）で限定した場合だけ、追加のnormal merge証拠・fresh readback・leaseを満たして退役できる。履歴証明やCLI引数だけでTask/M1/研究refの退役権限は成立しない。`audit/`も保持する。古い汎用`--apply`は`--retire-merged`無しでは拒否を維持し、未統合分を一括削除しない。
- マージ済みPR source refの退役は`tools.branch_hygiene --retire-merged --apply`の一経路に統一する。旧retention記録を前提とする個別手動削除ツールは稼働対象に残さない。未統合・保護・研究refは削除せず、Coordinatorのregistry/lease認証・`cleanup_authorized`の成立、経済実験やlive注文の認可をbranch整理から推定しない。
- `docs/` は現行正本のみ。完了済みplan/specは必要な契約を昇格してGit historyに残し、`docs/history`・`docs/archive`を増やさない。`LICENSE`・`LICENSES/`と、immutable研究artifactは通常のdocs整理から除外する。
