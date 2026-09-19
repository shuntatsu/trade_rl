# Agent entry point

Trade RL の変更を行う Agent は、この入口を読んだら、詳細文書・実装へ進む前に `main/current HEAD`、作業branch、open PRのhead/base、recent commitsを確認する。重複・競合する実装線がないことを確かめてから次を読む。

1. `docs/README.md` — 現行ドキュメントの入口と正本の一覧
2. `docs/AGENTS.md` — 読む順序、更新先、保持・削除ルール、検証契約
3. 変更対象に応じて `docs/architecture/lean-core.md`、`docs/architecture/package-boundaries.md`、`docs/research/current-status.md`

Local repository tooling (`python -m tools.agent_repo`) は preflight / context / impact / semantic diff / verification routing をsource-derivedで要約するために利用してよい。ただし、その出力はsource review、GitHub open-PR/branch overlap確認、final full CIの代替ではない。生成reportはcommitしない。


## Development loop

新規のproduction function/classは、production sourceへ追加する前に、可能な範囲でlocalの最小mock/fake/stubを使い、意図した入出力・状態変化・主要failureを実行確認する。実装後はtargeted test/smokeとfinal diff/statusをlocalで確認してからpushする。mock/smokeは重要境界の実Integration/E2Eを置き換えない。

CIは `push` でFast Push（lint/format/type）のみ、`pull_request` to `main` でfull Core + PPO Runtime + Guideを実行する。PPO Runtimeは`train-sb3` extraの実SB3/Torchを使うintegration smokeである。統合判断にはexact final PR HEADのfull CIだけを有効な証拠として使う。

## Branch / verification budget

Remote branch・temporary workflow・Issue/PRは**検証用の使い捨て状態ではなく、共有されたdurable state**として扱う。新しいremote stateを増やす前に、local/mockで代替できない理由を確認する。

### Branch creation gate

新しいremote branchを作る前に、必ずcurrent `main`、open PR、同一Issue/目的のbranch、recent commits、queued/in-progress Actionsを再取得する。同じObjectiveを持つactive branchが1本でもあれば、新規branchを作らず、そのHEADをreview/continueする。並行更新を検出した場合も「競合回避用branch」を増やさず、現在HEADを読み直して既存作業へ合流する。

write Taskは原則として**durable implementation branch 1本**だけを所有する。exactly-onceな経済実行・外部side effectのように、implementation authorityと実行authorityを分離する必要がある場合だけ、sealed implementation後に**execution branchを追加で1本まで**許可する。それ以外の追加remote branchが必要なら、作成前にIssue/PRへ「既存branch・local worktree・run/artifactでは表現できない理由」を記録する。

次の目的だけでremote branchを作らない。

- RED / GREEN / formatter / Ruff / Mypy / py_compile
- targeted verification / exact-head verification
- seal / fresh reconstruction / audit / provenance recovery
- helper verification / runner preparation / workflow rerun
- GitHub Actionsを発火させるためだけのcommit/branch

これらは既存durable branch上のexact SHA、local worktree、またはActions run/artifactで表現する。verificationは原則read-onlyであり、検証対象branchから別のverification branchを派生させない。

### Local-first verification ladder

検証は次の安い順序で行う。後段を前段の代替にしない。

1. pure function / state machineをlocalの最小mock/fake/stubでRED/GREENする。
2. local targeted test、Ruff/Format/Mypy、必要なreal dependency smokeを行う。
3. 同じdurable branch/PRの既存CIを使う。
4. GitHub permissions、Artifact API、workflow run identity、remote exactly-once claimなど**GitHub固有semantics**だけを専用Actionsで確認する。
5. final PR HEADでfull CI / integration / E2Eを行う。

formatter・lint・type checkのためにtemporary workflowやbot formatter branchを作らない。local execution surfaceが利用できない場合も、同じdurable branchへ最小patchを入れて既存CIで確認し、helper branchを増やさない。

GitHub固有semanticsの検証でtemporary workflowが避けられない場合は、**同一durable branch上で1ファイルを更新して使い回す**。失敗ごとに `tmp-*-v2` / `v3` を追加しない。final candidate HEADではtemporary workflowを削除し、durable diffに残っていないことを確認する。

### Exactly-once preflight

one-shot training/economic execution/irreversible external side effectの前には、real executionより先にmock/fake/stubで少なくとも次をfailure-injectionする。

- optional dependency欠落・version mismatch
- source/artifact/authority identity mismatch
- duplicate activation / duplicate result slot
- evaluator開始直前の例外
- evaluator開始後・publish前のcrash window

必要runtime dependencyはevaluator/training境界より前にinstall/import/version検証する。execution権を消費する処理では、durable activation/claimをevaluator開始前に作り、claim後の再実行をfail-closedにする。mockはこの順序保証のoracleとして使い、実Integration/E2Eを置き換えない。

## Git / PR boundary

Agentによる実装作業は専用branchまたはworktreeで行い、PRを通常の統合経路とする。`main` を通常の作業branchとして直接変更しない。Integration invariant: tested PR head contains current `main`. merge直前にcurrent `main` のSHAを再確認し、tested PR headがそのcommitを包含していることと、その同一PR HEADに対する最新CI結果を確認する。`main` が進んだ場合、古いGreenを再利用せず、non-force merge/rebase等でcurrent `main` を含む新しいPR HEADを作って再検証する。

Agent作業で `main` へのforce-push、history rewrite、branch削除を行わない。Acceptance Criteriaと必要なverificationを満たし、current `main`を包含するfinal PR HEADのrequired checksがGreenなら、Integratorは追加のユーザー確認なしに通常のPR経路でmergeしてよい。branch protection / rulesetを有効化したと報告する場合は、GitHub側から設定をread-backして確認する。

## Remote branch hygiene

Remote branchは「作業履歴の保管庫」として増やさない。activeなwrite Task / PRごとにdurableなremote branchを原則1本だけ持ち、RED・format・verification・one-shot automationのための補助branchは可能な限りlocal branch/worktreeまたはGitHub Actionsのrun/artifactで扱う。remote補助branchが必要だった場合も、そのtipをdurable anchorへ取り込める形で終了し、孤立した一時refを恒久保存しない。

`main`、open PRのhead/base、実行中またはqueuedのGitHub Actionsが参照するbranch、protected branch、および `provenance/branch-retention` は自動cleanupのdurable anchorとして保持する。`research/`・`seal/`・`freeze/`・`run/` というbranch名だけでは永久anchorにしない。研究provenanceは、Issue/PR/Artifactに記録したexact commit SHAと、`provenance/branch-retention` のmerge-historyで保持する。

`.github/workflows/branch-hygiene.yml` は、非anchor branchを初めて観測したrunでは、そのexact tip SHAと元branch名を `provenance/branch-retention` のmerge-historyとcommit messageへ保存し、そのtip SHA自体もretention commitのparentとして到達可能にしたうえで元refを残す。同じbranch名・同じtip SHAが少なくとも1時間retentionで観測済みで、その間にopen PR / active workflow / protected anchorにならず、tip SHAがdurable anchorから到達可能である場合に限り元refを削除してよい。tipが変われば新しい観測として猶予をやり直す。したがってcompleted research refを削除しても、`git branch <元branch名> <記録済みSHA>` で復元できるexact provenanceを失わない。

削除直前にはopen PR head/baseとactive workflow branchを再取得し、削除自体は計画時のexact tip SHAを `git push --force-with-lease=<ref>:<sha> --atomic` で条件付き実行する。tipが変化した、protectedになった、open PR/active workflowから参照されるようになったbranchは削除しない。削除後にもbranch一覧を再取得し、削除対象refが残っていればworkflowを失敗させる。これはAgentによる手動branch削除の許可ではない。

このRepositoryの現treeは現行システムだけを表す。完了済み設計・migration経緯・旧世代を保存するための `docs/history` / `docs/archive` は作らず、過去の内容は Git history から参照する。

コード構造を変更した場合は対応するcurrent docsと `tests/architecture/` の契約を同じ変更で更新する。研究上の前提・比較対象・評価手順・研究状態を変更した場合は `docs/research/current-status.md` を更新する。

ライセンス・provenance・third-party noticeは通常docsとは別の永久保持契約であり、`LICENSE` と `LICENSES/` を正本とする。明示的な法務目的なしに移動・削除・要約置換しない。

完了を判断する前に、対象テストだけでなくRepositoryのquality gateを確認する。少なくとも現在のCIが要求する Ruff、Format、Mypy、repository-tooling type check、full `tests/`、build、tracked source / sdist / wheel closure、clean-installed package identity と、変更内容に応じたarchitecture/contract testを実行し、未検証事項を成功扱いしない。