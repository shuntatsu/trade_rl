# Agent documentation contract

## 目的

この文書は、AgentがTrade RLを変更するときに**何を先に読み、どの文書を更新し、何をcurrent treeに残さないか**を定義する。

## 読む順序

1. root `AGENTS.md`
2. `main/current HEAD`、作業branch、open PRのhead/base、recent commitsを確認し、進行中の同目的の変更・後継実装・廃止予定の作業線を識別する
3. `docs/README.md`
4. data/strategy/risk/execution/accountingを変更する場合は `docs/architecture/lean-core.md`
5. packageの責務・依存方向・public APIに影響する場合は `docs/architecture/package-boundaries.md`
6. 研究仮説、data availability、capital/observation/action/reward/risk/execution/accounting semanticsを変更する作業なら `docs/architecture/research-assurance.md` を読み、G0-G2をresult-blindに定義・反証する
7. Study/Experiment/EvidenceSetやdevelopment研究workflowに関わる作業なら `docs/architecture/controlled-experiment-loop.md`
8. final/unused-data authorization、WINNER binding、one-shot publicationに関わる作業なら `docs/architecture/final-evaluation-authorization.md`
9. 研究判断・候補・評価・データscopeに関わる作業なら `docs/research/current-status.md`
10. 対象source、public facade、nearest tests、CIを照合する

文書だけを根拠にsourceを推測しない。現行source、public API、`tests/architecture/`、関連contract testsとdocsを突き合わせる。

`python -m tools.docs_source_consistency --check` は、登録済みのsource/doc semantic contractを、GitHub・market-data・networkを読まずに検証する。退役inventory・共有契約の現行配置・退役sourceの再導入拒否と、現行harnessの登録およびsource/specのprotocol digestを検証する。旧AGY v1の `BACKCAST_2020 INVALID` / `FORWARD HOLD`、Learning Lab、BREADTHの旧source/spec/合成oracleは[一括退役記録](plans/repository-split.md#一括退役記録)の固定Git履歴に保全し、退役後の現行CLI・CIで実行するものではない。新しいプロトコルを追加する場合は対応するbindingと独立した反例を同じsource/test/docs変更に登録する。Fast/Full CIのsemantic gateが失敗したら仕様を結果後に緩めず、現行sourceを正本に失敗原因と歴史的証拠の保持を確認して是正する。成功は**登録した部分契約の検証**のみであり、他の自然文の正しさ、全PIT、G0–G5/G4、unused/final、経済成績や実取引を認可しない。

## 必要時参照

`docs/AGENTS.md` が詳細指示の唯一の正本であり、節の抽出は原文を省略表示するだけで契約を変更しない。作業に関係するsource・tests・正本は別途直接確認する。

```bash
python -m tools.agent_docs_context --topic core
python -m tools.agent_docs_context --topic core guide
python -m tools.agent_docs_context --topic core claude
python -m tools.agent_docs_context --topic core tooling coordination
python -m tools.agent_docs_context --topic core eval
python -m tools.agent_docs_context --topic all
python -m tools.agent_docs_context --check
```

`core` は共通の変更・Git/PR/研究認可・検証・保持規則を含む。`guide` はHuman Guide、`claude` はClaude Code host、`tooling` はRepository inspection、`coordination` は複数AgentのTask/Lease、`eval` はAgent UX評価の詳細である。複数topicを同時選択でき、専門topicだけの指定でも必ず`core`を含める。未知/重複/欠落した第2レベル見出し、未閉鎖コードフェンス、未閉鎖HTMLコメントはfail-closedで拒否し、HTMLコメント内の見出しは節の登録数に含めない。原文を自動修正しない。これはこの文書のATX `##` 見出しを対象とする節抽出であり、汎用Markdownレンダラーではない。

CLIが利用できない場合は本文の対象節（必要なら全文）を直接読む。抽出結果やローカルHookからcurrent GitHub main、PR、review、CI、実験・sealed/unusedデータ、実売買の認可を推定しない。Agent指示を更新するときは `--check` と近傍testを実行し、見出しを追加したらtopic分類を明示的に更新する。

## Human Guide update contract

Root `guide/` は人間向けの**非正本**説明層であり、技術仕様・研究状態のauthorityにはしない。画面ではMarkdown本文・静的flow・表を主役にし、実identifier・型・file pathは記事末尾の実装参照で副表示する。人間向け本文は `guide/content/pages/*.md`、machine metadata / source fingerprint / CodeReferenceは `guide/content/meta/*.json` に置き、UI componentへ研究固有の真実を直接埋め込まない。

Guide manifestは `document-guide-v2` とし、navigation groupとdefault `reading_order`を明示する。`overview` / `status` は `code_references=[]` を必須にし、`reference` はcode referenceを1件以上持つ。replay/PPOの主要処理順は初期DOMのMarkdownだけで読める状態を維持し、interactive step selectionを理解の前提にしない。MDX、raw HTML実行、custom Markdown UI directiveを導入しない。

正本Markdownとの対応はsection fingerprintで、Python実装との対応はAST由来の `guide/.generated/code-symbols.json` でfail-closedに検証する。browser用 `.generated/code-symbols-runtime.json` はreview済み `content/meta/*.json` の参照symbolだけへ射影する。生成indexはproduction moduleをimport/executeせず、current treeへcommitしない。

Guideがbindしている `docs/architecture/*` または `docs/research/current-status.md` のsectionを変更した場合、対応pageの説明が新しい正本と一致することを人間が確認してから、対象topicだけを明示的にrefreshする。Guideが参照するPython symbolを変更した場合も、日本語説明・variables・関連testを読み直してから対象topicだけ `--refresh-code` する。refreshを単なるCI通過手段として実行しない。

```bash
python3 guide/tools/content_contract.py --refresh-code <topic-id>
python3 guide/tools/content_contract.py --refresh <topic-id>
python3 guide/tools/content_contract.py --check
npm --prefix guide run check
npm --prefix guide run e2e
```

`--refresh-code` は検証済みCodeReferenceのsource digestだけを更新し、説明を自動生成しない。`--refresh` はMarkdown section fingerprintだけを更新する。正本sectionやsourceが変わっていない通常のGuide UI変更ではfingerprintを更新しない。

Guide source linkはfloating `main` ではなくbuildした**exact revision**へ固定する。CIはPR head SHA、Pages build/smokeはverified `workflow_run.head_sha` を `GUIDE_SOURCE_REV` として渡す。production JavaScript chunkは500,000 bytes以下をhard budgetとし、Markdown renderer追加を理由にwarning thresholdを上げない。

Guide側の説明と正本が食い違う場合は、正本をGuideへ合わせず、現行source・contract test・正本docsから契約を再確認してGuideを修正する。

## Claude Code host preflight (#1158)

`python -m tools.claude_host_preflight` from a **clean trusted `shuntatsu/trade_rl` root** checks the literal `origin` repository identity, the merged Claude Code integration commit ancestry, `CLAUDE.md` / `AGENTS.md`, the `.claude/settings.json` SessionStart exec hook and the manually invoked `research-challenge` Skill. It then checks `claude --version` and invokes the *Python hook directly* using the configured executable and argument vector. It rechecks Git worktree cleanliness after each subprocess and compares `.git/index` SHA-256 bytes and modification time before/after; an unexpected untracked or modified file fails the preflight. It prints a bounded JSON record (booleans, current HEAD SHA, CLI version) and fails closed on a dirty/unrelated repository or unavailable executable. It does not print branch names, remote URLs, untracked filenames, personal file paths, Git stderr or hook stdout.

The Skill prerequisite recognizes only the repository's flat frontmatter subset: physical delimiter lines, unique literal `name` and `disable-model-invocation` fields, and an optional single-line JSON-quoted printable description. Unknown fields, duplicate keys, multiline or ambiguous syntax are refused; this is not a general YAML parser.

**Scope limit:** `preflight_ok: true` is local Python/subprocess evidence **only**, not proof of Claude Code's actual session loading. On the operator's installed/trusted Claude Code host, independently check `/context`, `/hooks`, `/skills`, `claude doctor` and real startup/resume/fork execution as supported; attach only sanitized evidence to [Issue #1158](https://github.com/shuntatsu/trade_rl/issues/1158). The preflight never reads sealed data, runs economic simulations, sends orders, installs dependencies, grants review authority, or changes CI policy.

## Local repository tooling

Repository-localなsource-derived inspectionには次を使ってよい。

```bash
uv run python -m tools.agent_repo preflight --base main
uv run python -m tools.agent_repo context trade_rl/evaluation/runs
uv run python -m tools.agent_repo impact trade_rl/evaluation/runs/config.py
uv run python -m tools.agent_repo diff --base main
uv run python -m tools.agent_repo verify --base main
uv run python -m tools.agent_repo eval-list
uv run python -m tools.agent_repo source-packet --base BASE_COMMIT_SHA --head HEAD_COMMIT_SHA --output C:/temp/source-packet
```

- `preflight`: local branch/HEAD/base/worktree/Active docs/workflow rosterを表示する。
- `context`: capability、facade、imports/consumers、近傍test/docs、schema/effect signalをsourceから導出する。SHA-256に関連する名称には共通Validatorの`used`（実際の静的呼出し）または`candidate`（候補のみ）を表示する。候補表示は当該フィールドの意味や正しさの証明ではない。
- `impact`: 複数pathのcontextを決定的順序で表示する。
- `diff`: public/data-shape/schema/dependency/effectのreview signalを表示する。dataclassの注釈フィールドとliteralな`JSON_FIELDS`永続化一覧の変更も示すが、serialization互換性の保証ではない。signal自体を仕様違反判定には使わない。
- `verify`: Fast / Required final / Extended / Coverage signalを分けて表示する。実ファイルが存在するときだけRun config/CLI・共通SHA Validator・Dataset scope・Run artifactの旧互換性・Binanceのフォールバックとcache integrityの回帰テストを追加する。静的なimport検出では見えない負の制約を補助するだけで、final full gateや実行時E2Eの代わりにはならない。
- `eval-list` / `eval-show` / `eval-score`: versionedなAgent-UX task promptとgeneric rubricを表示・Evaluator入力から採点する。task-specificなEvaluator answer keyはchecked-in corpusへ保存しない。Evaluatorのscore JSONはdecoded keyの重複（escape aliasを含む）を全object階層で拒否し、critical failureを後続memberで上書きしない。拒否診断に未信頼keyを出力しない。
- `source-packet`: full commit SHAで固定した二つのGit snapshotからsource transport packetを作る。changed fileはGit blobのraw bytes、patch rendererはmanifestでGit versionとempty-tree attribute policyを固定し、lazy fetchは無効にする。patchのcross-version byte identityは主張しない。Git filesystem pathはOSのpath encodingで保持し、行区切りのalternates形式で単一pathとして表現できないobject directory（LFを含むか末尾がCR）は拒否する。出力TARに含めるchanged HEAD fileはcase-insensitiveなfile/directory prefix collisionを拒否し、削除済みbase pathは判定対象にしない。outputは全てのregistered Git worktree外にある未使用directoryに限り、生成物をcommitしない。

これらはnetwork-free local toolingであり、GitHub上のopen PR/branch overlapは別途確認する。出力は一時情報であり、生成JSON/Markdown reportをcurrent treeへcommitしない。

`python -m tools.stage_claims_status LEDGER_DIR [LEDGER_DIR ...] [--schedule FILE] [--json | --html]` は、single-use stage claim ledgerの**read-only状態表**である。POSIXの既存stage root検査とdescriptor-relative no-follow読み取りを使い、単一リンクのregular fileと開いたinode/size/mtime/ctimeを照合する。digest名namespaceのみを走査し、重複・不整合・foreign・symlinkはINVALID、不完全な走査はledger errorとする。旧形式の`<STAGE>.claim` / `<STAGE>.result.json`は別観測契約として保ち、`single_use_stage_claim_v1`は既存`StageClaim`の検証を再利用し、stage markerと唯一のcomment markerのrepository/comment id、stage、digest、commit、正規化UTC時刻が一致する場合だけCLAIMEDとする。旧形式・v1のclaim時刻は秒までの拡張ISO形式（Tまたは空白区切り、任意の小数秒、明示Zまたは±HH:MM）を受理し、小数秒の7桁目以降が非ゼロの時刻・曖昧なoffset表記は丸めて照合せずINVALIDにする。microsecondで正確に表せる同じUTC instantは等価とする。result mtimeは同じ開いたファイルの観測値であり、result digestやlabelの真正性を証明しない。result JSONはstrict decodeし、明示allowlist（`stage`・`label`・`protocol_digest`・`result_sha256`）以外を表示しない。未知のpath名・parse errorの原文・非canonical labelは表示しない。`--schedule`のcaller提供日付はUTC暦日。全入力ledgerの走査にerrorまたはINVALIDがなく、protocol digestが一つのledger・namespaceにのみ観測された場合だけ、scheduleを対応付けてNOT_CLAIMEDを推定する。対応できないprotocolは`schedule_unmatched_protocols`に残し、別ledgerのclaim不在を推測しない。HTMLもJSONもtextもstdoutのみで、書込み先や`--replace`は提供しない。shellで保存する場合はcallerがledger外を指定する。この表示は原子的な全ledger snapshot、研究認可、実験実行、mergeの根拠ではない。strict POSIX descriptorがない環境では拒否する。Full CIのsharded testsとmypy対象に含める。

GitHub上のIssue/PR/branchを読取照合するときは `python -m tools.github_work_inventory --repository shuntatsu/trade_rl` を使う。この外部adapterは `tools.agent_repo` の外に置き、read-onlyの非正本snapshotを返す。open IssueコメントのTask projectionは発行者の真正性が未確認であり、未登録作業やclosed Issueも網羅しないため、結果をbranch削除・Issue close・研究認可の根拠にしない。Issue一覧のID・node_id・repository URL・状態・comment件数・updated_atを3回読み、各open Issueの全commentのID・投稿者ID・body・updated_atを2回照合する。コメント取得は1Issueあたり最大1000件・GET最大11ページまでで、件数不一致・重複/不正ID・コメント変更・Issue更新をfail-closedで拒否する。 PR rosterは初回・Issue再照合前・Issue/任意workflow再照合後の3回照合する。既存branchのみを使うPRの作成・終了・base変更も検出対象とする。監査中にPR/Issue rosterまたはbranch SHA/protectionが変化した場合はsnapshot生成を拒否し、GitHub Repository名の大文字小文字を区別せず同一Repositoryを照合し、PR head SHAは現在のbranch tipと照合する。Issue/PR/commentのroster・JSON object・keyはnative list/dict/str、照合するRepository URL・commit SHA・Issue node/state/updated_at・comment body/updated_at・PR local/base refはnative str、Issue/PR/comment IDと番号はnative intを要求する。PR base/headの非null `repository.full_name`はASCII `owner/name` の2要素形式を必須とし、不正なUnicodeのcasefold同一視や余分なslash・前後空白は拒否する。branch rosterはnative tuple/BranchInfo、name/SHAはnative str、protectionはnative boolを要求し、repository/default/branchを再照合する。foreign subclassのlookup・比較・formatで別のidentityを採用しない。PRのbase ref自体は現在の同一repositoryのbranch rosterに存在することを要求する。一方、PR base.shaは過去の参照時点であり、現在のbase branch tipとの一致を要求しない。forkやhead repository不明のPRはローカルbranch tipとの照合ができず、`--compare-prs`でも`nonlocal_head_uncompared`と記録して祖先関係を推測しない。これも原子的な状態証明や削除許可ではない。

`uv run python -m tools.github_work_inventory --repository shuntatsu/trade_rl --include-workflow-refs` は任意のread-only補助監査として、既存`GitHubApi.active_workflow_branch_names()`で`requested/waiting/pending/queued/in_progress`の全paginationを列挙し、未完了workflowが参照するbranch名を二度確認する。対象branchがGitHub branch一覧に存在しない、両観測が変化した、APIが欠落・失敗した場合はsnapshotを拒否する。既知branch名および二度の観測はGitの`refs/heads/`文法、Unicode scalar、同じref namespaceで共存不能な親子名を検証し、初回の不正値や不在refを二度目のAPI呼出し前に拒否する。参照されたbranchは`unfinished_workflow_ref`として保全対象に表示するが、open PRや保護branch等の優先区分は変更しない。既定のinventory JSONは不変とし、opt-inでも`workflow_activity_checked=false`、`task_registry_complete=false`、`cleanup_authorized=false`を維持する。これはbranch名のみの時点観測であり、run attempt/job/artifactの完全列挙、Task/leaseの認証、registry writer、branch削除の許可ではない。

GitHub Actionsの単一runをread-onlyで追加確認する場合は `uv run python -m tools.github_workflow_attempts --repository shuntatsu/trade_rl --run-id <ID>` を使う。この外部adapterは実runの `run_attempt` に対応する `/attempts/{n}/jobs` を全page取得し、run/attempt/head/branch/job identityと重複・ページ件数・前後のrun readbackを検証する。 取得したrun・page・job objectはnative dict/native文字列key、job rowsはnative list、SHA・branch・job名・更新時刻などのidentity文字列はnative strに限定し、foreign subclassのlookup・比較・strip・列挙に依存しない。共通の `validate_workflow_state` / `validate_attempt_job_states` も直接呼出し時にnative dictとnative文字列key、job rosterのnative list/tuple、run status/conclusionの組合せをlookup・列挙・比較前に検証する。空のrosterでも不正なrun状態を拒否し、roster完全性は呼出し側の別契約で検査する。通常の `/runs/{id}/jobs` が空でも作業不在とは判定しない。取得が欠落・変化・矛盾した場合はfail-closedで拒否する。出力の `workflow_activity_checked=false`、`task_registry_complete=false`、`cleanup_authorized=false` を維持し、単一runの観測を全Task・全workflowの証明やbranch削除許可へ昇格させない。

`uv run python -m tools.github_workflow_artifacts --repository shuntatsu/trade_rl --run-id <ID>` は、単一runの現存artifactメタデータをGET-onlyで二度全ページ取得し、run・head SHA/branch・repository/head-repository ID・artifact ID/サイズ/期限状態/digest形式・前後runとrepositoryの同一性を照合する。対象のJSON objectはnative dict/native文字列key、artifact rowsはnative list、identity文字列もnative strに限定し、foreign subclassのlookup・比較・列挙に依存しない。曖昧・部分的・変動した一覧は拒否するが、GitHub RESTのartifacts一覧はrun単位であり、過去attemptへの所属や削除/期限切れで失われた履歴、artifact bytesの真正性は証明しない。`artifact_count=0`も過去の成果物不存在の証拠ではない。`run_attempt_observed`は現行runのattempt番号にすぎず、`artifact_attempt_binding_verified=false`、`artifact_bytes_verified=false`、`artifact_history_complete=false`、`workflow_activity_checked=false`、`task_registry_complete=false`、`lease_reassignment_authorized=false`、`cleanup_authorized=false`を維持する。Task/lease再割当・branch削除・研究/実運用の許可には転用しない。

`tools.github_lease_reconciliation.inspect_expired_lease_evidence` は、期限切れwrite LeaseRecord、read-only RegistryRefSnapshot、workflow-ref opt-in `github_work_inventory_v1`、単一run attempt/job観測とrun-scoped artifact観測を**渡された範囲だけ**照合するnetwork-freeの保留判定である。exact `main`/lease base、registry digest・Task/owner/epoch/branch SHA、inventory上のregistry ref名とOID、open PR refと`open_pr_count`の件数一致、未完了runが参照するbranchのworkflow一覧への登録、run/attempt/artifact間のID・head一致と重複・部分欠落を検査し、矛盾・型不正・状態変化は拒否する。 Registry refはsnapshot adapterと同一の制限付きGit ref文法で検証し、run/job/artifactのstatus・conclusionも共通validatorを使用する。未完了状態には明示的なnull conclusionを要求し、欠落をnullに補完しない。job名・状態・結論の不正、完了run内の未完了job、成功run内の失敗等のjobを拒否する。観測objectはnative dict/native文字列key、rosterはnative list/tupleに限定する。workflow branch一覧はnative文字列だけを受理し、外部objectの特殊hook（__class__を含む）・比較・stripを型検査前に呼ばない。Fast PushとFull CIの`Repository tooling types`で`tools/github_lease_reconciliation.py`のmypy検証を必須とする。成功でも`decision=HOLD`、`trusted_roster_complete=false`、`authoritative_registry_established=false`、`lease_reassignment_authorized=false`、`cleanup_authorized=false`を固定し、`LeaseReconciliation`や書込み権限を発行しない。入力の真正性・全closed Issue/Task/PR/workflow/削除済みartifactの網羅・原子的freshnessは証明できず、空の観測やIssue IDも非正本である。`main`がlease baseから進んだ場合も保守的に拒否する。これは完全な本番Coordinator/CASまたは使い捨てGitHub並行競合・中断E2Eではなく、一般branch cleanupの根拠にはならない。

`tools.github_coordination_ref_cas` は、将来の**使い捨てGitHub integration test専用**のGraphQL `updateRefs` CAS primitiveである。`beforeOid` / `afterOid`・authenticated viewer・repository identity・独立readbackを検証する。repository・ref・OID・loginは組込みの文字列型のみ、GraphQL responseと参照するobjectは組込みdictと文字列keyのみを受理する。派生型の比較・lookup・文字列操作に認証・対象判定を委ねず、planの型とidentity fieldもobject read前に拒否する。GraphQL errorsはnullまたは組込み空listに限り、mutation IDのreadbackも型と値を照合する。監査済みrepository node IDを明示した場合は、payload-bound CASを経由しても最終mutation前のGraphQL ref readで同一IDを要求し、別IDならmutationを送らない。source-bound enrollment transactionは初回REST/GraphQL照合で確認したIDをこの引数へ必ず引き継ぐ。標準HTTP adapterの既存repository ID固定も維持する。GraphQL の repository lookup は `followRenames: false` で旧名からの転送を禁止し、`refs/heads/coordination/*` 以外、ref削除、no-opを拒否する。通常呼出しではmutationをhard-refuseし、明示的なtest-only opt-inがある場合だけ `force=false` の1-ref updateを構成する。**production repository `shuntatsu/trade_rl` 自体はopt-inの有無にかかわらずmutation targetとして拒否し、さらにrepository名が `-coordination-disposable` で終わる明示的な使い捨てintegration repositoryだけを対象にする。** 成功しても `authoritative_registry_established=false`、`task_registry_complete=false`、`cleanup_authorized=false` を維持する。これはprotected registry、single Coordinator、完全なroster、実GitHub並行競合E2Eの代替ではなく、production repository上のTask登録CLIとして使わない。

`tools.github_coordination_ref_cas.compare_and_swap_registry_plan` は同じdisposable/test-only mutation gateに従う追加検証である。既存pure `RegistryTransitionPlan` をexact before/after commit OIDに結び付け、caller-supplied object readerから取得したcanonical registry bytesをstrict parseし、before/after digest・1-Task transition・CAS後payload再readbackを照合する。読み出し元の認証・Git object identityをこのfakeable Protocol自体が証明するわけではなく、post-mutation readback失敗時はref変更済みの可能性があるため成功やrollbackを主張しない。`shuntatsu/trade_rl`でのmutation禁止・既定hard-refuse・authority/roster-complete/cleanup=falseは不変で、実GitHub disposable E2Eと保護されたproduction writerの代替ではない。

`tools.github_coordination_disposable_e2e.GitHubEnrollmentReadHttpApi` は、使い捨てrepository専用の固定GitHub HTTPS GET transportであり、既存Git object readerのOID固定GETに加え、repository root、正のIssue番号、`main`、`coordination/task-registry`、fenced `agent/{task}/e{epoch}-{owner}` branch refのみを許す。HTTP method・suffix・tree queryのkey/valueは組込み型の完全一致を要求する。任意URL、POST、redirect、予期しないquery/参照は通信前に拒否し、token漏えいを防ぐ。`GitHubRegistryObjectReader`と組合せてsource-bound transactionの本物のGET adapterに使用できるが、認証・外部保護・atomic multi-GET証明ではない。

`tools.github_coordination_disposable_e2e` は、既存の `compare_and_swap_registry_plan` を**明示的な使い捨てrepositoryで実行するためだけ**の実GitHub HTTPS transport/CLIであり、本番Coordinatorではない。固定 `https://api.github.com/graphql` へのtoken認証POSTは、transport自身もdisposable repository・coordination ref・expected actor・before/after OIDへ束縛する。GraphQL documentは既存のexact `CoordinationRefState` queryと `UpdateCoordinationRef` mutationのみ許可し、document自体とread queryのowner/name/ref、mutationのrepository ID・single ref・OID・`force=false`を送信予定JSON bytesの独立snapshotでHTTP前に照合する。重複JSON keyを拒否し、送信documentと同一のsnapshotで応答も分類する。caller側の独自比較・可変入力・重複key解釈の相違から通信許可やmutationを昇格させない。mutationは既定で無効、認証済みactor/repository/refの事前readback後に限り1回だけ許可し、応答が不明でも同一transportから再送しない。固定 `https://api.github.com/repos/` を使う使い捨てrepository専用・GET限定のredirect拒否REST transportを介して既存 `GitHubRegistryObjectReader` のGET commit→tree→blobを行う。一般の `GitHubApi` はRESTのredirect先へAuthorizationを転送する可能性があるため、このtest-only実行器では直接使用しない。最初に既存のtest-only target guardで `shuntatsu/trade_rl`、suffix不一致、非coordination ref、不正40-hex SHA、no-op、mutation opt-inなしを**通信前に拒否**する。CLIは `--allow-disposable-mutation` か `--inspect-interrupted-mutation` を排他的に明示し、`GH_TOKEN`/`GITHUB_TOKEN`、`--repository OWNER/NAME-coordination-disposable`、`--ref-name refs/heads/coordination/task-registry`、`--registry-path coordination/registry.json`、`--expected-oid <exact40OID>`、`--after-oid <exact40OID>`、`--expected-coordinator-login <GitHub login>` を要求する。実行者が準備した変更前・後の**不変Git commit**上のcanonical registry payloadから削除なしの厳密な1-Task遷移だけを再構成し、GraphQL `updateRefs(beforeOid/afterOid, force=false)` のreadbackとpost-CAS canonical bytesを検証する。ゼロOIDは初回ref不在の前提条件としてのみ使用でき、after_oidの削除は許さない。token/response bodyはエラー表示しない・HTTP redirect/retryは拒否する。失敗後はremote mutationが済んだ可能性があり、**自動再試行もrollbackも行わない**。`--inspect-interrupted-mutation` はrefとactor/repositoryの二重readbackを行うだけで、`AFTER_OBSERVED_UNATTRIBUTED` / `BEFORE_OBSERVED_NOT_PROOF_UNAPPLIED` / `CONFLICTING_OID_OBSERVED` のいずれも自分の書込みや安全なretryの証明としない。すべての観測は `authoritative_registry_established=false`、`task_registry_complete=false`、`cleanup_authorized=false` 等の非権威契約を維持する。実際のdisposable repositoryの準備・外部protected ref・production single writer・全roster・並行/中断GitHub E2Eは本実装だけでは未成立であり、test fakeの成功をE2E実行済みと誤記しない。branch削除は行わない。

Fast Pushの`Coordination disposable CAS smoke`では `uv run pytest -q` により `tests/tools/test_github_coordination_ref_cas.py`、`tests/tools/test_github_coordination_registry_cas_payload.py`、`tests/tools/test_github_coordination_disposable_e2e.py`、`tests/tools/test_github_coordination_disposable_http.py`、`tests/tools/test_github_coordination_enrollment_audit.py`、`tests/tools/test_github_coordination_enrollment_transaction.py`、`tests/tools/test_github_coordination_cas_native_boundary.py` のfake/no-network回帰テストを実行する。これは従来のRuff/Format/Mypyに**追加**する短い早期失敗検知であり、pull_request/review-eventの全Core/PPO/Guide/PostgreSQL/Generic Full CIや実disposable GitHub並行CAS・中断E2Eに代わるものではない。Fast成功だけでPR統合、Coordinator権限、roster完全性、cleanupや研究・取引権限を認めない。

`tools.github_coordination_registry_reader.GitHubRegistryObjectReader` は将来のdisposable test統合のための **read-only Git object adapter** で、明示されたrepositoryとexact commit SHAからGitHub RESTの GET commit → GET recursive tree → GET blob を辿る。配置pathはcallerが指定し、NULを含むpathは通信前に拒否する。取得objectはnative dict/native文字列key、tree rowsはnative list、参照するidentity文字列はnative strに限定し、型検証前に外部objectのlookup・比較・列挙hookを呼ばない。nontruncated tree内に普通の `100644` blobが一意であること、commit/tree/blobの返却OIDが要求値と一致すること、blob sizeとstrict Base64 decode後bytesのGit SHA-1一致、canonical registry parseとrepository一致をfail-closedで確認する。APIに対してはGET以外を呼ばない。API object自体の認証済みrepository binding・commit/treeのraw-object暗号学的証明はこのadapter単独では成立せず、許可信頼できるCoordinator・protected ref・remote CAS・GitHub実E2E・complete roster・cleanup authorityの代替ではない。canonical payloadへの実readbackに使用できても `authoritative_registry_established=false`、`task_registry_complete=false`、`cleanup_authorized=false` を維持する。

`tools.github_coordination_registry_snapshot.observe_registry_snapshot` は、このGET-only object readerを使って `refs/heads/coordination/*` の**read-only時点観測**を作る。最初と最後にrepositoryのexact name / GitHub node ID / native numeric IDを確認し、同じref名・commit OIDを2回確認して、その間にcommit→tree→blobからcanonical registry digestを取得する。ref・repository identity変動、存在しないref、非native object/key/identity文字列、payload不一致、API不達はfail-closedで拒否し、部分取得をempty registryや完全rosterへ変換しない。前後一致は瞬間的な観測であって、認証済みCoordinatorの証明、外部protected ref、roster完全性、全workflow/Taskリコンサイル、remote CAS linearizabilityやinterruption E2Eの保証ではない。`RegistryRefSnapshot` の `authoritative_registry_established` / `task_registry_complete` / `cleanup_authorized` は常にfalseで、このsnapshot単独ではbranch削除を認めない。統合済みPR sourceの個別削除は、後述の独立した厳格な判定のみを使用する。

`python -m tools.worktree_hygiene` は、`<repo-root>/.claude/worktrees/` 内の**ローカルlinked worktreeだけ**を対象にする古いAgent作業ディレクトリ整理ツールである（remote branchは扱わず、それは`tools.branch_hygiene`の範囲）。既定はread-onlyのplanで、各worktreeをREMOVABLE/KEEPと理由付きで表示し`--json`にも対応する。`--apply`は次を**すべて**満たすREMOVABLEだけを`git worktree remove`（`--force`なし）で削除する：lockされておらず、ツール自身が動くworktreeでない／rebase・merge・cherry-pick・revert・bisect（`sequencer/`含む）が進行中でない／`git status --porcelain --untracked-files=all --ignored=matching`にtracked変更・untrackedが無い／HEADが`origin/main`の祖先（`--fetch`指定時のみ`git fetch origin main`）／HEADがopen PRのheadと一致せず、branch名もopen PRのhead branchでない／他の登録worktreeや`.git`（ignored領域を含め走査）を内包しない／symlink・junction・reparse pointを一つも含まない（`--allow-ignored-data`の有無に関わらず、削除直前にも再確認。Git for Windowsは`git worktree remove`中にjunctionを辿り対象外のtarget内fileを削除し得るため。cache名のjunctionも対象）／ignored fileが再生成可能なcache（`__pycache__`・`*.pyc`・`.pytest_cache`・`.mypy_cache`・`.ruff_cache`）だけ／ignored fileを含む全file・directoryの最新mtimeが`--min-idle-hours`（既定24）より古い。`--allow-ignored-data`はcache以外のignored data（artifacts・data・outputs・`.env`等）の**削除を許す破壊的**オプションで、削除物は復元できない。`gh`失敗・open PR一覧が上限(200)に達した場合・`origin/main`が解決できない場合は何も削除しない（fail closed、終了コード2）。削除直前にlock・HEAD・進行中操作・status・内包worktreeを再確認し、directory renameの往復probeで使用中のhandleを検出して失敗時はKEEP `in-use`とする（Windowsでのみ有効で、POSIXでは検出できない）。rename probeが元に戻せなかった場合は即座に全体を停止して終了コード1とし、`git worktree prune`も実行せずprobe pathを表示する（人が手動で復元する）。その残骸（`*.hygiene-probe-*`）が対象worktreeの隣に残っている間は`--apply`の開始を拒否する（終了コード2、planは可）。それ以外のworktree単位の失敗は報告して残りを処理し終了コード1、終了時に`git worktree prune`を実行する。**生きているsessionの検出はlock・ツール自身のworktree・idle時間のheuristic・Windowsのrename probeに限られ、他のsessionを網羅的に検出するものではない**。local branchは`--delete-local-branches`指定時のみ`git branch -d`で削除し（`-D`は使わず、拒否は報告のみ）、remote branchとstashには触れない。成功は個々のworktreeの安全条件の確認に限られ、Task/lease・研究・実売買の認可やGitHub上のbranch整理の成立を意味しない。

## Research Assurance Gate (#687)

新規economic research protocol・material changeではG0–G3 result-blind author packet
research_assurance_v1 と別個の research_assurance_review_v1 を扱う。
python -m tools.agent_repo assurance inspect assurance.json は独立性を自動証明しない。
python -m tools.agent_repo assurance check assurance.json はPASS以外で非0終了する。
外部認証済みのレビューprincipal、record digest、protocol/implementation HEADが
一致しないとPASSはなく、PASSもG4/G5/unused/final/liveを自動認可しない。
G1–G2 mechanism/runtime監査のIssue #667とは目的が違い、並行して確認する。
正本は architecture/research-assurance.md、例は
tools/agent_repo/examples/research_assurance.json に置く。

## Agent Coordination Plane

複数Agentが同時に作業する場合、Repository Control Planeのsource-derived inspectionとは別に、`tools/agent_repo/coordination/` の **Agent Coordination Plane** を使ってTask契約・依存・競合・ownership・evidence freshnessを扱う。Coordination Planeはdomain/research/artifact authorityではなく、Repository上の作業を安全に分離・統合するrepository toolingである。

Network-free operator surfaceは次を提供する。

```bash
uv run python -m tools.agent_repo task digest task.json
uv run python -m tools.agent_repo task ready snapshot.json
uv run python -m tools.agent_repo task status-render status.json
uv run python -m tools.agent_repo task status-parse status-comment.txt
uv run python -m tools.agent_repo task dashboard dashboard.json
```

- Task contractは `task_revision` とcanonical **Task Contract digest** の両方へbindする。revision bump漏れでもdigestが変われば古いlease/review/verification evidenceをstaleとして扱う。
- Task状態は単一statusではなく `phase + condition` で表す。`review / stale` や `executing / blocked` を区別する。
- `execution_mode=read_only` はimmutable snapshotを共有してよく、writable lease/branchを持たない。`write` Taskはactive ownerを1つに限定し、isolated writable worktree/branchを使う。
- write ownershipはCoordinator-issued **lease epoch** をfencing tokenとして扱う。reassignment時はepochを増加し、new epoch専用branch/worktreeを作る。expired leaseはbranch/PR/CI/artifact side effectをreconcileするまで即reassignしない。
- `LeaseReconciliation` の `pr_ids`・`ci_ids`・`artifact_ids` は順序付きlist/tuple内の**native文字列**のみを受理し、list入力は不変tupleへコピーする。null、数値、真偽値、map/set/generator、単体文字列を証跡IDとして黙って文字列化しない。これらは外部Coordinatorによるreconciliation・Task/lease完全列挙の証明ではなく、read-onlyのHOLD観測を再割当/削除権限へ昇格させない。
- `WriteScope.allow/deny` と `TaskPacket.non_goals/resource_keys/acceptance_criteria/test_oracle` は直接Python生成でもnative文字列のlist/tupleだけを受理する。暗黙の`str()`変換、単体文字列・unordered set・map・generatorを拒否し、caller-owned listを検証前にtupleへ固定する。これらの文字列、TaskPacketのtitle/objective/deliverableはstrict UTF-8で表現可能でなければならず、deliverableは文字列に限定する。Task構築時点で拒否し、READY後にdigest計算が失敗する状態を許さない。旧来の正常なcanonical digestや公開JSON schemaは変えない。直接Pythonからの厳格なJSON decoderもnative dict/list/str/int（optionalのNoneを含む）に限定し、subclassの列挙・lookup・比較を利用しない。 `TaskPacket.write_scope` はnative `WriteScope`、`dependencies` はnative `TaskDependency` のlist/tupleに限定し、依存rosterは検証前にtupleへ固定する。mutableなduck-typed objectやsubclassで生成済みTaskのdigestを変更できる状態を許さない。
- `LeaseRecord` は正の有限TTLと最終heartbeatからの厳密なexpiry算出を要求する。expiryはUTC timeline上の経過時間で計算し、元timezoneの表示を保持する。`lease_is_current` とgrant/reconcile/heartbeat・期限切れHOLD preflightはUTC instantを比較し、DSTのgap/foldによるwall-clock差をTTLや前後関係と混同しない。最終heartbeat以降、expiryより前だけを有効とする。lease/reconciliationのTask ID・owner・branch・SHA・digestはnative文字列とし、独自比較による異なるfenceの受理を拒否する。時刻の逆行や延長された見かけのexpiryをTask登録・再割当権限へ昇格させない。
- file overlapだけでなく `authority:` / `identity:` / `schema:` / `workflow:` / `artifact:` / `side-effect:` Resource Keyでsemantic collisionを判定する。TaskPacket構築時点で既定の7 namespaceとfile resource文法を検証し、未知namespaceをREADYに進めない。 不正resourceの拒否診断と未知status/lease task IDの拒否診断に未信頼のkey/path/ID本文を含めない。`DependencyGraph` と `ready_tasks` はcaller-owned packet rosterを入口でtupleに固定し、`ready_tasks` はstatus Mappingもdictに一度コピーする。依存検証・hard conflict・選択でその同じlocal snapshotを使い、元Mappingを再観測しない。DependencyGraphはsnapshot内の全要素がnative TaskPacketであり、Task IDと依存先IDがnative文字列であることを、graphのkey生成・依存比較より前に検証する。 hard/integration依存の直接判定でも取得した非null状態はnative TaskStatusに限定し、状態属性参照前に拒否する。欠落状態は従来どおり未充足とする。これは本番registryの完全性や外部snapshotの原子性を証明しない。hard conflictは同時実行しない。
- CoordinatorがTask phase/condition/leaseのsingle writerであり、Worker/Reviewer/Verifier/Integratorの自然言語報告だけでstateを確定しない。status commentはCoordinator-owned durable projectionとして扱う。 projectionのJSON parseでは重複fieldをfail-closedで拒否する（最後の値を採用しない）。
- Review/verificationはTask revision/digest/base/**exact HEAD**へbindする。HEADが変わった後のapproval/Greenを再利用しない。`main`が進んだ場合はintegration evidenceをstaleにし、tested PR head contains current `main` を満たす新HEADで再検証する。
- Workerは自分でTaskを`complete`にしない。review → verification → integration → post-merge oracleを通し、child Taskが全部completeでもparent Acceptance Criteriaとcross-task invariantを最後に再評価する。 親完了predicateにはnative dict[str, TaskStatus]の観測を渡し、foreign mapping/statusを拒否する。親oracleはnative Trueのみ受理し、callerのrosterだけで全Taskの完全性は証明しない。
- `task ...` CLIはlocal JSON/textを解析・表示するだけでGitHubへclaim/comment/mergeを書き込まない。authorized GitHub writeはCoordinator/Integrator側の明示的な外部actionとして扱う。
- `tools.agent_repo.coordination.enrollment` は既存の `TaskPacket`・`LeaseRecord`・`TaskStatusRecord` とIssue identity / main・branch SHAの整合性を検証する、**非正本・read-only preflight** である。 `EnrollmentProjection.issue_node_id` は構築時点でstrict UTF-8表現可能な非空文字列に限定し、registryのcanonical bytes生成前に不正surrogateを拒否する。node IDの発行元真正性・外部認証はこの検査では成立しない。引数に渡されたIssue open状態、Task roster、registry OID、readbackは認証済み・網羅済みの証拠ではない。外部Coordinator認証、GitHub上のatomic CAS、実際の読み戻しと競合E2Eが揃うまでは、preflight成功をTask登録、`task_registry_complete=true`、cleanup/delete許可として扱わない。Task rosterとして渡された `enumerated_task_ids` は1回だけimmutable tupleへ固定し、その同一snapshotで型・重複・membershipを判定する。caller-owned mutable/stateful Sequenceを複数回再観測して登録可否を変えない。同一leaseのままTask revisionを増やす更新や、`complete`後の状態変更もこのpreflightでは拒否する。実際のrevision更新には新lease・外部reconciliation・transaction検証が必要である。
- `tools.github_coordination_issue_projection.observe_issue_projection` は、既存のEnrollmentProjectionとcaller指定のGitHub投稿者IDを使い、Issue識別子・open状態・最大1000件のコメントページを二度GETで照合する**read-only projection観測**。Taskコメントはcanonicalな1件のみ許容し、異なる作者・コメント内容・コメント数・Issue更新・ページ欠落を拒否する。 入力は構築時に検証済みのnative EnrollmentProjection、responseのnative dict/list/strとnative string keyに限定する。型の異なるcontainer・textはlookup・比較・iterator等のhookを実行する前に拒否する。これはatomic snapshotでも認証済みCoordinator single-writerやclosed/unregistered Taskの完全列挙でもなく、返却する`authoritative_registry_established`・`task_registry_complete`・`cleanup_authorized`は常にfalse。GitHub Issueの状態コメントだけで登録・lease再割当・削除を許可しない。
- `tools.agent_repo.coordination.registry` は将来の専用Coordinator refへ保存する**off-tree canonical payload schema**だけを定義する。既存 `EnrollmentProjection` をTask entryとして再利用し、同一repository・Task ID一意性・Task ID順のdeterministic JSON・canonical UTF-8 bytes / SHA-256 digest・strict parseを保証する。重複JSON key、未知/欠落field、schema不一致、cross-repository entry、malformed nested status、non-canonical bytesをfail-closedで拒否する。empty registryを表現できても完全列挙の証拠ではなく、checked-in mutable ledger、Coordinator認証、CAS成功、`task_registry_complete=true`、cleanup/delete許可にはしない。
- registryの更新候補は `plan_registry_transition` でpureに計画する。plannerはregistry自身からTask roster/previous entryを導出し、既存enrollment transition ruleを再利用し、callerが指定したexact before payload SHA-256とdeterministic after registry/digestをbindする。stale digest、no-op、cross-repository、illegal phase、revision/lease/head driftを拒否するが、これはremote OID CAS・GitHub認証・完全列挙・branch ancestry verificationの代替ではない。返却planのauthority/registry-complete/cleanup flagは常にfalseである。
- `tools.github_coordination_enrollment_transaction.execute_source_bound_disposable_enrollment` は、**本番では常に書込み禁止**の使い捨てGitHub検証用に、既存GET-only登録監査のTask/Issue/lease/registry観測と、RESTとGraphQLのrepository node ID・Coordinator actor照合、および既存GraphQL `beforeOid`/`afterOid` CASの単一Task変更案を同じトランザクション候補に束ねる。まず本番repository・非coordination ref・未明示のtest-only opt-in・不正native契約を**通信前に拒否**する。外側の初回source観測（repository numeric/node ID、Issue identity/updated_at、`main`/lease branch）を内側の登録監査にも固定し、各registry snapshotの初回repository identityもその固定値と照合する。監査前後とCAS直前のIssue/`main`/lease branch・registry refを照合し、変更前/変更後のcanonical Git payloadをCASが検証する。CAS後は別のREST Git-object readerで、初回と同じrepository numeric/node IDを要求してregistry ref→commit→tree→blobと完全なafter registry、Issue/`main`/lease branchを再照合する。観測された内側だけの不一致も拒否するが、観測されないABAの排除は保証しない。不一致・GitHub API失敗・競合・CAS直後の通信断は拒否し、成功扱い/自動再試行/ロールバック/lease再割当はしない。複数GETの原子性・ABA排除・二つのtransportの同一認証主体・外部ruleset保護・真の単一writer・完全Task roster・実GitHub並行/中断E2Eは**このfake testでは証明されない**。`enrollment_persisted=false`、`authoritative_registry_established=false`、`task_registry_complete=false`、`lease_reassignment_authorized=false`、`cleanup_authorized=false`はすべて固定し、実GitHubの使い捨てrepositoryでのauth/CAS/中断・競合試験とproduction権限確立までIssue #1067をOPENにする。
- `tools.github_coordination_enrollment_audit.audit_task_enrollment` は1件のTask登録候補について、GitHub GETでrepository ID/default main・open Issue番号/不変node ID/更新時刻・main SHA・fenced lease branch SHAを前後二度確認し、その間に既存の専用registry ref→commit→tree→blobのcanonical snapshotを検証する。RESTのrepository/Issue/ref responseに含まれるobjectと照合対象のidentity fieldは組込み型に限定し、transactionの期待Issue node IDもsource GET前に検証する。期待するIssue node IDは呼出し側で明示し、registry全体から既存Taskを照合した後に既存`prepare_enrollment_projection`と`plan_registry_transition`のread-only計画だけを返す。closed/PR/別Issue・変更されたsource、欠落したbranch、main/lease/registry drift、API失敗、不正状態は例外でHOLDする。ただし複数GETはatomic snapshotではなく、actor/transportの独立認証・Issue projection publish、Task全件/closed Issueの完全列挙、production writer・CAS・並行E2Eを証明しない。`enrollment_persisted=false`、`authoritative_registry_established=false`、`task_registry_complete=false`、`lease_reassignment_authorized=false`、`cleanup_authorized=false`は固定。Issue #1057のmerged-source条件付きbranch整理は独立した既存gateであり、この観測を削除権限に変換しない。
- Task/Issue/lease identityのrevision・Issue番号・epochはPython上でも**native positive integer**を要求し、`1.0 == 1`のようなinteger-shaped floatを受理しない。GitHub projectionやhandoffも同じ型境界を共有し、malformed inputを例外型の漏れや暗黙の同値比較で通さない。また、`HandoffPacket`のverified/pending/known_failures/do_not_repeat/evidence_idsはlist/tupleの**文字列要素のみ**を受理し、元listの変更が反映されないtupleに正規化する。整数・真偽値・null・unordered set・mapping・単体文字列を黙って`str()`に変換して証跡として受け入れない。
- 以下の通常退役規則に対し、2026-11-27のowner判断（[#1057のexact対象](https://github.com/shuntatsu/trade_rl/issues/1057#issuecomment-7212724691)）だけを限定手動例外とする。実行入口は同じ `tools.branch_hygiene`。
- **統合済み作業branchの通常終了は自動**：`tools.branch_hygiene --retire-merged --apply`（`.github/workflows/retire-merged-pr-branches.yml` のmain push/定期/manual実行）は、最新mainのGit履歴に到達可能で、原則として同じbranch名とexact source SHAを持つclosed/merged PRが確認できるrefを対象とする。**mainと同じSHAで新設された未着手のbranchは、マージ済みPRの対応記録がなければ削除しない。** 集約済みsourceは自動workflowと独立した手動CLIの `--include-consolidated` で読み取り専用previewする。自動workflowのpreview/applyはsource自身のmerged PR証拠に限定する。集約済みsourceのapplyは、Task/M1/研究の保全対象でないことを個別に確認してownerが承認したexact branch/SHAを `--apply --include-consolidated --consolidated-ref BRANCH=SHA`（繰り返し指定可）で限定する。未指定・重複・不一致・非適格は拒否し、選択外の通常merged sourceもこの実行では削除しない。履歴証明やCLI引数自体はowner承認を証明しない。`audit/`も保全する。GitHubのclosed local PRのexact branch名/HEADを、完全なGit main first-parent履歴の最初の到達点と照合する。その到達点がmain向けclosed/merged local carrier PRのexact merge_commit_shaで、実commitが2親、第二親がcarrierのexact final headである場合にだけ、元PRをmergedと偽らず集約済みと判定する。first-parent上のsource tip、未統合、squash/patch同一性だけ、fork、証明不明は保持する。各削除batchでclosed source/carrier証跡を再取得し、その後open PR/未完了workflow・branch/mainの完全readbackを再確認する。coordination/・task/・lease/も保全する。既存のexact lease、atomic push、readback、M1・研究保全例外は変更しない。GitHub branch一覧（exact SHA・保護状態）、closed/merged PRのexact head証跡、全open PR head/base、未完了workflow（requested/waiting/pending/queued/in_progress）の現況を再照合し、保護ref・研究/封印/provenance系・未統合のrefを削除しない。GitHub tokenでのatomic pushと各refのexact-SHA leaseで同時更新を拒否し、各バッチのreadbackが一致しない場合は停止する。重複名、欠落/予期しないref、SHA/保護状態の差異は件数と上限付きbranch名サンプルを診断に出すが、差異を許容する再試行・削除認可・rollbackを意味しない。 初回preflightと各削除batchのGit push直前に、pushと同じeffective Git設定でeffective fetch URLとpush URLを全件列挙し、それぞれ正規trade_rl HTTPS URLがちょうど1件であることを要求する。複数URL・別repoへの書換え・観測時の設定driftを拒否する。個別subprocess間の同時local設定書換えに対する原子的lockではない。workflow statusページの件数変動だけは、requested/waiting/pending/queued/in_progressの全ページを先頭から最大8試行で取得し直し、途中の不完全snapshotを破棄して、完全な最終snapshotだけを使用する。試行間のsleepは最大7回・合計25.4秒で打ち切り、8試行とも件数不一致なら削除前にfail-closedで停止する。malformed/unknown/auth失敗は直ちに拒否して削除しない。失敗/unknownは成功扱いしない。
- この限定経路はmerged PRのfinal headがmain履歴に保存済みと確認できるため、旧式の`provenance/branch-retention`へのPR別追加parentや一律1時間待機を新たに要求しない。mainの履歴にないunique commitsや研究証跡を持つbranchには**適用しない**。分離した未統合作業はIssue/PRへ用途・残作業を記載して引き継ぎ、不要と確定した後に別途履歴保全を行う。Task/leaseの権威registryが未完成であることは依然事実であり、全Taskを列挙済み・`cleanup_authorized=true`とは主張しない。
- 過去の一回限りのarchive退役例外は廃止し、現行の証拠付きsource退役経路だけを実行可能にする。固定archive ref・研究証拠は保持し、過去の対象、実装、承認・結果はGit historyとIssue #1057から参照する。旧コメントの再実行や、未統合refの一般削除は認めない。
- **M1 source branchの例外**：M1監査PRの引継ぎ用 `fix/m1-*`、Windows試験用 `fix/windows-job-child-pid-marker-20261105`、`fix/issue1112-m1-audit-20261123`、`fix/issue1112-junit-nested-counts-20261124` は、merged済み・main祖先のexact source tipであっても自動退役から除外し、plannerは `m1-source-preservation` の理由でkeepする。これは既存merged-branch自動削除方針の**限定的な例外**であり、他のsource ref、未統合ref、研究/provenance refの安全条件やatomic SHA leaseを緩和しない。経済評価/G0–G5・Coordinator/Task権限の成立も意味しない。
- `tools.branch_hygiene`の旧モード（`--retire-merged`なし）はGitHubへ接続する前に拒否し、旧`apply_cleanup()` APIは除去した。main祖先か不明な全branchを一括削除する経路は作らない。旧retention recordを要求する手動削除経路は廃止し、定常の統合済みPR source ref退役は`tools.branch_hygiene --retire-merged --apply`に一本化する。未統合・保護・研究refにこの経路を転用しない。Repository/GITHUB_TOKENのcontents writeが使えなければジョブ失敗を報告し、権限を迂回して削除しない。
- **2026-11-27限定手動例外**：`--retire-merged --retire-approved-20261127 --backup-git-dir PATH` は、sourceに固定された16個のexact branch/SHAだけをpreviewする。sourceのrepository rootで実行し、Git repository/config環境変数による上書きを拒否する。追加の `--apply` は現行mainの同一sourceを要求する。対象のうち9個はmainの祖先かつexact merged/consolidated PR証明を要求し、M1の他の保全例外は解除しない。旧Cloud7個は後継 `prepare/cloud-regime-switch-screen-20261124` を保持し、sourceに固定したexact SHAの履歴への到達性と既存の独立したローカルGit退避先（Cloud7個と後継のexact remote-tracking ref、full Git fsckによるobject整合性、alternate/shallow・ref/object metadataのsymlink不可）を確認する。新しいarchiveは作らない。全候補を単一atomic SHA leaseで削除し、事前/事後の全branch readbackの直後にopen PR・未完了workflowを再照合し、観測差異を拒否する。事後の差異は削除のrollbackや自動再試行を意味しない。候補の変更・保護・open PR・未完了workflow・後継不一致・証明欠落・通信失敗時は削除前に停止し、事後不一致も成功とみなさない。保持refの同時変更やPR/workflow参照の新設を原子的に排除する契約ではないため、既存バックアップを別に保持し、他の作業を停止してownerが実行する。対象外、任意prefixの一括削除、通常自動workflowへの転用、研究実行/消費authorityやTask cleanup認可の変更は不可。
- 旧read-only保持期間/retention記録plannerは現行実装から除去した。観測は`tools.branch_hygiene --retire-merged`のdry-runに統一し、結果は削除許可やTask完全性の証明ではない。実際の削除は`--retire-merged --apply`だけで、fresh readback・exact-SHA lease・事後照合を必須とする。

## Agent Eval protocol

Agent向けRepository構造の改善を評価する場合、baselineとcomparisonで同一suite/rubricを使い、**評価対象変更以外のRepository差分を混ぜない**。

1. 評価pairを作る時点のcurrent PR base/current `main` SHAをbaselineとし、comparisonにはそのexact baselineをancestorとして含むPR HEADを使う。実装開始時の古いbaseを、後からmain更新を取り込んだHEADと比較しない。
2. pairの実行完了前に`main`が進んだ場合、その比較証拠をstaleとして破棄し、current baseを含むcomparison HEADへ同期してbaseline/comparisonの両側を組み直す。
3. baseline側とcomparison側の各taskは、それぞれexact snapshotから作ったcleanな使い捨てbranch/worktreeで独立実行する。前taskの変更を次taskへ持ち越さない。
4. task-specificな過去会話を持たないfresh Agent sessionを開始し、通常のRepository accessだけを与える。
5. Agentへはtask promptと通常のRepository instructionsだけを渡す。generic rubricを明示的な攻略指示として追加せず、task-specificなsemantic goal / expected owner / critical failure / review question等のanswer keyはevaluated checkoutへ置かない。
6. 完了後にdiff、選択したcommands/tests、public/private imports、残存riskを観測する。
7. 全generic rubric dimensionを短いobservable evidence付きで採点する。critical failureはgeneric critical dimensionsの0点から決定し、task-specificな隠し正解をRepository内へ追加しない。
8. aggregate score、critical failure、changed-path count、目立つ不要探索だけをPR/Issue summaryへ記録する。
9. transcript、model reasoning、run-specific log、generated patchをcurrent treeへcommitしない。

Agent Evalはv1ではExtended verificationでありPR hard gateではない。点数上昇だけを品質証明にせず、critical failureやauthority/boundary/scopeの回帰を優先して見る。benchmarkへ過適合するためにprompt/rubricを弱めない。

## Git / PR integration safety

Agent実装は専用branchまたはworktreeで行い、PRを通常の統合経路とする。`main` への直接変更を通常経路にしない。Agentはforce-push、history rewrite、`main`削除を行わない。Acceptance Criteriaと必要なverificationを満たし、current `main`を包含するfinal PR HEADのrequired checksがGreenなら、Integratorは追加のユーザー確認なしに通常のPR経路でmergeしてよい。

Integration invariant: tested PR head contains current `main`. PRの成功証拠は**現在のfinal HEAD**に束縛し、merge直前のcurrent `main` SHAをそのtested PR headがancestorとして包含していることも確認する。PR headが動いた後の古いCIだけでなく、`main` が進んだ後の古いPR-head Greenも現在の統合証拠として扱わない。`main` が進んだ場合は、non-force merge/rebase等でcurrent `main` を含む新しいPR HEADを作り、その新HEADでpermanent CIを再実行する。将来merge queueを採用する場合は、current target branchを含むmerge-group SHAのrequired checkを同等のoracleとしてよい。

Branch protection / rulesetはGit treeとは別のGitHub設定である。保護を導入・変更した場合は、GitHubから設定をread-backし、required check、force-push/deletion、PR requirement、maintainer/admin bypass挙動を確認してから有効化済みと報告する。設定変更surfaceが利用できない場合は、proseやarchitecture testを代替にせず**未設定/未検証**と報告する。

統合後の `main` push は実際に反映された commit SHA で Fast Push・全Core shard/aggregate・PPO・Guide・PostgreSQL を再検証し、`Main Integrated Software Verification` がすべての success を要求する。PR以外の作業branch pushは従来のFastのみ、PRとレビューのFullおよびreview gateは従来どおりとする。main上のpushにはPR review receiptを捏造・流用しない。新mainによる旧runのcancelは旧commitの成功証明ではない。実際のmerge treeが検証済みPR treeと異なれば、事後の成功もpremergeの祖先条件を遡及的に充足させない。GitHubのexpected-head指定は対象branchのCASではなく、真の並行merge防止には外部の単一Integratorまたはmerge queue/保護が必要である。追加のmain Fullによるrunner負荷とGuide deploy開始の遅延は、実統合treeの検証を優先する明示的なtrade-offである。

通常のCI workflowの`concurrency.group = ci-${{ github.workflow }}-${{ github.ref }}-${{ github.event_name }}` は、同じPR refでも通常`pull_request` Fullと`pull_request_review` Fullを異なる待機・実行グループに分離し、通常PRの再実行がレビューFullのpending枠を置換することを防ぐ。`cancel-in-progress`のtruth tableは各イベント群内で維持する：レビュー`submitted`かつ`commented`はfalse、レビュー`edited`/`dismissed`や`changes_requested`/`approved`はtrue、新HEADの`pull_request/synchronize`やpushもそれぞれの群内でtrue。異なるイベント群は並行実行可能になる一方、新HEADのPR同期は旧HEADのレビューイベントFullを途中cancelせず、不要なrunner時間が生じ得る。GitHub既定の単一pending枠は**同じレビューイベント群内**には残り、重複コメント間のpending置換までは解決しない。GitHubの実スケジューラでの全挙動は静的testだけでは保証できない。両Fullの**exact final HEAD**での全job SUCCESS、review receipt有効性、unresolved blocking 0、最新main祖先関係を必須とし、旧HEAD・取消run・Fast Push・一部job Greenではマージしない。

### Independent Research Review / PR lifecycle

現行coreの `Generic Independent Research Review` のライフサイクル契約は次の通りとする。前提はLean Core・PPO Runtime・Human Guide・Market Store PostgreSQLの4ジョブすべての`success`であり、`skipped`・空値・未定義・失敗を許容しない。旧4h PPO専用研究review/triggerの実装は固定履歴に退役し、Generic gateは旧 `research/ppo-4h-indicator-smoke` refをBLOCKEDにする。専用研究認可をGenericへ移管せず、再開には別途承認済みの研究gateを要する。

1. **レビューゲート待機 (PENDING)**: 先行CIジョブ（core, ppo-runtime, guide, market-store）がパスしたPR exact HEADに対し、独立レビューゲート（`independent-research-review-audit`）が未完了（PENDING）の状態で外部レビューを待機する。
2. **software model reviewの実行**: Generic software PRでは、PR exact HEADの差分・関連source/tests/contractsをAI reviewerがread-onlyで監査する。`software_source_review_v2`では実装担当と同じconfigured model（例: `gpt-5.6-sol`）を許容し、fresh contextや別providerは必須にしない。これはsoftware source reviewであり、model independence・provider実行・研究result-blind independenceの証明ではない。歴史的4h PPO専用reviewの独立性要件は保存し、現行のG0–G5/economic reviewをGenericで代替しない。
3. **指摘と差し戻し (BLOCKED)**: 重大な懸念・脆弱性・契約違反（Medium / High severity）が検出された場合、指摘内容をレビュー本文に記録し `### Disposition: BLOCKED` で投稿する。PRはマージされず、修正対応（差し戻し）となる。
4. **承認とマージ・クローズ (APPROVED)**:
   - 未解消のblocking指摘がないことを確認し、当該receipt形式の有効な証拠が揃った場合だけ `### Disposition: APPROVED` を記録する。Lowの改善提案はreportへ残してよい。
   - `tested PR head contains current main` を満たし、全必須CIチェックがGreenとなった段階で、PRをマージ（`gh pr merge`）してクローズする。

Generic gateはGitHub review APIのauthenticated actorとexact `commit_id`へbindし、PR head refが欠落・不正ならPENDING、旧4h専用refならBLOCKEDとする。従来の`independent_model_review_v1`は互換のため受理し、新しい`software_source_review_v2`は`reviewer_model_id`・`review_result=PASS`・`blocking_findings=0`・具体的reportだけをpublisher-attested software review metadataとして受理する。同じconfigured modelを許容し、外部review job、provider attestation、diff digest、実装Task ID、model independenceは要求・主張しない。旧`software_source_review_v1`は未検証のjob/diff値を権威ある証拠のように見せるためfail-closedのまま廃止する。G0–G5・sealed/unused/final・live・歴史的4h PPO専用reviewはこの軽量化の対象外。旧経路の退役は認可の代替・移管ではない。

## 更新matrix

| 変更 | 必ず確認・更新する場所 |
|---|---|
| causal data、availability、artifact identity | `architecture/lean-core.md`, data tests |
| strategy interface、quantity hold、strategy/risk責任 | `architecture/lean-core.md`, strategy/replay tests |
| execution/accounting/fill/funding/borrow/liquidation | `architecture/lean-core.md`, simulation/risk/evaluation tests |
| package移動、責務境界、依存方向、public facade | `architecture/package-boundaries.md`, `tests/architecture/` |
| Study/Experiment/EvidenceSet、controlled factor、lineage、freeze | `architecture/controlled-experiment-loop.md`, experiment contract/workflow tests |
| final/unused-data authorization、WINNER binding、one-shot publication | `architecture/final-evaluation-authorization.md`, `tests/evaluation/final_test/`, boundary tests |
| economic hypothesis、causal story、capital/observation/action/reward/risk/execution/accounting semantics | `architecture/research-assurance.md`, G0-G2 counterexample / independent-oracle tests |
| 候補strategy/control、fit scope、evaluation scope | `research/current-status.md`, candidate/strategy tests |
| M1/M2/M3状態、development/final/stress手順 | `research/current-status.md` |
| Guideがbindする正本section / Python symbol | 対応する `guide/content/pages/*.md` と `guide/content/meta/*.json`, `--refresh` / `--refresh-code`, `guide/tools/content_contract.py --check` |
| Guide Pages deployment / public URL | `.github/workflows/deploy-guide.yml`, `guide/README.md`, Pages state read-back |
| docsの入口・保持ルール | `docs/README.md`, `docs/AGENTS.md`, root `AGENTS.md` |
| Markdownの相対リンク・導線 | `python -m tools.markdown_links --check`（全追跡Markdown、外部URL/fragmentは対象外）、各文書の正本・Guide content contract |
| license/provenance/third party | `LICENSE`, `LICENSES/`, package metadata。通常cleanupとは分離する |

## フォルダと保持ルール

- `architecture/`: current authoritative architectureのみ。
- `research/`: current research status/evidence protocolのみ。
- `specs/`: 未実装またはreview中で、独立してnormativeなdesignが存在するときだけ作る。
- `plans/`: 実装中の独立planが必要な間だけ作る。
- `specs/` / `plans/` に置く各Markdownは先頭付近に `Status: Active` を明記する。Activeでないfileをcurrent treeに残さない。
- 完了した `specs/` / `plans/` は、耐久的な内容を `architecture/` / `research/` へ反映した後に削除する。
- `docs/history` と `docs/archive` は作らない。
- 過去の設計・migration経緯・削除済み文書は **Git history** から取得する。

削除する前に、非再生成可能なsource data、未記録の実験結果、exact reproductionに必要なcommit/lock/config、legal noticeを失わないことを確認する。

## 法務の境界

`LICENSES/` は通常docsではない。license text、licensing policy、provenance、third-party noticeは永久保持するcompliance materialであり、完了済みplan/specと同じ理由で削除しない。

## 実装時の原則

- 現行の良い責務境界を優先し、将来使うかもしれない抽象化を追加しない。
- private old pathを残すだけのcompatibility shimは、明示的public contractの証拠がない限り作らない。
- 新規の個別研究（仮説、戦略、比較条件、固有runner、判定）は [trade_rl_research](https://github.com/shuntatsu/trade_rl_research) の責務とする。このrepoは共通データ・時間軸・指標・学習・risk・執行・会計・評価契約を所有する。研究側からの不足機能やbugは、再現条件と期待契約を示す短いIssueから着手できる。基盤は研究packageに依存しない。既存のsource-bound実行経路は移行と承認が完了するまで維持する。
- 小規模なdocs/test/意味保存refactor・fabricated-input検証は、変更の目的と影響範囲に応じたoracleを短く記録すればよい。独立planや新しい経済研究packetを作る必要はない。挙動や共通契約を変える場合は事前にTest Oracleとfailure modeを決める。経済mechanism変更、実データ権限、Full CI、独立reviewの要件は変更しない。
- 研究・経済mechanismを変更する場合は、経済結果を確認する前に `architecture/research-assurance.md` のG0-G2、falsifier、semantic invariant、Known limitationsを定義する。P&Lやwinner判定でG0-G2の不備を救済しない。
- **AI semantic review is mandatory**。上記変更ではeconomic result valuesを生成・閲覧する前にAIへresult-blind G0-G2 adversarial reviewを実施させる。既定はWorkerと異なるconfiguredモデルのfresh read-only AI reviewerであり、G0/G1のpremise・causal story・Mechanism ContractをAIが反証する。独立review surfaceがない場合もAI review自体は省略せず独立性を`NOT ESTABLISHED`とするが、新しいG4 economic result生成へは進まない。AIの肯定だけでmachine oracleやsource-bound testを代替しない。
- refactor/bugfixはTDDでREDを観測してからproductionを変える。
- Greenにするためにassertionを弱めたり、skipや古いforwarderを残したりしない。
- 移動だけのrefactorでは、可能ならAST/serialization/public facade等の独立oracleでsemantic driftを反証する。

## Verification contract

完了報告前に、変更内容に応じたtargeted testと全体quality gateを確認する。現在の標準gateは次である。

```bash
uv run python -m tools.agent_docs_context --check
uv run python -m tools.markdown_links --check
uv run ruff check trade_rl tests tools
uv run ruff format --check trade_rl tests tools
uv run mypy trade_rl
uv run mypy tools/agent_repo tests/architecture/distribution.py
uv run mypy tools/independent_research_review.py tools/docs_source_consistency.py tools/ci_test_shard.py
uv run pytest -q tests
```

さらに `uv build`、tracked production Python sourceとsdist/direct wheel/sdist再build wheelのpath・bytes一致、checkout外での非editable installとpublic import/CLI smoke、package identity、関連architecture/contract test、GitHub Actionsの**同一final HEAD**の結果を確認する。 Full PR CIは`Lean Core`・`PPO Runtime`・`Human Guide`・`Market Store PostgreSQL`の4ジョブを要求し、Market Storeでは実PostgreSQL serviceによるintegration test（skipなし）とmigrate/status smokeも確認する。古い3ジョブのみのGreenは現行Full gateの代替としない。PPO/SB3境界を含む変更では、PR-only `PPO Runtime` jobで`train-sb3` extraの実Stable-Baselines3/Torchを導入し、environment API・実rollout・interleaved fit・model save/loadをsmokeする。旧Learning Labのcore runnerと専用テストは固定履歴へ退役した。研究側でのLearning Labの変更は、研究repoのlocked runtime実importと合成agent学習・checkpoint保存/読込・評価・real-data拒否を含む独立した実行検証を要し、core CIは代替しない。Core側のoptional skipだけで実runtime検証を代替しない。fake/mockだけをintegration oracleにしない。古いHEADのGreenを現在HEADの証拠にしない。merge直前は同時に、tested PR head contains current `main` を満たすことを再確認する。

Hosted Full CIの `Lean Core tests` は全 `tests` を各runnerで収集し、pytest node IDの安定SHA-256で8分割する（実測の高負荷・fixture再利用に基づく現行2 moduleだけは `tools/ci_test_shard.py` の明示的な配置に従い、同一shardでfixtureを再利用する）。各shardの成功時だけ同一HEAD・同一workflow attemptに紐付く `ci_test_shard_v2` receiptを発行し、完了node ID一覧・全収集件数/digestに加えて全nodeのsetup/call/teardownの結果（passed/skipped/xfailed/xpassed）とskip/xfail理由、収集段階のskip node ID/理由を保存する。完了はpassを意味しない。各nodeのphase rosterと独立全収集のcollection skip rosterも完全一致を要求し、旧v1 receiptを現行v2の証跡として受理しない。旧履歴は当時のsourceとreceiptのまま保全し、失われた旧skip詳細や別runtimeでのpassを推定しない。 各nodeのfinish通知だけでは完了とせず、setup/call/teardownの実行reportまたは正当なsetup skipとteardownを照合する。setup-only/setup-plan等の非実行modeを拒否し、従来のskip/xfailは維持する。`Lean Core` は全8shardの成功、重複・漏れのない完了一覧と全収集digestおよび独立した通常pytest全収集との一致を必須とし、その後も従来の静的検査・配布物・installed smokeを行う。失敗・中断・artifact欠落・異なる収集roster・異なるHEADは拒否する。部分rerunの旧attempt receiptは再利用せず、必要なら全jobsをrerunする。通常のローカル `uv run pytest -q tests` は分割しない。退役した専用PPO研究認可は復活させず、現行の必須CI/レビュー条件は維持する。各shardで遅い25件とskip理由を記録し、実際のhosted wall time・待ち時間・runner minutesで改善を評価する。

Python依存関係の監視は、Dependabotのuv更新がHOLD中でも継続する。 `.github/workflows/ci-uv-advisory-audit.yml` (#1441 / #1175) は固定uvとそのversionを強制する隔離configで公開OSVの全locked extrasを読み取り専用監査し、registry package rosterとreport件数、scanner終了状態、manifest/lock不変を検証する。PRではfindingsを明示するreport-onlyであり、scheduled/manual scanはfindingsで非0終了、scanner failureは全eventで非0終了する。通常の必須source Full CIとは別の監視で、成功は有効なreportの生成を意味し、脆弱性不在・更新復旧・研究実行認可ではない。新規advisoryと修正候補は#1441で追跡し、過去のlock、実行環境、receiptを変更しない。GitHub-hosted Dependabotのuv version updatesは停止したまま、以下のrepo-owned候補生成で更新を管理する。

Python依存関係の更新候補はtrade_rl内で管理する（#1175）。 `.github/workflows/uv-lock-proposals.yml` は毎週火曜04:45 JST、mainでの手動実行、およびworkflow・候補生成/監査tool・pyproject.tomlの変更がmainに統合されたときに起動し、固定uv 0.13.10を使い、committed HEADのGit archiveを別の一時ディレクトリへ展開して `uv lock --upgrade-package` を実行する。公開PyPIの同一major・既存package roster・不変project contractに限定した候補のみを生成し、同一releaseのhash/metadata変化、downgrade、prerelease/epoch/local、package追加削除、registry変更、対象外packageの変更、solver失敗を拒否する。これは保守的な候補範囲であって互換性・脆弱性不在を保証しない。元のmanifest/lock・研究履歴は変更しない。出力artifactの `report.json` はexact HEAD、uv version、manifest/新旧lock SHA-256と変更一覧を保持し、変更がある場合だけ `uv-lock.patch` と候補 `uv.lock` を添付する。GitHub権限はcontents: read、push・PR作成・自動mergeを行わない。通常の同一repository PRとして候補を採用するときはfresh main上で再生成し、source/lock hash照合、`git apply --check`、差分・advisory inventory・全必須CI・exact-head software reviewを確認してから統合する。単独packageが必要なら手動workflowのpackagesにnameを空白区切りで指定する。ローカルでもrepository rootから `python -m tools.uv_lock_proposal --head "$(git rev-parse HEAD)" --packages "ruff" --output /path/to/new-output` を固定uv/Python 3.12で実行できる。outputはrepo外の未存在pathに限定する。直接pin変更、major/roster変更、security修正に必要な例外は明示的な別PRで扱う。GitHub-hosted Dependabotのuv version updatesはlimit 0を維持し、外部repoへの修正提出やupstream mergeをこの運用の完了条件にしない。

Coverageはbranch coverage 80%を目標signalとするが、80%未満だけを理由にmerge不可とはしない。重要Failure Mode、変更行、Error/Retry/Timeout/Fallback、Assertion品質を優先し、数値を上げるだけの低価値testを追加しない。

テストGreenだけでは正しさを宣言しない。最終diff、削除物、public API、重要failure mode、未検証事項、残存riskを再確認する。

## 作業線の終了と変更の証拠

自身のmerged PRのexact final headに一致する不要なmain祖先branchのみ、`tools.branch_hygiene --retire-merged`のpreviewと、`--apply`の再照合・exact-SHA lease・削除後readbackで定期的に整理する。集約済みclosed PR sourceは独立した手動CLIの`--include-consolidated`で導入証明をpreviewし、ownerが保全不要と個別承認したexact branch/SHAの`--apply --include-consolidated --consolidated-ref BRANCH=SHA`指定でのみ退役できる。自動workflowはこの追加経路を呼ばない。open PR head/base・未完了workflow・保護ref・研究/封印/provenance・未統合historyは保存する。旧式の個別retention記録を毎回追加する必要はない。GitHub API/認証が不完全な場合は停止する。通常`--apply`と直接`apply_cleanup()`の一括汎用cleanupは禁止のまま。ブランチ整理はCoordinator authority/Task完全登録や研究・live認可を示さない。

ソース移動やPython docstring変更でも、Runのimplementation manifestは相対パスとraw bytesにbindするため実装digestが変化する。同じStudyの比較途中へ移行を混ぜず、元のsource/runtimeを保持する。計算の意味保存と、期待されるprovenance変化は別々に検証する。
