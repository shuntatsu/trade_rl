# Interactive Human Guide Pages v1

Status: Active

## Objective

公開済み `guide/` を、Repository内のソース閲覧だけでなく、人間がURLから直接利用できる静的WebサイトとしてGitHub Pagesへ配信する。

公開は既存Guideの説明責務・正本境界を変えず、`docs/` を技術・研究のauthority、`guide/` を非正本の説明層として維持する。

## Non-goals

- `guide/` を技術仕様・研究状態のauthorityへ昇格しない。
- trading、研究実行、artifact生成、live/Production操作をWeb UIへ追加しない。
- custom domain、analytics、authentication、server-side API、databaseを追加しない。
- `gh-pages` branchへ生成物をcommitしない。
- Firefox/WebKit E2Eをこの公開変更の必須scopeへ広げない。
- Pages公開を理由に既存Lean Core CIを弱めたり、Human Guide CIを置き換えたりしない。

## Current state

- Repositoryはpublicで、GitHub Pagesは現時点で未有効化である。
- `guide/` はReact + TypeScript + Vite + Tailwindで静的build可能である。
- `guide/vite.config.ts` は `base: "./"` を使用しており、project Pages配下でもasset URLを相対解決できる。
- Guide navigationはhash-basedであり、Pages側のSPA fallback rewriteを要求しない。
- Guideの「正本」リンクはGitHub `main/docs/...` へのabsolute URLであり、Pages配下へ公開しても意味を変えない。
- Permanent `ci.yml` はLean CoreとHuman Guideを検証し、Human Guideではsource freshness / lint / typecheck / unit / production build / Chromium E2E / axeを実行する。

## Proposed architecture

### Separation of concerns

品質検証と公開を別workflowに分離する。

```text
push / PR
   |
   v
.github/workflows/ci.yml
   |  Lean Core + Human Guide full quality gate
   |
   +-- same-repository main push CI success only
           |
           v
.github/workflows/deploy-guide.yml
   |  checkout exact verified main SHA
   |  npm ci
   |  source-check + production build
   |  Pages artifact upload
   v
GitHub Pages deployment
   |
   +--> public URL browser smoke
   v
https://shuntatsu.github.io/trade_rl/
```

`ci.yml` は品質判定のsingle permanent CI workflowであり続ける。`deploy-guide.yml` はdeployment workflowであり、テスト成功の代替oracleにはしない。

### Trigger contract

`deploy-guide.yml` は `workflow_run` で workflow name `CI` の `completed` を監視する。

実deploymentは次の条件をすべて満たす場合だけ行う。

1. `workflow_run.conclusion == success`
2. `workflow_run.event == push`
3. `workflow_run.head_branch == main`
4. `workflow_run.head_repository.full_name == github.repository`
5. checkout対象が `workflow_run.head_sha` とexact一致する

PR run、fork由来run、failure/cancelled CI、古いmain SHAからはdeployしない。

`workflow_run`は前段workflowより強い権限を持ち得るため、branch名だけでdeploy eligibilityを決めない。same-repositoryのmain pushであることをfail-closedに静的contract testする。

mainがCI完了後にさらに進んだ場合でも、そのrunは検証済みのexact SHAだけをdeployする。後続mainは自身のCI成功後に別deploymentで更新される。

### Pages build

Build jobは次だけを行う。

1. exact verified SHA checkout
2. Node 24 setup
3. GitHub Pages configuration/readiness確認
4. `npm ci` under `guide/`
5. `python3 guide/tools/content_contract.py --check`
6. `npm run build`
7. `guide/dist/` をPages artifactとしてupload

Human Guide full test suiteは前段CIですでに実行済みであるため、deployment workflowでは再度full Playwright suiteを実行しない。ただしsource freshnessとproduction buildはdeploy artifactそのものに対するfail-closed verificationとして再実行する。

### Pages deploy

Deploy jobはbuild jobに依存し、GitHub推奨の `github-pages` environmentを使用する。

権限はjob単位で最小化する。

- build: `contents: read` とPages設定readに必要なpermissionだけ
- deploy: `pages: write`, `id-token: write`、必要なread permissionのみ

GitHub公式Pages actionsは実装時点の最新安定releaseを再確認し、Repository既存方針に合わせてmutable major tagではなくexact commit SHAへpinする。

### Public URL smoke

Deploy成功後、deployment stepが返すpublic URLを対象に小さいChromium smokeを実行する。

確認対象は公開経路固有の失敗に限定する。

- root documentが取得できる
- overviewが表示される
- 1つ以上のtopic hash routeへ遷移できる
- Light/Dark toggleが機能する
- 「正本」linkがGitHub `main/docs/...` を指す
- 320px viewportで重大なhorizontal overflowがない

これは既存Human Guide full E2Eの代替ではなく、「Pagesから実際に配信されたartifactが利用可能」というpost-deploy oracleである。

### One-time repository setting

GitHub Pages custom workflowを使うには、Repository側で一度だけPages sourceをGitHub Actionsへ設定する必要がある。

```text
Settings
  -> Pages
  -> Build and deployment
  -> Source: GitHub Actions
```

この設定はGit tree外のGitHub stateであり、workflow fileだけでは「Pages有効化済み」とみなさない。

現在利用可能なGitHub connectorにはPages設定write surfaceがないため、実装完了時にこの一回限りの設定だけはユーザー操作が必要になる可能性がある。

`actions/configure-pages` の自動enablementへPAT/追加secretを導入する方式は採用しない。理由は、公開機能のためだけに長期secretや追加administration authorityをworkflowへ持たせる必要がないためである。

## Source URL / routing invariants

- Public root URLは原則 `https://shuntatsu.github.io/trade_rl/` とする。
- Asset URLはVite `base: "./"` によりproject pathから相対解決する。
- Topic navigationはhash routeを維持する。
- direct navigationはroot + hashで完結し、server-side fallbackを要求しない。
- source linksはGitHub `main` の正本docを指し、Pages artifact内へMarkdown authorityを複製しない。

## Failure modes and handling

### Pages disabled

Pages configuration/readinessまたはdeployがfailする。これをコードfailureと混同せず、「Repository Pages source未設定」として扱う。

公開完了を宣言する前にRepository metadata `has_pages=true` と実URL到達性をread-backする。

### CI failed or cancelled

Deployment workflowはskipし、古い公開siteを維持する。failureしたmain SHAをpublishしない。

### PR/fork workflow_run obtains deployment workflow context

Eligibility guardがfalseとなり、build/deployを実行しない。`head_branch == main`だけをsecurity boundaryにしない。

### Source fingerprint stale

Build jobのsource-checkでfailし、deploy artifactを作らない。Guideのcontentを正本と照合してからfingerprintをrefreshする。

### Production build failure

artifact upload/deployへ進まない。

### Deployment API / Pages outage

main codeやCI successを巻き戻さない。deploy runをfailureとして残し、再実行可能なdeployment問題として分離する。

### Public smoke failure

「deploy API success」と「人間が使える公開site」を分けて扱う。smoke failure時は公開完了とせず、asset path、cache、routing、Pages responseを調査する。

### New main commit arrives during deployment

deploymentはverified `workflow_run.head_sha`へbindするため、途中で対象SHAを変えない。新mainは次のsuccessful CIからdeployされる。

## Security and permissions

- deploy workflowはfork PRや通常PRから直接Pages write権限を使わない。
- deployment eligibilityはsuccessful same-repository main push CIに限定する。
- deploy sourceはsuccessful CIのexact SHAに限定する。
- long-lived PAT / deploy key / cloud secretを導入しない。
- `GITHUB_TOKEN` permissionsはjob単位で最小化する。
- deployment environmentは `github-pages` を使う。
- third-party deployment serviceを追加しない。

## Acceptance Criteria

1. `deploy-guide.yml` がsuccessful same-repository main push CIだけからdeploymentを開始する。
2. `workflow_run.event == push`、`head_branch == main`、same repositoryを全て検証する。
3. deployment checkout SHAがCI `workflow_run.head_sha`とexact一致する。
4. PR / fork / failed / cancelled CIはdeployしない。
5. build artifactは`guide/dist/`だけで、Repository全体をpublishしない。
6. source freshness checkがdeploy artifact build前に必須である。
7. Pages workflow actionsはexact commit SHAへpinされる。
8. Pages job権限は必要最小限である。
9. current `ci.yml` のLean Core / Human Guide quality gateを弱めない。
10. `guide/`のauthority境界・research claim境界を変更しない。
11. Pages sourceをGitHub Actionsへ設定後、Repository stateでPages有効をread-backできる。
12. `https://shuntatsu.github.io/trade_rl/` がHTTP成功で到達できる。
13. 公開siteでoverview、theme toggle、最低1つのtopic hash route、正本linkが機能する。
14. 320px public viewportで重大なhorizontal overflowがない。
15. post-deploy oracleはdeployed commit SHA / deployment run / public URLを記録する。
16. 完了後、このActive spec/planはcurrent treeから削除し、恒久運用契約だけを`guide/README.md` / docs routingへ残す。

## Test Oracle

Repository implementation:

- architecture contract testでdeployment trigger、same-repository/main-push guard、permissions、exact-SHA checkout、artifact path、action pinningを静的検証する。
- security falsificationとしてPR/fork/failed CI相当payloadではdeployment conditionが成立しないことを検証する。
- existing Lean Core CIをfull実行する。
- existing Human Guide CIをfull実行する。
- deploy workflowのbuild pathをPR上で副作用なしに検証できる構成にする。Pages write自体はmain merge後のみ行う。

External / post-merge:

- main merge SHAのpush CIがsuccess。
- deploy workflowがそのCIのexact SHAをdeploy。
- Repository `has_pages` がtrue。
- public URLがHTTP成功。
- browser smokeでroot + hash navigation + theme + source link + 320px overflowを確認。

## Implementation boundaries

想定変更先は次に限定する。

```text
.github/workflows/deploy-guide.yml
tests/architecture/...pages deployment contract test...
guide/e2e/...public Pages smoke...
guide/README.md
README.md            # public Guide URL導線。公開確認後のみ
docs/AGENTS.md       # deployment contractがAgent更新対象なら最小追記
```

`trade_rl/**`、研究計算、Dataset/Study/Run artifact、strategy/simulation/evaluation semanticsは変更しない。

## Completion rule

「workflowがGreen」だけでは完了としない。

次をすべて満たした時点で完了とする。

- current mainを包含したexact PR HEADでCI Green
- merge後main CI Green
- Pages source設定のread-back
- Pages deployment success
- public URL到達性とbrowser smoke
- final diff / permanent workflow roster / temporary branch cleanup
- durable docs反映とActive spec/plan削除

Pages設定が未実施の場合、Repository実装が完成していても「公開完了」とは報告しない。
