# Machine Run Report

この文書は、research / training artifactから**LLM分析なし**で貼り付け可能なreportを生成する維持対象契約の正本です。Reporterは**read-only**で、既存artifactをsource of truthとして読み取り、閾値の再評価、candidateの再ranking、収益性の解釈、次experimentのrecommendationを行いません。

## 1. Entry point

Chatへ貼るMarkdownは次で生成します。

```bash
uv run python scripts/build_run_report.py \
  --root var/causal-alpha-v3-post-v2-full \
  --profile chat \
  --output -
```

同じartifactからversioned JSONを保存する場合は次を使います。

```bash
uv run python scripts/build_run_report.py \
  --root var/causal-alpha-v3-post-v2-full \
  --profile json \
  --output report/run-report.json
```

`--profile chat`はdeterministic Markdown、`--profile json`は`run_report_v1` JSONを生成します。同じ入力artifact treeに対して、artifact自体が変化しない限り出力内容とorderingは決定論的です。

`--output -`はstdoutへ出力します。Fileへ書く場合、**output must stay outside** the source artifact rootです。`--root`配下へのreport書込みは、source evidenceを観測処理で汚染するためexit code 2で拒否します。

## 2. Stage contract

Reportのstage順序は固定です。

```text
signal
selection
teacher_admission
teacher_package
behavior_cloning
critic_warm_start
ppo
zero_shot
sealed_evaluation
```

Stage statusは次の6種類だけです。

```text
PASS / REJECT / IN_PROGRESS / NOT_RUN / MISSING / INVALID
```

意味は次のとおりです。

- `PASS`: persisted terminal evidenceがそのstageの成功を示す。
- `REJECT`: persisted terminal evidenceがそのstageのrejectを示す。
- `IN_PROGRESS`: authoritative progress artifactが存在するがterminal evidenceはまだ存在しない。
- `NOT_RUN`: 上流のpersisted rejectionによって、このstageへ進まなかったことが証明できる。
- `MISSING`: stage evidenceが存在しない。単なるabsenceから未実行理由を推測しない。
- `INVALID`: known artifactのschema、digest、identity、state transitionが現在のcontractと整合しない。

`MISSING`を`NOT_RUN`へ読み替えません。逆に、上流rejectが永続化されている場合はdownstreamを`MISSING`ではなく`NOT_RUN`として表現します。

## 3. V3 authoritative artifacts

Causal Alpha V3についてReporterは既存のartifact graphを直接読みます。

```text
execution-identity.json
run-manifest.json
authored-config.json
signal/*.json
signal/rejection.json
selection/evidence.json
selection/rejection.json
selection/progress.json
admission/evidence.json
admission/rejection.json
teacher/package.json
```

Run / execution identityは維持対象の`from_payload` contractで検証し、cross-run digest driftやschema mismatchはfail closedします。Signal、Selection、Teacher AdmissionのthresholdをReporter内で再計算しません。Persist済みの`passed`、`rejection_reasons`、gross/net return、confidence interval、turnover、trade countなどをreportへコピーします。

Selectionがterminalになる前に`selection/progress.json`だけが存在する場合は`IN_PROGRESS`です。Candidate/symbol progress tableもpersisted progressの値をそのまま表示します。

## 4. Downstream training evidence

BC以降のstageは、必要に応じて次のstrict generic evidenceを読みます。

```text
reporting/stages/behavior_cloning.json
reporting/stages/critic_warm_start.json
reporting/stages/ppo.json
reporting/stages/zero_shot.json
reporting/stages/sealed_evaluation.json
```

Schemaは`run_report_stage_evidence_v1`です。

```json
{
  "artifact_digest": "<sha256>",
  "artifact_digests": {
    "checkpoint": "<sha256>"
  },
  "metrics": {
    "loss": 0.125,
    "steps": 100
  },
  "reasons": [],
  "schema_version": "run_report_stage_evidence_v1",
  "stage": "behavior_cloning",
  "status": "PASS"
}
```

このadapterはReporterがtraining結果を推定するためのものではありません。Producer側がpersistした事実を共通report schemaへ載せるための読み取り契約です。Malformed schema、digest mismatch、unknown status、上流rejectionと矛盾するdownstream PASSは`INVALID`です。

## 5. 出力の責務境界

Machine Run Reportが保証するのは、persisted artifactを決定論的に検証・整形したことです。以下はreport生成の責務外です。

- alphaが統計的・経済的に十分かという追加解釈
- profitability / Production GOの判断
- next experimentの提案
- LLMによるdiagnosis / summary
- Signal / Selection / Admission thresholdの再計算
- model training、checkpoint更新、artifact migration

したがって、ここへ結果を共有するときは`--profile chat --output -`の出力をそのまま貼れます。追加分析が必要な場合も、Machine Run Report自体はfact-only evidenceとして保持します。
