# モデル意思決定ログ

本ログは追記専用とし、既存エントリを上書き・削除しない。閾値、特徴量、モデル、データ期間、検証方法、デプロイ判断を変更するたびに1件追加する。

## 一覧

| Record ID | Decision time (UTC) | Stage | Model version | Decision | Decision owner | Evidence bundle/run |
|---|---|---|---|---|---|---|
| DEC-YYYYMMDD-NNN | YYYY-MM-DDTHH:MM:SSZ | research/shadow/canary/production/rollback | version | accepted/rejected/hold/rollback | owner | run ID / artifact URI |

## エントリテンプレート

### 識別・責任
- Record ID:
- Decision time (UTC):
- Author:
- Decision owner:
- Independent reviewer:
- Stage:
- Decision: `accepted` / `rejected` / `hold` / `rollback`
- Reason and alternatives considered:

### 候補モデルと再現性
- Model version:
- Previous model version:
- Git commit SHA:
- Model artifact SHA-256:
- Configuration SHA-256:
- Data manifest SHA-256:
- Dependency lock SHA-256:
- Training/validation/test periods:
- Data seeds / model seeds:
- Training command or workflow run:

### 検証証拠
- Exact candidate CI run and conclusion:
- Shadow run ID / report SHA-256:
- Drift report ID / SHA-256:
- Incident report ID / SHA-256:
- Canary run ID / parent Shadow run / report SHA-256:
- Replay-versus-live calibration:
- Bootstrap and multiple-testing treatment:
- Pre-trade risk verification:
- GameDay evidence and recovery timings:
- Known limitations and unresolved risks:

### 承認・運用
- Approval ticket:
- GitHub Environment approver:
- Deployment Gate decision JSON / SHA-256:
- Effective capital and risk limits:
- Rollback target and trigger conditions:
- Follow-up actions, owners, and due dates:
- Related incident/RCA links:
