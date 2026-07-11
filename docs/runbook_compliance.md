# コンプライアンス・監査証跡運用手順書

## 1. 適用範囲

本書はモデル開発、デプロイ、取引リスク、インシデント対応に関する内部監査証跡を定義する。法的助言ではなく、適用法令・登録要否・保存年限は運営主体、顧客資産の有無、法域、取引所、商品区分に基づき法務・コンプライアンス担当が確定する。

## 2. 必須証跡

各候補モデルについて次を保存する。

- モデルversion、モデルartifact、SHA-256。
- Git commit、設定hash、データhash、期間、seed、依存関係lock。
- Shadow、Drift、Incident、Canaryの各JSONレポート。
- 証拠生成workflowのrun ID、head SHA、実行結果。
- Deployment Gateの判定JSON。
- Production承認チケットとGitHub Environment承認者。
- 事前リスク拒否、注文、約定、取消、ポジション照合の監査ログ。
- インシデント、flatten、ロールバック、再同期、GameDayの記録。
- `model_decision_log.md` の意思決定記録。

証拠Artifactの構造は`deployment_evidence.md`に従う。

## 3. 整合性と関連性

SHA-256の形式確認だけでは不十分である。Deployment Gateは実ファイルからdigestを再計算し、次の同一性を検証する。

- すべてのレポートが同一model version、Git SHA、model digestを指す。
- Canaryが検証済みShadow run IDを親として参照する。
- 証拠生成runのhead SHAがcandidate Git SHAと一致する。
- Productionで検証したmodel artifactそのものをデプロイする。

証拠の差し替えや古い成功レポートの再利用は禁止する。

## 4. 保管

監査証跡はアクセス制御、versioning、削除保護、暗号化、時刻同期を備えた不変または追記専用ストレージへ保存する。ローカル`output/`は開発用であり、Production監査の唯一の保管先にしてはならない。

保存年限は法務・コンプライアンス担当が文書化して承認する。承認前の内部安全既定として7年を候補にできるが、普遍的な法的義務として扱わない。

## 5. 権限分離

- モデル作成者だけでProductionを承認してはならない。
- Production GitHub Environmentには指定reviewerを設定する。
- 証拠生成、承認、デプロイ、監査ログ管理の権限を可能な限り分離する。
- 緊急overrideを行った場合は理由、実行者、期間、復旧条件を記録する。

## 6. Production前法務チェック

- 運営法人・居住地・取引主体。
- 自己資金か顧客資金か。
- 運用助言・ファンド・代理執行等への該当可能性。
- 利用取引所と対象商品、地域制限、API利用規約。
- 税務、会計、記録保存、個人情報、サイバーセキュリティ要件。

未確定項目がある場合、技術テスト合格だけでProduction GOにしてはならない。
