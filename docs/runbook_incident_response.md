# インシデント対応運用手順書

## 1. 適用範囲

本手順書はShadow、Canary、Productionにおけるモデル、データ、注文、ポジション、デプロイ証跡の異常へ対応する。コード実装のみをもってProduction運用可能とは判定しない。

## 2. 重大度

| 重大度 | 例 | 初動 |
|---|---|---|
| SEV1 | 意図しないリスク増加注文、ポジション不整合、証拠改ざん、Productionデータ破損 | 新規リスク停止、全注文取消、ポジション縮小、ロールバック |
| SEV2 | Canary損失・スリッページ・ドリフト閾値超過、再同期失敗 | 昇格停止、原因調査、必要に応じてフラット化 |
| SEV3 | Shadow性能低下、非本番メトリクス異常 | 次回昇格前にトリアージ |

## 3. 初動

1. デプロイ昇格を停止する。
2. モデルversion、Git SHA、設定、注文、約定、ポジション、取引所時刻、証拠Artifactを保存する。
3. 新規リスク注文を禁止する。
4. Productionまたは重大なポジション不整合では緊急flattenを実行する。
5. 必要に応じて検証済みモデルへロールバックする。
6. 全操作後に取引所と内部台帳を再照合する。

## 4. 緊急flatten

`flatten` は実行Adapterなしでは成功しない。次の例で `<adapter_module>:<factory>` は実環境のAdapter factoryへ置換する。

```bash
python -m mars_lite.trading.guardrails \
  --action flatten \
  --executor <adapter_module>:<factory> \
  --idempotency-key SEV1-<incident-id>-<timestamp> \
  --reason "SEV1: emergency risk reduction" \
  --output-format json
```

成功条件はすべて満たす必要がある。

- 新規リスクが遮断された。
- 未約定注文が0件になった。
- 残存ポジションが銘柄別許容誤差以内になった。
- reduce-only注文ID、取消注文ID、最終照合結果が保存された。
- CLI終了コードが0で、出力JSONの`success`が`true`である。

Adapter未設定、冪等キー未設定、注文残存、ポジション残存、照合失敗のいずれかでは成功扱いにしない。

## 5. モデルロールバック

```bash
python -m mars_lite.server.model_registry list
python -m mars_lite.server.model_registry rollback --target-version <verified-version>
```

レジストリ変更後、サービングプロセスを安全に再読込または再起動し、API・ログ・メトリクスが同じversionとartifact hashを示すことを確認する。レジストリだけの変更を復旧完了とみなさない。

## 6. 復旧基準

- open ordersが0または承認済みのreduce-only注文のみ。
- 内部台帳と取引所の注文・ポジション差が許容誤差以内。
- アクティブモデルversion、Git SHA、artifact hashが一致。
- データ鮮度、ドリフト、リスク検査が正常。
- インシデント証拠が保存され、Production再開が別チケットで承認された。

## 7. 連絡先

Production前に実在するOn-call、リスク責任者、コンプライアンス担当、取引所障害連絡先を運用秘密管理基盤へ登録する。プレースホルダー連絡先のままProductionへ進んではならない。

## 8. GameDay必須項目

データ停止、WebSocket切断、ACK消失、部分約定と取消競合、プロセスクラッシュ、証拠改ざん、任意versionロールバック、緊急flattenをテストネットで実行し、検知・停止・フラット化・再同期・復旧時間を記録する。
