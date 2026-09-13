## 何をするシステムか

Trade RLは、その時点で利用可能だった市場データ（point-in-time data）を固定し、**同じhard risk・約定・会計条件の下で複数の売買戦略を比較する研究基盤**です。

狙いは、バックテスト上の数字だけを大きくすることではありません。入力データ、売買判断、実行コスト、会計、検証証拠、研究判断を同じ経路で追跡できる状態を作り、unused dataでも維持される利益だけを残すことです。

## 全体の流れ

```text
causalな市場証拠
        ↓
MarketDatasetを構築・固定
        ↓
戦略が SHORT / FLAT / LONG の意図を出す
        ↓
hard riskで実行可能な目標へ制約
        ↓
MarketExecutor + BookStateで約定・会計
        ↓
銘柄ごとの独立リプレイ
        ↓
metrics / comparison / robustness / immutable evidence
```

この流れでは、戦略が直接P&Lを作りません。戦略は売買意図を出し、riskとexecution/accountingを必ず通った後の結果だけを比較します。

## 現在どこまで分かっているか

| 区分 | 現在の状態 |
| --- | --- |
| 検証済み | `market_build_v3` と `portable_feature_numerics_v1` を固定した実データDataset、PPO Observation v2、共通execution/accounting、結果前の事前登録からartifact後の独立検証までの研究経路 |
| 観測したが不十分 | Portable Controlled Experiment 0001ではmean-reversionがbaseline比で5 / 5銘柄改善しturnoverも低下したが、candidate total returnが正だったのは1 / 5銘柄 |
| 正式判断 | Experiment 0001は事前登録したruleに従い **KEEP_BASELINE** |
| 未証明 | 継続的なprofitability、winner strategy、unused future dataでの勝利、Production/live order authorization |

**再現可能な研究基盤が成立したことと、儲かる戦略が見つかったことは別です。** 現時点では後者を主張しません。

## 次に読む順序

1. **データと特徴量** — 未来情報を混ぜず、同じ入力を再現する方法。
2. **1本のバーを追う** — 観測からrisk、約定、BookState更新までの実行順。
3. **PPOの実装を追う** — PPOが何を観測し、actionがどう実際の売買へ変わるか。
4. **約定・会計** — fee、spread、participation、borrowをどこで計上するか。
5. **実験と証拠** — 一因子だけを変え、結果を見る前に判断ruleを固定する方法。
6. **研究の現在地** — 何が確定し、何が未証明か。

実装class/functionを引きたい場合だけ、ナビゲーションの **参照 → コード地図** を使います。
