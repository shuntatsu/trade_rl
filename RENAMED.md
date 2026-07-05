# Trade RL — 使い方ガイド（ポートフォリオ配分RL v3）

7銘柄のポートフォリオ配分を1時間ごとに決定するRLエージェントの使い方。
設計の全根拠は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## RLの強さは引き出せているか？ → 証拠付きで「Yes」（合成データ上）

RLをやる意味の判定基準は「**チューニング済みルールに勝てるか（ゲート2）**」。
勝てないならルールで十分で、RLは不要。この基準で3つの異なる市場すべてで勝っている:

| 市場 | RLフルスタック | 最良ルール（クロスモメンタム則） | 判定 |
|---|---|---|---|
| 強モメンタム（cross） | +2129% | +1436% | RL勝ち |
| **平均回帰（meanrev）** | **+175%** | **−48%（損失）** | **RL圧勝** |
| 弱シグナル（IC 0.11） | +81.5% | +56% | RL勝ち |

**核心は平均回帰ケース**: 固定ルールは市場の型が変わると損失を出すが、RLは適応して黒字。
これが「RLがルールに対して持つ本質的優位」の実証であり、RLの強さが引き出せている証拠。
加えて、強モメンタムではオラクル則と同等の捕捉率を**より低いDD・低回転**で達成している
（コスト・リスクを織り込んだ滑らかな執行をRLが学習した）。

**正直な限界**: 上記はすべて合成データ。実データ（IC 0.02〜0.05想定）での証明は
これから（下記P1）。ボトルネックはもうRL機構ではなく、データにアルファがあるかどうか。

---

## セットアップ

Python **3.12推奨**（3.13/3.14はtorch wheel要確認）。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  /  Linux・Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## クイックスタート（データ不要・どこでも動く）

まず健全性試験(P0)で「学習システムが正しく機能しているか」を確認する:

```bash
python scripts/train_portfolio.py --phase p0 --timesteps 300000
```

- **合格条件**: ①アルファ有データでB&H・フラット両方に勝つ ②ノイズデータでほぼ取引しない
- 出力: `output/portfolio/p0_report.json`、エクイティカーブ図
- 所要: CPUで約20〜30分

## 本番ワークフロー（実データ・ローカルPCで実行）

Binanceは一部地域からブロックされるため、データ取得はローカルPCで。

```bash
# 1. 先物データ取得（kline + funding rate + オーダーフロー集計）
python scripts/fetch_futures.py --symbols BTCUSDT ETHUSDT XRPUSDT BNBUSDT SUIUSDT PAXGUSDT --days 180 --to csv

# 2. 学習（品質ゲート → リーク自己検査 → ICゲート → 学習を自動実行）
python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000 --ensemble 3

# 3. ウォークフォワード検証（多シード・コスト2倍感度）
python scripts/train_portfolio.py --phase wf --source csv --data ./data --timesteps 500000
```

**重要**: 手順2は**ゲート1（OOSランクIC≥0.02）に不合格なら自動で停止**する。
実データに予測力がなければRL学習に進まない（これが撤退基準）。強制続行は `--skip-gate`。

## 主なオプション（既定は検証済みの推奨値）

| フラグ | 既定 | 説明 |
|---|---|---|
| `--phase` | p0 | `p0`健全性 / `train`学習 / `wf`ウォークフォワード |
| `--source` | synthetic | `synthetic` / `csv`（fetch出力） / `postgres`（Platform DB） |
| `--ensemble` | 1 | シードアンサンブル数。**弱シグナルでは3推奨**（不確実性縮小が効く） |
| `--gamma` | 0.5 | 割引率。行動効果が即時のため低い値が正解（0.995は崩壊する） |
| `--postproc` | full | 後処理（平滑/集中上限/ボラ目標/DDデリスク）。`legacy`で無効化 |
| `--target-vol` | 0.5 | 年率ボラ目標。0以下で無効 |
| `--feature-mask` | off | IC安定性による特徴選別（実データのジャンク特徴対策・オプトイン） |
| `--timesteps` | 300000 | 学習ステップ。実データ本番は200万〜 |

学習は既定で **TFゲート抽出器 + Ridge教師BCウォームスタート + 検証ベースモデル選択** を使う
（すべてベンチマークで根拠づけられた既定値。詳細はARCHITECTURE.md §2.1）。

## 結果の読み方（ゲート体系）

学習は複数のゲートを順に通す。各ゲートの意味:

1. **品質ゲート**: 欠損率・スパイク・timestamp整合。不合格銘柄は自動除外
2. **リーク自己検査**: 評価コード自体の健全性（shuffleでIC消失／未来シフトでIC増大）
3. **ゲート1（IC）**: 特徴に予測力があるか。**ここが実データでの本当の勝負**
4. **ベースライン比較**: RL成績は常に4種（フラット/B&H/ボラ逆数/クロスモメンタム則）と並記。
   **クロスモメンタム則に勝てて初めてRLの意味がある**
5. **ウォークフォワード**: コスト2倍でも中央値がプラスか（頑健性）

## ダッシュボード（学習の可視化・バックテスト）

```bash
# ターミナル1（リポジトリルートから）
python scripts/run_server.py                 # http://localhost:8001

# ターミナル2
cd frontend && npm install && npm run dev    # http://localhost:5173
```

- ダッシュボードから `mode: portfolio` で学習起動・メトリクス監視・バックテスト
- **`GET /api/signal/latest`**: 学習済みモデルの推奨ウェイトを返す。
  Trade Platform のBots画面はこれをポーリングするだけで統合できる

```json
{
  "weights": {"BTCUSDT": 0.28, "ETHUSDT": -0.11, ...},  // 後処理・ガードレール適用後
  "raw_weights": {...},                                   // 方策の生出力
  "net_exposure": 0.35, "gross_exposure": 0.82,
  "data_age_hours": 0.5, "stale": false,
  "guardrail": {"action": "proceed", "triggered": []}
}
```

## Trade Platform との連携

接点は2つだけ（Django側のコード改修は不要）:
1. 同一PostgreSQLの `rl_funding_rate` / `rl_orderflow_1m` テーブル
   （`fetch_futures.py --to postgres --dsn <DSN>` が作成・更新）
2. `GET /api/signal/latest`（Platform の Bots 画面から fetch）

## テスト

```bash
python -m pytest tests/ -v
```

## 次の一手（ロードマップ抜粋・詳細は ARCHITECTURE.md §4）

1. **実データP1**: 上記ワークフロー手順1〜2。ゲート1が本当の勝負
2. 基準特徴の拡充（建玉残高・清算・L/S比率をfetch_futuresに追加）
3. 弱シグナル領域: `--ensemble 3` で不確実性スケーリングを活用
4. 紙上運用2週間 → バックテストとの乖離分析 → 資金投入判断
