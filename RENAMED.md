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

**ホールド（B&H）に負けないか**: 純粋な上昇相場（方向性ベータのみ）では、当初
市場中立的に振る舞い**B&Hに負けて損失を出していた（−9.87%）**。これを修正済み:
方向性を持つ時系列モメンタム教師＋持続トレンドゲート（ランダムウォークの偽トレンドは
fold符号一致で排除）を追加し、**−9.87%→+20.54%（黒字化、SharpeはB&Hと同等）**。
相対アルファ市場では逆にB&Hを圧倒（RL +2327% vs B&H −24%）。教師はデータのレジームに
応じて自動選択（相対アルファ→Ridge、方向性ベータ→時系列モメンタム、無信号→フラット）。

**資産クラス・銘柄数**: 銘柄共有エンコーダは任意銘柄数に対応（10銘柄で+1288%を確認）。
株式等でfunding/オーダーフローが無くても動く（欠損はゼロ埋め）。仮想通貨・株式混在の
グループも可。ただし株式の立会時間・ギャップは品質ゲートで検知される（session対応は今後）。

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

## 本番ワークフロー（実データ）

### 最短ルート: Hyperliquid（認証不要・この環境からも取得可）

Hyperliquid公開APIは上位足（15m/1h/4h/1d）と funding をネイティブ配信する。
`--source hyperliquid` はデータを自動取得・キャッシュするため、取得と学習を
1コマンドで繋げられる（`data/hyperliquid/` にCSVキャッシュ）。

```bash
# 任意: 事前取得（キャッシュを温める。省略しても学習時に自動取得）
python scripts/fetch_hyperliquid.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT --days 180

# P1 シグナル検証 → 合格なら学習（ICゲートは自動）
python scripts/train_portfolio.py --phase train --source hyperliquid \
    --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT --days 180 --timesteps 2000000

# ウォークフォワード検証（多シード・コスト2倍感度・オラクル対比）
python scripts/train_portfolio.py --phase wf --source hyperliquid \
    --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT --days 180
```

銘柄は `BTCUSDT` でも `BTC` でも可（末尾USDT/USDを自動でコイン名へ正規化）。

> **実測（180日・7銘柄の生特徴）**: OOSランクIC ≈ 0.006 で**ICゲート不合格**。
> 素の価格・テクニカル特徴だけでは実データに予測力がほぼ無く、システムは
> RL学習に進まず停止する（これが正しい撤退動作）。ボトルネックはRL機構では
> なく**データのアルファ量**。ここを超えるには建玉残高/清算/L・S比率/funding
> 予測などの追加アルファ（`--source` 側で供給）や独自シグナルが要る。

### Binance先物（funding + aggTradesオーダーフローも必要な場合）

```bash
python scripts/fetch_futures.py --symbols BTCUSDT ETHUSDT XRPUSDT BNBUSDT SUIUSDT PAXGUSDT --days 180 --to csv
python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000 --ensemble 3
python scripts/train_portfolio.py --phase wf --source csv --data ./data --timesteps 500000
```

**重要**: 手順2は**ゲート1（OOSランクIC≥0.02）に不合格なら自動で停止**する。
実データに予測力がなければRL学習に進まない（これが撤退基準）。強制続行は `--skip-gate`。

## 主なオプション（既定は検証済みの推奨値）

| フラグ | 既定 | 説明 |
|---|---|---|
| `--phase` | p0 | `p0`健全性 / `train`学習 / `wf`ウォークフォワード / `pbt`HP探索 / `regime`レジーム特化 |
| `--source` | synthetic | `synthetic` / `csv`（fetch出力） / `postgres`（Platform DB） |
| `--ensemble` | 1 | シードアンサンブル数。**弱シグナルでは3推奨**（不確実性縮小が効く） |
| `--gamma` | 0.5 | 割引率。行動効果が即時のため低い値が正解（0.995は崩壊する） |
| `--postproc` | full | 後処理（平滑/集中上限/ボラ目標/DDデリスク）。`legacy`で無効化 |
| `--target-vol` | 0.5 | 年率ボラ目標。0以下で無効 |
| `--feature-mask` | off | IC安定性による特徴選別（実データのジャンク特徴対策・オプトイン） |
| `--htf-gate` | off | 階層MTF: 上位足(4h)トレンドで方向を制約し1hはサイジング担当 |
| `--timesteps` | 300000 | 学習ステップ。実データ本番は200万〜 |
| `--pbt-pop` / `--pbt-gen` / `--pbt-steps` | 6 / 4 / 40000 | PBT探索の個体数・世代数・個体あたりステップ（`--phase pbt`） |
| `--regime-bars` | 120 | レジーム専門家のエピソード長（`--phase regime`、5日=120本） |

### 収益最適化フェーズ（項目2〜5）

```bash
# 執行コストは既定でsqrt-impact + TWAP分割モデル（項目2、常時ON）

# 項目4: PBTでハイパーパラメータ（gamma/ent_coef/λ_turnover/reward_scale/lr）を自動探索
python scripts/train_portfolio.py --phase pbt --source csv --data ./data --pbt-pop 6 --pbt-gen 4 --pbt-steps 40000

# 項目3: bull/bear/range 専門家を学習し推論時に現レジームへルーティング
python scripts/train_portfolio.py --phase regime --source csv --data ./data --timesteps 500000

# 項目5: 階層MTF（4hトレンドが方向を決め、1hがサイジング）
python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000 --htf-gate
```

- **項目2（執行モデル）**: 大口リバランスを平方根則で不利に評価し、TWAP分割の恩恵をモデル化。`mars_lite/trading/execution.py`。学習コストが実運用の執行コスト構造を反映（train/serve一致）。
- **項目3（レジーム特化）**: `btc_trend`グローバル特徴で相場を bull/bear/range に分類。各レジームで専門家を短窓学習し、`RegimeEnsemble`が観測から現レジームを読んでルーティング（`agent.predict`互換）。
- **項目4（PBT）**: 軽量Population-Based Trainingで手動チューニング済みでないHP空間を進化探索。`pbt_result.json`に最良HPを出力。
- **項目5（階層MTF）**: 上位足トレンドと逆行する方向を遮断し、整合する方向のサイジングは1h方策に委ねる。方向の一貫性を上げつつ短期の機動力を保つ。

学習は既定で **TFゲート抽出器 + Ridge教師BCウォームスタート + 検証ベースモデル選択** を使う
（すべてベンチマークで根拠づけられた既定値。詳細はARCHITECTURE.md §2.1）。

## 結果の読み方（ゲート体系）

学習は複数のゲートを順に通す。各ゲートの意味:

1. **品質ゲート**: 欠損率・スパイク・timestamp整合。不合格銘柄は自動除外
2. **リーク自己検査**: 評価コード自体の健全性（shuffleでIC消失／未来シフトでIC増大）
3. **ゲート1（IC）**: 特徴に予測力があるか。**ここが実データでの本当の勝負**
4. **ベースライン比較**: RL成績は常に4種（フラット/B&H/ボラ逆数/クロスモメンタム則）と並記。
   **クロスモメンタム則に勝てて初めてRLの意味がある**
5. **オラクル則（手数料込みDP上限）**: `oracle_dp` は「未来を完全に知った上で、
   手数料を払ってでもポジションを持つ/反転する価値があるか」を動的計画法で厳密に
   解いた**理論上限**（各銘柄を1/N等資本の独立サブアカウントとし、状態{-1,0,+1}上で
   Viterbi型の最適経路を探索。微小な山谷は手数料に負けて取らない＝閾値超えの山と谷
   だけを取る）。`capture rate = RL収益 / oracle収益` が捕捉率。同一コスト・同一
   レバレッジ制約下での分母なので、RLの伸びしろが一目でわかる。
6. **ウォークフォワード**: コスト2倍でも中央値がプラスか（頑健性）

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
