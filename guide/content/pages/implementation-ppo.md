## このページで答える問い

PPOは何を観測し、actionをどう売買意図へ変え、どの時点でhard risk・約定・reward計算を通るのか。

学習時だけ使える情報をpolicyへ混ぜず、実行時も同じObservation schemaを使うことが中心契約です。歴史的なv2は5区分、保有期間を扱うv3は約定後の保有年齢を追加します。

## PPO Observation v2/v3

Observation v2のpolicy inputは次の5区分を固定順序で持ちます。

| 順番 | 区分 | 意味 |
| ---: | --- | --- |
| 1 | `local_values` | 選択した現在時点のローカル特徴量 |
| 2 | `local_available` / finite mask | 値をpolicyへ渡してよいか |
| 3 | `local_staleness` | 選択featureと同じ順序の鮮度遅延 |
| 4 | `current_intent` | 現在のSHORT / FLAT / LONG状態 |
| 5 | `current_weight` | 現在のportfolio weight |
| 6 | `position_age_bars`（v3のみ） | 実際の約定数量から数えた保有年齢 |

```text
local_values
    + local_available / finite
    + local_staleness
    + current_intent
    + current_weight
    + position_age_bars (v3)
              ↓
       _encode_observation
              ↓
      PPO observation vector
```

Observation v3はv2の5区分をそのまま保ち、末尾に
`min(position_age_bars, 504) / 504` を加えます。年齢は直近のactionではなく実際のsigned fillから更新します。最初の非ゼロfillは1、保有が続く区間は1ずつ進み、実際にflatになれば0、保有方向が反転したら1へ戻ります。

初回canonical M2のpolicy inputにはsymbol IDやdataset-global aggregateを入れません。さらにfit対象を一部銘柄へ絞る場合、選択したlocal featureもfit-scope外symbolへ依存していないかDataset build identityから確認します。cross-sectional rank/dispersionはstrict subsetではrejectし、reference-relative featureはreference symbolがfit scope内にある場合だけ許します。これはrowを除くだけでholdout symbolの情報が残る経路を閉じるためです。

## PPOの最小保有期間

`minimum_hold_bars`は、実際の保有年齢が指定bar数に達するまで、policyが自発的にFLATまたは反対方向へ変える意図を抑えます。年齢は予測actionではなく実約定数量から進めるため、partial fillやflattenも状態へ反映します。hard riskは期間中も常に適用され、必要なら縮小・flattenできます。したがって最小保有期間は取引を必ずその長さ保持する保証ではありません。

正の保有期間には、年齢を観測できるObservation v3が必要です。比較ではH=0のbaselineも候補と同じv3を使い、seedごとにPPOを新しく学習します。この要因が測るのは同じ売買を単純に長く持つ効果だけではなく、最低滞在ルールを与えたPPOシステム全体の変化です。

## 学習時の銘柄スケジュールは2方式

`fit_ppo_strategy`の既定は従来どおり`sequential`です。1つの`PPOTradingEnv`がfit対象銘柄をfull-window episodeごとにround-robinするため、既存のStudyやcandidateの意味は変わりません。

fit対象が複数銘柄なら、requested `total_timesteps`をSB3の2048-step rollout単位へ切り上げた実効budgetが、全fit銘柄へ最低1 full agent episodeずつ届くことも事前に確認します。足りなければ学習を始めずfail closedにします。terminal settlementの内部barはagentがactionを選ばないため、このcoverage step数には含めません。

`interleaved`は明示的に選ぶ別layoutです。fit対象の各銘柄について1銘柄だけに固定した同じ`PPOTradingEnv`を1個ずつ作り、`DummyVecEnv`で同じPPO policyへ渡します。 各envの**active symbol**は1銘柄ですが、featureが参照してよい**information scope**は全fit銘柄rosterです。envを直接作る場合はinformation scopeを省略するとactive symbolだけをscopeとして検証するため、holdout依存のcross-asset featureを迂回できません。`rollout_steps_per_env`はcallerが明示し、全envを合わせたrollout sample数がminibatch size 64で割り切れなければfail closedにします。`total_timesteps`、Studyで固定したObservation schema、reward、hard risk、約定・会計、network、entropy係数は変えません。

```text
sequential（既定）
  1つのenv: BTC full window → ETH full window → ...

interleaved（opt-in）
  BTC固定env ─┐
  ETH固定env ─┼─ DummyVecEnv → 同じPPO policy update
  ...         ─┘
```

これは学習データの並べ方だけを変えるopt-in capabilityです。結果盲検のCPU synthetic timing（2銘柄、各513 bars、8 features、seed 11、2,048 requested/realized transitions、warm-up後3回交互測定）では、interleaved/512はsequentialより中央値で13.9%短かった一方、policy hashは異なりました。この速度測定は合成データ上に限られ、実データの学習時間・経済的優位性・利益を示しません。interleavedのseed安定性やproduction適性も未評価です。candidate evidenceはrequested/realized transition数を保存し、layout比較では両条件のrealized transition数を揃えます。developmentで比較するときはlayoutと`rollout_steps_per_env`を結果を見る前に別実験として固定します。学習deviceはCPUへ固定し、実行マシンのGPU有無だけでpolicy学習経路が変わらないようにします。なお、`DummyVecEnv`はsub-envへ異なるreset seedを配るため、execution乱数まで同時に変えないよう`slippage_std > 0`の確率的slippageはinterleavedではfail closedです。

PPOへ渡す主要constructor/policy設定もコードで明示します。learning rate、rollout長、batch/epoch、discount/GAE、clip、advantage normalization、entropy/value係数、gradient clip、gSDE/target-KLに加え、MlpPolicyのTanh、orthogonal init、FlattenExtractor、shared feature extractor、Adam epsを固定します。これはSB3 2.3.2で既に有効だった値を明文化するもので、performanceを見た調整ではありません。ただしSB3/PyTorch内部の学習実装そのものを複製しているわけではないため、依存versionは引き続きimplementation identityの一部です。

## 学習stepの全体像

```text
現在時点の市場・portfolio状態
            │
            │ 1. 選択したObservation v2/v3を符号化
            ▼
      PPO observation
            │
            │ 2. policyが離散actionを選ぶ
            ▼
         PPO action
            │
            │ 3. SHORT / FLAT / LONGへ変換
            ▼
        proposal_weight
            │
            │ 4. hard risk
            ▼
        target_weight
            │
            │ 5. MarketExecutorで約定・会計
            ▼
 interval_net_return + BookState
            │
            │ 6. reward = log1p(interval_net_return)
            ▼
      次のObservation
```

actionが直接rewardになるわけではありません。必ずriskとexecution/accountingを通ります。

## 1. 観測を符号化する

利用不能またはnon-finiteなlocal valueはmaskし、stalenessを同じfeature順序で持ちます。現在のintentとweightも含め、固定順序の`float32` vectorへ連結します。

入力値の標準化は明示的に選べる追加機能です。学習対象の期間・銘柄だけで平均と標準偏差を計算し、学習中も評価中も同じ係数を使います。欠損値は標準化後も0とし、利用可否・鮮度・保有状態の意味は変えません。既定では従来の値をそのまま使います。

保存したPPOは、標準化の有無にかかわらずモデルbytesだけでは再利用しません。fit済みstrategy自身がtraining Datasetから得た選択feature名を保持し、inference bundleへObservation schema、選択featureのindex/name、model bytes、必要なら標準化係数を一つのdigestで固定します。保存時はcallerが渡したfeedの選択feature名をstrategy自身のbindingと照合し、読み込み時もmanifestからそのbindingを復元して現在のfeature順序と照合します。featureの意味がずれた、model bytesが置き換わった、必要な係数が一致しない、といった場合はpolicyを読み込む前にfail closedにします。過去のnormalized-only bundleは読み取り互換を維持しますが、単独のhistorical `model.zip`を自動的に安全なdeployment artifactへ昇格させません。利益が改善するかは、別の実データ比較で判定します。

保存は一時領域で完成させてからまとめて公開します。Windowsの短いファイルロックには有限回の再試行を行い、保存先が現れた場合や再試行が尽きた場合は失敗として扱います。途中までコピーしたモデルを成功として公開することはありません。

このcontractを固定することで、学習時にだけ便利な情報を後から追加してバックテストを有利にする余地を減らします。

## 2. policyがactionを選ぶ

PPO policyは選択されたObservation schemaから離散actionを返します。このactionはまだ注文ではありません。

## 3. actionを売買意図へ変換する

離散actionを`SHORT` / `FLAT` / `LONG`へ写像し、現在のportfolio状態からproposal weightを作ります。

同じintentを維持する場合、価格driftだけを理由に毎decisionで機械的に元weightへ戻すのが標準ではありません。標準はquantity-preserving holdです。

分割では実保有と未約定分を含む目標数量を同じ比率で換算します。また、評価用のマーク価格と取引価格が異なっても、保有継続だけで数量は増減しません。数量換算には口座の評価価格を使い、注文の価格基準は取引価格のまま維持します。

## 4. hard riskを通す

PPOもrule strategyと同じrisk上限を守ります。既定設定で提案が有限かつ実行上限内で、drawdownが`[0, 1]`なら、risk変換が厳密に何もしないため環境は投影呼び出しを省いて同じtargetを渡します。異常なdrawdown、上限を超える提案、明示されたrisk設定は`PreTradeRisk`の検証・投影へ進みます。

学習用のリスク設定は明示的に指定でき、episodeをresetしても同じ設定を使います。省略時は従来の設定を保ちます。学習と評価で保有上限や下落時の縮小条件が違うと、同じ売買意図でも約定やコストが変わります。両者を合わせる実験ではリスク設定だけを変更し、報酬・観測・学習データの並べ方は別の比較として扱います。

PPOでも `max_turnover` と `drawdown_deleveraging` はそのstepのtransient risk projectionです。`max_turnover` が静的hard capと同時に発火しても、そのstepのtargetを `desired_quantity` へ書き戻しません。これにより反転actionの元proposalを保持したまま次stepのturnover projectionを再計算できます。transient理由がない静的hard capだけはbounded proposalとして再束縛します。このproposal/risk state分離はcanonical replayと同じです。

## 5. 約定・会計を通す

制約済みtargetを`MarketExecutor`へ渡し、fill、cost、funding、borrow、BookState、区間net returnを共通経路で更新します。

stepの`info`では、risk後の`target_weight`と約定後の`realized_weight`を分け、risk理由、fill ratio、requested/filled turnover、cost・funding・borrow・dividend・cash-interestの金額、termination理由も返します。policy判断・risk制約・partial fill・carryを同じ値として扱わないための診断情報です。

Directional PPOでは学習とdevelopment評価が同じbase execution設定を共有し、Datasetにあるborrowも両方で課します。過去のPPO実験は当時の実装へ固定された証拠であり、この修正後の学習経済へ自動的に読み替えません。

## 6. net returnからrewardを作る

### Directional terminal settlement

通常stepのrewardは約定・コスト反映後の区間returnから計算します。Directional PPOでは評価時のterminal FLATと経済endpointを合わせるため、最後のagent decision後にsettlement専用区間を予約します。agentが選んだactionをFLATへ上書きするのではなく、環境が外生的な`FLAT` proposalを同じhard riskと`MarketExecutor`へ1 barずつ流します。latency・capacity・partial fill・turnover制約も同じ状態機械で処理し、完全flatにならない残余はそのまま残します。

terminal transitionのrewardは、最後のagent intervalのlog returnにsettlement各intervalの実現log wealth changeを加算します。settlement barはagent stepではなくepisode末尾のterminal economic costとして1つのterminal transitionへ畳み込むため、settlement内部へPPOのdiscount factorを別途掛けません。これによりforced settlementをagent actionとして記録せず、episodeを終える時点の実行可能な経済結果だけをterminal rewardへ含めます。標準化を使う場合も、統計fit範囲はagentが実際に観測するdecision rowsまでです。なおtrainingはper-symbol account、developmentはshared-cashであるため、cross-symbol accountingまで一致したという意味ではありません。

rewardは約定・コスト反映後の区間returnから計算します。

```text
reward = log1p(interval_net_return)
```

したがって、取引コストを無視したpolicy scoreを別経路で最大化しているわけではありません。

## 学習時と実行時で同じ観測契約を使う

`PPOIntentStrategy`の実行時は`StrategyObservation`を`_encode_observation`へ渡します。学習時の`PPOTradingEnv`は、Studyで選んだ同じObservation schemaの入力フィールドをデータセットから直接読み、`_encode_observation_fields`で符号化します。学習中は公開レコードの生成を省きますが、policyへ渡すベクトルの意味と順序は実行時と一致します。

```text
学習時: feature slices + availability + staleness + current intent/weight → _encode_observation_fields → PPO
実行時: StrategyObservation → _encode_observation → deterministic predict
```

学習時と実行時で観測schemaを変えないことが重要です。テストでは学習用fast pathの出力を`StrategyObservation`経由の符号化と比較しています。

## 不変条件

- policy inputはv2の5区分またはv3の6区分で、Study内では同じschemaを使う。
- unavailable/non-finite valueはそのままpolicyへ渡さない。
- fit-scope外symbol由来のdataset-global aggregateを初回policy inputへ入れない。
- PPO actionを直接P&L/rewardへ変換しない。
- hard riskと共通execution/accountingを必ず通す。
- 実行時も学習時と同じencoderを使う。

## まだ保証していないこと

Observation contractがcausalであることと、PPOが儲かることは別です。現行研究はPPO superiorityやproduction profitabilityを証明していません。

## 純利益を目的にする配分PPO候補

Issue #810の追加候補は、独立口座の現在残高と凍結済みの予測・費用情報から、数量保持、負方向、中央、正方向の4つの行動を選びます。直接配分の中央はFLAT要求、残差配分の中央は非RL配分器の目標です。行動を目標へ写した後、hard riskと共通の約定・会計処理を通します。

報酬は、その処理で実際に増減した純資産を固定初期資本で割った値です。初期資本1000の口座がショートの価格急変で純資産-501になった場合も、損失を0へ丸めません。有限の経済期間が終わると実際の保有と注文を残したまま評価を止め、rollout bufferの切れ目では口座をresetしません。recipeは金融時計、risk、約定条件、観測と行動の意味を束縛します。

固定した合成信号・費用のみの市場では、実SB3学習と保存前後の全約定履歴の一致を検査しています。この既定v1観測は未正規化の部分観測で、注文の細部や保有年齢をすべて含みません。保存bundleは推論用であり、optimizer・口座・RNGを含む厳密な学習再開ではありません。連続walk-forward、費用校正、正式レビューと実市場での純利益比較は未完了です。

## 口座・注文の観測契約を整える

配分方策に渡す口座状態の準備として、不変のスナップショット契約を追加しています。選んだ1銘柄の数量・評価価格・倍率、口座のcash・equity・drawdown、未約定注文の方向・期限・reduce-only・約定済み数量を保持します。数量の正確な分数と表示用floatは別々に残し、注文を出す前の処理や、約定可能になる前の部分約定を拒否します。未処理の発注状態に処理記録がある場合や、待機状態に処理記録がない場合も拒否します。約定済み数量が0なら、約定金額も0を要求します。

この契約では、呼び出し側が申告した事実の構造と時刻関係を検査します。実口座からの読み出しは別のobserverが担当し、選択銘柄の公開済み価格、口座の処理時刻、既存の注文検証、更新済みのmarginを確認します。既存の注文処理が完了とみなす残量や、実行器が受け付けない微小な約定済み数量も渡しません。下限を超える合法な少量約定は保持します。検査用コピーを使い、元の口座や注文・乱数を変更しません。

明示した配分観測v2では、下記の環境接続がこのobserverを使います。digestは内容の一致を確認するためのもので、口座履歴の真正性を証明しません。保有年齢や予約資金も、この契約から推測して追加しません。bootstrap時も、呼び出し側が共通accountingでmarginを更新してから渡します。

## 配分口座の数値列schemaを定義する

独立した`allocation_account_observation_v2`は、入力特徴名、注文枠の上限、固定初期資本、正規化に使うepisode長を必須にする不変の宣言です。列数は入力特徴Fに、予測・配分目標9列、口座13列、注文枠ごと24列を加えます。現在と過去最大のdrawdown、注文の符号付き数量、時刻が未設定かどうかも別の列名にしています。

payloadの射影・ソート・padding指定を、別のpure encoderが実装しています。schemaの宣言を変更するとdigestが変わります。下記の明示v2接続を追加し、既存の配分PPO v1の観測・保存recipeと、Directional PPO Observation v2/v3は維持します。このsoftware検証だけで学習や純利益の改善を確認した状態にはなりません。

## 口座スナップショットを数値列へ写す

encoderは、スナップショットと配分判断の口座・時刻・数量・資産・固定資本が一致するか確認してから、新しい数値配列を返します。例えば初期資本100、価格100の買い注文1単位に対し、約定済み1/3、残り2/3なら、対応する3列は1、1/3、2/3です。分数の経済比率を先に計算し、float32では絶対量を水増ししない方向に丸めます。表現不能な値や、非ゼロがゼロになる過小値は拒否します。

注文は正確な経済factで並べてから数値化します。この並べ方は実行時のID優先順位を再現しません。IDを数値入力へ入れず、時刻が未設定の場合とindex 0の場合はmaskで区別します。注文枠を超える入力は拒否し、余った枠は0です。既存v1のpending集計を、exactな注文残量と同じものとは扱いません。予約資金や保有年齢を推測せず、完全なMarkov状態や口座履歴の真正性も主張しません。

## 明示v2で環境・保存・読み込みをつなぐ

schemaを指定した環境だけがv2を使い、省略した環境は従来v1のままです。開始時に共通処理でmarginを更新し、生存中の口座をobserverからencoderへ渡します。同じ行動なら、会計・約定・報酬の処理は同じです。保存するv2 recipeとbundleは別のversionになり、特徴名・資本・期間・列順が合わないものをモデル読み込み前に拒否します。

実際の経済期間の終了や債務を残した終了では、口座の読み出しを試さず、全0の終了通知を返します。SB3はその後に新しい口座をresetしますが、終了フラグによってcritic値を報酬へ足しません。終了していないrollout境界では、現在の生存状態をcriticへ渡して続きを推定します。時間切れのtruncationは生成しません。

v2の入力記録は、actorへ渡されたtensor、collector境界のnew_obsと終了フラグ、別の全0終了通知を区別してhashと件数を残します。現在のSB3でcriticへ渡される値との一致は、別の監視テストで確認します。環境が生成しただけの値を「学習で消費済み」とは扱いません。短い合成テストで保存前後の約定履歴も確認します。明示した学習設定と凍結前処理は下記で接続します。真正性や利益の証明、連続口座のwalk-forwardは後続作業です。

## 学習設定を省略できない宣言にする

配分PPO用の別の不変契約では、rollout・batch・epoch、割引・GAE、学習率・clip、損失係数、ネットワークとAdamの19設定を必須にします。例えばAdam epsilonを1e-5と明示すると、その値をpayloadとdigestへ保持します。省略して別の既定値へ落ちることや、固定したCPU用の識別子・フラグの変更をreaderが拒否します。可変なtuple派生型や、JSON以外のオブジェクトへの変換も受け付けません。

宣言自体は学習器を作りません。下記の明示接続でも、v1/v2のrecipe・既定動作は変更しません。raw actionと約定の記録、数日後の利益を学ぶ検査は後続作業です。この宣言のテストを、学習成功や市場利益の証拠として扱いません。

## 明示した学習設定と更新の記録をつなぐ

設定を指定した配分観測v2、または下記の凍結前処理を指定したv3が明示学習契約を使います。rollout・割引・GAEを金融時計へ一致させ、実際のPPO・ネットワーク・Adamへ全設定を渡します。既定dtypeを勝手に変えず、CPU float32と導入済みの依存版を確認します。生の特徴を使う環境のrecipeはv2のままで、保存bundleと学習receiptをv3にします。凍結前処理を指定した場合はrecipe v3とbundle・学習receipt v4です。前処理と設定を省略またはNoneにした既存経路は従来どおりです。

Adamが戻った呼び出しを数え、最後の学習更新が終わってから記録を確定します。KLによる停止は更新前に起きるため、例えば1エポックに入ってもAdam呼び出しが0回の場合があります。例外時はフックを解除し、成功した記録として保存しません。呼び出しが戻ったことは、パラメータ改善や学習成功を意味しません。

v3の設定・件数・versionが合わない保存物はモデル読み込み前に拒否し、読み込み後も実ネットワークやoptimizer設定を検査します。短い合成テストで2回・4回の呼び出し、KL停止時の無更新、既存モードと保存前後の行動履歴を確認します。市場利益や完全な学習再開、連続walk-forwardの完了を示すものではありません。

## 学習前の特徴だけから変換を凍結する

配分用の前処理は、学習開始より前の指定範囲で、選んだ特徴がすべて利用可能な行だけを使います。値が[0,1000,2]でも、中央の1000がその行の判断時刻に未公開なら除外し、平均1・標準偏差1です。後の特徴値4は3へ変換します。銘柄ごとの有効行数が違っても、銘柄を等しく重み付けします。

範囲・特徴順・公開時刻・有効行・係数を凍結し、適用時に再学習や欠損補完をしません。特徴生成と元の正規化設定の一致を確認しますが、そのhashで元データの真正性や特徴生成の因果性を証明しません。将来データを学習時のDataset IDへ固定する設計ではありません。

特徴だけを変換する部品を、下記の観測・recipe宣言と明示した環境・保存物へ組み合わせます。学習窓の切替は後続作業です。既存の生の予測・配分・会計や通常のv1/v2/v3保存物の意味は維持します。前処理テストの成功を、市場利益や学習成功とは扱いません。

## 特徴だけを変える観測v3を定義する

別版の観測v3では、平均1・標準偏差1の凍結した係数で、生の特徴4を3へ一度だけ変えます。特徴の後ろに続く資本・保有・未約定注文の列は、v2と同じbytesです。元の判断・予測・配分候補も変えません。複数特徴の順序を固定し、既存の幅F+22+24Kへ特徴数を再加算しません。

recipe v3は係数・前処理digest・観測の意味を束縛し、readerは版や列順の混在を拒否します。判定前のroot/tag検査も追加し、型を偽装したmapping・文字列は従来版でも拒否します。通常のJSON入力と従来のrecipe bytesは変えません。下記の明示した環境・学習器だけが使い、保存bundle v4と学習receipt v4へ対応付けます。

生の特徴0は-1になるため、全0の終了通知をこのlive encoderへ渡してはいけません。現在の環境は真の終了時に変換を迂回します。宣言からの特徴・生成設定・時刻の検査と、実Datasetで学習prefixを確認する処理は別です。推論Dataset IDをfit時のIDへ固定せず、実際の特徴順・生成設定・適用時刻を上位で確認します。純粋なencoderの時刻検査はfit-as-ofであり、学習開始や運用の承認ではありません。利益や学習成功は未検証です。

## 凍結した特徴を実際の学習入力と保存物へつなぐ

前処理を指定した環境は、実際の特徴名・生成設定・元の正規化設定と、判断時刻がfit完了・宣言した初回判断時刻以降かを確認します。将来のDataset IDや列番号が変わっても、この意味が一致すれば使えます。学習器の構築・終了時には別に、実Datasetのfit範囲・共同で利用可能だった行・係数・初回判断を再計算して凍結宣言と照合します。resetやactor入力でfitしません。前処理を指定した学習には明示設定が必須です。

平均1・標準偏差1の有限例では、actorが消費する生の特徴4・6は3・5になります。生存中の次の特徴8はcriticへ7として渡り、終了後の新しい口座は3へresetします。別の終了通知は全0です。実際にactorへ渡ったtensorとcollector境界の配列を別々に記録し、特徴後方の経済列と生の予測・配分判断は維持します。同じ固定行動の約定・会計・報酬が一致することも合成例で確認します。

保存物はrecipe v3、bundle・学習receipt v4、入力receipt v3を組み合わせます。凍結宣言のdigestとfit元のID・時刻、実入力のdigestを結び、混在や古い参照をモデル読み込み前に拒否します。読み込み後は実ネットワークの幅、PPO・Adam設定、epoch件数も検査します。判定前のmanifest root/tag検査は従来版の型偽装にも適用しますが、有効な既存JSONのbytesは変えません。一貫して別の係数とpinへ置換した保存物の真正性まで証明する仕組みではありません。

ここで確認したのは有限なソフトウェア検査です。数日後の利益を学べたこと、複数の学習窓・銘柄を網羅したこと、連続口座や市場利益、研究実行の承認は示しません。
