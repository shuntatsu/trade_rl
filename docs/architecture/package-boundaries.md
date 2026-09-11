# Package boundaries

## 結論

Trade RLはtop-level responsibilityを `artifacts / data / integrations / risk / simulation / strategies / evaluation` に分ける。各package内の物理フォルダは研究・実行責務と一致させ、旧private pathを残すためだけのforwarding shimは置かない。

`trade_rl/_validation.py` はstandard-library-onlyの最下層validation authorityである。

## Current package tree

```text
trade_rl/
├── __init__.py
├── _version.py
├── _validation.py
├── artifacts/
│   ├── canonical.py
│   ├── hashing.py
│   ├── atomic_pointer.py
│   ├── atomic_write.py
│   ├── store.py
│   └── verified_file.py
├── data/
│   ├── market.py
│   ├── contracts.py
│   ├── identity.py
│   ├── source.py
│   ├── view.py
│   ├── artifacts/{codec.py,publication.py}
│   ├── build/{config.py,builder.py}
│   └── features/{core.py,cross_asset.py,economic.py,multitimeframe.py}
├── integrations/
│   └── binance/
│       ├── types.py
│       ├── transport.py
│       ├── cache.py
│       ├── vision.py
│       ├── metadata.py
│       └── dataset.py
├── risk/
│   ├── inputs.py
│   ├── portfolio.py
│   ├── pretrade.py
│   └── emergency.py
├── simulation/
│   ├── accounting.py
│   ├── execution.py
│   ├── bar_path.py
│   ├── liquidity.py
│   ├── orders/{model.py,admission.py,reconciliation.py}
│   ├── stateful/{runtime.py,execution.py,bar_lifecycle.py,order_transitions.py,symbol_fills.py}
│   ├── targets/{execution.py,exposure_controller.py}
│   └── diagnostics/{execution_stress.py,funding.py,runtime_performance.py,runtime_performance_io.py}
├── strategies/
│   ├── dataset_scope.py
│   ├── interface.py
│   ├── position_intent.py
│   ├── controls.py
│   ├── rules/{trend.py,mean_reversion.py}
│   ├── forecasts/{controller.py,supervised.py,ridge.py,lightgbm.py}
│   └── rl/ppo.py
└── evaluation/
    ├── replay.py
    ├── metrics.py
    ├── evidence.py
    ├── series.py
    ├── gates/{models.py,resolve.py}
    ├── comparison/{bootstrap.py,paired.py,seed_robustness.py,strategies.py}
    ├── robustness/
    │   ├── capacity.py
    │   ├── closed_trades.py
    │   ├── fold_metrics.py
    │   ├── perfect_information/{bound.py,solver.py}
    │   └── walk_forward/{capabilities.py,folds.py,sealed_test.py,stitching.py}
    ├── runs/{candidate.py,candidate_suite.py,config.py,execute.py,provenance.py,artifact.py}
    └── experiments/
        ├── errors.py
        ├── codec.py
        ├── inspection.py
        ├── store.py
        ├── evidence.py
        ├── analysis.py
        ├── delta.py
        ├── workflow.py
        ├── contracts/
        └── bootstrap/{__init__.py,config.py,binance.py,workflow.py,cli.py}
```

`evaluation/experiments/` はdevelopment-onlyのhigher-level Study lifecycleを所有し、`evaluation/runs/` のverified Run Coreを再利用する。`evaluation/experiments/bootstrap/` はそのStudyを実行する前のcanonical preparationだけを所有する。

## Ownership

### `artifacts`

汎用のcanonical encoding、digest、atomic publication primitive、verified fileを持つ。market data、strategy、evaluation等のupper layerを知らない。

### `data`

`MarketDataset`、point-in-time contract/source、identity、bounded view、artifact codec/publication、dataset build、causal feature computationを持つ。strategy/evaluation/simulationへ依存しない。

### `integrations`

外部venue/providerを内部data contractへ変換するadapter層。Binanceはtransport、cache、Vision archive、metadata、dataset assemblyを分離する。strategy/evaluationを知らない。`BinancePublicTransport`の既定値はnetwork-enabledの既存互換を維持し、bootstrapだけがsource freeze後に`allow_network=False`を明示してcache-only化する。

### `risk`

portfolio/pretrade/emergencyのhard safety・feasibilityを持つ。strategyのalpha判断やevaluationを所有しない。

### `simulation`

execution/accountingの経済正本と、order/stateful/target/diagnosticsを持つ。strategy/evaluationから独立することで、同じexecution semanticsを複数研究候補で共有できる。

### `strategies`

small strategy interfaceとlogical intent、controls、rule、forecast、teacher-free RLを持つ。evaluationを知らない。`dataset_scope.py` はdatasetに束縛されたfeature/symbol selection validationの単一ownerであり、forecastとRLのsibling familyが互いの内部実装へ依存せず共有する。model自身やcandidate config自身の不変条件validationは各ownerに残す。

### `evaluation`

lower layerを利用してReplay・metrics・gate・comparison・robustness・concrete runを構成する。

`evaluation/runs/` の責務は一回の計算とimmutable Run evidenceである。

- `candidate_suite.py`: 5 candidates + 3 controlsのfit/replay構成。
- `config.py`: Run JSONの単一parse/resolution authority。
- `execute.py`: resolved specから既存candidate suiteを一度実行するin-memory seam。
- `provenance.py`: implementation/runtime/research-context provenance生成。
- `artifact.py`: summary/raw returns/provenanceのpublication、verified load、semantic identity。
- `candidate.py`: 上記を順番に呼ぶ薄いfilesystem CLI/facade。

`trade_rl.evaluation.runs` はcandidate-run contract、execution、artifact inspection/publication、provenance constructionのTier-2 public facadeである。`config.py`、`candidate_suite.py`、`execute.py`、`artifact.py`、`provenance.py` は引き続き実装ownerであり、facadeはこれらをwrapperなしでre-exportするだけとする。production codeは `evaluation/runs/` の外からRun Coreを利用するときfacadeを経由し、package内部は循環を避けるためowner moduleを直接参照してよい。Tier-1 `trade_rl.evaluation` の公開面はこの規則によって拡大しない。candidate-runのpersisted schema互換契約はPython import pathとは独立して維持する。

`runs` はhigher-level experiment lifecycleを知らない。`evaluation/experiments/` はStudy/Experiment contract、append-only store、multi-seed EvidenceSet、analysis、controlled delta、lineage/budget/freeze workflowを所有する。 `codec.py` はpersisted JSONから既存contractへのfail-closed decodeとstable payload/identity変換を所有し、`inspection.py` はdisk graphからのread-only state reconstruction・tamper validation・`inspect_study`を所有する。`workflow.py` はmutation lock下のcommand orchestrationだけを所有し、各mutation前のdisk再構築と既存failure-injection seamを維持する。

`evaluation/experiments/bootstrap/` は次だけを所有する。

- `config.py`: strict `CanonicalM2BootstrapConfig` parse/normalization/preflightと単一seed-policy authority。
- `binance.py`: exact exchange-info / Vision source freeze、raw-source roster、cache-only transport composition。
- `workflow.py`: source → canonical dataset → immutable StudyPlanをwhole-root stagingで構築し、manifest検証後に一回だけpublishする。
- `cli.py`: `--config` / `--output` をparseしてworkflowを呼ぶだけのfilesystem adapter。
- `__init__.py`: intentionally narrow public facade。

`trade_rl.evaluation.experiments` から公開するbootstrap APIは `CanonicalM2BootstrapConfig`、`CanonicalM2BootstrapResult`、`bootstrap_canonical_m2_study`、`inspect_canonical_m2_bootstrap` の4つだけである。source-freeze private helperはpublic contractではない。

Bootstrapはpreparation-onlyであり、baseline、Controlled Experiment、winner freeze、sealed final-test authorizationを実行しない。`evaluation/runs -> evaluation/experiments` の逆依存を作らず、`integrations`から`evaluation`へ依存させず、`evaluation/experiments/bootstrap`からsealed final-test ownerへ依存させない。

## Dependency direction

`tests/architecture/test_lean_dependency_boundaries.py` が実行可能な正本であり、少なくとも次を禁止する。

```text
_validation -> standard library only
artifacts   -X-> data/risk/simulation/strategies/evaluation/integrations
data        -X-> strategies/evaluation/simulation
integrations -X-> strategies/evaluation
risk        -X-> strategies/evaluation
simulation  -X-> strategies/evaluation
strategies  -X-> evaluation
strategies/rl -X-> strategies/forecasts
evaluation/runs -X-> evaluation/experiments
evaluation/experiments/bootstrap -X-> sealed final-test authorization
trade_rl -X-> tools/agent_repo
```

`evaluation` はlower core packagesを利用してよい。ただしlower layerからbootstrapへ逆依存しない。strategy family間で共有するdataset-bound selectionはroot `strategies/dataset_scope.py` を経由し、RLからforecast内部へ依存させない。依存方向を逆転させる必要が出た場合、循環依存や責務漏れを先に疑う。

### Static import ownership gate

`tools/agent_repo/source_index.py` の `ImportCollector` は、production sourceを実行せず、physical module treeを使って絶対・相対import、親packageからの子module import、選択したsymbolのstatic import re-exportを解決する。module名の区切りまで比較し、似たprefixの別moduleやfacade内の無関係なexportを禁止依存にしない。module scopeの条件分岐は保守的に両方検査し、関数・classのlocal importを公開exportと混同しない。

star importはliteral `__all__`、または明示的なpublic import re-exportを追跡する。動的に組み立てた`__all__`は実行して推測せず検査を失敗させる。循環re-exportも有限に走査する。source-derived mapは一回のscan内だけに保持し、生成catalogをcurrent treeへ保存しない。

`ImportCollector.collect()` はstatic re-exportを追跡してsemantic ownerまで展開する一方、`collect_direct()` はsourceに直接綴られたmodule pathだけを解決し、facade symbolのre-export先までは追わない。semantic dependencyとdirect-import policyは異なるoracleとして使い分け、facade経由の正当な利用をowner moduleの直接依存と誤認しない。

このgateはstatic import ownershipの検査であり、runtime sandboxや任意のPython到達可能性の証明ではない。動的import、実行時のattribute再束縛、反射や関数実行で生じる依存は別のreview/contract testが必要である。

### Repository tooling boundary

`tools/agent_repo/` はAgentのpreflight/context/impact/semantic diff/verification routingを行う**repository-local development tooling**であり、`trade_rl` runtime packageの一部ではない。Git/source/current docsを読む側であり、domain owner/schema registry/public runtime APIにはならない。

- `trade_rl/**` から `tools/agent_repo` への依存は禁止する。
- `tools/agent_repo` はproduction wheelへ入れない。`setuptools.packages.find.include = ["trade_rl*"]` を維持する。
- source-derived index/reportはmemory/stdoutだけに保持し、generated catalog/reportをcurrent treeへcommitしない。
- toolingのstatic analysisはruntime reachabilityの完全証明ではなく、source reviewとfinal full CIを置き換えない。
- local `verify` は開発中の検査選択を支援するが、完了判定ではpermanent CIのfull gateへ収束する。

## Repository integration boundary

GitHub PR / CI / branch-protection・ruleset設定はRepository統合の安全性を管理するが、`trade_rl` runtime packageのauthorityではない。checked-in architecture testはPR/CI policy fileの契約を検証できるが、実際のbranch protection状態はGitHub側のread-backで別途確認する。

Integration invariant: tested PR head contains current `main`. merge直前のcurrent `main` commitがtested PR headのancestorであり、その同一PR HEADにpermanent CI successが存在することを統合証拠とする。`main` が進んだ後の古いPR-head Greenは再利用せず、current `main` を含む新HEADを再検証する。将来merge queueを採用する場合は、current target branchを含むmerge-group SHAのrequired checkを同等の証拠としてよい。

このGit tree内のproseやarchitecture testだけでbranch protectionが有効とは判断しない。ruleset/protectionの設定変更後はGitHub stateをread-backし、required check、PR requirement、force-push/deletion、maintainer/admin bypassを確認する。管理surfaceが利用できない場合は未設定/未検証として扱う。

## Public API policy

Intentionally maintainedなpackage-level importは、内部private file移動より優先して安定させる。

例:

```python
from trade_rl.data import MarketDataset
from trade_rl.strategies import RidgeForecastStrategy
from trade_rl.evaluation import UniversalStrategyComparison
from trade_rl.evaluation.experiments import bootstrap_canonical_m2_study
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.risk import PreTradeRisk
```

一方、historicalに存在したprivate module pathはpublic contractの証拠ではない。private path移動のためだけのshimは原則追加しない。

## Architecture change rule

Packageを追加・移動・削除するときは同じ変更で次を行う。

1. 責務と依存方向をこの文書へ反映する。
2. `tests/architecture/` でrequired/forbidden pathと依存を契約化する。
3. intentionally publicなpackage facadeをsnapshot/contract testで確認する。
4. 単なるmoveならnon-import AST、serialized bytes/digest、golden behavior等でsemantic driftを可能な範囲で反証する。
5. 旧private pathや一時migration helperをfinal treeに残さない。

## Distribution source closure

構造変更では、working treeだけでなくGit HEADのproduction `.py` roster、sdist、direct wheel、sdistから再buildしたwheelの相対pathとSHA-256が一致することを検証する。`tests/architecture/distribution.py` は未追跡・ignoreされたsource、worktree差分、sourceの欠落・混入・改変、重複member、不正path、symlink sourceを拒否し、archiveを展開・実行しない。

CIはbuilt wheelをcheckout外の新規venvへ非editable installし、isolated Pythonでpackage identity、public facade import、candidate/bootstrap CLI helpを確認する。source closureはPython sourceの配布契約であり、optional trainerの実学習、全platform動作、すべてのnon-code resourceを保証するものではない。license/provenanceの恒久保持は別の既存gateも維持する。