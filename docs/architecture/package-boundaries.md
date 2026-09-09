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
    └── experiments/{errors.py,store.py,evidence.py,analysis.py,delta.py,workflow.py,contracts/}
```

`evaluation/experiments/` はdevelopment-onlyのhigher-level Study lifecycleを所有し、`evaluation/runs/` のverified Run Coreを再利用する。

## Ownership

### `artifacts`

汎用のcanonical encoding、digest、atomic publication primitive、verified fileを持つ。market data、strategy、evaluation等のupper layerを知らない。

### `data`

`MarketDataset`、point-in-time contract/source、identity、bounded view、artifact codec/publication、dataset build、causal feature computationを持つ。strategy/evaluation/simulationへ依存しない。

### `integrations`

外部venue/providerを内部data contractへ変換するadapter層。Binanceはtransport、cache、Vision archive、metadata、dataset assemblyを分離する。strategy/evaluationを知らない。

### `risk`

portfolio/pretrade/emergencyのhard safety・feasibilityを持つ。strategyのalpha判断やevaluationを所有しない。

### `simulation`

execution/accountingの経済正本と、order/stateful/target/diagnosticsを持つ。strategy/evaluationから独立することで、同じexecution semanticsを複数研究候補で共有できる。

### `strategies`

small strategy interfaceとlogical intent、controls、rule、forecast、teacher-free RLを持つ。evaluationを知らない。

### `evaluation`

lower layerを利用してReplay・metrics・gate・comparison・robustness・concrete runを構成する。

`evaluation/runs/` の責務は一回の計算とimmutable Run evidenceである。

- `candidate_suite.py`: 5 candidates + 3 controlsのfit/replay構成。
- `config.py`: Run JSONの単一parse/resolution authority。
- `execute.py`: resolved specから既存candidate suiteを一度実行するin-memory seam。
- `provenance.py`: implementation/runtime/research-context provenance生成。
- `artifact.py`: summary/raw returns/provenanceのpublication、verified load、semantic identity。
- `candidate.py`: 上記を順番に呼ぶ薄いfilesystem CLI/facade。

`runs` はhigher-level experiment lifecycleを知らない。`evaluation/experiments/` はStudy/Experiment contract、append-only store、multi-seed EvidenceSet、analysis、controlled delta、lineage/budget/freeze workflowを所有する。`evaluation/runs -> evaluation/experiments` の逆依存は作らない。experiments層からsealed unused-future authorizationへも依存しない。

## Dependency direction

`tests/architecture/test_lean_dependency_boundaries.py` が実行可能な正本であり、少なくとも次を禁止する。

```text
_validation -> standard library only
artifacts   -X-> data/risk/simulation/strategies/evaluation/integrations
data        -X-> strategies/evaluation/simulation
integrations-X-> strategies/evaluation
risk        -X-> strategies/evaluation
simulation  -X-> strategies/evaluation
strategies  -X-> evaluation
```

`evaluation` はlower core packagesを利用してよい。依存方向を逆転させる必要が出た場合、循環依存や責務漏れを先に疑う。

## Public API policy

Intentionally maintainedなpackage-level importは、内部private file移動より優先して安定させる。

例:

```python
from trade_rl.data import MarketDataset
from trade_rl.strategies import RidgeForecastStrategy
from trade_rl.evaluation import UniversalStrategyComparison
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
