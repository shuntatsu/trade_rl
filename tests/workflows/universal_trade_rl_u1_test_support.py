from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from tests.rl.universal_trade_test_support import (
    make_u1_feature_specs,
    make_u1_market,
    make_u1_wrapper,
)
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.rl.universal_normalization import (
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)


@dataclass(frozen=True, slots=True)
class U1WorkflowFixture:
    manifest: UniversalTradeRLUniverseManifest
    u0_identity: UniversalTradeRLRunIdentity
    normalizer_provenance: UniversalTradeRLFitProvenance
    environment: UniversalTradeEnvironment

    def build_contract(self) -> Any:
        module = importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u1_contract"
        )
        return module.build_universal_trade_rl_u1_contract(
            manifest=self.manifest,
            u0_identity=self.u0_identity,
            normalizer_provenance=self.normalizer_provenance,
            environment=self.environment,
        )


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def build_u1_workflow_fixture() -> U1WorkflowFixture:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=6200, feature_level=0.0)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=6200, feature_level=0.25)
    first_timestamp_ns = _timestamp_ns(btc.timestamps[0])
    last_timestamp_ns = _timestamp_ns(btc.timestamps[-1])

    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    sources = (
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT",
            dataset_digest=btc.dataset_id,
            first_timestamp_ns=first_timestamp_ns,
            last_timestamp_ns=last_timestamp_ns,
            row_count=btc.n_bars,
        ),
        UniversalTradeRLSymbolSource(
            symbol="ETHUSDT",
            dataset_digest=eth.dataset_id,
            first_timestamp_ns=_timestamp_ns(eth.timestamps[0]),
            last_timestamp_ns=_timestamp_ns(eth.timestamps[-1]),
            row_count=eth.n_bars,
        ),
        UniversalTradeRLSymbolSource(
            symbol="SOLUSDT",
            dataset_digest="d" * 64,
            first_timestamp_ns=first_timestamp_ns,
            last_timestamp_ns=last_timestamp_ns,
            row_count=btc.n_bars,
        ),
        UniversalTradeRLSymbolSource(
            symbol="XRPUSDT",
            dataset_digest="f" * 64,
            first_timestamp_ns=first_timestamp_ns,
            last_timestamp_ns=last_timestamp_ns,
            row_count=btc.n_bars,
        ),
    )
    manifest = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=sources,
    )
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    knowledge_cutoff_ns = _timestamp_ns(btc.timestamps[6000])
    provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=access,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        source_symbols=config.train_symbols,
        knowledge_cutoff=knowledge_cutoff_ns,
    )
    policy_contract = UniversalTradePolicyContract(
        feature_specs=make_u1_feature_specs()
    )
    normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": btc, "ETHUSDT": eth},
        contract=policy_contract,
        source_dataset_digests=provenance.source_dataset_digests,
        knowledge_cutoff_ns=provenance.knowledge_cutoff,
        universe_manifest_digest=manifest.digest,
        provenance_digest=provenance.digest,
    )
    u0_identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
        universe_manifest_digest=manifest.digest,
        model_config_digest=None,
        fit_provenance_digests=(),
    )
    environment = make_u1_wrapper(
        dataset=btc,
        contract=policy_contract,
        normalizer=normalizer,
    )
    return U1WorkflowFixture(
        manifest=manifest,
        u0_identity=u0_identity,
        normalizer_provenance=provenance,
        environment=environment,
    )
