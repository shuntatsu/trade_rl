from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = SOURCE.read_text(encoding="utf-8")
lines = text.splitlines()
tree = ast.parse(text)
nodes = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
}


def definition(name: str) -> str:
    node = nodes.get(name)
    if node is None:
        raise SystemExit(f"causal alpha definition disappeared: {name}")
    starts = [node.lineno]
    starts.extend(item.lineno for item in node.decorator_list)
    start = min(starts) - 1
    if node.end_lineno is None:
        raise SystemExit(f"causal alpha definition has no end line: {name}")
    return "\n".join(lines[start : node.end_lineno])


def definitions(names: tuple[str, ...]) -> str:
    return "\n\n\n".join(definition(name) for name in names) + "\n"


contracts_names = (
    "_readonly",
    "CausalAlphaEpisodePartition",
    "CausalAlphaSymbolSamples",
    "CausalAlphaExpandingFit",
    "CausalAlphaEpisodeEvidence",
    "CausalAlphaBatchEvidence",
    "CausalAlphaCandidateConfig",
    "CausalAlphaCandidateEpisodeMetrics",
    "CausalAlphaCandidateEvidence",
    "CausalAlphaSelectionEvidence",
    "UniversalCausalAlphaTeacherPackage",
)
contracts = '''"""Immutable contracts and evidence for the Universal causal alpha teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch, OracleEpisodeContract

_CAUSAL_ALPHA_EPISODE_PARTITION_SCHEMA = "universal_causal_alpha_episode_partition_v1"
_CAUSAL_ALPHA_SYMBOL_SAMPLES_SCHEMA = "universal_causal_alpha_symbol_samples_v1"
_CAUSAL_ALPHA_EXPANDING_FIT_SCHEMA = "universal_causal_alpha_expanding_fit_v1"
_CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA = "universal_causal_alpha_batch_evidence_v1"


''' + definitions(contracts_names) + '''
__all__ = [
    "CausalAlphaBatchEvidence",
    "CausalAlphaCandidateConfig",
    "CausalAlphaCandidateEpisodeMetrics",
    "CausalAlphaCandidateEvidence",
    "CausalAlphaEpisodeEvidence",
    "CausalAlphaEpisodePartition",
    "CausalAlphaExpandingFit",
    "CausalAlphaSelectionEvidence",
    "CausalAlphaSymbolSamples",
    "UniversalCausalAlphaTeacherPackage",
]
'''
Path("trade_rl/workflows/universal_causal_alpha_contracts.py").write_text(
    contracts, encoding="utf-8"
)

fitting_names = (
    "_train_range",
    "build_chronological_episode_partition",
    "_sample_int_vector",
    "latest_complete_episode_split",
    "validate_universal_causal_alpha_partitions",
    "_prefix_forward_label",
    "build_causal_alpha_symbol_samples",
    "_validated_sample_scope",
    "fit_expanding_causal_alpha_models",
    "build_causal_alpha_episode_batch",
)
fitting = '''"""Chronological data preparation and expanding fits for causal alpha."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import (
    UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    universal_feature_schema_digest_from_names,
)
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaRidgeConfig,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
    forward_log_return_label,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch, OracleEpisodeContract
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaBatchEvidence,
    CausalAlphaEpisodeEvidence,
    CausalAlphaEpisodePartition,
    CausalAlphaExpandingFit,
    CausalAlphaSymbolSamples,
)


''' + definitions(fitting_names) + '''
__all__ = [
    "build_causal_alpha_episode_batch",
    "build_causal_alpha_symbol_samples",
    "build_chronological_episode_partition",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "validate_universal_causal_alpha_partitions",
]
'''
Path("trade_rl/workflows/universal_causal_alpha_fitting.py").write_text(
    fitting, encoding="utf-8"
)

selection_names = (
    "_candidate",
    "default_causal_alpha_candidate_grid",
    "_candidate_evidence",
    "rank_causal_alpha_candidates",
    "_causal_alpha_target_for_contract",
)
selection = '''"""Train-only candidate grid and ranking for the Universal causal alpha teacher."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEvidence,
    CausalAlphaSelectionEvidence,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    fit_expanding_causal_alpha_models,
)


''' + definitions(selection_names) + '''
__all__ = [
    "default_causal_alpha_candidate_grid",
    "rank_causal_alpha_candidates",
]
'''
Path("trade_rl/workflows/universal_causal_alpha_selection.py").write_text(
    selection, encoding="utf-8"
)

facade = '''"""Public orchestration facade for the Universal causal alpha teacher."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_oracle_bc import evaluate_episode_action_path
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaBatchEvidence,
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEvidence,
    CausalAlphaEpisodeEvidence,
    CausalAlphaEpisodePartition,
    CausalAlphaExpandingFit,
    CausalAlphaSelectionEvidence,
    CausalAlphaSymbolSamples,
    UniversalCausalAlphaTeacherPackage,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    _validated_sample_scope,
    build_causal_alpha_episode_batch,
    build_causal_alpha_symbol_samples,
    build_chronological_episode_partition,
    fit_expanding_causal_alpha_models,
    latest_complete_episode_split,
    validate_universal_causal_alpha_partitions,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    _causal_alpha_target_for_contract,
    default_causal_alpha_candidate_grid,
    rank_causal_alpha_candidates,
)


''' + definitions(("evaluate_causal_alpha_selection", "build_universal_causal_alpha_teacher_package")) + '''
__all__ = [
    "CausalAlphaBatchEvidence",
    "CausalAlphaCandidateConfig",
    "CausalAlphaCandidateEpisodeMetrics",
    "CausalAlphaCandidateEvidence",
    "CausalAlphaEpisodeEvidence",
    "CausalAlphaEpisodePartition",
    "CausalAlphaExpandingFit",
    "CausalAlphaSelectionEvidence",
    "CausalAlphaSymbolSamples",
    "UniversalCausalAlphaTeacherPackage",
    "build_causal_alpha_episode_batch",
    "build_causal_alpha_symbol_samples",
    "build_chronological_episode_partition",
    "build_universal_causal_alpha_teacher_package",
    "default_causal_alpha_candidate_grid",
    "evaluate_causal_alpha_selection",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "rank_causal_alpha_candidates",
    "validate_universal_causal_alpha_partitions",
]
'''
SOURCE.write_text(facade, encoding="utf-8")
