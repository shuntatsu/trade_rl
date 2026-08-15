from __future__ import annotations

from trade_rl.workflows import universal_causal_alpha_v3_contracts as contracts
from trade_rl.workflows import universal_causal_alpha_v3_store as store


def test_hardened_v3_contracts_do_not_export_obsolete_v1_runtime_types() -> None:
    assert not hasattr(contracts, "CausalAlphaV3RunManifest")
    assert not hasattr(contracts, "CausalAlphaV3AdmissionRecord")
    assert not hasattr(contracts, "UniversalCausalAlphaV3TeacherPackage")


def test_hardened_v3_store_has_no_obsolete_v1_admission_persistence_path() -> None:
    assert not hasattr(store.CausalAlphaV3RecordStore, "write_admission_record")
    assert not hasattr(store.CausalAlphaV3RecordStore, "load_admission_records")
