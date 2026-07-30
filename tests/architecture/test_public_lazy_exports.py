from __future__ import annotations

import importlib

import pytest

import trade_rl.rl as rl


@pytest.mark.parametrize("name", rl.__all__)
def test_public_lazy_export_resolves_to_a_real_attribute(name: str) -> None:
    module_name = rl._EXPORTS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name in {"gymnasium", "stable_baselines3", "sb3_contrib"}:
            pytest.skip(f"optional training dependency is unavailable: {error.name}")
        raise

    assert hasattr(module, name), f"{module_name} does not define public export {name}"
    assert getattr(rl, name) is getattr(module, name)
