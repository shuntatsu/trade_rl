from __future__ import annotations

from tests.rl.test_universal_trade_u2_environment import U2EnvironmentFixture
from tests.rl.test_universal_trade_u2_environment_factory import _construct_factory

pytest_plugins = ("tests.rl.test_universal_trade_u2_environment",)


def test_u2_high_level_factory_exposes_exact_source_closure_digest(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    factory, closure, _locators, _source_loads, _child_builds = _construct_factory(
        u2_environment_fixture
    )

    assert factory.source_closure_digest == closure.digest
