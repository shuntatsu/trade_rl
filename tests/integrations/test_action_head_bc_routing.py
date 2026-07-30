from __future__ import annotations

from types import SimpleNamespace

from trade_rl.integrations.sb3_training import _uses_hierarchical_actor_head


def test_direct_head_does_not_enter_hierarchical_bc_when_method_exists() -> None:
    policy = SimpleNamespace(
        shared_actor_head="shared_target_v1",
        hierarchical_actor_outputs=lambda observations: observations,
    )

    assert not _uses_hierarchical_actor_head(policy)


def test_hierarchical_head_enters_hierarchical_bc() -> None:
    policy = SimpleNamespace(
        shared_actor_head="hierarchical_gate_target_v1",
        hierarchical_actor_outputs=lambda observations: observations,
    )

    assert _uses_hierarchical_actor_head(policy)
