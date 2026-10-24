"""Role- and pair-scoped digest reuse during artificial plan declaration."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import FrameType

import pytest

from tools.tier_d_nonrl import tier_d_native, tier_d_runner
from tools.tier_d_nonrl.tier_d_fixture import write_artificial_dataset
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation._simple_stream_digest import (
    SimpleStreamDigestMemo,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


def _prepared_pair_inputs(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset = write_artificial_dataset(dataset_root)
    protocol = tier_d_native.synthetic_protocol(
        tier_d_runner.market_protocol(max_workers=1), dataset
    )
    protocol["scenarios"] = ["BASEL2"]
    protocol["budget"]["native_arms"] = 2
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    (prepared_root / "forecasts").mkdir()
    prepared, tables = tier_d_runner.prepare_inputs(
        prepared_root, dataset_root, dataset, protocol
    )
    config = protocol["configurations"][0]
    inputs = prepared[config["id"]]
    estimates = tier_d_runner.symbol_estimates(
        tables, config["horizon_hours"], "BASEL2", dataset.symbols[0]
    )
    return inputs, estimates, protocol, config


def _declare(inputs, estimates, protocol, config):
    return tier_d_native.declare_pair(
        inputs,
        estimates,
        protocol,
        horizon=config["horizon_hours"],
        symbol_index=0,
        account=protocol["accounts"][0],
        scenario="BASEL2",
        configuration=config["id"],
        provenance=build_candidate_run_provenance(),
    )


def _count_stream_serializations(operation: Callable[[], object]):
    digest = FrozenSimpleReturnStream.digest
    assert isinstance(digest, property) and digest.fget is not None
    digest_code = digest.fget.__code__
    calls = 0

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is digest_code:
            calls += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = operation()
    finally:
        sys.setprofile(previous)
    return result, calls


def test_pair_declarations_reuse_live_memos_without_changing_canonical_plans(
    tmp_path, monkeypatch
):
    inputs, estimates, protocol, config = _prepared_pair_inputs(tmp_path)
    memo_factory_calls = []
    initial_global_memos = []
    terminal_memos = []
    nested_global_memos = []

    def memo_factory():
        memo = SimpleStreamDigestMemo()
        memo_factory_calls.append(memo)
        return memo

    monkeypatch.setattr(
        tier_d_native, "SimpleStreamDigestMemo", memo_factory, raising=False
    )
    original_initial_global = tier_d_native.declare_global_allocation_execution
    original_terminal = tier_d_native.declare_terminal_allocation_execution

    def initial_global_spy(*args, **kwargs):
        initial_global_memos.append(kwargs.get("_stream_digest_memo"))
        return original_initial_global(*args, **kwargs)

    def terminal_spy(*args, **kwargs):
        terminal_memos.append(kwargs.get("_stream_digest_memo"))
        return original_terminal(*args, **kwargs)

    monkeypatch.setattr(
        tier_d_native, "declare_global_allocation_execution", initial_global_spy
    )
    monkeypatch.setattr(
        tier_d_native, "declare_terminal_allocation_execution", terminal_spy
    )

    import trade_rl.evaluation.allocation_terminal_execution as terminal_module

    original_nested_global = terminal_module.declare_global_allocation_execution

    def nested_global_spy(*args, **kwargs):
        nested_global_memos.append(kwargs.get("_stream_digest_memo"))
        return original_nested_global(*args, **kwargs)

    monkeypatch.setattr(
        terminal_module, "declare_global_allocation_execution", nested_global_spy
    )

    def declare_route(*, force_direct: bool):
        with monkeypatch.context() as selected_route:
            if force_direct:
                selected_route.setattr(
                    SimpleStreamDigestMemo,
                    "get",
                    lambda _self, stream: stream.digest,
                )
            return _count_stream_serializations(
                lambda: _declare(inputs, estimates, protocol, config)
            )

    direct, direct_calls = declare_route(force_direct=True)
    first_memos = tuple(memo_factory_calls)
    first_initial = tuple(initial_global_memos)
    first_terminal = tuple(terminal_memos)
    first_nested = tuple(nested_global_memos)

    memoized, memoized_calls = declare_route(force_direct=False)
    second_memos = tuple(memo_factory_calls[len(first_memos) :])
    second_initial = tuple(initial_global_memos[len(first_initial) :])
    second_terminal = tuple(terminal_memos[len(first_terminal) :])
    second_nested = tuple(nested_global_memos[len(first_nested) :])

    next_pair, next_pair_calls = declare_route(force_direct=False)
    third_memos = tuple(memo_factory_calls[len(first_memos) + len(second_memos) :])
    third_initial = tuple(
        initial_global_memos[len(first_initial) + len(second_initial) :]
    )
    third_terminal = tuple(terminal_memos[len(first_terminal) + len(second_terminal) :])
    third_nested = tuple(nested_global_memos[len(first_nested) + len(second_nested) :])

    assert canonical_json_bytes(direct.controls.payload) == canonical_json_bytes(
        memoized.controls.payload
    )
    assert memoized.controls.digest == direct.controls.digest
    assert canonical_json_bytes(next_pair.controls.payload) == canonical_json_bytes(
        memoized.controls.payload
    )
    assert next_pair.controls.digest == memoized.controls.digest
    assert direct_calls > memoized_calls
    assert memoized_calls == next_pair_calls

    for memos, initial, terminal, nested in (
        (first_memos, first_initial, first_terminal, first_nested),
        (second_memos, second_initial, second_terminal, second_nested),
        (third_memos, third_initial, third_terminal, third_nested),
    ):
        assert len(memos) == 2
        assert len({id(memo) for memo in memos}) == 2
        assert initial == terminal == nested == memos
    assert len({id(memo) for memo in (*first_memos, *second_memos, *third_memos)}) == 6


def test_pair_terminal_redeclaration_rejects_stream_mutation(tmp_path, monkeypatch):
    inputs, estimates, protocol, config = _prepared_pair_inputs(tmp_path)
    original_terminal = tier_d_native.declare_terminal_allocation_execution
    mutated = False
    observed_memos = []

    def mutate_between_declarations(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            observed_memos.append(kwargs.get("_stream_digest_memo"))
            packet = inputs.stream.packets[-1]
            object.__setattr__(
                packet, "expected_simple_return", packet.expected_simple_return + 0.01
            )
            mutated = True
        return original_terminal(*args, **kwargs)

    monkeypatch.setattr(
        tier_d_native,
        "declare_terminal_allocation_execution",
        mutate_between_declarations,
    )
    with pytest.raises(
        ValueError, match="terminal current native prefix differs from frozen plan"
    ):
        _declare(inputs, estimates, protocol, config)
    assert mutated
    assert len(observed_memos) == 1
    assert type(observed_memos[0]) is SimpleStreamDigestMemo
