"""Bounded artificial prefix diagnostics; no native execution or qualification."""

from __future__ import annotations

import cProfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from tools.tier_d_nonrl.progress import ArmProgressSpec

_Advance = Callable[[str, int, int], None]


class _PrefixStop(Exception):
    """An individual invocation's predeclared technical stopping signal."""


@dataclass(frozen=True)
class PrefixObservation:
    expected_stop: bool
    count: int
    actor_advances: int
    reserve_advances: int
    processing_index: int
    elapsed_seconds: float
    error: Exception | None
    returned: object | None
    profile_publication_failure: str | None


def _make_profiler() -> cProfile.Profile:
    return cProfile.Profile()


def observe_prefix(
    invoke: Callable[[_Advance], object],
    spec: ArmProgressSpec,
    *,
    target: int,
    on_advance: _Advance | None = None,
    profile_path: Path | None = None,
    clock: Callable[[], float] = perf_counter,
) -> PrefixObservation:
    """Observe actual contiguous callbacks, stopping with an owned exception at N.

    ``invoke`` supplies execution; this helper owns no account or native state.
    A valid event counts even if forwarding it fails. The exact owned stopping
    exception remains available for its caller-attached partial receipt. Counts
    and normal returns confer no financial completion or settlement status.
    Elapsed time excludes profile publication; profile files are fresh-only.
    """
    if type(spec) is not ArmProgressSpec:
        raise ValueError("prefix observation requires an exact arm progress spec")
    spec.__post_init__()
    if type(target) is not int or not 1 <= target <= spec.target:
        raise ValueError("prefix target must be an ordinary int within the arm")
    if (
        not callable(invoke)
        or not callable(clock)
        or (on_advance is not None and not callable(on_advance))
        or (profile_path is not None and not isinstance(profile_path, Path))
    ):
        raise ValueError("prefix observation requires callables and an optional Path")

    count = actor = reserve = 0
    processing_index = spec.start_index
    stop = _PrefixStop("predeclared artificial native prefix reached")

    def advance(phase: str, start: int, processing: int) -> None:
        nonlocal count, actor, reserve, processing_index
        expected_phase = "actor" if processing_index < spec.actor_stop else "reserve"
        if (
            type(phase) is not str
            or phase != expected_phase
            or type(start) is not int
            or type(processing) is not int
            or start != processing_index
            or processing != start + 1
            or processing > spec.settlement_stop
            or count >= target
        ):
            raise ValueError("prefix callback is not a contiguous declared advance")
        try:
            if on_advance is not None:
                on_advance(phase, start, processing)
        finally:
            # The native return is known even if its journal publication fails.
            count += 1
            actor += phase == "actor"
            reserve += phase == "reserve"
            processing_index = processing
        if count == target:
            raise stop

    profiler = _make_profiler() if profile_path is not None else None
    error: Exception | None = None
    returned: object | None = None
    profile_failure = None
    started = clock()
    try:
        if profiler is not None:
            profiler.enable()
        returned = invoke(advance)
    except Exception as failure:
        error = failure
    finally:
        if profiler is not None:
            try:
                profiler.disable()
            except Exception as failure:
                profile_failure = f"{type(failure).__name__}: {failure}"
        elapsed = clock() - started
        if profiler is not None and profile_path is not None:
            try:
                # dump_stats overwrites its destination; reserve a new owned file.
                with profile_path.open("xb"):
                    pass
                profiler.dump_stats(str(profile_path))
            except Exception as failure:
                profile_failure = (
                    profile_failure or f"{type(failure).__name__}: {failure}"
                )
    return PrefixObservation(
        expected_stop=error is stop,
        count=count,
        actor_advances=actor,
        reserve_advances=reserve,
        processing_index=processing_index,
        elapsed_seconds=elapsed,
        error=error,
        returned=returned,
        profile_publication_failure=profile_failure,
    )
