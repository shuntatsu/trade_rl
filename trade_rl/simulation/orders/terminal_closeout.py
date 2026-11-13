"""Runtime-validated authority for the terminal-settlement minimum-notional waiver.

A risk-owned terminal close is a reduce-only exit of an existing holding, so a venue
does not apply its minimum notional to it. That exemption must never be assertable
by an ordinary caller: it is carried by a :class:`TerminalCloseoutAuthority`, which
can only be issued by :func:`_issue_terminal_closeout` after validating that the
interval really lies inside the reserved terminal-settlement window of a run and
that every flagged symbol holds a position. Only the terminal-settlement entry
points (the shared portfolio session and the shared replay) call the issuer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.execution import MarketExecutor

_ISSUER_KEY = object()


class TerminalCloseoutAuthority:
    """Opaque token naming the symbols a validated terminal close may exit."""

    __slots__ = ("_symbols", "_start_index")

    def __init__(
        self, *, _key: object, symbols: tuple[int, ...], start_index: int
    ) -> None:
        if _key is not _ISSUER_KEY:
            raise TypeError(
                "TerminalCloseoutAuthority is issued only by terminal settlement"
            )
        self._symbols = symbols
        self._start_index = start_index

    @property
    def symbols(self) -> frozenset[int]:
        return frozenset(self._symbols)

    @property
    def start_index(self) -> int:
        return self._start_index


def _issue_terminal_closeout(
    executor: MarketExecutor,
    book: BookState,
    *,
    start_index: int,
    agent_stop_index: int,
    stop_index: int,
    symbols: tuple[int, ...],
) -> TerminalCloseoutAuthority:
    """Issue the waiver authority for one reserved terminal-settlement interval."""

    values = (start_index, agent_stop_index, stop_index)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("terminal settlement indices must be integers")
    latency = int(executor.cost.order_latency_bars)
    if agent_stop_index != stop_index - latency - 1:
        raise ValueError(
            "agent_stop_index is not the reserved terminal window of this run"
        )
    if not agent_stop_index <= start_index < stop_index:
        raise ValueError("interval is not inside the terminal-settlement window")
    exact = book.exact_quantities
    if len(set(symbols)) != len(symbols):
        raise ValueError("terminal close-out symbols must be unique")
    for symbol in symbols:
        if (
            isinstance(symbol, bool)
            or not isinstance(symbol, int)
            or not 0 <= symbol < len(exact)
            or exact[symbol] == 0
        ):
            raise ValueError("terminal close-out symbols must hold a position")
    return TerminalCloseoutAuthority(
        _key=_ISSUER_KEY, symbols=tuple(sorted(symbols)), start_index=start_index
    )


def _require_terminal_closeout(
    authority: object, *, start_index: int, bars: int
) -> frozenset[int]:
    """Return the authorized symbols after checking the token matches this call."""

    if authority is None:
        return frozenset()
    if type(authority) is not TerminalCloseoutAuthority:
        raise TypeError("terminal_closeout must be issued by terminal settlement")
    if authority.start_index != start_index or bars != 1:
        raise ValueError("terminal close-out authority does not match this interval")
    return authority.symbols


__all__ = ["TerminalCloseoutAuthority"]
