from decimal import Decimal, localcontext
from fractions import Fraction

from tools.market_account_oracle import FillEconomics, ReferenceAccount


def _third() -> ReferenceAccount:
    return ReferenceAccount(
        cash="1000", quantities=("1/3",), marks=(3.0,), multipliers=(1.0,)
    )


def test_inactive_settlement_multiplies_exact_inventory_before_reporting_money() -> (
    None
):
    account = _third()
    proceeds = account.settle(mask=(True,), prices=(3.0,), recovery=(1.0,))
    assert proceeds == Decimal("1")
    assert account.cash == Decimal("1001")
    assert account.quantities == (Fraction(0),)


def test_full_close_is_independent_of_the_callers_decimal_precision() -> None:
    account = _third()
    with localcontext() as caller_context:
        caller_context.prec = 2
        account.apply_fill(
            symbol_index=0,
            reported_quantity=-0.3333333333333333,
            execution_price=3.0,
            economics=FillEconomics(),
            market_notional=1000.0,
        )
    assert account.quantities == (Fraction(0),)
