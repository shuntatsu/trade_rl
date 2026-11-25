"""Binary64-compatible scalar admission for native decision-mark cash budgets."""

from __future__ import annotations

import math

import numpy as np


def finite_cash_scalar(value: object, name: str) -> float:
    # Admit the same binary64-compatible native scalar domain as market
    # inputs. Validate before narrowing: e.g. a negative longdouble can
    # become -0.0, bypassing the long-only spot boundary.
    # Reject subclasses that can override conversion and hide an invalid value.
    if isinstance(value, np.generic) and type(value) is not value.dtype.type:
        raise TypeError(f"{name} must be a native real scalar")
    if type(value) is int or (
        isinstance(value, np.integer) and value.dtype.kind in "iu"
    ):
        if abs(int(value)) > 2**53:
            raise ValueError(f"{name} must be binary64-compatible")
        v = float(value)
    elif type(value) is float or isinstance(value, np.floating):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            raise ValueError(f"{name} must be binary64-compatible")
        v = float(value)
    else:
        raise TypeError(f"{name} must be a native real scalar")
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v
