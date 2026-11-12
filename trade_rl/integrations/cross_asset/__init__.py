"""Point-in-time cross-asset market context (FRED, Yahoo, Stooq).

This package builds a causal data and feature layer only. It never reads crypto
prices and is not exported from :mod:`trade_rl.integrations` so that importing
the integrations facade does not pull in any network code.

Not to be confused with ``trade_rl.data.features.cross_asset``, which computes
BTC-relative features among the crypto symbols of one dataset; nothing here
shares code or semantics with it.
"""
