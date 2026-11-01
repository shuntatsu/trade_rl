"""Reusable PostgreSQL market-data store with a Binance Vision ingester (Issue #810).

The package root deliberately imports nothing: ``plan`` and dry-run ``ingest`` need
neither PostgreSQL nor the optional ``store`` extra, and only ``store`` imports
``psycopg``. Entry point: ``python -m trade_rl.integrations.market_store``.
"""
