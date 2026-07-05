import os
import sys

import psycopg

dsn = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "PLATFORM_DB_URL", "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl"
)
with psycopg.connect(dsn) as conn, conn.cursor() as cur:
    cur.execute(
        "SELECT source, symbol, timeframe, count(*) "
        "FROM rl_klines GROUP BY 1,2,3 ORDER BY 1,2,3"
    )
    print("rl_klines:", cur.fetchall())
    cur.execute(
        "SELECT source, symbol, count(*) "
        "FROM rl_funding_rate GROUP BY 1,2 ORDER BY 1,2"
    )
    print("rl_funding_rate:", cur.fetchall())
    for t in ("rl_orderflow_1m", "rl_derivatives"):
        cur.execute(f"SELECT count(*) FROM {t}")
        print(f"{t}:", cur.fetchone()[0])
