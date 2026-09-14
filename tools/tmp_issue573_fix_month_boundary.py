from __future__ import annotations

from pathlib import Path

path = Path("trade_rl/integrations/binance/transport.py")
text = path.read_text(encoding="utf-8")
old = '''            rows = _strict_index_price_archive_rows(
                payload,
                source=url,
                expected_member=member,
                interval_ms=interval_ms,
            )
            for row in rows:
                open_ms = _normalize_epoch_ms(row[0])
'''
new = '''            rows = _strict_index_price_archive_rows(
                payload,
                source=url,
                expected_member=member,
                interval_ms=interval_ms,
            )
            month_start_ms = int(month.timestamp() * 1000)
            if month.month == 12:
                next_month = month.replace(
                    year=month.year + 1,
                    month=1,
                    day=1,
                )
            else:
                next_month = month.replace(month=month.month + 1, day=1)
            month_end_ms = int(next_month.timestamp() * 1000)
            for row in rows:
                open_ms = _normalize_epoch_ms(row[0])
                if not month_start_ms <= open_ms < month_end_ms:
                    raise BinanceTransportError(
                        f"index-price row lies outside archive calendar month: {url}"
                    )
'''
if text.count(old) != 1:
    raise RuntimeError("expected exactly one index-price archive row loop")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
