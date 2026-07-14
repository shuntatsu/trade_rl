from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "trade_rl/evaluation/metrics.py",
    "        periods_per_year=returns.periods_per_year,\n",
    "        periods_per_year=int(round(returns.annualization_periods_per_year)),\n",
)

replace_once(
    "trade_rl/serving/normalizer.py",
    '''    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"serving normalizer sidecar is invalid: {error}") from error
''',
    '''    except (KeyError, TypeError, ValueError) as error:
        detail = str(error)
        if "digest" in detail:
            raise ValueError(f"serving normalizer digest mismatch: {detail}") from error
        raise ValueError(f"serving normalizer sidecar is invalid: {detail}") from error
''',
)
