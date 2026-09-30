"""Extended FX universe and rate panel for carry research.

Round 1 established that carry is the only strategy here with a credible
signature. It is strongest where rate differentials are widest, so this module
extends the G10 universe with the high-yielders we have both spot and rate
data for.

Exclusions are deliberate and documented -- a silently dropped currency is a
silently changed result.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import data as fxdata

# G10 leg: already validated in round 1.
G10 = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]

# High-yield / high-beta leg. Every one of these has BOTH a cached spot series
# and a live FRED rate series.
EXTENDED = ["USDMXN", "USDZAR", "USDNOK", "USDSEK"]

# Why USDTRY is absent despite being cached and being the classic carry trade:
# FRED's Turkish 3M interbank series (IR3TIB01TRM156N) stops in 2008-04. There
# is no rate data to compute a differential from after that, and carry without
# the differential is not carry. Including it with a stale or assumed rate
# would fabricate the exact quantity the strategy trades on.
EXCLUDED = {
    "USDTRY": "FRED rate series IR3TIB01TRM156N ends 2008-04; no modern differential",
}

EXTRA_RATE_SERIES = {
    "MXN": "IR3TIB01MXM156N",   # Mexico 3M interbank, 1997-2026
    "ZAR": "IR3TIB01ZAM156N",   # South Africa 3M, 1980-2026
    "NOK": "IR3TIB01NOM156N",   # Norway 3M, 1979-2026
    "SEK": "IR3TIB01SEM156N",   # Sweden 3M, 1982-2026
}

CACHE = fxdata.CACHE


def load_rates(refresh: bool = False) -> pd.DataFrame:
    """G10 rates plus the extended currencies, daily, in percent."""
    path = CACHE / "rates_extended.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    base = fxdata.load_rates()
    cols = {c: base[c] for c in base.columns}
    for ccy, sid in EXTRA_RATE_SERIES.items():
        try:
            cols[ccy] = fxdata.fetch_fred(sid)
        except Exception as e:
            print(f"  ! rate {ccy} ({sid}): {type(e).__name__}: {e}")

    df = pd.DataFrame(cols).sort_index()
    idx = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="D")
    df = df.reindex(idx).ffill()
    df.index.name = "date"
    df.to_parquet(path)
    return df


def load_prices(pairs: list[str]) -> tuple[dict[str, pd.Series], pd.DatetimeIndex]:
    """Close series for `pairs`, aligned on a common index."""
    mk = fxdata.load_all()
    missing = [p for p in pairs if p not in mk]
    if missing:
        print(f"  ! no spot data for: {missing}")
    prices = {p: mk[p]["close"] for p in pairs if p in mk}
    idx = None
    for s in prices.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = pd.DatetimeIndex(sorted(idx))
    return {k: v.reindex(idx).ffill() for k, v in prices.items()}, idx


def rate_coverage(rates: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    """When each pair's differential actually becomes computable.

    A carry backtest that starts before both legs have rate data is silently
    trading a differential against a forward-filled constant.
    """
    rows = []
    for p in pairs:
        b, q = p[:3], p[3:]
        if b not in rates.columns or q not in rates.columns:
            rows.append({"pair": p, "status": "MISSING RATE", "from": None})
            continue
        both = rates[[b, q]].dropna()
        rows.append({"pair": p, "status": "ok",
                     "from": both.index[0].date() if len(both) else None,
                     "last": both.index[-1].date() if len(both) else None})
    return pd.DataFrame(rows)
