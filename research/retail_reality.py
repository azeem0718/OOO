"""Can a retail account actually capture the carry premium?

The backtest earns the INTERBANK differential. A retail broker does not pay
that. It pays less than market on the currency you are long and charges more
than market on the one you are short, keeping the difference as a swap markup.
That markup is subtracted from exactly the quantity this strategy trades.

This is the difference between a real edge and a tradeable one.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import portfolio as pf
from research import universe as U

OOS = pd.Timestamp("2017-01-01")
START = pd.Timestamp("2006-06-01")


def main():
    rates = U.load_rates()
    prices, idx = U.load_prices(U.G10)
    idx = idx[idx >= START]
    prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
    pairs = list(prices)
    end = idx[-1] + pd.Timedelta(days=1)

    basket = pf.spot_returns(prices).reindex(idx).mean(axis=1)
    rv = basket.rolling(60).std() * np.sqrt(252)
    th = rv.rolling(504, min_periods=252).quantile(0.80)
    calm = (rv.shift(1) <= th.shift(1)).astype(float).fillna(1.0)

    r = rates.shift(pf.RATE_LAG_DAYS).reindex(idx).ffill()
    diff = pd.DataFrame({p: (r[p[:3]] - r[p[3:]]) for p in pairs}, index=idx)
    w = np.sign(diff).resample("ME").last().reindex(idx).ffill().fillna(0.0)
    w = w.mul(calm, axis=0)

    print("=" * 88)
    print("RETAIL SWAP MARKUP: how much of the carry premium survives the broker")
    print("=" * 88)
    print("  A markup of X% per year is charged on every open position, long or short,")
    print("  because the broker skims both sides of the swap.\n")
    print(f"{'annual swap markup':>20}{'OOS Sharpe':>13}{'OOS CAGR%':>12}{'vs gross':>11}")

    base_stats = None
    rows = []
    for markup in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        # Haircut applies to held exposure, scaled by gross book size.
        adj = pf.total_returns(prices, rates).copy()
        daily = markup / 100.0 / 252.0
        haircut = pd.DataFrame(
            np.where(w.reindex(adj.index).fillna(0.0).abs() > 0, daily, 0.0),
            index=adj.index, columns=adj.columns)
        adj = adj - haircut

        i = idx[(idx >= OOS) & (idx < end)]
        sim = pf.simulate(w.reindex(i), adj.reindex(i), prices, target_vol=0.10)
        st = pf.stats(sim["ret"])
        if base_stats is None:
            base_stats = st
        delta = st["cagr"] - base_stats["cagr"]
        rows.append((markup, st["sharpe"], st["cagr"]))
        print(f"{markup:>19.1f}%{st['sharpe']:>13.2f}{st['cagr']:>12.2f}{delta:>11.2f}")

    print()
    print("=" * 88)
    print("WHAT THIS MEANS AT REAL ACCOUNT SIZES")
    print("=" * 88)
    # Use the walk-forward estimate (0.31), not the favourable fixed-split 0.67.
    wf_sharpe, vol = 0.31, 0.10
    print(f"  Using the WALK-FORWARD Sharpe ({wf_sharpe}), not the fixed-split 0.67,")
    print(f"  at a {vol:.0%} volatility target:\n")
    exp_ret = wf_sharpe * vol
    print(f"    expected return  ~{100*exp_ret:.1f}%/yr")
    print(f"    expected vol     ~{100*vol:.0f}%/yr")
    print(f"    a 1-in-6 year    ~{100*(exp_ret-vol):+.1f}%  (one sd below)")
    print(f"    observed maxDD   -13.0% (OOS), -36.9% (full sample incl. 2008)\n")
    print(f"{'account':>12}{'expected $/yr':>16}{'1sd-down $/yr':>16}{'maxDD at -13%':>16}")
    for acct in (500, 5_000, 25_000, 100_000):
        print(f"{acct:>11,}{acct*exp_ret:>16,.0f}{acct*(exp_ret-vol):>16,.0f}"
              f"{-acct*0.13:>16,.0f}")

    print()
    print("  Minimum viable book: 7 pairs x 0.01 lots = 7,000 units notional.")
    print("  At 10% vol target the book needs roughly 1-3x leverage, so margin")
    print("  is not the binding constraint -- the binding constraint is that")
    print("  a 3% expected return on a small account is a rounding error,")
    print("  while the -13% to -37% drawdowns are not.")


if __name__ == "__main__":
    main()
