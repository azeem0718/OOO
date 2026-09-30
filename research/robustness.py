"""Stress the leading candidate. This is NOT another search.

The distinction matters. A search reports its maximum, which is biased upward
and inflates the multiple-testing bar. A robustness check reports the ENTIRE
surface and asks a different question: is the result a broad plateau, or a
knife-edge cell that happens to look good?

A plateau means the effect is real but the exact parameters do not matter.
A single good cell surrounded by bad ones means we found noise.

Candidate: G10 carry, sign-weighted, crash-filtered, monthly rebalance,
10% vol target -- OOS Sharpe 0.67.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import portfolio as pf, validate
from research import universe as U

OOS = pd.Timestamp("2017-01-01")
START = pd.Timestamp("2006-06-01")


def build():
    rates = U.load_rates()
    prices, idx = U.load_prices(U.G10)
    idx = idx[idx >= START]
    prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
    return rates, prices, idx


def make_calm(prices, idx, lookback=60, window=504, q=0.80):
    basket = pf.spot_returns(prices).reindex(idx).mean(axis=1)
    rv = basket.rolling(lookback).std() * np.sqrt(252)
    th = rv.rolling(window, min_periods=252).quantile(q)
    return (rv.shift(1) <= th.shift(1)).astype(float).fillna(1.0)


def carry_w(rates, pairs, idx, rebalance, lag):
    r = rates.shift(lag).reindex(idx).ffill()
    diff = pd.DataFrame({p: (r[p[:3]] - r[p[3:]]) for p in pairs}, index=idx)
    return np.sign(diff).resample(rebalance).last().reindex(idx).ffill().fillna(0.0)


def sharpe_of(w, rets, prices, idx, lo, hi, target_vol=0.10, vol_lb=60):
    i = idx[(idx >= lo) & (idx < hi)]
    sim = pf.simulate(w.reindex(i), rets.reindex(i), prices,
                      target_vol=target_vol, vol_lookback=vol_lb)
    return pf.stats(sim["ret"])["sharpe"], sim["ret"]


def surface(title, cells, note=""):
    vals = [v for _, v in cells]
    pos = sum(1 for v in vals if v > 0)
    print(f"\n{title}")
    if note:
        print(f"  ({note})")
    for label, v in cells:
        bar = "#" * max(0, int(round(v * 20)))
        print(f"    {label:<34}{v:>7.2f}  {bar}")
    print(f"    -> {pos}/{len(vals)} positive | median {np.median(vals):.2f} | "
          f"min {min(vals):.2f} | max {max(vals):.2f}")
    return vals


def main():
    rates, prices, idx = build()
    rets = pf.total_returns(prices, rates)
    end = idx[-1] + pd.Timedelta(days=1)
    pairs = list(prices)
    allvals = []

    print("=" * 92)
    print("ROBUSTNESS OF G10 CRASH-FILTERED CARRY  (out-of-sample 2017+, full surface shown)")
    print("=" * 92)

    # 1. Rebalance frequency -- the single most arbitrary choice.
    cells = []
    for rb, name in (("W", "weekly"), ("ME", "monthly"), ("QE", "quarterly")):
        w = carry_w(rates, pairs, idx, rb, pf.RATE_LAG_DAYS)
        calm = make_calm(prices, idx)
        s, _ = sharpe_of(w.mul(calm, axis=0), rets, prices, idx, OOS, end)
        cells.append((f"rebalance={name}", s))
    allvals += surface("1. Rebalance frequency", cells,
                       "monthly was the original choice; others are untuned")

    # 2. Rate publication lag -- guards against same-day leakage.
    cells = []
    for lag in (1, 5, 21, 42):
        w = carry_w(rates, pairs, idx, "ME", lag)
        calm = make_calm(prices, idx)
        s, _ = sharpe_of(w.mul(calm, axis=0), rets, prices, idx, OOS, end)
        cells.append((f"rate lag={lag}d", s))
    allvals += surface("2. Rate publication lag", cells,
                       "if this collapses at longer lags the edge was leakage")

    # 3. Crash-filter threshold.
    cells = []
    for q in (0.70, 0.80, 0.90, 1.01):
        w = carry_w(rates, pairs, idx, "ME", pf.RATE_LAG_DAYS)
        calm = make_calm(prices, idx, q=min(q, 1.0)) if q <= 1.0 else pd.Series(1.0, index=idx)
        s, _ = sharpe_of(w.mul(calm, axis=0), rets, prices, idx, OOS, end)
        cells.append((f"vol cutoff q={q:.2f}" + (" (no filter)" if q > 1 else ""), s))
    allvals += surface("3. Crash-filter aggressiveness", cells)

    # 4. Vol-target machinery.
    cells = []
    for tv in (0.05, 0.10, 0.15):
        for lb in (40, 60, 90):
            w = carry_w(rates, pairs, idx, "ME", pf.RATE_LAG_DAYS)
            calm = make_calm(prices, idx)
            s, _ = sharpe_of(w.mul(calm, axis=0), rets, prices, idx, OOS, end,
                             target_vol=tv, vol_lb=lb)
            cells.append((f"target={tv:.0%} vol_lb={lb}", s))
    allvals += surface("4. Volatility-targeting parameters", cells,
                       "Sharpe should be nearly invariant to the target level")

    # 5. Leave-one-currency-out: is it one pair carrying everything?
    cells = []
    for drop in pairs:
        keep = [p for p in pairs if p != drop]
        sub = {k: prices[k] for k in keep}
        w = carry_w(rates, keep, idx, "ME", pf.RATE_LAG_DAYS)
        calm = make_calm(sub, idx)
        s, _ = sharpe_of(w.mul(calm, axis=0), rets[keep], sub, idx, OOS, end)
        cells.append((f"without {drop}", s))
    allvals += surface("5. Leave-one-out by currency pair", cells,
                       "a big drop means one pair carried the result")

    print()
    print("=" * 92)
    print("OVERALL PLATEAU")
    print("=" * 92)
    a = np.array(allvals)
    print(f"  {len(a)} configurations tested, all reported (no cherry-picking)")
    print(f"  positive: {100*(a>0).mean():.0f}%   median {np.median(a):.2f}   "
          f"IQR [{np.percentile(a,25):.2f}, {np.percentile(a,75):.2f}]   "
          f"range [{a.min():.2f}, {a.max():.2f}]")
    print(f"  verdict: {'BROAD PLATEAU - result is not parameter-specific' if (a>0).mean()>0.9 and np.median(a)>0.3 else 'FRAGILE - depends on specific parameters'}")

    # 6. Walk-forward: tradeable in real time, with no fixed OOS wall.
    print()
    print("=" * 92)
    print("WALK-FORWARD  (5y train / 2y test, rolling; params re-chosen each window)")
    print("=" * 92)
    gridp = [{"rb": rb, "q": q} for rb in ("W", "ME", "QE") for q in (0.7, 0.8, 0.9)]

    def run_fn(sub_df, p):
        i = sub_df.index
        sub_prices = {k: prices[k].reindex(i).ffill() for k in pairs}
        w = carry_w(rates, pairs, i, p["rb"], pf.RATE_LAG_DAYS)
        calm = make_calm(sub_prices, i, q=p["q"])
        sim = pf.simulate(w.mul(calm, axis=0), rets.reindex(i), sub_prices, target_vol=0.10)
        st = pf.stats(sim["ret"])
        return {"sharpe": st.get("sharpe", np.nan), "net": st.get("cagr", np.nan),
                "n_trades": st.get("n", 0), "profit_factor": np.nan, "ret": sim["ret"]}

    frame = pd.DataFrame(index=idx)
    wf = validate.walk_forward(frame, gridp, run_fn, train_years=5, test_years=2)
    if len(wf):
        print(f"{'test window ends':<20}{'chosen params':<28}{'IS SR':>8}{'OOS SR':>9}")
        for _, r in wf.iterrows():
            print(f"{str(r['test_end'])[:10]:<20}{str(r['params']):<28}"
                  f"{r['is_obj']:>8.2f}{r['oos_sharpe']:>9.2f}")
        oos_srs = wf["oos_sharpe"].dropna()
        print(f"\n  walk-forward windows: {len(oos_srs)}   "
              f"positive: {int((oos_srs>0).sum())}/{len(oos_srs)}   "
              f"mean OOS Sharpe {oos_srs.mean():.2f}   median {oos_srs.median():.2f}")
        print(f"  (in-sample Sharpes averaged {wf['is_obj'].mean():.2f} -> "
              f"decay of {wf['is_obj'].mean()-oos_srs.mean():.2f})")
    else:
        print("  insufficient history for walk-forward")


if __name__ == "__main__":
    main()
