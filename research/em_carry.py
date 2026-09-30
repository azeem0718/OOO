"""Does widening the carry universe to high-yielders improve it?

Round 1 finding: carry was the only strategy with a credible signature, but at
OOS Sharpe 0.67 it sat below the search-adjusted bar of 0.86. Carry theory says
the premium scales with the size of the interest differential, so the natural
test is whether adding high-yielders strengthens it -- or just adds crash risk.

Design is unchanged from round 1 so the comparison is like-for-like: total
return (spot + differential), 10% vol target, monthly rebalance, costs on
turnover, parameters fixed before the 2017 out-of-sample wall.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import portfolio as pf, validate
from research import universe as U

OOS = pd.Timestamp("2017-01-01")
START = pd.Timestamp("2006-06-01")   # first date all G10 spot series exist

# Conservative retail round-turn spreads for the wide pairs. The majors table
# in fx/costs.py is calibrated for majors and is far too tight for EM.
EM_SPREADS = {"USDMXN": 50.0, "USDZAR": 80.0, "USDNOK": 60.0, "USDSEK": 60.0}

# Round 1 spent 43 trials. Everything added here counts on top of that.
PRIOR_TRIALS = 43
PRIOR_OOS_SHARPES = [0.14, 0.08, -0.01, -0.19, -0.52,
                     0.45, 0.31, -0.61, -0.08, 0.24,
                     0.45, 0.39, 0.67, 0.59]


def crash_filter(prices: dict, idx: pd.DatetimeIndex) -> pd.Series:
    """1 when trailing basket vol is calm, 0 when it is in its top quintile.

    Thresholds are computed on trailing data only.
    """
    basket = pf.spot_returns(prices).reindex(idx).mean(axis=1)
    rv = basket.rolling(60).std() * np.sqrt(252)
    th = rv.rolling(504, min_periods=252).quantile(0.80)
    return (rv.shift(1) <= th.shift(1)).astype(float).fillna(1.0)


def xs_rank(rates: pd.DataFrame, pairs: list[str], idx: pd.DatetimeIndex,
            k: int = 3) -> pd.DataFrame:
    """Cross-sectional carry: long the k widest positive differentials,
    short the k widest negative ones. Dollar exposure roughly nets out."""
    r = rates.shift(pf.RATE_LAG_DAYS).reindex(idx).ffill()
    diff = pd.DataFrame(
        {p: (r[p[:3]] - r[p[3:]]) for p in pairs
         if p[:3] in r.columns and p[3:] in r.columns}, index=idx)
    w = pd.DataFrame(0.0, index=diff.index, columns=diff.columns)
    rk_hi = diff.rank(axis=1, ascending=False)
    rk_lo = diff.rank(axis=1, ascending=True)
    w[rk_hi <= k] = 1.0
    w[rk_lo <= k] = -1.0
    return w.resample("ME").last().reindex(idx).ffill().fillna(0.0)


def tail_stats(r: pd.Series) -> dict:
    x = r.dropna()
    m = x.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    return {"skew": float(x.skew()), "kurt": float(x.kurtosis()),
            "worst_month": 100 * float(m.min()),
            "p01_day": 100 * float(x.quantile(0.01))}


def main():
    rates = U.load_rates()
    pairs_all = U.G10 + U.EXTENDED
    prices, idx = U.load_prices(pairs_all)
    idx = idx[idx >= START]
    prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
    rets = pf.total_returns(prices, rates)

    g10 = [p for p in U.G10 if p in prices]
    ext = [p for p in pairs_all if p in prices]
    print(f"G10 universe      ({len(g10)}): {', '.join(g10)}")
    print(f"Extended universe ({len(ext)}): {', '.join(ext)}")
    print(f"excluded: {', '.join(U.EXCLUDED)}  ({list(U.EXCLUDED.values())[0]})")
    print(f"period: {idx[0].date()} -> {idx[-1].date()}   OOS wall {OOS.date()}\n")

    calm_g10 = crash_filter({p: prices[p] for p in g10}, idx)
    calm_ext = crash_filter(prices, idx)

    variants: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for name, plist, calm in (("G10", g10, calm_g10), ("EXT", ext, calm_ext)):
        sign_w = pf.w_carry(rates, plist, idx, rebalance="ME", mode="sign")
        rank_w = xs_rank(rates, plist, idx, k=3)
        variants[f"{name} carry sign"] = (sign_w, plist)
        variants[f"{name} carry sign +crash"] = (sign_w.mul(calm, axis=0), plist)
        variants[f"{name} carry rank3"] = (rank_w, plist)
        variants[f"{name} carry rank3 +crash"] = (rank_w.mul(calm, axis=0), plist)

    def seg(w, plist, lo, hi, sp=EM_SPREADS):
        i = idx[(idx >= lo) & (idx < hi)]
        return pf.simulate(w.reindex(i), rets[plist].reindex(i), prices,
                           target_vol=0.10, spread_pips=sp)

    end = idx[-1] + pd.Timedelta(days=1)
    print("=" * 104)
    print("CARRY: G10 vs EXTENDED UNIVERSE   (10% vol target, conservative EM spreads, costs charged)")
    print("=" * 104)
    print(f"{'variant':<26}{'full SR':>9}{'IS SR':>8}{'OOS SR':>8}{'OOS CAGR%':>11}"
          f"{'OOS maxDD%':>11}{'OOS +yrs':>10}{'skew':>7}{'wrstMo%':>9}")
    oos_sims = {}
    for name, (w, plist) in variants.items():
        f = pf.stats(seg(w, plist, START, end)["ret"])
        i = pf.stats(seg(w, plist, START, OOS)["ret"])
        osim = seg(w, plist, OOS, end)
        o = pf.stats(osim["ret"])
        oos_sims[name] = osim
        t = tail_stats(osim["ret"])
        print(f"{name:<26}{f['sharpe']:>9.2f}{i['sharpe']:>8.2f}{o['sharpe']:>8.2f}"
              f"{o['cagr']:>11.2f}{o['max_dd']:>11.1f}{o['pos_years']*100:>9.0f}%"
              f"{t['skew']:>7.2f}{t['worst_month']:>9.1f}")

    # Honest multiple-testing bar for the cumulative search.
    new_srs = [pf.stats(s["ret"])["sharpe"] for s in oos_sims.values()]
    trials = PRIOR_TRIALS + len(variants)
    all_srs = PRIOR_OOS_SHARPES + new_srs
    bar = validate.expected_max_sharpe(trials, float(np.std(all_srs, ddof=1)))
    best = max(oos_sims, key=lambda k: pf.stats(oos_sims[k]["ret"])["sharpe"])
    bstat = pf.stats(oos_sims[best]["ret"])
    ci = validate.block_bootstrap_sharpe(oos_sims[best]["ret"], n_boot=3000, block=20)

    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    print(f"  best OOS variant     : {best}")
    print(f"  OOS Sharpe           : {bstat['sharpe']:.2f}   95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}]"
          f"   P(SR>0)={ci['p_gt_0']:.3f}")
    print(f"  OOS CAGR / vol / DD  : {bstat['cagr']:.2f}% / {bstat['vol']:.2f}% / {bstat['max_dd']:.1f}%")
    print(f"  cumulative trials    : {trials}  ->  search-adjusted bar Sharpe {bar:.2f}")
    print(f"  clears that bar      : {'YES' if bstat['sharpe'] > bar else 'NO'}")
    print(f"  round-1 benchmark    : G10 carry crash-filtered, OOS Sharpe 0.67")

    print()
    print("=" * 104)
    print("COST SENSITIVITY  (EM spreads scaled; does the extension survive wider spreads?)")
    print("=" * 104)
    print(f"{'variant':<26}{'1x':>9}{'2x':>9}{'4x':>9}{'8x':>9}")
    for name, (w, plist) in variants.items():
        if not name.startswith("EXT"):
            continue
        row = []
        for mult in (1, 2, 4, 8):
            sp = {k: v * mult for k, v in EM_SPREADS.items()}
            row.append(pf.stats(seg(w, plist, OOS, end, sp)["ret"])["sharpe"])
        print(f"{name:<26}" + "".join(f"{v:>9.2f}" for v in row))

    print()
    print("=" * 104)
    print(f"{best}: OOS year by year (%)")
    print("=" * 104)
    yr = oos_sims[best]["ret"].resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    print("  " + "  ".join(f"{i.year}:{v:+.1f}" for i, v in yr.items()))

    pd.DataFrame({k: v["ret"] for k, v in oos_sims.items()}).to_csv(
        "/home/user/OOO/research/em_carry_oos_returns.csv")
    print("\nwrote research/em_carry_oos_returns.csv")


if __name__ == "__main__":
    main()
