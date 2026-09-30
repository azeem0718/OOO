"""Carry, trend, and their combination -- tested as total-return portfolios.

These two are the FX anomalies with real documented persistence, so they get a
fair test: total return including the interest differential, volatility
targeting, monthly rebalancing, and transaction costs on turnover.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import data, portfolio as pf, validate

OOS_START = "2017-01-01"
G10 = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]


def build():
    mk = data.load_all()
    rates = data.load_rates()
    prices = {p: mk[p]["close"] for p in G10 if p in mk}
    idx = None
    for s in prices.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = pd.DatetimeIndex(sorted(idx))
    prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
    rets = pf.total_returns(prices, rates)
    return prices, rates, rets, idx


def row(label, st, extra=""):
    print(f"{label:<30}{st.get('cagr',float('nan')):>8.2f}{st.get('vol',float('nan')):>8.2f}"
          f"{st.get('sharpe',float('nan')):>8.2f}{st.get('max_dd',float('nan')):>9.1f}"
          f"{st.get('calmar',float('nan')):>8.2f}{st.get('pos_years',float('nan'))*100:>9.0f}%  {extra}")


def hdr(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(f"{'strategy':<30}{'CAGR%':>8}{'vol%':>8}{'Sharpe':>8}{'maxDD%':>9}{'Calmar':>8}{'+years':>10}")


def main():
    prices, rates, rets, idx = build()
    print(f"universe: {len(prices)} pairs  {idx[0].date()} -> {idx[-1].date()}")
    print(f"pairs: {', '.join(prices)}")

    # Carry is meaningless before the euro exists and before CHF/NZD rates start.
    start = pd.Timestamp("2006-06-01")
    idx2 = idx[idx >= start]
    rets2 = rets.reindex(idx2)

    variants = {}
    trials = 0
    for mode in ("sign", "rank"):
        w = pf.w_carry(rates, list(prices), idx2, rebalance="ME", mode=mode)
        variants[f"carry[{mode}]"] = w
        trials += 1
    for lb in (120, 250):
        w = pf.w_trend(prices, list(prices), idx2, lookback=lb, rebalance="ME")
        variants[f"trend[{lb}d]"] = w
        trials += 1
    # Combination: average the two signal sets (diversification, not fitting).
    variants["carry+trend"] = (
        (variants["carry[sign]"] + variants["trend[250d]"]) / 2.0)
    trials += 1

    hdr("FULL PERIOD (2006-06 onward), vol-targeted 10%, costs charged")
    full = {}
    for name, w in variants.items():
        sim = pf.simulate(w, rets2, prices, target_vol=0.10)
        st = pf.stats(sim["ret"], name)
        full[name] = sim
        row(name, st)

    hdr(f"IN-SAMPLE (2006-06 .. {OOS_START})")
    is_idx = idx2[idx2 < pd.Timestamp(OOS_START)]
    for name, w in variants.items():
        sim = pf.simulate(w.reindex(is_idx), rets2.reindex(is_idx), prices, target_vol=0.10)
        row(name, pf.stats(sim["ret"], name))

    hdr(f"OUT-OF-SAMPLE ({OOS_START} onward)  <-- the number that counts")
    oos_idx = idx2[idx2 >= pd.Timestamp(OOS_START)]
    oos = {}
    for name, w in variants.items():
        sim = pf.simulate(w.reindex(oos_idx), rets2.reindex(oos_idx), prices, target_vol=0.10)
        oos[name] = sim
        row(name, pf.stats(sim["ret"], name))

    # Statistical honesty on the best OOS performer.
    best = max(oos, key=lambda k: pf.stats(oos[k]["ret"])["sharpe"])
    bs = pf.stats(oos[best]["ret"])
    ci = validate.block_bootstrap_sharpe(oos[best]["ret"], n_boot=3000, block=20)
    srs = [pf.stats(v["ret"])["sharpe"] for v in oos.values()]
    bar = validate.expected_max_sharpe(trials, float(np.std(srs, ddof=1)) or 0.3)

    print()
    print("=" * 96)
    print("STATISTICAL VERDICT on the best out-of-sample strategy")
    print("=" * 96)
    print(f"  best OOS: {best}   Sharpe {bs['sharpe']:.2f}  CAGR {bs['cagr']:.2f}%  maxDD {bs['max_dd']:.1f}%")
    print(f"  block-bootstrap 95% CI on Sharpe: [{ci['lo']:.2f}, {ci['hi']:.2f}]   P(SR>0)={ci['p_gt_0']:.3f}")
    print(f"  trials={trials}, cross-trial Sharpe sd={np.std(srs,ddof=1):.3f} -> luck bar {bar:.2f}")
    print(f"  clears multiple-testing bar: {'YES' if bs['sharpe']>bar else 'NO'}")

    # Cost drag: how much of the gross edge the spread eats.
    print()
    print("=" * 96)
    print("COST DRAG (OOS): gross vs net")
    print("=" * 96)
    print(f"{'strategy':<30}{'gross SR':>10}{'net SR':>10}{'drag':>8}")
    for name, w in variants.items():
        g = pf.simulate(w.reindex(oos_idx), rets2.reindex(oos_idx), prices,
                        target_vol=0.10, charge_costs=False)
        n = oos[name]
        gs, ns = pf.stats(g["ret"])["sharpe"], pf.stats(n["ret"])["sharpe"]
        print(f"{name:<30}{gs:>10.2f}{ns:>10.2f}{gs-ns:>8.2f}")

    # Carry decomposition: is it the differential or the spot move?
    print()
    print("=" * 96)
    print("WHERE CARRY'S RETURN COMES FROM (full period)")
    print("=" * 96)
    spot = pf.spot_returns(prices).reindex(idx2)
    wc = variants["carry[sign]"]
    s_only = pf.simulate(wc, spot, prices, target_vol=0.10)
    print(f"  spot-only Sharpe   {pf.stats(s_only['ret'])['sharpe']:>6.2f}   "
          f"(betting on direction alone)")
    print(f"  spot+carry Sharpe  {pf.stats(full['carry[sign]']['ret'])['sharpe']:>6.2f}   "
          f"(the actual anomaly)")

    yr = oos[best]["ret"].resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    print()
    print(f"{best} OOS year by year (%):")
    print("  " + "  ".join(f"{i.year}:{v:+.1f}" for i, v in yr.items()))


if __name__ == "__main__":
    main()
