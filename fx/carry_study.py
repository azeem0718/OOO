"""Characterise carry honestly: stability, crash risk, and two motivated fixes.

The fixes are chosen for economic reasons, not by search:
  * dollar-neutral   -- carry against USD is partly a bet on the dollar. Netting
                        the average position out removes that shared exposure.
  * crash-filtered   -- carry crashes coincide with volatility spikes (investors
                        unwind funding trades together). Cutting exposure when
                        realised vol is high targets the known failure mode.
Every variant tested is counted as a trial and reported.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import data, portfolio as pf, validate

G10 = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
START, OOS = pd.Timestamp("2006-06-01"), pd.Timestamp("2017-01-01")


def main():
    mk = data.load_all()
    rates = data.load_rates()
    prices = {p: mk[p]["close"] for p in G10 if p in mk}
    idx = None
    for s in prices.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = pd.DatetimeIndex(sorted(idx))
    idx = idx[idx >= START]
    prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
    rets = pf.total_returns(prices, rates)

    base = pf.w_carry(rates, list(prices), idx, rebalance="ME", mode="sign")

    # dollar-neutral: subtract the cross-sectional mean each day
    dn = base.sub(base.mean(axis=1), axis=0)

    # crash filter: flatten when trailing vol of an equal-weight FX basket is
    # in its own top quintile, measured on trailing data only
    basket = pf.spot_returns(prices).reindex(idx).mean(axis=1)
    rv = basket.rolling(60).std() * np.sqrt(252)
    thresh = rv.rolling(504, min_periods=252).quantile(0.80)
    calm = (rv.shift(1) <= thresh.shift(1)).astype(float).fillna(1.0)
    cf = base.mul(calm, axis=0)
    dncf = dn.mul(calm, axis=0)

    variants = {"carry": base, "carry dollar-neutral": dn,
                "carry crash-filtered": cf, "carry DN+crash": dncf}

    def seg(w, lo, hi):
        i = idx[(idx >= lo) & (idx < hi)]
        return pf.simulate(w.reindex(i), rets.reindex(i), prices, target_vol=0.10)

    print("=" * 100)
    print("CARRY VARIANTS  (vol-targeted 10%, costs charged, monthly rebalance)")
    print("=" * 100)
    print(f"{'variant':<24}{'  full SR':>9}{'IS SR':>8}{'OOS SR':>8}"
          f"{'OOS CAGR%':>11}{'OOS maxDD%':>11}{'OOS Calmar':>11}{'+yrs':>7}")
    end = idx[-1] + pd.Timedelta(days=1)
    oos_sims = {}
    for name, w in variants.items():
        f = pf.stats(seg(w, START, end)["ret"])
        i = pf.stats(seg(w, START, OOS)["ret"])
        o_sim = seg(w, OOS, end)
        o = pf.stats(o_sim["ret"])
        oos_sims[name] = o_sim
        print(f"{name:<24}{f['sharpe']:>9.2f}{i['sharpe']:>8.2f}{o['sharpe']:>8.2f}"
              f"{o['cagr']:>11.2f}{o['max_dd']:>11.1f}{o['calmar']:>11.2f}"
              f"{o['pos_years']*100:>6.0f}%")

    # Multiple-testing bar for the WHOLE study, not just these four variants.
    #
    # Using only the carry variants' dispersion understates the bar badly: they
    # are near-identical strategies, so their Sharpes barely differ and the
    # implied bar collapses. The search that produced this winner spanned every
    # family tried, so the bar must use that full dispersion and trial count.
    SWEEP_OOS_SHARPES = [0.14, 0.08, -0.01, -0.19, -0.52]   # 34 trials beneath
    ANOMALY_OOS_SHARPES = [0.45, 0.31, -0.61, -0.08, 0.24]  # 5 trials
    TRIALS = 34 + 5 + 4
    srs = [pf.stats(s["ret"])["sharpe"] for s in oos_sims.values()]
    all_srs = SWEEP_OOS_SHARPES + ANOMALY_OOS_SHARPES + srs
    bar = validate.expected_max_sharpe(TRIALS, float(np.std(all_srs, ddof=1)) or 0.3)
    best = max(oos_sims, key=lambda k: pf.stats(oos_sims[k]["ret"])["sharpe"])
    bstat = pf.stats(oos_sims[best]["ret"])
    ci = validate.block_bootstrap_sharpe(oos_sims[best]["ret"], n_boot=3000, block=20)

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  best OOS variant : {best}")
    print(f"  OOS Sharpe       : {bstat['sharpe']:.2f}   95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}]   P(SR>0)={ci['p_gt_0']:.3f}")
    print(f"  OOS CAGR / vol   : {bstat['cagr']:.2f}% / {bstat['vol']:.2f}%   maxDD {bstat['max_dd']:.1f}%")
    print(f"  trials (whole study): {TRIALS}  ->  search-adjusted luck bar Sharpe {bar:.2f}")
    print(f"  clears that bar  : {'YES' if bstat['sharpe'] > bar else 'NO'}")
    print(f"  standalone signif: bootstrap P(SR>0)={ci['p_gt_0']:.3f} on its own returns,")
    print(f"                     which is independent of how many strategies were tried")

    # Rolling stability of plain carry over the whole sample.
    print()
    print("=" * 100)
    print("ROLLING 3-YEAR SHARPE of plain carry (full sample) -- is it ever reliable?")
    print("=" * 100)
    full = pf.simulate(base, rets, prices, target_vol=0.10)
    r = full["ret"]
    roll = r.rolling(756).mean() / r.rolling(756).std() * np.sqrt(252)
    ann = roll.resample("YE").last().dropna()
    print("  " + "  ".join(f"{i.year}:{v:+.2f}" for i, v in ann.items()))
    print(f"\n  fraction of 3y windows with Sharpe > 0 : {100*(roll.dropna()>0).mean():.0f}%")
    print(f"  fraction with Sharpe > 0.5             : {100*(roll.dropna()>0.5).mean():.0f}%")
    print(f"  worst 3y Sharpe                        : {roll.min():.2f}")

    # Drawdown profile including the 2008 carry crash.
    eq = (1 + r).cumprod()
    ddser = eq / eq.cummax() - 1
    print(f"\n  full-sample maxDD {100*ddser.min():.1f}% on {ddser.idxmin().date()}")
    wy = r.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    print("  yearly %: " + "  ".join(f"{i.year}:{v:+.1f}" for i, v in wy.items()))

    out = pd.DataFrame({"carry_ret": r, "equity": eq, "dd": ddser})
    out.to_csv("/home/user/OOO/fx/carry_track_record.csv")
    print("\nwrote fx/carry_track_record.csv")


if __name__ == "__main__":
    main()
