"""Does the emacross edge survive regimes it was never fitted on?

The published result used M5 data covering 2025-03 -> 2026-08 -- a period when
gold ran from ~2900 to ~4200. A trend-following cross cannot lose in that tape.
H1 reaches back to 2008-12, covering the 2011 blowoff, the 2013-15 bear market,
and long stretches of chop. That is the honest test.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import data, engine, metrics, strategies
from fx.costs import XAUUSD

THEIR_SPREAD, MEASURED_SPREAD, COMMISSION = 14.0, 6.0, 47.0

BASE = dict(entry_valid_bars=5, sizing="fixed", lots=0.02, trail_arm_r=1.5,
            trail_offset=2.0, take_profit_r=0.0, start_equity=500.0)


def run(df, spread, commission, min_stop, lots=0.02):
    cfg = engine.ExecConfig(**dict(BASE, spread_points=spread, lots=lots,
                                   commission_per_lot=commission,
                                   min_stop_dist=min_stop))
    sigs = strategies.emacross(df, 9, 15, use_vwap=True, entries="E1")
    tr, cv = engine.run(df, sigs, XAUUSD, cfg)
    return engine.trades_frame(tr), cv, cfg


print("=" * 92)
print("STEP 2  Same strategy, every timeframe the broker serves, full history")
print("=" * 92)
print(f"{'TF':<5}{'from':<12}{'to':<12}{'bars':>9}{'trades':>8}{'WR%':>7}"
      f"{'PF':>7}{'net$':>12}{'avgR':>7}{'DD%':>8}{'Sharpe':>8}")

frames = {}
for tf, mstop in [("H1", 10.0), ("H4", 10.0), ("M30", 10.0), ("M15", 10.0)]:
    df = data.load_xau(tf)
    frames[tf] = df
    tfr, cv, cfg = run(df, THEIR_SPREAD, COMMISSION, mstop)
    s = metrics.summarize(tfr, cv, cfg.start_equity, tf)
    print(f"{tf:<5}{str(df.index[0])[:10]:<12}{str(df.index[-1])[:10]:<12}"
          f"{len(df):>9,}{s['n_trades']:>8}{s['win_rate']:>7.1f}"
          f"{min(s['profit_factor'],99):>7.2f}{s['net']:>12,.0f}"
          f"{s['avg_r']:>7.3f}{s['max_dd_pct']:>8.1f}{s['sharpe']:>8.2f}")

print()
print("=" * 92)
print("STEP 3  H1 (2008-2026) year by year  --  the regime question")
print("=" * 92)
h1 = frames["H1"]
tfr, cv, cfg = run(h1, THEIR_SPREAD, COMMISSION, 10.0)
yb = metrics.equity_by_year(tfr)
gold = h1["close"].resample("YE").last()
gold_ret = gold.pct_change() * 100
print(f"{'year':<6}{'trades':>8}{'WR%':>7}{'net$':>11}{'avgR':>8}{'cum$':>11}   {'gold yr%':>9}")
for y, row in yb.iterrows():
    gr = gold_ret.get(pd.Timestamp(f"{y}-12-31"), np.nan)
    gs = f"{gr:+.1f}" if np.isfinite(gr) else "  n/a"
    print(f"{y:<6}{int(row['trades']):>8}{row['win_rate']:>7.1f}{row['net']:>11,.0f}"
          f"{row['avg_r']:>8.3f}{row['cum_net']:>11,.0f}   {gs:>9}")

pos = int((yb["net"] > 0).sum())
print(f"\n  profitable years: {pos}/{len(yb)}")
s = metrics.summarize(tfr, cv, cfg.start_equity, "H1 full")
print(f"  full-period net ${s['net']:,.0f} over {s['years']:.1f}y | "
      f"PF {s['profit_factor']:.2f} | Sharpe {s['sharpe']:.2f} | maxDD {s['max_dd_pct']:.1f}%")
print(f"  tail dependence: drop top 5 -> ${s['net_drop_top5']:,.0f} | "
      f"drop top 20 -> ${s['net_drop_top20']:,.0f}")
print(f"  break-even cost/trade ${s['breakeven_cost_per_trade']:.2f} "
      f"(modelled ${tfr['cost'].mean():.2f})")

print()
print("=" * 92)
print("STEP 4  Cost sensitivity on H1 (commission/lot is NOT calibrated)")
print("=" * 92)
print(f"{'spread_pts':>11}{'comm/lot':>10}{'cost/trade':>12}{'net$':>12}{'PF':>7}{'avgR':>8}")
for sp in (6.0, 14.0, 25.0):
    for cm in (0.0, 20.0, 47.0, 75.0):
        t2, c2, g2 = run(h1, sp, cm, 10.0)
        s2 = metrics.summarize(t2, c2, g2.start_equity, "")
        cpt = t2["cost"].mean() + (sp * XAUUSD.point * XAUUSD.contract_size * 2 * 0.02)
        print(f"{sp:>11.0f}{cm:>10.0f}{cpt:>12.2f}{s2['net']:>12,.0f}"
              f"{min(s2['profit_factor'],99):>7.2f}{s2['avg_r']:>8.3f}")
