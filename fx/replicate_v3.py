"""Replicate the reported V3 backtest, then stress it over the full history.

Purpose: prove the ported strategy + new engine reproduce the published
numbers before trusting either on data the original never saw.

Reported (REPORT_v3.txt, M5 2025-03-10 -> 2026-08-06, 0.02 lots, $500):
    306 trades | 29.4% WR | PF 1.81 | net +$5,155.75 | max DD -$860.61
"""
import sys

import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import data, engine, metrics, strategies
from fx.costs import XAUUSD

# Their cost model: $47/lot commission + $28/lot round-turn spread.
# $28/lot round turn = $14/lot per side = 0.14 USD/oz = 14 points per side.
THEIR_SPREAD_POINTS = 14.0
THEIR_COMMISSION = 47.0

V3 = dict(entry_valid_bars=5, min_stop_dist=10.0, sizing="fixed", lots=0.02,
          trail_arm_r=1.5, trail_offset=2.0, take_profit_r=0.0,
          start_equity=500.0)


def run_one(df, label, spread_points, commission, **over):
    cfg_kw = dict(V3, spread_points=spread_points, commission_per_lot=commission)
    cfg_kw.update(over)
    cfg = engine.ExecConfig(**cfg_kw)
    sigs = strategies.emacross(df, fast=9, slow=15, use_vwap=True, entries="E1")
    trades, curve = engine.run(df, sigs, XAUUSD, cfg)
    tf = engine.trades_frame(trades)
    s = metrics.summarize(tf, curve, cfg.start_equity, label)
    s["n_signals"] = len(sigs)
    return s, tf, curve


def show(s):
    print(f"  signals {s['n_signals']:>6}   trades {s['n_trades']:>5}   "
          f"WR {s['win_rate']:>5.1f}%   PF {s['profit_factor']:>5.2f}")
    print(f"  net ${s['net']:>12,.2f}   expectancy ${s['expectancy']:>7.2f}   "
          f"avgR {s['avg_r']:>6.3f}")
    print(f"  maxDD ${s['max_dd']:>11,.2f} ({s['max_dd_pct']:.1f}%)   "
          f"Sharpe {s['sharpe']:>5.2f}   years {s['years']:.1f}")


print("=" * 78)
print("STEP 1  Replicate the published M5 run (their cost model)")
print("=" * 78)
m5 = data.load_xau("M5", start="2025-03-10", end="2026-08-06")
print(f"  bars {len(m5):,}  {m5.index[0]} -> {m5.index[-1]}  (broker time)")
s, tf_m5, cv_m5 = run_one(m5, "M5 replication", THEIR_SPREAD_POINTS, THEIR_COMMISSION)
show(s)
print(f"\n  REPORTED: trades 306 | WR 29.4% | PF 1.81 | net $5,155.75 | DD -$860.61")
print(f"  DELTA   : trades {s['n_trades']-306:+d} | WR {s['win_rate']-29.4:+.1f}pp | "
      f"PF {s['profit_factor']-1.81:+.2f} | net ${s['net']-5155.75:+,.2f}")
