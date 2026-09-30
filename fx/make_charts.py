"""Equity curves for the report. Log scale, because linear flatters compounding."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import data, engine, metrics, portfolio as pf, strategies
from fx.costs import XAUUSD

G10 = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
OOS = pd.Timestamp("2017-01-01")

mk = data.load_all()
rates = data.load_rates()
prices = {p: mk[p]["close"] for p in G10 if p in mk}
idx = None
for s in prices.values():
    idx = s.index if idx is None else idx.union(s.index)
idx = pd.DatetimeIndex(sorted(idx))
idx = idx[idx >= pd.Timestamp("2006-06-01")]
prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
rets = pf.total_returns(prices, rates)

base = pf.w_carry(rates, list(prices), idx, rebalance="ME", mode="sign")
basket = pf.spot_returns(prices).reindex(idx).mean(axis=1)
rv = basket.rolling(60).std() * np.sqrt(252)
th = rv.rolling(504, min_periods=252).quantile(0.80)
calm = (rv.shift(1) <= th.shift(1)).astype(float).fillna(1.0)
cf = base.mul(calm, axis=0)
trend = pf.w_trend(prices, list(prices), idx, lookback=250, rebalance="ME")

curves = {
    "Carry (crash-filtered)": pf.simulate(cf, rets, prices, target_vol=0.10)["ret"],
    "Carry (plain)": pf.simulate(base, rets, prices, target_vol=0.10)["ret"],
    "Trend 250d": pf.simulate(trend, rets, prices, target_vol=0.10)["ret"],
}

fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
colors = {"Carry (crash-filtered)": "#1b7f5a", "Carry (plain)": "#3d6fb4",
          "Trend 250d": "#b4553d"}
for name, r in curves.items():
    eq = (1 + r).cumprod()
    ax[0].plot(eq.index, eq, label=name, lw=1.6, color=colors[name])
    ax[1].plot(eq.index, 100 * (eq / eq.cummax() - 1), lw=1.2, color=colors[name])

for a in ax:
    a.axvline(OOS, color="#888", ls="--", lw=1.2)
    a.grid(alpha=0.25, lw=0.6)
ax[0].set_yscale("log")
ax[0].set_ylabel("Equity (log, 1.0 = start)")
ax[0].set_title("G10 FX carry vs trend — total return, 10% vol target, costs charged\n"
                "dashed line = start of out-of-sample period (params fixed before it)",
                fontsize=11)
ax[0].legend(loc="upper left", frameon=False)
ax[0].text(OOS, ax[0].get_ylim()[1] * 0.97, "  out-of-sample →",
           va="top", fontsize=9, color="#555")
ax[1].set_ylabel("Drawdown %")
ax[1].set_xlabel("")
fig.tight_layout()
fig.savefig("/home/user/OOO/fx/carry_vs_trend.png", dpi=130)
print("wrote fx/carry_vs_trend.png")

# Gold: the regime chart that explains the 1,031% report.
try:
    h1 = data.load_xau("H1")
    cfg = engine.ExecConfig(entry_valid_bars=5, min_stop_dist=10.0, sizing="fixed",
                            lots=0.02, trail_arm_r=1.5, trail_offset=2.0,
                            spread_points=14.0, commission_per_lot=47.0,
                            start_equity=500.0)
    sigs = strategies.emacross(h1, 9, 15, use_vwap=True, entries="E1")
    tr, cv = engine.run(h1, sigs, XAUUSD, cfg)
    tf = engine.trades_frame(tr)
    fig, ax2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1]})
    ax2[0].plot(h1.index, h1["close"], color="#c9a227", lw=0.8)
    ax2[0].set_ylabel("XAUUSD")
    ax2[0].set_title("emacross159 (EMA9/15 + VWAP) on XAUUSD H1, 2008–2026\n"
                     "the published +1,031% came from the shaded window alone",
                     fontsize=11)
    ax2[1].plot(cv.index, cv["equity"], color="#333", lw=1.2)
    ax2[1].axhline(500, color="#888", ls=":", lw=1)
    ax2[1].set_ylabel("Equity $ (0.02 lots, $500)")
    for a in ax2:
        a.axvspan(pd.Timestamp("2025-03-10"), pd.Timestamp("2026-08-06"),
                  color="#4a90d9", alpha=0.16)
        a.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig("/home/user/OOO/fx/gold_regime.png", dpi=130)
    print("wrote fx/gold_regime.png")
    print(f"  H1 net ${tf['net'].sum():,.0f} over {len(tf)} trades")
except Exception as e:
    print(f"  ! gold chart skipped: {type(e).__name__}: {e}")
