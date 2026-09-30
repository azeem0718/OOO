"""Round-2 chart: the plateau and the walk-forward record."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import portfolio as pf
from research import universe as U

START, OOS = pd.Timestamp("2006-06-01"), pd.Timestamp("2017-01-01")
rates = U.load_rates()
prices, idx = U.load_prices(U.G10)
idx = idx[idx >= START]
prices = {k: v.reindex(idx).ffill() for k, v in prices.items()}
pairs = list(prices)
rets = pf.total_returns(prices, rates)

basket = pf.spot_returns(prices).reindex(idx).mean(axis=1)
rv = basket.rolling(60).std() * np.sqrt(252)
th = rv.rolling(504, min_periods=252).quantile(0.80)
calm = (rv.shift(1) <= th.shift(1)).astype(float).fillna(1.0)
r = rates.shift(pf.RATE_LAG_DAYS).reindex(idx).ffill()
diff = pd.DataFrame({p: (r[p[:3]] - r[p[3:]]) for p in pairs}, index=idx)
w = np.sign(diff).resample("ME").last().reindex(idx).ffill().fillna(0.0).mul(calm, axis=0)

sim = pf.simulate(w, rets, prices, target_vol=0.10)
eq = (1 + sim["ret"]).cumprod()

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.28, wspace=0.22)

ax = fig.add_subplot(gs[0, :])
ax.plot(eq.index, eq, color="#1b7f5a", lw=1.7)
ax.axvline(OOS, color="#888", ls="--", lw=1.2)
ax.set_yscale("log")
ax.set_ylabel("Equity (log)")
ax.grid(alpha=0.25, lw=0.6)
ax.set_title("G10 crash-filtered carry — total return, 10% vol target, costs charged\n"
             "real but modest: walk-forward Sharpe 0.31, not the 0.67 a fixed split flatters it to",
             fontsize=11)
ax.text(OOS, eq.max() * 0.98, "  out-of-sample →", va="top", fontsize=9, color="#555")

# Rate-lag robustness: the anti-leakage test.
ax2 = fig.add_subplot(gs[1, 0])
lags, srs = [1, 5, 21, 42], [0.68, 0.67, 0.59, 0.54]
ax2.bar([str(x) for x in lags], srs, color="#1b7f5a", alpha=0.85)
ax2.axhline(0, color="#333", lw=0.8)
ax2.set_xlabel("Signal delayed by N days")
ax2.set_ylabel("OOS Sharpe")
ax2.set_title("Degrades gracefully, never collapses\n→ signal, not look-ahead", fontsize=10)
ax2.grid(alpha=0.25, lw=0.6, axis="y")

# Walk-forward windows.
ax3 = fig.add_subplot(gs[1, 1])
wins = ["2013", "2015", "2017", "2019", "2021", "2023", "2025", "2026"]
vals = [0.81, -0.35, 0.21, 0.69, 0.48, -0.02, -0.20, 0.89]
ax3.bar(wins, vals, color=["#1b7f5a" if v > 0 else "#b4553d" for v in vals], alpha=0.85)
ax3.axhline(0, color="#333", lw=0.8)
ax3.axhline(np.mean(vals), color="#333", ls=":", lw=1.2)
ax3.set_xlabel("Walk-forward test window ends")
ax3.set_ylabel("OOS Sharpe")
ax3.set_title(f"5/8 windows positive, mean {np.mean(vals):.2f}\n"
              "expect multi-year losing stretches", fontsize=10)
ax3.grid(alpha=0.25, lw=0.6, axis="y")

fig.savefig("/home/user/OOO/research/carry_robustness.png", dpi=130, bbox_inches="tight")
print("wrote research/carry_robustness.png")
