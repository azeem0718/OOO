"""Performance statistics.

Sharpe is annualised from the equity curve's periodic returns, not from trade
returns, so it reflects time in the market rather than trade frequency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANN = {"D": 252.0, "H1": 252.0 * 23, "H4": 252.0 * 6, "M15": 252.0 * 92,
        "M5": 252.0 * 276, "M30": 252.0 * 46, "M1": 252.0 * 1380}


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualisation factor from the median bar spacing."""
    if len(index) < 3:
        return 252.0
    med = pd.Series(index).diff().dt.total_seconds().median()
    if not med or not np.isfinite(med) or med <= 0:
        return 252.0
    # Trading time, not wall-clock: FX runs ~120h/week of 168.
    bars_per_week = (5 * 24 * 3600) / med
    return float(bars_per_week * 52.0)


def max_drawdown(eq: pd.Series) -> tuple[float, float]:
    """(absolute drawdown, fractional drawdown of peak)."""
    peak = eq.cummax()
    dd = eq - peak
    frac = (dd / peak.replace(0, np.nan)).min()
    return float(dd.min()), float(frac if np.isfinite(frac) else 0.0)


def summarize(trades: pd.DataFrame, curve: pd.DataFrame,
              start_equity: float, label: str = "") -> dict:
    """Headline stats for one backtest run."""
    eq = curve["equity"]
    out: dict = {"label": label, "n_trades": int(len(trades))}

    if len(trades) == 0:
        out.update({k: 0.0 for k in
                    ("net", "ret_pct", "win_rate", "profit_factor", "expectancy",
                     "avg_r", "sharpe", "sortino", "max_dd", "max_dd_pct",
                     "calmar", "cagr")})
        return out

    net = float(trades["net"].sum())
    wins = trades[trades["net"] > 0]["net"]
    losses = trades[trades["net"] < 0]["net"]
    gp, gl = float(wins.sum()), float(-losses.sum())

    out["net"] = net
    out["ret_pct"] = 100.0 * net / start_equity
    out["win_rate"] = 100.0 * len(wins) / len(trades)
    out["profit_factor"] = (gp / gl) if gl > 0 else np.inf
    out["expectancy"] = net / len(trades)
    out["avg_r"] = float(trades["r_multiple"].mean())
    out["median_r"] = float(trades["r_multiple"].median())

    ppy = periods_per_year(curve.index)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) > 2 and rets.std() > 0:
        out["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(ppy))
        downside = rets[rets < 0].std()
        out["sortino"] = float(rets.mean() / downside * np.sqrt(ppy)) if downside and downside > 0 else np.inf
    else:
        out["sharpe"] = 0.0
        out["sortino"] = 0.0

    dd, ddp = max_drawdown(eq)
    out["max_dd"] = dd
    out["max_dd_pct"] = 100.0 * ddp

    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    final = float(eq.iloc[-1])
    out["years"] = years
    out["cagr"] = (100.0 * ((final / start_equity) ** (1 / years) - 1)
                   if final > 0 and start_equity > 0 else -100.0)
    out["calmar"] = (out["cagr"] / abs(out["max_dd_pct"])) if out["max_dd_pct"] else np.inf
    out["trades_per_year"] = len(trades) / years

    # Tail dependence: how much of the edge sits in a handful of trades.
    srt = trades["net"].sort_values(ascending=False)
    for k in (1, 5, 10, 20):
        out[f"net_drop_top{k}"] = float(srt.iloc[k:].sum()) if len(srt) > k else 0.0
    out["top5_share"] = (float(srt.iloc[:5].sum()) / net) if net > 0 else np.nan

    # Break-even cost per trade: how much cost the edge can absorb.
    out["breakeven_cost_per_trade"] = (net / len(trades)) + float(trades["cost"].mean())

    streak = best = 0
    for v in trades["net"]:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    out["longest_loss_streak"] = best
    return out


def equity_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    """Net PnL and trade count per calendar year -- the regime check."""
    if len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    t["year"] = pd.to_datetime(t["exit_time"]).dt.year
    g = t.groupby("year").agg(
        trades=("net", "size"), net=("net", "sum"),
        win_rate=("net", lambda s: 100.0 * (s > 0).mean()),
        avg_r=("r_multiple", "mean"),
    )
    g["cum_net"] = g["net"].cumsum()
    return g
