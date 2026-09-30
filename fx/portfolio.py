"""Portfolio-level total-return simulation for carry and trend.

Why this exists alongside engine.py: carry and volatility-targeted trend are
continuous-weight, portfolio-level strategies whose return comes partly from
the interest differential. A stop-and-target trade engine cannot represent
them. Running them through one measures spot direction instead of the anomaly.

Conventions:
  * A weight of +1 on EURUSD means long 1 unit of notional EUR funded in USD.
  * Total return of that position = spot return + (r_EUR - r_USD)/252.
  * Weights are ALWAYS lagged one day before being applied to returns.
  * Costs are charged on |change in weight| at the pair's half-turn spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import _FX_SPREAD_PIPS, fx_spec

# Rates are published with a lag and monthly series are stamped at month start,
# so we shift before use. Differentials are highly persistent, making this
# nearly free -- and it removes any doubt about same-day leakage.
RATE_LAG_DAYS = 5


def carry_daily(pair: str, rates: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Daily interest differential earned by a +1 position in `pair`."""
    base, quote = pair[:3], pair[3:]
    if base not in rates.columns or quote not in rates.columns:
        return pd.Series(0.0, index=index)
    r = rates.shift(RATE_LAG_DAYS).reindex(index).ffill()
    return ((r[base] - r[quote]) / 100.0 / 252.0).fillna(0.0)


def spot_returns(prices: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({k: v.pct_change() for k, v in prices.items()}).sort_index()


def total_returns(prices: dict[str, pd.Series], rates: pd.DataFrame) -> pd.DataFrame:
    """Spot return plus carry, per pair."""
    sp = spot_returns(prices)
    out = {}
    for pair in sp.columns:
        out[pair] = sp[pair].fillna(0.0) + carry_daily(pair, rates, sp.index)
    return pd.DataFrame(out, index=sp.index)


def cost_bp(pair: str, price_level: float) -> float:
    """One-way cost in decimal, from the round-turn spread in pips."""
    spec = fx_spec(pair)
    round_turn_pips = _FX_SPREAD_PIPS.get(pair, 1.5)
    one_way_price = (round_turn_pips / 2.0) * spec.pip
    return one_way_price / price_level if price_level > 0 else 0.0


def simulate(weights: pd.DataFrame, rets: pd.DataFrame,
             prices: dict[str, pd.Series],
             target_vol: float | None = 0.10,
             vol_lookback: int = 60,
             max_leverage: float = 3.0,
             charge_costs: bool = True) -> dict:
    """Run a weight schedule and return the resulting track record.

    target_vol scales the whole book to an annualised volatility estimated from
    trailing realised vol -- estimated on data available at the time, never on
    the full sample.
    """
    w = weights.reindex(rets.index).ffill().fillna(0.0)
    common = [c for c in w.columns if c in rets.columns]
    w, r = w[common], rets[common]

    # Equal-weight the book, then lag so today's weight uses yesterday's signal.
    n = w.abs().sum(axis=1).replace(0, np.nan)
    w_norm = w.div(n, axis=0).fillna(0.0)
    w_lag = w_norm.shift(1).fillna(0.0)

    gross = (w_lag * r).sum(axis=1)

    if target_vol:
        realized = gross.rolling(vol_lookback).std() * np.sqrt(252)
        scale = (target_vol / realized.shift(1)).clip(upper=max_leverage)
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        scale = pd.Series(1.0, index=gross.index)

    net = gross * scale

    if charge_costs:
        turn = (w_lag.diff().abs().fillna(0.0))
        cb = pd.DataFrame(
            {c: [cost_bp(c, p) for p in
                 pd.Series(prices[c]).reindex(rets.index).ffill().bfill().to_numpy()]
             for c in common}, index=rets.index)
        cost = (turn * cb).sum(axis=1) * scale
        net = net - cost
    else:
        cost = pd.Series(0.0, index=net.index)

    eq = (1.0 + net).cumprod()
    return {"ret": net, "equity": eq, "scale": scale, "cost": cost,
            "gross_ret": gross}


def stats(ret: pd.Series, label: str = "") -> dict:
    r = ret.dropna()
    if len(r) < 30:
        return {"label": label, "n": len(r)}
    eq = (1 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    sd = r.std()
    downside = r[r < 0].std()
    return {
        "label": label,
        "n": int(len(r)),
        "years": yrs,
        "cagr": 100.0 * (eq.iloc[-1] ** (1 / yrs) - 1.0),
        "vol": 100.0 * sd * np.sqrt(252),
        "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(252)) if downside and downside > 0 else np.nan,
        "max_dd": 100.0 * dd,
        "calmar": (100.0 * (eq.iloc[-1] ** (1 / yrs) - 1.0)) / abs(100.0 * dd) if dd else np.nan,
        "hit_rate": 100.0 * (r > 0).mean(),
        "best_yr": 100.0 * r.resample("YE").apply(lambda x: (1 + x).prod() - 1).max(),
        "worst_yr": 100.0 * r.resample("YE").apply(lambda x: (1 + x).prod() - 1).min(),
        "pos_years": float((r.resample("YE").apply(lambda x: (1 + x).prod() - 1) > 0).mean()),
    }


# ───────────────────────── weight generators ──────────────────────────────

def w_carry(rates: pd.DataFrame, pairs: list[str], index: pd.DatetimeIndex,
            rebalance: str = "ME", mode: str = "sign") -> pd.DataFrame:
    """Long the high-yielder. 'sign' takes the differential's sign;
    'rank' goes long the top third and short the bottom third."""
    r = rates.shift(RATE_LAG_DAYS).reindex(index).ffill()
    diff = pd.DataFrame(
        {p: (r[p[:3]] - r[p[3:]]) for p in pairs
         if p[:3] in r.columns and p[3:] in r.columns}, index=index)
    if mode == "sign":
        w = np.sign(diff)
    else:
        rk = diff.rank(axis=1, pct=True)
        w = pd.DataFrame(0.0, index=diff.index, columns=diff.columns)
        w[rk >= 2 / 3] = 1.0
        w[rk <= 1 / 3] = -1.0
    return w.resample(rebalance).last().reindex(index).ffill().fillna(0.0)


def w_trend(prices: dict[str, pd.Series], pairs: list[str],
            index: pd.DatetimeIndex, lookback: int = 250,
            rebalance: str = "ME") -> pd.DataFrame:
    """Time-series momentum: sign of the trailing `lookback` return."""
    cols = {}
    for p in pairs:
        if p not in prices:
            continue
        s = pd.Series(prices[p]).reindex(index).ffill()
        cols[p] = np.sign(s.pct_change(lookback))
    w = pd.DataFrame(cols, index=index)
    return w.resample(rebalance).last().reindex(index).ffill().fillna(0.0)
