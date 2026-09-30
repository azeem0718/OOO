"""Strategy library. Each function returns a list of signal dicts.

A signal is {idx, side, trigger|market, stop, kind}. `idx` is the bar the
signal was computed on; the engine arms it from idx+1 onward. No function may
read past `idx` -- that is the whole discipline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ───────────────────────── indicators ─────────────────────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, span: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to each broker trading day, weighted by tick volume.

    Resetting daily is what makes it a session VWAP rather than a cumulative
    all-history mean. Falls back to typical price when volume is absent.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    if "volume" not in df.columns:
        return typical.groupby(df.index.normalize()).transform(
            lambda x: x.expanding().mean())
    vol = df["volume"].replace(0, np.nan).ffill().fillna(1.0)
    day = df.index.normalize()
    pv = (typical * vol).groupby(day).cumsum()
    cv = vol.groupby(day).cumsum()
    return pv / cv


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ───────────────────── 1. emacross159 (ported) ────────────────────────────

def emacross(df: pd.DataFrame, fast: int = 9, slow: int = 15,
             use_vwap: bool = True, entries: str = "E1",
             entry2_mode: str = "touch_reclaim", atr_prox: float = 0.1,
             atr_span: int = 14) -> list[dict]:
    """EMA(fast)/EMA(slow) cross with optional session-VWAP gate.

    Faithful port of strategy_emacross159.generate_signals, generalised to any
    fast/slow pair and timeframe.

    E1: the cross bar itself, closing with the trend. Stop = min(low, EMAslow).
    E2: trend already established, price retraces to an EMA and reclaims it.
    """
    ef = ema(df["close"], fast).to_numpy()
    es = ema(df["close"], slow).to_numpy()
    vw = session_vwap(df).to_numpy() if use_vwap else np.full(len(df), np.nan)
    a = atr(df, atr_span).to_numpy()
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))

    want_e1 = "E1" in entries
    want_e2 = "E2" in entries
    sigs: list[dict] = []

    for i in range(slow + 1, len(df)):
        if not np.isfinite(es[i]):
            continue
        if use_vwap and not np.isfinite(vw[i]):
            continue
        above = (c[i] > vw[i]) if use_vwap else True
        below = (c[i] < vw[i]) if use_vwap else True
        up = ef[i] > es[i] and ef[i - 1] <= es[i - 1]
        dn = ef[i] < es[i] and ef[i - 1] >= es[i - 1]
        green, red = c[i] > o[i], c[i] < o[i]

        if want_e1:
            if above and up and green:
                sigs.append(dict(idx=i, side=1, kind="E1",
                                 trigger=h[i], stop=min(l[i], es[i])))
                continue
            if below and dn and red:
                sigs.append(dict(idx=i, side=-1, kind="E1",
                                 trigger=l[i], stop=max(h[i], es[i])))
                continue

        if not want_e2:
            continue
        t_up, t_dn = ef[i] > es[i], ef[i] < es[i]
        if above and t_up and not up:
            if entry2_mode == "proximity":
                hit = l[i] <= min(ef[i], es[i]) + atr_prox * a[i] and c[i] > o[i]
            else:
                hit = l[i] <= max(ef[i], es[i]) and c[i] > max(ef[i], es[i])
            if hit:
                sigs.append(dict(idx=i, side=1, kind="E2",
                                 trigger=h[i], stop=es[i]))
        elif below and t_dn and not dn:
            if entry2_mode == "proximity":
                hit = h[i] >= max(ef[i], es[i]) - atr_prox * a[i] and c[i] < o[i]
            else:
                hit = h[i] >= min(ef[i], es[i]) and c[i] < min(ef[i], es[i])
            if hit:
                sigs.append(dict(idx=i, side=-1, kind="E2",
                                 trigger=l[i], stop=es[i]))
    return sigs


# ───────────────────── 2. Donchian breakout ───────────────────────────────

def donchian(df: pd.DataFrame, lookback: int = 55, exit_lookback: int = 20,
             atr_span: int = 14, atr_mult: float = 2.0,
             long_only: bool = False) -> list[dict]:
    """Classic channel breakout: buy a new N-bar high, stop at ATR multiple.

    Signal is the prior channel, so the trigger is known before the bar opens.
    """
    hh = df["high"].rolling(lookback).max().shift(1)
    ll = df["low"].rolling(lookback).min().shift(1)
    a = atr(df, atr_span)
    sigs = []
    for i in range(lookback + 1, len(df)):
        up, dn, av = hh.iat[i], ll.iat[i], a.iat[i]
        if not (np.isfinite(up) and np.isfinite(dn) and np.isfinite(av) and av > 0):
            continue
        sigs.append(dict(idx=i - 1, side=1, kind="DCup",
                         trigger=float(up), stop=float(up - atr_mult * av)))
        if not long_only:
            sigs.append(dict(idx=i - 1, side=-1, kind="DCdn",
                             trigger=float(dn), stop=float(dn + atr_mult * av)))
    return sigs


# ───────────────────── 3. Time-series momentum ────────────────────────────

def ts_momentum(df: pd.DataFrame, lookback: int = 60, atr_span: int = 14,
                atr_mult: float = 3.0, rebalance: int = 20) -> list[dict]:
    """Go with the sign of the trailing return; re-evaluate every `rebalance`.

    The canonical FX/managed-futures anomaly. Entered at the next open.
    """
    r = df["close"].pct_change(lookback)
    a = atr(df, atr_span)
    sigs = []
    for i in range(lookback + 1, len(df), max(1, rebalance)):
        v, av = r.iat[i], a.iat[i]
        if not (np.isfinite(v) and np.isfinite(av) and av > 0) or v == 0:
            continue
        side = 1 if v > 0 else -1
        px = float(df["close"].iat[i])
        sigs.append(dict(idx=i, side=side, kind="TSMOM", market=True,
                         trigger=px, stop=float(px - side * atr_mult * av)))
    return sigs


# ───────────────────── 4. Bollinger mean reversion ────────────────────────

def bollinger_revert(df: pd.DataFrame, n: int = 20, k: float = 2.0,
                     atr_span: int = 14, atr_mult: float = 2.0,
                     rsi_n: int = 14, rsi_gate: float = 0.0) -> list[dict]:
    """Fade a close outside the band, stop beyond it. Optional RSI confirmation."""
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std(ddof=0)
    up, lo = m + k * sd, m - k * sd
    a = atr(df, atr_span)
    rs = rsi(df["close"], rsi_n)
    c = df["close"]
    sigs = []
    for i in range(n + 1, len(df)):
        av = a.iat[i]
        if not (np.isfinite(up.iat[i]) and np.isfinite(av) and av > 0):
            continue
        px = float(c.iat[i])
        if c.iat[i] < lo.iat[i] and (rsi_gate <= 0 or rs.iat[i] < rsi_gate):
            sigs.append(dict(idx=i, side=1, kind="BBlong", market=True,
                             trigger=px, stop=float(px - atr_mult * av)))
        elif c.iat[i] > up.iat[i] and (rsi_gate <= 0 or rs.iat[i] > 100 - rsi_gate):
            sigs.append(dict(idx=i, side=-1, kind="BBshort", market=True,
                             trigger=px, stop=float(px + atr_mult * av)))
    return sigs


# ───────────────────── 5. ATR volatility breakout ─────────────────────────

def atr_breakout(df: pd.DataFrame, atr_span: int = 14, entry_mult: float = 1.0,
                 stop_mult: float = 2.0, trend_ma: int = 200) -> list[dict]:
    """Buy prior close + k*ATR, filtered by a long-term trend MA."""
    a = atr(df, atr_span)
    ma = df["close"].rolling(trend_ma).mean() if trend_ma else None
    c = df["close"]
    sigs = []
    for i in range(max(atr_span, trend_ma or 0) + 1, len(df)):
        av = a.iat[i - 1]
        if not (np.isfinite(av) and av > 0):
            continue
        ref = float(c.iat[i - 1])
        bull = True if ma is None else (np.isfinite(ma.iat[i - 1]) and ref > ma.iat[i - 1])
        if bull:
            trig = ref + entry_mult * av
            sigs.append(dict(idx=i - 1, side=1, kind="ATRup",
                             trigger=trig, stop=float(trig - stop_mult * av)))
        else:
            trig = ref - entry_mult * av
            sigs.append(dict(idx=i - 1, side=-1, kind="ATRdn",
                             trigger=trig, stop=float(trig + stop_mult * av)))
    return sigs


REGISTRY = {
    "emacross": emacross,
    "donchian": donchian,
    "ts_momentum": ts_momentum,
    "bollinger_revert": bollinger_revert,
    "atr_breakout": atr_breakout,
}
