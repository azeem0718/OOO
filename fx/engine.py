"""Bar-by-bar backtest engine for stop-entry, R-multiple-managed trades.

Design constraints, all chosen to avoid flattering the result:

1. NO LOOK-AHEAD. A signal is computed from bars <= i and can only fill on
   bar i+1 or later. The engine never reads bar i+1 while deciding on bar i.
2. PESSIMISTIC INTRABAR. Within one bar we cannot know whether the high or the
   low came first. If a bar could have hit both the stop and the target, we
   always resolve it as the stop. This understates results; the opposite
   assumption manufactures them.
3. GAP-HONEST FILLS. A stop order that gaps through its trigger fills at the
   bar's open, not the trigger price. A protective stop that gaps fills at the
   open too. Both cost money, which is what happens live.
4. COSTS ON BOTH SIDES. Half-turn spread is paid at entry and at exit, plus
   round-turn commission and per-night swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import Spec


@dataclass
class ExecConfig:
    """Execution and money-management rules."""

    # Entry
    entry_valid_bars: int = 5        # stop order expires after N bars
    min_stop_dist: float = 0.0       # skip setups with a tighter stop (price units)
    max_stop_dist: float = np.inf    # skip absurdly wide stops

    # Sizing: fixed lots, or risk a fraction of equity per trade
    sizing: str = "fixed"            # "fixed" | "risk_pct"
    lots: float = 0.02
    risk_pct: float = 0.01
    max_lots: float = 100.0
    min_lots: float = 0.01
    lot_step: float = 0.01

    # Exit management
    trail_arm_r: float = 1.5         # arm trail once peak R >= this
    trail_offset: float = 2.0        # stop = (floor(peakR) - offset) * R
    take_profit_r: float = 0.0       # 0 disables a hard target
    scale_frac: float = 0.0          # fraction closed at take_profit_r
    time_stop_bars: int = 0          # 0 disables

    # Costs
    commission_per_lot: float | None = None   # None -> spec value
    spread_points: float | None = None        # None -> data column or spec
    use_data_spread: bool = False             # prefer the broker spread column

    start_equity: float = 10_000.0
    one_position_at_a_time: bool = True


@dataclass
class Trade:
    side: int
    entry_i: int
    entry_time: pd.Timestamp
    entry: float
    stop: float
    initial_stop: float
    lots: float
    exit_i: int | None = None
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    peak_r: float = 0.0
    gross: float = 0.0
    cost: float = 0.0
    net: float = 0.0
    r_multiple: float = 0.0
    bars_held: int = 0
    nights: int = 0
    kind: str = ""
    # Derived at fill time: spread-adjusted entry, and risk in USD.
    entry_eff: float = 0.0
    risk_usd: float = 0.0


def _round_lots(x: float, step: float, lo: float, hi: float) -> float:
    if step <= 0:
        return float(np.clip(x, lo, hi))
    return float(np.clip(np.floor(x / step) * step, lo, hi))


def _quote_to_usd(pnl_quote: np.ndarray | float, spec: Spec, price: float,
                  conv: float | None) -> float:
    """Convert quote-currency PnL to USD.

    USD-quoted (EURUSD, XAUUSD): already USD.
    USD-based  (USDJPY):         divide by the pair's own rate.
    Cross      (EURJPY):         needs an external quote-ccy/USD rate.
    """
    if spec.quote == "USD":
        return float(pnl_quote)
    if spec.base == "USD":
        return float(pnl_quote) / price
    if conv is None or not np.isfinite(conv) or conv <= 0:
        return float("nan")
    return float(pnl_quote) * conv


def run(
    bars: pd.DataFrame,
    signals: list[dict],
    spec: Spec,
    cfg: ExecConfig,
    conv: pd.Series | None = None,
) -> tuple[list[Trade], pd.DataFrame]:
    """Execute `signals` over `bars`.

    signals: dicts with idx, side, trigger, stop, and optional kind. `idx` is
    the bar the signal was computed on; the order is live from idx+1.
    conv: quote-currency -> USD rate aligned to bars.index, for cross pairs.

    Returns (trades, equity_curve).
    """
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    idx = bars.index
    n = len(bars)

    # Per-bar half-turn spread in price units.
    if cfg.spread_points is not None:
        sp_pts = np.full(n, float(cfg.spread_points))
    elif cfg.use_data_spread and "spread" in bars.columns:
        raw = bars["spread"].to_numpy(float)
        # The broker column reads 0 where it was never populated; those are
        # missing values, not free trades. Fall back to the spec there.
        raw = np.where(raw > 0, raw, spec.spread_points)
        sp_pts = raw
    else:
        sp_pts = np.full(n, float(spec.spread_points))
    half_spread = sp_pts * spec.point

    comm = spec.commission_per_lot if cfg.commission_per_lot is None else cfg.commission_per_lot
    conv_arr = None
    if conv is not None:
        conv_arr = conv.reindex(idx).ffill().to_numpy(float)

    # Pending stop orders keyed by the bar they become live.
    pend: dict[int, list[dict]] = {}
    for s in signals:
        first = int(s["idx"]) + 1
        if first < n:
            pend.setdefault(first, []).append(s)

    trades: list[Trade] = []
    equity = cfg.start_equity
    eq_curve = np.full(n, np.nan)

    open_tr: Trade | None = None
    live: list[dict] = []          # armed orders: {sig, expires_at}

    def _close_if_hit(t: Trade, i: int, entry_bar: bool = False) -> None:
        """Set exit fields if bar i resolves the trade. Stop beats target."""
        risk = abs(t.entry - t.initial_stop)
        tp = (t.entry + t.side * cfg.take_profit_r * risk
              if cfg.take_profit_r > 0 and risk > 0 else None)
        hit_stop = (l[i] <= t.stop) if t.side > 0 else (h[i] >= t.stop)
        hit_tp = False if tp is None else ((h[i] >= tp) if t.side > 0 else (l[i] <= tp))
        if hit_stop:
            # A gap through the stop fills at the open, which is worse. On the
            # entry bar the fill already happened at/after the open, so the
            # stop level itself is the honest fill.
            gapped = (not entry_bar) and ((o[i] < t.stop) if t.side > 0 else (o[i] > t.stop))
            t.exit_i, t.exit_time = i, idx[i]
            t.exit, t.reason = float(o[i] if gapped else t.stop), "stop"
        elif hit_tp:
            t.exit_i, t.exit_time, t.exit, t.reason = i, idx[i], float(tp), "target"
        elif cfg.time_stop_bars and (i - t.entry_i) >= cfg.time_stop_bars:
            t.exit_i, t.exit_time, t.exit, t.reason = i, idx[i], float(c[i]), "time"

    def _book(t: Trade, i: int) -> None:
        """Compute costs/PnL for a resolved trade and bank it."""
        nonlocal equity
        t.bars_held = i - t.entry_i
        eff_exit = t.exit - t.side * half_spread[i]
        gross_q = (eff_exit - t.entry_eff) * t.side * t.lots * spec.contract_size
        cv = None if conv_arr is None else conv_arr[i]
        t.gross = _quote_to_usd(gross_q, spec, float(t.exit), cv)
        t.nights = int(max(0, (idx[i].normalize() - idx[t.entry_i].normalize()).days))
        t.cost = t.lots * (comm + spec.swap_per_lot_night * t.nights)
        t.net = t.gross - t.cost
        t.r_multiple = (t.net / t.risk_usd) if t.risk_usd > 0 else 0.0
        equity += t.net
        trades.append(t)

    for i in range(n):
        # ---- 1. arm orders whose signal bar has closed -------------------
        for s in pend.get(i, []):
            ref = o[i] if s.get("market") else s["trigger"]
            dist = abs(ref - s["stop"])
            if dist < cfg.min_stop_dist or dist > cfg.max_stop_dist:
                continue
            live.append({"sig": s, "expires_at": i + cfg.entry_valid_bars})

        # ---- 2. exit check, using the stop IN FORCE from prior bars ------
        # The protective stop may only reflect information through bar i-1.
        # Updating it from bar i's own high and then testing it against bar
        # i's low is hindsight: it lets a trade exit at a price that occurred
        # before the extreme which justified the new stop.
        if open_tr is not None:
            _close_if_hit(open_tr, i)
            if open_tr.exit_i is not None:
                _book(open_tr, i)
                open_tr = None

        # ---- 3. look for an entry fill on THIS bar -----------------------
        if open_tr is None and live:
            still: list[dict] = []
            filled = None
            for od in live:
                s = od["sig"]
                trig, side = float(s["trigger"]), int(s["side"])
                # market=True enters at the next open instead of on a stop.
                if s.get("market"):
                    if filled is None:
                        filled = (s, float(o[i]))
                    continue
                touched = (h[i] >= trig) if side > 0 else (l[i] <= trig)
                if touched and filled is None:
                    gapped = (o[i] > trig) if side > 0 else (o[i] < trig)
                    filled = (s, float(o[i] if gapped else trig))
                elif i <= od["expires_at"]:
                    still.append(od)
            live = still

            if filled is not None:
                s, fill = filled
                side = int(s["side"])
                stop0 = float(s["stop"])
                risk_px = abs(fill - stop0)
                if risk_px > 0:
                    entry_eff = fill + side * half_spread[i]
                    if cfg.sizing == "risk_pct":
                        risk_q = risk_px * spec.contract_size
                        r_usd_per_lot = _quote_to_usd(
                            risk_q, spec, fill,
                            None if conv_arr is None else conv_arr[i])
                        lots = (
                            _round_lots(equity * cfg.risk_pct / r_usd_per_lot,
                                        cfg.lot_step, cfg.min_lots, cfg.max_lots)
                            if r_usd_per_lot and np.isfinite(r_usd_per_lot)
                            and r_usd_per_lot > 0 else 0.0)
                    else:
                        lots = cfg.lots
                    if lots > 0:
                        t = Trade(side=side, entry_i=i, entry_time=idx[i],
                                  entry=fill, stop=stop0, initial_stop=stop0,
                                  lots=lots, kind=str(s.get("kind", "")))
                        t.entry_eff = entry_eff
                        risk_q = risk_px * lots * spec.contract_size
                        t.risk_usd = _quote_to_usd(
                            risk_q, spec, fill,
                            None if conv_arr is None else conv_arr[i])
                        open_tr = t
                        live = []   # one position at a time

                        # Same-bar adverse move. We cannot know whether the
                        # adverse extreme came before or after the fill, so we
                        # take the pessimistic reading and stop out here.
                        _close_if_hit(open_tr, i, entry_bar=True)
                        if open_tr.exit_i is not None:
                            _book(open_tr, i)
                            open_tr = None

        # ---- 4. advance the trail for FUTURE bars ------------------------
        if open_tr is not None:
            t = open_tr
            risk = abs(t.entry - t.initial_stop)
            fav = (h[i] - t.entry) if t.side > 0 else (t.entry - l[i])
            if risk > 0:
                t.peak_r = max(t.peak_r, fav / risk)
                if cfg.trail_arm_r > 0 and t.peak_r >= cfg.trail_arm_r:
                    steps = np.floor(t.peak_r) - cfg.trail_offset
                    cand = t.entry + t.side * steps * risk
                    t.stop = max(t.stop, cand) if t.side > 0 else min(t.stop, cand)

        # ---- 5. expire stale orders --------------------------------------
        live = [od for od in live if i <= od["expires_at"]]
        eq_curve[i] = equity

    # Mark an unclosed position to market so the curve is complete.
    if open_tr is not None:
        t = open_tr
        t.exit_i, t.exit_time, t.exit, t.reason = n - 1, idx[-1], float(c[-1]), "open_at_end"
        t.bars_held = (n - 1) - t.entry_i
        eff_exit = c[-1] - t.side * half_spread[-1]
        gross_q = (eff_exit - t.entry_eff) * t.side * t.lots * spec.contract_size
        cv = None if conv_arr is None else conv_arr[-1]
        t.gross = _quote_to_usd(gross_q, spec, float(c[-1]), cv)
        t.cost = t.lots * comm
        t.net = t.gross - t.cost
        t.r_multiple = (t.net / t.risk_usd) if t.risk_usd > 0 else 0.0
        equity += t.net
        eq_curve[-1] = equity
        trades.append(t)

    curve = pd.DataFrame({"equity": eq_curve}, index=idx).ffill()
    curve["equity"] = curve["equity"].fillna(cfg.start_equity)
    return trades, curve


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "side", "kind", "entry", "exit",
                     "initial_stop", "lots", "gross", "cost", "net", "r_multiple",
                     "peak_r", "bars_held", "reason"]
        )
    return pd.DataFrame([
        {
            "entry_time": t.entry_time, "exit_time": t.exit_time, "side": t.side,
            "kind": t.kind, "entry": t.entry, "exit": t.exit,
            "initial_stop": t.initial_stop, "lots": t.lots, "gross": t.gross,
            "cost": t.cost, "net": t.net, "r_multiple": t.r_multiple,
            "peak_r": t.peak_r, "bars_held": t.bars_held, "reason": t.reason,
        }
        for t in trades
    ])
