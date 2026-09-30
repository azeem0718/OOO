"""Cross-market strategy sweep with a hard in-sample / out-of-sample wall.

Method, chosen so the OOS numbers mean something:

  * Parameters are chosen ONCE per strategy family, on the pooled in-sample
    performance across every market. No per-market tuning -- that is where
    most "robust" FX systems quietly die.
  * The chosen set is then run untouched on the out-of-sample years.
  * Every parameter combination tried counts as a trial, and the trial count
    feeds the deflated-Sharpe bar. We report that bar next to the result.
  * Position sizing is a fixed fraction of equity risked per trade, so a
    $/pip-heavy pair like GBPJPY cannot dominate the portfolio by accident.
"""
from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
warnings.filterwarnings("ignore")

from fx import costs, data, engine, metrics, strategies, validate

OOS_START = "2017-01-01"
RISK_PCT = 0.01
START_EQ = 10_000.0

# ── markets ──────────────────────────────────────────────────────────────
def load_markets() -> dict[str, pd.DataFrame]:
    m = data.load_all()
    try:
        g = data.load_xau("D1")
        g.index = g.index.normalize()
        m["XAUUSD"] = g
    except Exception as e:
        print(f"  ! XAUUSD D1 unavailable: {e}")
    return m


def conv_for(sym: str, mkts: dict[str, pd.DataFrame]) -> pd.Series | None:
    """Quote-currency -> USD series, needed only for crosses."""
    quote = sym[3:] if sym != "XAUUSD" else "USD"
    if quote == "USD":
        return None
    if sym[:3] == "USD":
        return None  # engine divides by the pair's own rate
    if quote == "JPY" and "USDJPY" in mkts:
        return 1.0 / mkts["USDJPY"]["close"]
    if quote == "GBP" and "GBPUSD" in mkts:
        return mkts["GBPUSD"]["close"]
    return None


# ── parameter grids (deliberately small: every combo is a trial) ──────────
def grid(fam: str) -> list[dict]:
    if fam == "donchian":
        return [dict(lookback=lb, atr_mult=am)
                for lb, am in itertools.product((20, 55, 100), (2.0, 3.0))]
    if fam == "ts_momentum":
        return [dict(lookback=lb, atr_mult=3.0, rebalance=20)
                for lb in (60, 120, 250)]
    if fam == "bollinger_revert":
        return [dict(n=20, k=k, atr_mult=2.0) for k in (2.0, 2.5)]
    if fam == "atr_breakout":
        return [dict(entry_mult=em, stop_mult=sm, trend_ma=200)
                for em, sm in itertools.product((0.5, 1.0), (2.0, 3.0))]
    if fam == "emacross":
        return [dict(fast=f, slow=s, use_vwap=False, entries="E1")
                for f, s in ((9, 15), (12, 26))]
    raise KeyError(fam)


EXITS = {
    "trail1.5R": dict(trail_arm_r=1.5, trail_offset=2.0, take_profit_r=0.0),
    "target2R":  dict(trail_arm_r=0.0, trail_offset=0.0, take_profit_r=2.0),
}


def run_market(df: pd.DataFrame, sym: str, fam: str, params: dict,
               exit_kw: dict, mkts: dict) -> tuple[dict, pd.Series]:
    """One backtest. Returns (summary, daily equity returns)."""
    spec = costs.spec_for(sym)
    cfg = engine.ExecConfig(
        entry_valid_bars=3, sizing="risk_pct", risk_pct=RISK_PCT,
        start_equity=START_EQ, min_stop_dist=0.0, lot_step=0.0,
        min_lots=0.0, max_lots=1e9, **exit_kw)
    sigs = strategies.REGISTRY[fam](df, **params)
    tr, cv = engine.run(df, sigs, spec, cfg, conv=conv_for(sym, mkts))
    tf = engine.trades_frame(tr)
    s = metrics.summarize(tf, cv, START_EQ, f"{sym}:{fam}")
    rets = cv["equity"].pct_change().replace([np.inf, -np.inf], np.nan)
    return s, rets


def portfolio_sharpe(ret_map: dict[str, pd.Series]) -> tuple[float, pd.Series]:
    """Equal-weight portfolio of per-market return streams."""
    if not ret_map:
        return np.nan, pd.Series(dtype=float)
    R = pd.DataFrame(ret_map).sort_index()
    port = R.mean(axis=1, skipna=True).fillna(0.0)
    sd = port.std()
    sr = float(port.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    return sr, port


def main():
    mkts = load_markets()
    print(f"markets: {len(mkts)}  ({', '.join(sorted(mkts))})\n")

    is_map, oos_map = {}, {}
    for sym, df in mkts.items():
        a, b = validate.split_is_oos(df, OOS_START)
        if len(a) > 400 and len(b) > 200:
            is_map[sym], oos_map[sym] = a, b
    print(f"usable markets: {len(is_map)}   IS < {OOS_START} <= OOS\n")

    trials = 0
    chosen: dict[str, dict] = {}
    print("=" * 96)
    print("IN-SAMPLE parameter selection (pooled across all markets, no per-market tuning)")
    print("=" * 96)
    print(f"{'family':<18}{'exit':<11}{'params':<44}{'pooled IS Sharpe':>18}")
    for fam in strategies.REGISTRY:
        best = (None, None, -np.inf)
        for params in grid(fam):
            for ename, ekw in EXITS.items():
                trials += 1
                rets = {}
                for sym, df in is_map.items():
                    try:
                        s, r = run_market(df, sym, fam, params, ekw, mkts)
                        if s["n_trades"] >= 5:
                            rets[sym] = r
                    except Exception:
                        pass
                sr, _ = portfolio_sharpe(rets)
                if np.isfinite(sr) and sr > best[2]:
                    best = (params, ename, sr)
                print(f"{fam:<18}{ename:<11}{str(params):<44}{sr:>18.3f}")
        chosen[fam] = {"params": best[0], "exit": best[1], "is_sharpe": best[2]}
        print(f"{'  -> CHOSEN':<18}{best[1]:<11}{str(best[0]):<44}{best[2]:>18.3f}\n")

    print(f"total parameter combinations tried (trials): {trials}\n")

    print("=" * 96)
    print(f"OUT-OF-SAMPLE ({OOS_START} onward) using the in-sample choice, untouched")
    print("=" * 96)
    results = []
    oos_port = {}
    for fam, ch in chosen.items():
        rets, rows = {}, []
        for sym, df in oos_map.items():
            try:
                s, r = run_market(df, sym, fam, ch["params"], EXITS[ch["exit"]], mkts)
                if s["n_trades"] >= 3:
                    rets[sym] = r
                    rows.append(s)
            except Exception:
                pass
        sr, port = portfolio_sharpe(rets)
        oos_port[fam] = port
        wins = sum(1 for r in rows if r["net"] > 0)
        results.append({
            "family": fam, "exit": ch["exit"], "params": ch["params"],
            "is_sharpe": ch["is_sharpe"], "oos_sharpe": sr,
            "markets": len(rows), "markets_profitable": wins,
            "oos_trades": int(sum(r["n_trades"] for r in rows)),
            "oos_net_pct": float(np.mean([r["ret_pct"] for r in rows])) if rows else np.nan,
            "oos_median_pf": float(np.median([min(r["profit_factor"], 99) for r in rows])) if rows else np.nan,
            "oos_max_dd_pct": float(np.mean([r["max_dd_pct"] for r in rows])) if rows else np.nan,
        })
    res = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
    print(f"{'family':<18}{'exit':<11}{'IS SR':>7}{'OOS SR':>8}{'mkts':>6}"
          f"{'prof':>6}{'trades':>8}{'avg ret%':>10}{'medPF':>7}{'avgDD%':>8}")
    for _, r in res.iterrows():
        print(f"{r['family']:<18}{r['exit']:<11}{r['is_sharpe']:>7.2f}{r['oos_sharpe']:>8.2f}"
              f"{r['markets']:>6}{r['markets_profitable']:>6}{r['oos_trades']:>8}"
              f"{r['oos_net_pct']:>10.1f}{r['oos_median_pf']:>7.2f}{r['oos_max_dd_pct']:>8.1f}")

    # Multiple-testing bar.
    sr_std = float(res["oos_sharpe"].std(ddof=1)) if len(res) > 1 else 0.5
    bar = validate.expected_max_sharpe(trials, sr_std if sr_std > 0 else 0.5)
    print(f"\nMultiple-testing bar: {trials} trials, cross-trial Sharpe sd {sr_std:.3f}")
    print(f"  a worthless strategy reaches OOS Sharpe ~{bar:.2f} by luck alone")
    best = res.iloc[0]
    print(f"  best observed OOS Sharpe: {best['oos_sharpe']:.2f} ({best['family']})")
    print(f"  verdict: {'CLEARS the bar' if best['oos_sharpe'] > bar else 'DOES NOT clear the bar'}")

    # Bootstrap CI on the best family's OOS portfolio.
    port = oos_port[best["family"]]
    ci = validate.block_bootstrap_sharpe(port, n_boot=2000, block=20, ppy=252.0)
    print(f"\nBlock-bootstrap 95% CI on {best['family']} OOS portfolio Sharpe: "
          f"[{ci['lo']:.2f}, {ci['hi']:.2f}]  P(SR>0)={ci['p_gt_0']:.3f}")

    res.to_csv("/home/user/OOO/fx/sweep_results.csv", index=False)
    print("\nwrote fx/sweep_results.csv")


if __name__ == "__main__":
    main()
