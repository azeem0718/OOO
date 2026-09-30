"""Hand-computed assertions on the engine.

Every number in the expected values below is derived by hand in the comment
above it. If the engine drifts, these fail loudly rather than quietly changing
a research conclusion.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/OOO")
from fx import engine
from fx.costs import Spec

# Frictionless gold-like instrument so arithmetic is exact and checkable.
FREE = Spec(symbol="TEST", contract_size=100.0, point=0.01, pip=0.01, digits=2,
            base="XAU", quote="USD", commission_per_lot=0.0, spread_points=0.0,
            swap_per_lot_night=0.0)


def bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def cfg(**kw):
    base = dict(entry_valid_bars=5, lots=1.0, trail_arm_r=0.0, trail_offset=0.0,
                spread_points=0.0, commission_per_lot=0.0, start_equity=10_000.0)
    base.update(kw)
    return engine.ExecConfig(**base)


fails = []


def check(name, got, want, tol=1e-6):
    ok = (want is None and got is None) or (
        got is not None and want is not None and abs(got - want) <= tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok:
        fails.append(name)


# ── 1. Basic long: fill at trigger, exit at stop ────────────────────────────
# Signal on bar0 (trigger 100, stop 99). Bar1 h=101 >= 100 -> fill @100.
# Bar2 l=98 <= 99 -> stop exit @99 (open 100 did not gap below 99).
# gross = (99-100)*1*1.0*100 = -100.00 ; risk = (100-99)*1*100 = 100 -> R = -1
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [100, 100.5, 98, 98.5]])
tr, cv = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE, cfg())
check("1a n_trades", len(tr), 1)
check("1b entry", tr[0].entry, 100.0)
check("1c exit", tr[0].exit, 99.0)
check("1d net", tr[0].net, -100.0)
check("1e R", tr[0].r_multiple, -1.0)

# ── 2. Pessimistic intrabar: bar hits BOTH stop and target -> stop wins ─────
# Entry @100 on bar1, stop 99, TP at 2R = 102. Bar2 spans 98..103: both hit.
# Engine must resolve as the stop: net = -100.
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [100, 103, 98, 102]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE,
                   cfg(take_profit_r=2.0))
check("2a reason is stop", 1.0 if tr[0].reason == "stop" else 0.0, 1.0)
check("2b net", tr[0].net, -100.0)

# ── 3. Target-only bar resolves as target ──────────────────────────────────
# Same setup but bar2 low 99.5 never reaches the stop; high 103 >= 102 -> TP.
# gross = (102-100)*100 = +200 -> R = +2
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [100, 103, 99.5, 102]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE,
                   cfg(take_profit_r=2.0))
check("3a reason is target", 1.0 if tr[0].reason == "target" else 0.0, 1.0)
check("3b net", tr[0].net, 200.0)
check("3c R", tr[0].r_multiple, 2.0)

# ── 4. Entry gap: bar opens ABOVE the buy-stop trigger -> fill at open ──────
# Trigger 100 but bar1 opens at 102 -> fill 102, not 100. Stop 99.
# Bar2 l=98 -> stop @99. gross = (99-102)*100 = -300.
df = bars([[100, 100, 100, 100], [102, 103, 101.5, 102.5], [102, 102.5, 98, 98.5]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE, cfg())
check("4a gap entry fill", tr[0].entry, 102.0)
check("4b net", tr[0].net, -300.0)

# ── 5. Exit gap: price gaps THROUGH the protective stop -> fill at open ─────
# Entry @100 bar1, stop 99. Bar2 opens 95 (below 99) -> exit @95, not 99.
# gross = (95-100)*100 = -500.
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [95, 96, 94, 95.5]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE, cfg())
check("5a gap exit fill", tr[0].exit, 95.0)
check("5b net", tr[0].net, -500.0)

# ── 6. Order expiry: never touched within entry_valid_bars -> no trade ──────
df = bars([[100, 100, 100, 100]] + [[90, 91, 89, 90]] * 8)
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE,
                   cfg(entry_valid_bars=3))
check("6a expired, no trade", len(tr), 0)

# ── 7. No look-ahead: a signal cannot fill on its own bar ──────────────────
# Bar0 itself trades up to 105, but the order is only live from bar1, and bars
# 1+ stay at 90. A look-ahead bug would fill on bar0.
df = bars([[100, 105, 100, 104]] + [[90, 91, 89, 90]] * 6)
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE, cfg())
check("7a no same-bar fill", len(tr), 0)

# ── 8. Costs charged on BOTH sides ─────────────────────────────────────────
# spread_points=6 (=$0.06/oz), commission 47/lot, 1.0 lot.
# Entry eff = 100 + 0.06 = 100.06 ; exit eff = 99 - 0.06 = 98.94
# gross = (98.94-100.06)*100 = -112.00 ; cost = 47 -> net = -159.00
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [100, 100.5, 98, 98.5]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE,
                   cfg(spread_points=6.0, commission_per_lot=47.0))
check("8a gross w/ spread", tr[0].gross, -112.0, tol=1e-6)
check("8b cost", tr[0].cost, 47.0)
check("8c net", tr[0].net, -159.0, tol=1e-6)

# ── 9. Short mirror ────────────────────────────────────────────────────────
# Sell stop trigger 100, stop 101. Bar1 l=99 -> fill @100. Bar2 h=102 -> stop @101.
# gross = (101-100)*(-1)*100 = -100.
df = bars([[100, 100, 100, 100], [100, 100.5, 99, 99.5], [100, 102, 99.5, 101.5]])
tr, _ = engine.run(df, [dict(idx=0, side=-1, trigger=100.0, stop=101.0)], FREE, cfg())
check("9a short entry", tr[0].entry, 100.0)
check("9b short net", tr[0].net, -100.0)

# ── 10. Ratcheting trail: peak 3R -> stop moves to +1R, exit there ─────────
# Entry @100, risk 1.0 (stop 99). trail_arm_r=1.5, offset=2.
# Bar2 high 103 -> peak_r=3 -> steps=floor(3)-2=1 -> stop=100+1*1=101.
# Bar2 low 100.5 does not hit 101 (stop set same bar, checked next bar).
# Bar3 low 100 <= 101 -> exit @101. gross=(101-100)*100=+100 -> R=+1.
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5],
           [100.5, 103, 100.5, 102.5], [102, 102, 100, 100.5]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.0)], FREE,
                   cfg(trail_arm_r=1.5, trail_offset=2.0))
check("10a trail exit px", tr[0].exit, 101.0)
check("10b trail net", tr[0].net, 100.0)
check("10c peak_r", round(tr[0].peak_r, 4), 3.0)

# ── 11. min_stop_dist filter skips tight setups ────────────────────────────
df = bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5], [100, 100.5, 98, 98.5]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=100.0, stop=99.9)], FREE,
                   cfg(min_stop_dist=1.0))
check("11a tight setup skipped", len(tr), 0)

# ── 12. USD-based pair conversion (USDJPY-like): PnL/exit_price ────────────
# 100k contract, 1 lot, entry 150.00 stop 149.00, exit at stop.
# gross_jpy = (149-150)*100000 = -100_000 JPY ; /149 = -671.14 USD
JPY = Spec(symbol="USDJPY", contract_size=100_000.0, point=0.001, pip=0.01,
           digits=3, base="USD", quote="JPY")
df = bars([[150, 150, 150, 150], [150, 150.5, 149.5, 150.2], [150, 150.2, 148.5, 148.8]])
tr, _ = engine.run(df, [dict(idx=0, side=1, trigger=150.0, stop=149.0)], JPY, cfg())
check("12a usdjpy net USD", round(tr[0].net, 2), round(-100_000 / 149.0, 2), tol=0.01)

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("ALL ENGINE TESTS PASSED")
