# FX & gold strategy research

A backtesting stack built to answer one question honestly: **which of these
strategies survives contact with out-of-sample data and real costs?**

## Layout

| file | role |
|---|---|
| `data.py` | Yahoo daily OHLC (15 pairs), FRED policy rates, broker CSV loader, bar sanitiser |
| `costs.py` | contract specs and cost models; XAUUSD spread **measured** from broker data |
| `engine.py` | bar-by-bar stop-entry engine with R-multiple trade management |
| `test_engine.py` | 26 hand-computed assertions — run this before trusting any result |
| `strategies.py` | emacross159 (ported), Donchian, TSMOM, Bollinger, ATR breakout |
| `portfolio.py` | total-return portfolio sim for carry/trend (spot **+** interest differential) |
| `validate.py` | IS/OOS split, walk-forward, block bootstrap, deflated-Sharpe bar |
| `sweep.py` | cross-market IS/OOS sweep |
| `anomalies.py` | carry vs trend |
| `carry_study.py` | carry stability and motivated variants |

## Running

```bash
python fx/test_engine.py                       # always first
python fx/sweep.py                             # technical strategies, 16 markets
python fx/anomalies.py                         # carry vs trend
python fx/carry_study.py                       # carry deep-dive
XAU_DATA_DIR=/path/to/XAUUSD/data python fx/regime_test.py
```

Broker CSVs are **not** in this repo — the archive they came from also carries
credentials. Point `$XAU_DATA_DIR` at them locally.

## Engine guarantees

Each is asserted in `test_engine.py`:

1. **No look-ahead.** A signal computed on bar *i* can only fill on *i+1* or later.
2. **Stops use prior-bar information only.** A trailing stop may not be raised
   using the same bar's high and then tested against that bar's low.
3. **Pessimistic intrabar.** If a bar could have hit both stop and target, it
   resolves as the stop.
4. **Gap-honest fills.** Orders gapping through their level fill at the open.
5. **Costs both sides.** Half-turn spread on entry and exit, plus commission
   and swap.

Rule 2 was a real bug caught by these tests; it had been inflating results.
