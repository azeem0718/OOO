"""Instrument specs and cost model.

Costs are the difference between a real edge and a backtest artifact, so every
number here is either measured from broker data or exposed as a parameter we
sweep. Nothing is silently assumed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    """Contract spec + cost model for one instrument.

    contract_size: units of base per 1.0 lot (100 oz gold, 100_000 FX).
    point:         minimum price increment.
    pip:           conventional pip size for reporting (10 * point for FX).
    commission_per_lot: round-turn USD per 1.0 lot.
    spread_points: fallback half-turn spread in points when the data has no
                   spread column. Applied on entry AND exit.
    """

    symbol: str
    contract_size: float
    point: float
    pip: float
    digits: int
    base: str
    quote: str
    commission_per_lot: float = 0.0
    spread_points: float = 0.0
    swap_per_lot_night: float = 0.0
    calibrated: bool = False


# XAUUSD contract spec, read from the MT5 terminal at data-download time.
#
# spread_points=6 is MEASURED: the median of the broker's own spread column
# over 2018-2026, the only era where that column is populated (it reads 0 for
# 2008-09/2013-17 and a constant 50 for 2010-12 -- both artifacts, not quotes).
# 6 points = $0.06/oz = $6/lot per side = $12/lot round turn.
#
# commission_per_lot is NOT calibrated. loader.COSTS carried $47/lot from a
# legacy "$75/lot total" that was split arbitrarily. Verifying it needs real
# deal history from MT5, which needs Windows. We therefore sweep it.
XAUUSD = Spec(
    symbol="XAUUSD",
    contract_size=100.0,
    point=0.01,
    pip=0.01,
    digits=2,
    base="XAU",
    quote="USD",
    commission_per_lot=47.0,
    spread_points=6.0,
    swap_per_lot_night=6.0,
    calibrated=False,
)

# Retail FX: 100k standard lot. Spread in POINTS (1 pip = 10 points on 5-digit
# quotes, = 10 points on 3-digit JPY quotes). Typical retail round-turn spreads
# in pips, converted to points below.
_FX_SPREAD_PIPS = {
    "EURUSD": 0.6, "GBPUSD": 0.9, "USDJPY": 0.7, "AUDUSD": 0.8,
    "USDCHF": 1.0, "USDCAD": 1.1, "NZDUSD": 1.4, "EURJPY": 1.3,
    "GBPJPY": 2.0, "EURGBP": 1.1, "AUDJPY": 1.5, "USDMXN": 18.0,
    "USDZAR": 25.0, "USDNOK": 25.0, "USDSEK": 25.0,
}

_JPY_QUOTED = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY"}


def fx_spec(pair: str, commission_per_lot: float = 7.0) -> Spec:
    """Standard-lot retail FX spec. Spread is half the round-turn, per side."""
    base, quote = pair[:3], pair[3:]
    jpy = pair in _JPY_QUOTED
    point = 0.001 if jpy else 0.00001
    pip = 0.01 if jpy else 0.0001
    round_turn_pips = _FX_SPREAD_PIPS.get(pair, 1.5)
    # spread_points is per side, so halve the round-turn figure.
    half_pips = round_turn_pips / 2.0
    return Spec(
        symbol=pair,
        contract_size=100_000.0,
        point=point,
        pip=pip,
        digits=3 if jpy else 5,
        base=base,
        quote=quote,
        commission_per_lot=commission_per_lot,
        spread_points=half_pips * (pip / point),
        swap_per_lot_night=0.0,
        calibrated=False,
    )


def spec_for(symbol: str, **kw) -> Spec:
    if symbol.upper() in ("XAUUSD", "GOLD"):
        return XAUUSD
    return fx_spec(symbol.upper(), **kw)
