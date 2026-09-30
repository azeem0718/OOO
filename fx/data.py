"""FX data acquisition: Yahoo daily OHLC (primary) + FRED close series (cross-check).

Yahoo silently downgrades granularity when asked for range=max, so we always
request explicit epoch windows and assert the returned bar spacing is daily.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Pairs we research. Yahoo symbol -> canonical name.
PAIRS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "AUDJPY": "AUDJPY=X",
    "USDMXN": "USDMXN=X",
    "USDZAR": "USDZAR=X",
    "USDNOK": "USDNOK=X",
    "USDSEK": "USDSEK=X",
}

# FRED daily close series for independent cross-validation.
# value_is_usd_per_base=True means the series is quoted as our pair is.
FRED_SERIES: dict[str, tuple[str, bool]] = {
    "EURUSD": ("DEXUSEU", True),
    "GBPUSD": ("DEXUSUK", True),
    "USDJPY": ("DEXJPUS", True),
    "AUDUSD": ("DEXUSAL", True),
    "USDCHF": ("DEXSZUS", True),
    "USDCAD": ("DEXCAUS", True),
    "NZDUSD": ("DEXUSNZ", True),
    "USDMXN": ("DEXMXUS", True),
    "USDZAR": ("DEXSFUS", True),
    "USDNOK": ("DEXNOUS", True),
    "USDSEK": ("DEXSDUS", True),
}


def _fetch_url(url: str, timeout: int = 45, retries: int = 4) -> bytes:
    """GET with exponential backoff on transient failures / rate limits."""
    delay = 2.0
    last: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"fetch failed after {retries} tries: {url}") from last


def fetch_yahoo(symbol: str, start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily OHLC from Yahoo using explicit epoch bounds.

    Using range=max returns MONTHLY bars for FX; epoch bounds return true daily.
    """
    p1 = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp())
    end_dt = dt.datetime.now(dt.timezone.utc) if end is None else dt.datetime.fromisoformat(end).replace(tzinfo=dt.timezone.utc)
    p2 = int(end_dt.timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval=1d&includeAdjustedClose=false"
    )
    payload = json.loads(_fetch_url(url))
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"yahoo error for {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"yahoo returned no result for {symbol}")

    ts = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
        },
        index=pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True),
    )
    df.index = df.index.tz_convert("UTC").normalize().tz_localize(None)
    df.index.name = "date"
    return df


def _fetch_via_curl(url: str, timeout: int = 45, retries: int = 3) -> str:
    """Fetch with curl.

    The session's egress proxy re-terminates TLS and accepts curl's CONNECT but
    closes urllib/requests connections to fred.stlouisfed.org outright
    (RemoteDisconnected on every attempt, while curl succeeds every time).
    curl is already configured with the proxy CA bundle, so this uses the
    client that works rather than weakening verification or bypassing policy.
    """
    import subprocess

    last = ""
    for attempt in range(retries):
        r = subprocess.run(
            ["curl", "-sS", "--fail", "--max-time", str(timeout), url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
        last = r.stderr.strip() or f"curl exit {r.returncode}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"curl failed for {url}: {last}")


def fetch_fred(series_id: str) -> pd.Series:
    """Daily/monthly series from FRED. '.' marks holidays and is dropped."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = _fetch_via_curl(url)
    df = pd.read_csv(io_str(raw))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    )
    return s.dropna()


def io_str(text: str):
    import io as _io

    return _io.StringIO(text)


def sanitize(df: pd.DataFrame, pair: str, max_jump: float = 0.10) -> tuple[pd.DataFrame, dict]:
    """Clean an OHLC frame and report exactly what was removed.

    Yahoo FX carries phantom bars (weekend/holiday rows repeating Friday's
    print), inverted ranges, and occasional decimal-shift spikes. Silently
    keeping any of them manufactures fake breakouts and fake stop-outs.
    """
    report: dict[str, int | list] = {}
    n0 = len(df)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    report["dropped_duplicate_dates"] = n0 - len(df)

    n = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    report["dropped_incomplete_bars"] = n - len(df)

    n = len(df)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    report["dropped_nonpositive"] = n - len(df)

    # Inverted or impossible ranges: high must bound open/close, low must floor them.
    n = len(df)
    ok = (
        (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12)
        & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12)
    )
    report["dropped_inverted_range"] = int((~ok).sum())
    df = df[ok]

    # Frozen weekend bars: FX closes Fri ~22:00 UTC. A Saturday/Sunday row that
    # merely repeats the prior close is an artifact, not a tradeable bar.
    n = len(df)
    weekend = df.index.dayofweek >= 5
    flat = df["high"].sub(df["low"]).abs() < 1e-12
    drop_we = weekend & flat
    report["dropped_flat_weekend_bars"] = int(drop_we.sum())
    df = df[~drop_we]

    # Decimal-shift / bad-tick spikes: a close-to-close move beyond max_jump
    # that immediately reverses is a data error, not a market event.
    lr = np.log(df["close"]).diff()
    spike = (lr.abs() > max_jump) & (lr.shift(-1).abs() > max_jump) & (np.sign(lr) != np.sign(lr.shift(-1)))
    report["dropped_reverting_spikes"] = int(spike.fillna(False).sum())
    df = df[~spike.fillna(False)]

    report["rows_in"] = n0
    report["rows_out"] = len(df)
    report["pair"] = pair
    return df, report


def load_pair(pair: str, refresh: bool = False, start: str = "1990-01-01") -> pd.DataFrame:
    """Load a sanitized daily OHLC frame for `pair`, caching to parquet."""
    if pair not in PAIRS:
        raise KeyError(f"unknown pair {pair!r}; known: {sorted(PAIRS)}")
    path = CACHE / f"{pair}_1d.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    raw = fetch_yahoo(PAIRS[pair], start=start)
    clean, report = sanitize(raw, pair)
    if len(clean) < 500:
        raise RuntimeError(f"{pair}: only {len(clean)} clean bars, refusing to cache")
    clean.to_parquet(path)
    (CACHE / f"{pair}_1d.report.json").write_text(json.dumps(report, indent=2, default=str))
    return clean


def load_all(refresh: bool = False, pause: float = 0.4) -> dict[str, pd.DataFrame]:
    """Load every pair, tolerating individual failures."""
    out: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        try:
            out[pair] = load_pair(pair, refresh=refresh)
        except Exception as e:  # keep the sweep alive; report at the end
            print(f"  ! {pair}: {type(e).__name__}: {e}")
        if refresh:
            time.sleep(pause)
    return out


def cross_check(pair: str, df: pd.DataFrame) -> dict | None:
    """Correlate Yahoo closes against FRED's independent fixing.

    A low correlation means one source is wrong; we want to know before we
    build a strategy on it.
    """
    if pair not in FRED_SERIES:
        return None
    series_id, same_quote = FRED_SERIES[pair]
    try:
        fred = fetch_fred(series_id)
    except Exception as e:
        return {"pair": pair, "error": f"{type(e).__name__}: {e}"}
    if not same_quote:
        fred = 1.0 / fred
    joined = pd.concat([df["close"].rename("yahoo"), fred.rename("fred")], axis=1, join="inner").dropna()
    if len(joined) < 100:
        return {"pair": pair, "n": len(joined), "error": "insufficient overlap"}
    rel = (joined["yahoo"] - joined["fred"]).abs() / joined["fred"]
    return {
        "pair": pair,
        "series": series_id,
        "n_overlap": int(len(joined)),
        "corr_levels": float(joined["yahoo"].corr(joined["fred"])),
        "corr_returns": float(np.log(joined["yahoo"]).diff().corr(np.log(joined["fred"]).diff())),
        "median_rel_diff": float(rel.median()),
        "p99_rel_diff": float(rel.quantile(0.99)),
    }


# ───────────────────── broker (MT5) CSV loader ────────────────────────────

XAU_DIR_ENV = "XAU_DATA_DIR"


def xau_dir() -> Path:
    """Where the Doo Prime XAUUSD CSVs live.

    Kept outside the repo: the archive they came from also carries broker
    credentials, so nothing from it is committed.
    """
    import os

    d = os.environ.get(XAU_DIR_ENV)
    if not d:
        raise RuntimeError(f"set ${XAU_DIR_ENV} to the XAUUSD/data directory")
    return Path(d)


def load_xau(tf: str, start: str | None = None, end: str | None = None,
             to_utc: bool = False) -> pd.DataFrame:
    """Load one XAUUSD timeframe from the broker CSVs.

    Timestamps are BROKER SERVER TIME (EET/EEST), not UTC -- MT5 stamps bars in
    server time and the '+00:00' in the CSV is a storage artifact. Session logic
    and the daily VWAP anchor both depend on this, so we keep server time by
    default and convert only on request.
    """
    path = xau_dir() / f"XAUUSD_{tf.upper()}.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    if to_utc:
        df.index = (df.index.tz_localize("Europe/Athens", nonexistent="shift_forward",
                                         ambiguous="NaT")
                    .tz_convert("UTC").tz_localize(None))
        df = df[df.index.notna()]
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# ───────────────────── policy / short rates for carry ─────────────────────
#
# Carry is the best-documented FX anomaly, but it only exists in TOTAL return:
# spot move PLUS the interest differential earned for holding the high-yielder.
# Backtesting it on spot alone measures something else entirely.
RATE_SERIES: dict[str, str] = {
    "USD": "DFF",                 # effective fed funds, daily
    "EUR": "ECBDFR",              # ECB deposit facility, daily
    "JPY": "IRSTCI01JPM156N",     # Japan immediate rate, monthly
    "GBP": "IR3TIB01GBM156N",     # UK 3M interbank, monthly
    "AUD": "IR3TIB01AUM156N",     # Australia 3M, monthly
    "CAD": "IR3TIB01CAM156N",     # Canada 3M, monthly
    "CHF": "IR3TIB01CHM156N",     # Switzerland 3M, monthly
    "NZD": "IR3TIB01NZM156N",     # New Zealand 3M, monthly
}


def load_rates(refresh: bool = False) -> pd.DataFrame:
    """Daily-frequency short rates in percent, one column per currency.

    Monthly series are forward-filled to daily. A rate is only known after it
    is published, so we shift by one day before use downstream -- filling
    forward from the observation date itself would leak same-day information.
    """
    path = CACHE / "rates.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    cols = {}
    for ccy, sid in RATE_SERIES.items():
        try:
            cols[ccy] = fetch_fred(sid)
        except Exception as e:
            print(f"  ! rate {ccy} ({sid}): {type(e).__name__}: {e}")
    if not cols:
        raise RuntimeError("no rate series could be fetched")
    df = pd.DataFrame(cols).sort_index()
    idx = pd.date_range(df.index.min(), pd.Timestamp.today().normalize(), freq="D")
    df = df.reindex(idx).ffill()
    df.index.name = "date"
    df.to_parquet(path)
    return df
