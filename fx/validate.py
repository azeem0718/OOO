"""Guards against mistaking data-mining for an edge.

Searching many strategy/parameter/market combinations and reporting the best
one guarantees an inflated result: the maximum of N noisy estimates is biased
upward even when every true edge is zero. These tools quantify that bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def split_is_oos(df: pd.DataFrame, oos_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(oos_start)
    return df[df.index < cut], df[df.index >= cut]


def block_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000,
                           block: int = 20, seed: int = 7,
                           ppy: float = 252.0) -> dict:
    """Confidence interval for annualised Sharpe via circular block bootstrap.

    Blocks preserve autocorrelation and volatility clustering, which an iid
    bootstrap destroys -- and destroying them makes the CI look tighter than
    reality.
    """
    r = returns.dropna().to_numpy(float)
    n = len(r)
    if n < block * 3:
        return {"sharpe": np.nan, "lo": np.nan, "hi": np.nan, "p_gt_0": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        take = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        x = r[take]
        sd = x.std()
        out[b] = (x.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0
    sd0 = r.std()
    sr = (r.mean() / sd0 * np.sqrt(ppy)) if sd0 > 0 else 0.0
    return {
        "sharpe": float(sr),
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "p_gt_0": float((out > 0).mean()),
    }


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Sharpe the best of `n_trials` worthless strategies reaches by luck.

    Bailey & Lopez de Prado's expected-maximum result. Anything below this line
    is indistinguishable from noise no matter how good it looks.
    """
    if n_trials < 2 or not np.isfinite(sr_std) or sr_std <= 0:
        return 0.0
    g = 0.5772156649015329  # Euler-Mascheroni
    a = stats.norm.ppf(1 - 1.0 / n_trials)
    b = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1 - g) * a + g * b))


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_std: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability the observed Sharpe is real after the search is accounted for.

    sr is per-period (not annualised). Returns a probability in [0, 1];
    below ~0.95 means the result does not clear the multiple-testing bar.
    """
    if n_obs < 10:
        return np.nan
    sr0 = expected_max_sharpe(n_trials, sr_std)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return np.nan
    se = np.sqrt(denom / (n_obs - 1))
    return float(stats.norm.cdf((sr - sr0) / se)) if se > 0 else np.nan


def walk_forward(df: pd.DataFrame, param_grid: list[dict], run_fn,
                 train_years: float = 5.0, test_years: float = 2.0,
                 objective: str = "sharpe") -> pd.DataFrame:
    """Anchored-window walk-forward: fit on train, trade the next test window.

    Each test window is genuinely out-of-sample for the parameters used in it,
    so concatenating the test windows gives a tradeable track record rather
    than a fitted one.
    """
    rows = []
    t0, tN = df.index[0], df.index[-1]
    train_end = t0 + pd.Timedelta(days=int(train_years * 365.25))
    while train_end < tN:
        test_end = min(train_end + pd.Timedelta(days=int(test_years * 365.25)), tN)
        tr = df[(df.index >= t0) & (df.index < train_end)]
        te = df[(df.index >= train_end) & (df.index < test_end)]
        if len(tr) < 250 or len(te) < 60:
            break
        best, best_val = None, -np.inf
        for p in param_grid:
            s = run_fn(tr, p)
            v = s.get(objective, -np.inf)
            if v is not None and np.isfinite(v) and v > best_val:
                best, best_val = p, v
        if best is not None:
            s_oos = run_fn(te, best)
            rows.append({
                "train_end": train_end, "test_end": test_end,
                "params": best, "is_obj": best_val,
                "oos_sharpe": s_oos.get("sharpe", np.nan),
                "oos_net": s_oos.get("net", np.nan),
                "oos_trades": s_oos.get("n_trades", 0),
                "oos_pf": s_oos.get("profit_factor", np.nan),
            })
        train_end = test_end
    return pd.DataFrame(rows)


def stability(per_year: pd.Series) -> dict:
    """How evenly the profit arrived. Concentration is fragility."""
    v = per_year.dropna()
    if len(v) == 0:
        return {"pos_frac": np.nan, "best_share": np.nan, "n_years": 0}
    total = v.sum()
    return {
        "n_years": int(len(v)),
        "pos_frac": float((v > 0).mean()),
        "best_share": float(v.max() / total) if total > 0 else np.nan,
        "worst_year": float(v.min()),
    }
