"""Statistics: does sentiment today say anything about returns tomorrow?"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests


def daily_sentiment(headlines: pd.DataFrame) -> pd.DataFrame:
    """Average headline sentiment per trading day.

    Weekend news is rolled forward to Monday because markets are closed.
    """
    d = pd.to_datetime(headlines["date"])
    shift = np.where(d.dt.weekday == 5, 2, np.where(d.dt.weekday == 6, 1, 0))
    day = d.dt.normalize() + pd.to_timedelta(shift, unit="D")
    g = (headlines.assign(day=day)
         .groupby("day")
         .agg(sentiment=("sentiment", "mean"), n_headlines=("sentiment", "size")))
    g.index.name = "date"
    return g


def build_dataset(prices: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Left join so every trading day stays; days without news have NaN sentiment."""
    return prices.join(daily, how="left")


def lag_correlations(ds: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Correlation between sentiment on day t and the return on day t+lag.

    lag 0 is same-day (partly reverse causation: prices move, headlines describe it).
    lag >= 1 is the honest 'predictive' question.
    """
    rows = []
    for lag in range(max_lag + 1):
        pair = pd.concat([ds["sentiment"], ds["ret"].shift(-lag)], axis=1).dropna()
        if len(pair) < 30:
            rows.append((lag, np.nan, np.nan, len(pair)))
            continue
        r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append((lag, r, p, len(pair)))
    out = pd.DataFrame(rows, columns=["lag_days", "correlation", "p_value", "n"])
    # We test several lags, so a plain p < 0.05 would produce false positives by chance.
    out["significant_bonferroni"] = out["p_value"] < (0.05 / (max_lag + 1))
    return out


def granger(ds: pd.DataFrame, maxlag: int = 3) -> pd.DataFrame:
    """Does past sentiment help predict returns beyond past returns alone?"""
    data = ds[["ret", "sentiment"]].copy()
    data["sentiment"] = data["sentiment"].fillna(0.0)
    data = data.dropna()
    if len(data) < 40:
        raise ValueError("Not enough overlapping days for a Granger test.")
    # Newer statsmodels removed the `verbose` argument; older versions print by default.
    # Silencing stdout works on both.
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        res = grangercausalitytests(data, maxlag=maxlag)
    return pd.DataFrame(
        [(lag, res[lag][0]["ssr_ftest"][1]) for lag in range(1, maxlag + 1)],
        columns=["lag_days", "p_value"],
    )


def walk_forward(ds: pd.DataFrame, min_train: int = 60, step: int = 5) -> dict:
    """Expanding-window backtest of next-day direction (up/down).

    Compares three things on the same out-of-sample days:
      - baseline: always guess the majority direction seen so far
      - control : logistic regression on today's return only
      - full    : control + sentiment features
    If 'full' does not beat 'control', sentiment added nothing.
    """
    d = ds.copy()
    d["sentiment"] = d["sentiment"].fillna(0.0)
    d["sent_ma3"] = d["sentiment"].rolling(3, min_periods=1).mean()
    d["ret_today"] = d["ret"]
    d["target"] = (d["ret"].shift(-1) > 0).astype(float)
    d = d.iloc[:-1]  # last day has no next-day return
    if len(d) < min_train + 10:
        raise ValueError("Not enough data for a walk-forward test. Use a longer history.")
    y = d["target"].values

    def run(cols):
        X = d[cols].values
        preds = np.full(len(d), np.nan)
        for i in range(min_train, len(d), step):
            if len(set(y[:i])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(X[:i], y[:i])
            j = min(i + step, len(d))
            preds[i:j] = model.predict(X[i:j])
        return preds

    def baseline():
        preds = np.full(len(d), np.nan)
        for i in range(min_train, len(d), step):
            j = min(i + step, len(d))
            preds[i:j] = float(y[:i].mean() >= 0.5)
        return preds

    p_full = run(["ret_today", "sentiment", "sent_ma3"])
    p_ctrl = run(["ret_today"])
    p_base = baseline()
    mask = ~np.isnan(p_full) & ~np.isnan(p_ctrl) & ~np.isnan(p_base)

    def acc(p):
        return float((p[mask] == y[mask]).mean())

    return {
        "n_test": int(mask.sum()),
        "acc_full": acc(p_full),
        "acc_control": acc(p_ctrl),
        "acc_baseline": acc(p_base),
        "predictions": pd.DataFrame(
            {"actual_up": y[mask], "pred_up": p_full[mask]}, index=d.index[mask]
        ),
    }
