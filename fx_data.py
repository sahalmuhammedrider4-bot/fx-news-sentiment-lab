"""Data loading: FX prices (Yahoo Finance) and news headlines (Google News RSS)."""
from __future__ import annotations

import hashlib
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf

PAIRS = {
    "USD/MYR": {"ticker": "MYR=X", "queries": ["ringgit", "Bank Negara Malaysia", "US dollar ringgit"]},
    "EUR/USD": {"ticker": "EURUSD=X", "queries": ["euro dollar", "ECB interest rates", "Federal Reserve rates"]},
    "USD/JPY": {"ticker": "JPY=X", "queries": ["yen dollar", "Bank of Japan", "Federal Reserve rates"]},
    "GBP/USD": {"ticker": "GBPUSD=X", "queries": ["pound sterling", "Bank of England", "Federal Reserve rates"]},
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily close prices and log returns, indexed by date."""
    df = yf.download(
        ticker, start=str(start), end=str(end + timedelta(days=1)),
        auto_adjust=True, progress=False,
    )
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}. Check your internet connection.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": "close"})
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    out["ret"] = np.log(out["close"]).diff()
    return out.dropna()


def _window_url(query: str, start: date, end: date) -> str:
    q = f"{query} after:{start} before:{end}"
    return f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"


def fetch_headlines(queries: list[str], start: date, end: date,
                    window_days: int = 7, pause: float = 0.4) -> pd.DataFrame:
    """Collect headlines week by week so we get history, not just the latest items.

    Results are cached on disk so repeated runs are fast and polite to the server.
    """
    key = hashlib.md5(f"{queries}{start}{end}".encode()).hexdigest()[:10]
    cache_file = CACHE_DIR / f"headlines_{key}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["date"])

    rows = []
    for q in queries:
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=window_days), end + timedelta(days=1))
            feed = feedparser.parse(_window_url(q, cur, nxt))
            for e in feed.entries:
                pp = e.get("published_parsed")
                if not pp:
                    continue
                rows.append({
                    "date": date(pp.tm_year, pp.tm_mon, pp.tm_mday),
                    "title": e.title,
                    "source": e.get("source", {}).get("title", ""),
                })
            cur = nxt
            time.sleep(pause)

    df = pd.DataFrame(rows, columns=["date", "title", "source"])
    if df.empty:
        raise ValueError("No headlines found. Try again later or widen the date range.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="title").sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df
