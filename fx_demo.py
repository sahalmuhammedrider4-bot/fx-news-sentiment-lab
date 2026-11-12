"""SYNTHETIC data generator, used only to test the pipeline offline.

A small effect is deliberately planted (news mood on day t nudges the return on t+1),
so the tests can check that the analysis is able to detect a real effect when one exists.
Never present results from this data as real market findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POS = [
    "Ringgit rallies as dollar weakens", "Currency gains on strong export data",
    "Markets surge on upbeat economic outlook", "Investors upbeat as growth beats forecasts",
]
NEG = [
    "Ringgit slides as dollar strengthens", "Currency tumbles on recession fears",
    "Markets plunge amid selloff", "Investors worry as crisis deepens",
]


def make_demo(days: int = 300, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    mood = rng.normal(0, 1, days)
    ret = rng.normal(0, 0.004, days)
    ret[1:] += 0.002 * mood[:-1]  # planted effect
    close = 4.5 * np.exp(np.cumsum(ret))
    prices = pd.DataFrame({"close": close, "ret": ret}, index=pd.Index(dates, name="date")).iloc[1:]

    rows = []
    for t, d in enumerate(dates):
        p_pos = 1 / (1 + np.exp(-1.5 * mood[t]))
        for _ in range(rng.integers(2, 6)):
            title = rng.choice(POS) if rng.random() < p_pos else rng.choice(NEG)
            rows.append({"date": d, "title": f"{title} ({rng.integers(1000)})", "source": "synthetic"})
    return prices, pd.DataFrame(rows)
