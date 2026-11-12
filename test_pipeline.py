"""Offline sanity test: the pipeline should detect the effect planted in synthetic data."""
from fx_analysis import build_dataset, daily_sentiment, granger, lag_correlations, walk_forward
from fx_demo import make_demo
from fx_sentiment import add_sentiment

prices, headlines = make_demo(days=400)
scored = add_sentiment(headlines)
daily = daily_sentiment(scored)
ds = build_dataset(prices, daily)

lags = lag_correlations(ds)
print(lags.round(4), "\n")
gr = granger(ds)
print(gr.round(4), "\n")
wf = walk_forward(ds)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in wf.items() if k != "predictions"})

lag1 = lags.loc[lags.lag_days == 1].iloc[0]
assert lag1["correlation"] > 0.1, "planted lag-1 effect not detected"
assert lag1["p_value"] < 0.01
assert gr.loc[0, "p_value"] < 0.05
print("\nOK: pipeline detects the planted effect.")
