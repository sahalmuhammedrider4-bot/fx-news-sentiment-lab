"""FX News Sentiment Lab: does news sentiment help explain currency moves?

Run:  streamlit run app.py
"""
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from fx_analysis import build_dataset, daily_sentiment, granger, lag_correlations, walk_forward
from fx_data import PAIRS, fetch_headlines, get_prices
from fx_demo import make_demo
from fx_sentiment import add_sentiment

st.set_page_config(page_title="FX News Sentiment Lab", layout="wide")
st.title("FX News Sentiment Lab")
st.caption("Tests whether headline sentiment says anything about currency moves, "
           "and reports honestly when it does not.")

with st.sidebar:
    pair = st.selectbox("Currency pair", list(PAIRS))
    days = st.slider("History (days)", 90, 365, 180, step=30)
    backend_label = st.radio("Sentiment model", ["VADER (fast)", "FinBERT (finance-tuned, slower)"])
    demo = st.checkbox("Use SYNTHETIC demo data (not real, for testing only)")
    if st.button("Run analysis", type="primary"):
        st.session_state["run"] = True

if not st.session_state.get("run"):
    st.info("Pick a pair in the sidebar and press **Run analysis**.")
    st.stop()

backend = "finbert" if backend_label.startswith("FinBERT") else "vader"


@st.cache_data(show_spinner="Fetching prices and headlines (first run takes a minute)...", ttl=3600)
def load(pair: str, days: int, demo: bool):
    if demo:
        return make_demo(days=max(days, 200))
    end = date.today()
    start = end - timedelta(days=days)
    cfg = PAIRS[pair]
    return get_prices(cfg["ticker"], start, end), fetch_headlines(cfg["queries"], start, end)


@st.cache_data(show_spinner="Scoring headlines...")
def score(headlines, backend: str):
    return add_sentiment(headlines, backend)


try:
    prices, headlines = load(pair, days, demo)
    scored = score(headlines, backend)
    daily = daily_sentiment(scored)
    ds = build_dataset(prices, daily)
    lags = lag_correlations(ds)
    gr = granger(ds)
    wf = walk_forward(ds)
except Exception as e:  # show a friendly message instead of a traceback
    st.error(f"Something went wrong: {e}")
    st.stop()

if demo:
    st.warning("Showing SYNTHETIC data with a planted effect. These are not real market results.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Headlines analysed", f"{len(scored):,}")
c2.metric("Trading days", f"{len(ds):,}")
c3.metric("Baseline accuracy", f"{wf['acc_baseline']:.1%}")
c4.metric("Model with sentiment", f"{wf['acc_full']:.1%}",
          delta=f"{(wf['acc_full'] - wf['acc_control']):+.1%} vs no-sentiment model")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price vs sentiment", "Lag analysis", "Backtest", "Headlines", "Limitations"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index, y=ds["close"], name=f"{pair} close"))
    fig.add_trace(go.Bar(x=ds.index, y=ds["sentiment"], name="Daily sentiment", yaxis="y2", opacity=0.4))
    fig.update_layout(height=450, legend=dict(orientation="h"),
                      yaxis=dict(title="Price"),
                      yaxis2=dict(title="Sentiment", overlaying="y", side="right", range=[-1, 1]))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("Correlation between sentiment on day *t* and the return on day *t + lag*. "
                "Lag 0 is same-day and partly reflects headlines **describing** a move that already happened. "
                "Lags of 1+ are the predictive question.")
    bar = go.Figure(go.Bar(x=lags["lag_days"], y=lags["correlation"],
                           customdata=lags[["p_value", "n"]],
                           hovertemplate="lag %{x}<br>r=%{y:.3f}<br>p=%{customdata[0]:.3f}<br>n=%{customdata[1]}<extra></extra>"))
    bar.update_layout(height=350, xaxis_title="Lag (days)", yaxis_title="Correlation")
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(lags.round(4), use_container_width=True, hide_index=True)
    st.markdown("**Granger causality** (does past sentiment improve forecasts of returns "
                "beyond past returns alone?)")
    st.dataframe(gr.round(4), use_container_width=True, hide_index=True)
    lag1 = lags.loc[lags["lag_days"] == 1].iloc[0]
    if lag1["p_value"] < 0.05 / len(lags):
        st.success(f"Lag-1 correlation r={lag1['correlation']:.3f} survives multiple-testing correction.")
    else:
        st.info(f"Lag-1 correlation r={lag1['correlation']:.3f} (p={lag1['p_value']:.3f}) "
                "is not significant after multiple-testing correction. "
                "Little evidence that sentiment predicts next-day moves.")

with tab3:
    st.markdown(f"Walk-forward test on **{wf['n_test']}** out-of-sample days. "
                "The model is retrained on past data only, then predicts the next-day direction.")
    st.table({
        "Strategy": ["Always guess majority direction", "Today's return only", "Return + sentiment"],
        "Accuracy": [f"{wf['acc_baseline']:.1%}", f"{wf['acc_control']:.1%}", f"{wf['acc_full']:.1%}"],
    })
    st.caption("If the last row does not clearly beat the other two, sentiment added no usable signal. "
               "Currency direction is close to a coin flip, so 50-55% is realistic.")

with tab4:
    st.markdown("Most negative and most positive headlines in the sample.")
    cols = ["date", "title", "source", "sentiment"]
    st.dataframe(scored.sort_values("sentiment").head(15)[cols], hide_index=True, use_container_width=True)
    st.dataframe(scored.sort_values("sentiment", ascending=False).head(15)[cols],
                 hide_index=True, use_container_width=True)

with tab5:
    st.markdown("""
- **Sentiment is not currency-directional.** "Fed raises rates" reads negative to a general model but is bullish for USD.
  Headlines also mix several currencies. A pair-aware model would be the next improvement.
- **Headlines follow prices as much as they lead them.** Hence the focus on lag >= 1 and out-of-sample testing.
- **Short history, few days.** Small samples give noisy correlations. Multiple lags are tested, so we correct for it.
- **Google News RSS** returns limited items per query, so coverage is a sample, not the full news flow.
- **No transaction costs or slippage** are modelled. This is research, not a trading system.
""")
