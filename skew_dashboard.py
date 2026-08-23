from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from skew_map import load_config, run_scan

st.set_page_config(page_title="Options Skew Map", page_icon="🧭", layout="wide")
st.title("🧭 Options Skew Map")
st.caption("Price × options positioning research layer — not a trade signal or financial advice.")

cfg = load_config()
with st.sidebar:
    st.header("Scanner")
    raw = st.text_area("Symbols", ", ".join(cfg["universe"]), height=180)
    symbols = [x.strip().upper() for x in raw.replace("\n", ",").split(",") if x.strip()]
    st.write(f"Target delta: ±{cfg['target_delta']:.2f}")
    st.write(f"Expiry window: {cfg['min_dte']}–{cfg['max_dte']} DTE")
    run = st.button("Run live scan", type="primary", use_container_width=True)

latest_path = Path("data/skew_latest.csv")
if run:
    with st.spinner("Reading price and option chains…"):
        df, errors = run_scan(symbols, cfg)
    if errors:
        st.warning(f"{len(errors)} symbols were skipped because a reliable reading could not be produced.")
elif latest_path.exists():
    df = pd.read_csv(latest_path)
else:
    df = pd.DataFrame()

if df.empty:
    st.info("Run the scanner to create the first board. Historical change signals become more useful as daily observations accumulate.")
    st.stop()

for col in ["return_1m", "return_vs_spy_1m", "normalized_skew", "raw_skew", "quality_score", "divergence_score"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

high_quality = df[df.quality_label != "LOW"]
contrarian = high_quality[high_quality.quadrant == "CONTRARIAN BID"]
hedged = high_quality[high_quality.quadrant == "HEDGED RALLY"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Names scanned", len(df))
c2.metric("Contrarian bids", len(contrarian))
c3.metric("Hedged rallies", len(hedged))
c4.metric("Catalyst flagged", int(df.catalyst_flag.fillna(False).astype(bool).sum()))

st.subheader("Price × skew radar")
fig = px.scatter(
    df, x="return_1m", y="normalized_skew", color="quadrant", size="quality_score",
    hover_name="symbol", hover_data=["sector", "return_vs_spy_1m", "skew_change_5obs", "quality_label", "catalyst_flag"],
    category_orders={"quadrant": ["CONTRARIAN BID", "CHASE", "HEDGED RALLY", "FEAR"]}
)
fig.add_hline(y=0, line_dash="dash", opacity=.5)
fig.add_vline(x=0, line_dash="dash", opacity=.5)
fig.update_xaxes(tickformat=".1%", title="1-month stock return")
fig.update_yaxes(tickformat=".1%", title="Normalized 25Δ skew (put IV − call IV) / ATM IV")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Research ranking")
show = df.copy().sort_values(["divergence_score", "quality_score"], ascending=False)
columns = ["symbol", "sector", "quadrant", "return_1m", "return_vs_spy_1m", "normalized_skew",
           "skew_change_5obs", "skew_change_20obs", "skew_percentile", "divergence_score",
           "quality_label", "quality_score", "catalyst_flag", "days_to_earnings", "expiry"]
st.dataframe(show[[c for c in columns if c in show]], use_container_width=True, hide_index=True,
             column_config={
                 "return_1m": st.column_config.NumberColumn(format="%.2f%%"),
                 "return_vs_spy_1m": st.column_config.NumberColumn(format="%.2f%%"),
                 "normalized_skew": st.column_config.NumberColumn(format="%.3f"),
             })

st.subheader("Sector agreement")
sector_rows = []
for sector, g in high_quality.groupby("sector"):
    if len(g) < 2:
        continue
    put_share = float((g.normalized_skew > 0).mean())
    call_share = 1 - put_share
    agreement = max(put_share, call_share)
    side = "PUTS BID" if put_share >= .5 else "CALLS BID"
    sector_rows.append({"sector": sector, "names": len(g), "raw_skew_avg": g.raw_skew.mean(),
                        "side": side, "agreement": agreement,
                        "trusted": agreement >= cfg["sector_agreement_threshold"]})
sector_df = pd.DataFrame(sector_rows)
if not sector_df.empty:
    st.dataframe(sector_df.sort_values("agreement", ascending=False), use_container_width=True, hide_index=True)
else:
    st.caption("Need at least two high/medium-quality names in a sector for an agreement reading.")

history_path = Path("data/skew_history.csv")
st.subheader("Single-name history")
if history_path.exists():
    history = pd.read_csv(history_path)
    available = sorted(history.symbol.unique())
    selected = st.selectbox("Symbol", available)
    h = history[history.symbol == selected].sort_values("run_date")
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=h.run_date, y=h.normalized_skew, mode="lines+markers", name="Normalized skew"))
    chart.add_hline(y=0, line_dash="dash", opacity=.5)
    chart.update_layout(yaxis_title="Normalized skew", xaxis_title="Observation date")
    st.plotly_chart(chart, use_container_width=True)

with st.expander("How to read this"):
    st.markdown("""
- **Contrarian Bid:** price down, calls relatively bid. Investigate; it is not automatically bullish.
- **Chase:** price up, calls relatively bid. Momentum and options pricing agree.
- **Hedged Rally:** price up, puts relatively bid. Downside insurance remains expensive.
- **Fear:** price down, puts relatively bid. Price and protection demand agree.

**Quality matters.** The scanner estimates delta from the chain, checks OI, spreads and delta fit, rejects implausible skew, and flags earnings occurring inside the selected expiry. Historical *change* is generally more informative than comparing absolute skew between unrelated companies.
""")
