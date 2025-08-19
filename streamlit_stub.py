
# streamlit_stub.py
# Minimal app to explore computed per-90 metrics and rating.
# Usage:
#   streamlit run streamlit_stub.py

import streamlit as st
import pandas as pd
import numpy as np

st.title("Arsenal Player Stats Explorer")
st.caption("Fill players_template.csv with real numbers, run core.py to create features.parquet, then use this app.")

@st.cache_data
def load_features(path: str):
    try:
        return pd.read_parquet(path)
    except Exception:
        st.warning("features.parquet not found. Upload it or run core.py first.")
        return None

df = load_features("features.parquet")
if df is None:
    st.stop()

player = st.selectbox("Player", sorted(df["player"].unique()))
seasons = sorted(df[df.player==player]["season"].unique())
season = st.selectbox("Season", seasons)

row = df[(df.player==player) & (df.season==season)].iloc[0]
st.subheader(f"{player} {season} · {row['position']}")
st.metric("Rating", row.get("rating", np.nan))

cols = [
    "non_pen_xg_p90","non_pen_goals_p90","xag_p90","assists_p90",
    "shots_p90","passes_prog_p90","prog_carries_p90","key_passes_p90",
    "tackles_p90","interceptions_p90","pressures_p90","duels_won_p90",
    "aerials_won_p90","turnovers_p90"
]
table = {c: row.get(c, np.nan) for c in cols if c in row.index}
st.write(pd.DataFrame([table]).T.rename(columns={0: "per90"}))

st.caption("This is the numeric foundation. Next we will generate text stat cards and index them with FAISS for RAG.")
