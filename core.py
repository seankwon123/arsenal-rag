
# core.py
# Compute per-90 rates, position percentiles, and a simple rating for each row.
# Usage:
#   python core.py --in players_template.csv --out features.parquet

import argparse
import numpy as np
import pandas as pd

RATE_COLS = [
    "non_pen_xg","non_pen_goals","assists","xag","shots",
    "passes_prog","prog_carries","key_passes","tackles",
    "interceptions","pressures","duels_won","aerials_won","turnovers"
]

WEIGHTS = {
    "RW": {"non_pen_xg":0.25,"non_pen_goals":0.25,"xag":0.15,"prog_carries":0.15,"passes_prog":0.10,"turnovers":-0.10},
    "LW": {"non_pen_xg":0.25,"non_pen_goals":0.25,"xag":0.15,"prog_carries":0.15,"passes_prog":0.10,"turnovers":-0.10},
    "CF": {"non_pen_xg":0.30,"non_pen_goals":0.30,"xag":0.10,"shots":0.15,"prog_carries":0.10,"turnovers":-0.05},
    "AM": {"xag":0.25,"assists":0.25,"key_passes":0.20,"passes_prog":0.20,"turnovers":-0.10},
    "CM": {"passes_prog":0.25,"key_passes":0.15,"tackles":0.15,"interceptions":0.15,"pressures":0.15,"duels_won":0.15},
    "RB": {"passes_prog":0.20,"key_passes":0.15,"tackles":0.20,"interceptions":0.15,"pressures":0.15,"duels_won":0.15},
    "LB": {"passes_prog":0.20,"key_passes":0.15,"tackles":0.20,"interceptions":0.15,"pressures":0.15,"duels_won":0.15},
    "CB": {"aerials_won":0.25,"duels_won":0.20,"tackles":0.20,"interceptions":0.20,"passes_prog":0.15},
}

def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    m_per90 = df["minutes"].replace(0, np.nan) / 90.0
    for c in RATE_COLS:
        df[f"{c}_p90"] = df[c] / m_per90
    return df

def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    def by_pos(g: pd.DataFrame) -> pd.DataFrame:
        for c in [f"{x}_p90" for x in RATE_COLS]:
            g[f"{c}_pct"] = g[c].rank(pct=True, na_option="keep")
        return g
    return df.groupby("position", group_keys=False).apply(by_pos)

def compute_rating(row: pd.Series) -> float:
    pos = row["position"]
    weights = WEIGHTS.get(pos, {})
    score = 0.0
    total = 0.0
    for base, w in weights.items():
        col = f"{base}_p90_pct"
        v = row.get(col, np.nan)
        if np.isnan(v):
            continue
        if w < 0:
            v = 1 - v
            w = -w
        score += w * v
        total += w
    if total == 0:
        return np.nan
    return round(100 * score / total, 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path_in", type=str, default="players_template.csv")
    ap.add_argument("--out", dest="path_out", type=str, default="features.parquet")
    args = ap.parse_args()

    df = pd.read_csv(args.path_in)
    df = add_rates(df)
    df = add_percentiles(df)
    df["rating"] = df.apply(compute_rating, axis=1)
    df.to_parquet(args.path_out, index=False)
    print(f"Wrote features to {args.path_out}")
