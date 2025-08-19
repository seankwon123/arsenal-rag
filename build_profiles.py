import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from unidecode import unidecode

# --------- CONFIG ---------
CSV_2324 = "arsenal_2023_24_fbref_standard.csv"  # REQUIRED (your 23-24 standard table export)
CSV_2425 = "arsenal_2024_25_fbref_standard.csv"  # OPTIONAL (same table for 24-25 if you have it)

TEAM = "Arsenal"
SEASON_2324 = "2023-24"
SEASON_2425 = "2024-25"

# Output artifacts
PROFILES_JSON = "player_profiles.json"
CARDS_CSV = "profiles_cards.csv"
FAISS_DIR = "faiss_index"
# --------------------------


# ---- Column mapping from your header to short snake_case we’ll use internally
COLMAP = {
    "Player": "player",
    "Nation": "nation",
    "Pos": "pos",
    "Age": "age",
    "Playing Time_Min": "min",
    "Performance_Gls": "gls",
    "Performance_Ast": "ast",
    "Performance_G-PK": "g_np",
    "Performance_PK": "pk",
    "Performance_PKatt": "pk_att",
    "Performance_CrdY": "y",
    "Performance_CrdR": "r",
    "Expected_xG": "xg",
    "Expected_npxG": "npxg",
    "Expected_xAG": "xag",
    "Expected_npxG+xAG": "npxg_xag",
    "Progression_PrgC": "prg_c",
    "Progression_PrgP": "prg_p",
    "Progression_PrgR": "prg_r",
    "Per 90 Minutes_Gls": "gls90",
    "Per 90 Minutes_Ast": "ast90",
    "Per 90 Minutes_G+A": "ga90",
    "Per 90 Minutes_G-PK": "g_np90",
    "Per 90 Minutes_G+A-PK": "ga_np90",
    "Per 90 Minutes_xG": "xg90",
    "Per 90 Minutes_xAG": "xag90",
    "Per 90 Minutes_xG+xAG": "xg_xag90",
    "Per 90 Minutes_npxG": "npxg90",
    "Per 90 Minutes_npxG+xAG": "npxg_xag90",
    "Matches": "matches",
    # from your helper columns in scrape we kept:
    "team": "team",
    "season": "season",
}

NUMERIC_COLUMNS = {
    "min","gls","ast","g_np","pk","pk_att","y","r",
    "xg","npxg","xag","npxg_xag",
    "prg_c","prg_p","prg_r",
    "gls90","ast90","ga90","g_np90","ga_np90",
    "xg90","xag90","xg_xag90","npxg90","npxg_xag90",
    "matches",
}

def norm_key(name: str) -> str:
    """Normalize player key for merging (accentless, lowercase, squashed spaces)."""
    s = unidecode(str(name)).lower().strip()
    s = " ".join(s.split())
    return s

def load_and_normalize(path: str, season_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Rename mapped columns; keep only ones we care about
    keep = []
    for src, dst in COLMAP.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
            keep.append(dst)
    # Ensure we keep identity columns
    for base in ["player","pos","nation","age"]:
        if base not in df.columns and base in keep:
            pass
    # Some scrapes already added team/season; enforce/overwrite for consistency
    df["team"] = TEAM
    df["season"] = season_label

    # Subset to the columns we know + any identity columns present
    cols = list({*keep, "player","pos","nation","age","team","season"})
    df = df[[c for c in cols if c in df.columns]].copy()

    # Cast numeric columns safely
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive a few helpful fields if missing
    if "gls" in df.columns and "ast" in df.columns:
        df["ga"] = df["gls"].fillna(0) + df["ast"].fillna(0)
    else:
        df["ga"] = pd.NA

    if "min" in df.columns:
        mins = df["min"].replace(0, pd.NA)
        df["ga90_calc"] = (df["ga"] / (mins/90)).round(3)
        df["xg90_calc"] = (df["xg"] / (mins/90)).round(3) if "xg" in df.columns else pd.NA
        df["xag90_calc"] = (df["xag"] / (mins/90)).round(3) if "xag" in df.columns else pd.NA
        df["npxg_xag90_calc"] = (df.get("npxg",0) + df.get("xag",0)) / (mins/90)
        df["npxg_xag90_calc"] = df["npxg_xag90_calc"].round(3)
    else:
        df["ga90_calc"] = pd.NA
        df["xg90_calc"] = pd.NA
        df["xag90_calc"] = pd.NA
        df["npxg_xag90_calc"] = pd.NA

    # Merge key
    df["player_key"] = df["player"].map(norm_key)

    # Keep only squad player rows (drop totals/header echoes if any slipped through)
    df = df[df["player"].notna()].copy()
    bad_labels = {"Player", "Squad Total", "Opponent Total", "Team Total"}
    df = df[~df["player"].astype(str).isin(bad_labels)]
    return df.reset_index(drop=True)

def load_both() -> pd.DataFrame:
    if not Path(CSV_2324).exists():
        raise FileNotFoundError(f"Missing {CSV_2324} (23–24 standard CSV).")
    df1 = load_and_normalize(CSV_2324, SEASON_2324)

    if Path(CSV_2425).exists():
        df2 = load_and_normalize(CSV_2425, SEASON_2425)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1

    return df

def profile_for_player(rows: pd.DataFrame) -> Dict:
    """
    Build a per-player profile dict that may include one or two seasons depending on availability.
    """
    # identity (pick most frequent pos/nation/age if duplicated)
    player = rows.iloc[0]["player"]
    key = rows.iloc[0]["player_key"]
    pos = rows["pos"].dropna().mode().iat[0] if rows["pos"].dropna().size else None
    nation = rows["nation"].dropna().mode().iat[0] if rows["nation"].dropna().size else None
    age = rows["age"].dropna().mode().iat[0] if rows["age"].dropna().size else None

    seasons: List[Dict] = []
    for season, grp in rows.groupby("season", sort=False):
        r = grp.iloc[0]
        def val(c): return None if c not in grp.columns else (None if pd.isna(r[c]) else (int(r[c]) if c in {"min","gls","ast","g_np","pk","pk_att","y","r","prg_c","prg_p","prg_r","matches"} else float(r[c]) if pd.api.types.is_number(r[c]) else r[c]))
        season_block = {
            "season": season,
            "team": TEAM,
            "min": val("min"),
            "gls": val("gls"),
            "ast": val("ast"),
            "ga": (val("gls") or 0) + (val("ast") or 0) if val("gls") is not None and val("ast") is not None else None,
            "xg": val("xg"),
            "npxg": val("npxg"),
            "xag": val("xag"),
            "npxg_xag": val("npxg_xag"),
            "prg_c": val("prg_c"),
            "prg_p": val("prg_p"),
            "prg_r": val("prg_r"),
            "ga90": val("ga90") if "ga90" in grp.columns and r["ga90"]==r["ga90"] else None,
            "ga90_calc": val("ga90_calc"),
            "xg90": val("xg90") if "xg90" in grp.columns and r["xg90"]==r["xg90"] else None,
            "xg90_calc": val("xg90_calc"),
            "xag90": val("xag90") if "xag90" in grp.columns and r["xag90"]==r["xag90"] else None,
            "xag90_calc": val("xag90_calc"),
            "npxg_xag90": val("npxg_xag90") if "npxg_xag90" in grp.columns and r["npxg_xag90"]==r["npxg_xag90"] else None,
            "npxg_xag90_calc": val("npxg_xag90_calc"),
        }
        seasons.append(season_block)

    return {
        "player": player,
        "player_key": key,
        "pos": pos,
        "nation": nation,
        "age": age,
        "seasons": seasons,   # 1 or 2 entries
    }

def make_card(profile: Dict) -> str:
    """
    Human-readable text used for retrieval. Includes one or two seasons, with key numbers.
    """
    head = f"{profile['player']} — {profile.get('pos','?')} | {TEAM}"
    chunks = [head]
    for s in profile["seasons"]:
        line = [s["season"]]
        if s.get("min") is not None:  line.append(f"Min {s['min']}")
        if s.get("gls") is not None:  line.append(f"G {s['gls']}")
        if s.get("ast") is not None:  line.append(f"A {s['ast']}")
        if s.get("ga") is not None:   line.append(f"GA {s['ga']}")
        if s.get("xg") is not None:   line.append(f"xG {s['xg']:.2f}")
        if s.get("xag") is not None:  line.append(f"xAG {s['xag']:.2f}")
        if s.get("npxg") is not None: line.append(f"npxG {s['npxg']:.2f}")
        if s.get("prg_c") is not None: line.append(f"PrgC {int(s['prg_c'])}")
        if s.get("prg_p") is not None: line.append(f"PrgP {int(s['prg_p'])}")
        # choose a GA/90 best available
        ga90 = s.get("ga90") if s.get("ga90") is not None else s.get("ga90_calc")
        if ga90 is not None: line.append(f"GA/90 {ga90:.2f}")
        chunks.append(" | ".join(line))
    return "\n".join(chunks)

def main():
    df = load_both()

    # Build profiles (1 or 2 seasons per player depending on data available)
    profiles: List[Dict] = []
    for key, rows in df.groupby("player_key"):
        profiles.append(profile_for_player(rows.sort_values("season")))

    # Save structured profiles
    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Saved {PROFILES_JSON} with {len(profiles)} players")

    # Build “cards” for retrieval and a quick CSV to eyeball them
    cards = []
    for p in profiles:
        cards.append({
            "player": p["player"],
            "pos": p.get("pos"),
            "card": make_card(p),
        })
    cards_df = pd.DataFrame(cards).sort_values("player")
    cards_df.to_csv(CARDS_CSV, index=False)
    print(f"Saved {CARDS_CSV}")

    # ---- Build FAISS index from cards
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document

    docs = [Document(page_content=row["card"], metadata={"player": row["player"], "pos": row.get("pos")})
            for _, row in cards_df.iterrows()]

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(FAISS_DIR)
    print(f"FAISS index saved to {FAISS_DIR}")

    # ---- Tiny retrieval demo (no LLM, just nearest cards)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    for q in [
        "Who led Arsenal in progressive passes in 24-25 season?",
        "Compare Saka vs Martinelli on xG and GA/90 across 23-24 and 24-25",
    ]:
        hits = retriever.get_relevant_documents(q)
        print("\nQ:", q)
        for h in hits:
            print("-", h.page_content.splitlines()[0])

if __name__ == "__main__":
    main()
