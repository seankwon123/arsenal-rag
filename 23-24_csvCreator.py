import io
import re
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

FBREF_URL = "https://fbref.com/en/squads/18bb7c10/2023-2024/Arsenal-Stats"
TABLE_DIV_ID = "all_stats_standard"   # wrapper that contains the 'Standard Stats' table
OUT_CSV = "arsenal_2023_24_fbref_standard.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://fbref.com/",
}

def http_get(url, headers=None, retries=3, backoff=0.8, timeout=30):
    headers = headers or {}
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise last

def find_commented_html_with_id(soup: BeautifulSoup, target_id: str) -> str:
    """
    FBref often wraps tables inside HTML comments.
    We scan all comments and return the first comment text that contains the target_id.
    """
    for comm in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if target_id in comm:
            return str(comm)
    # Sometimes the table is not commented—try direct lookup as a fallback:
    div = soup.find("div", id=target_id)
    if div:
        return str(div)
    raise RuntimeError(f"Could not find commented block or div with id={target_id}")

def flatten_columns(columns) -> list:
    """
    FBref tables often parse to MultiIndex columns. This flattens them nicely:
    ('Unnamed: 0_level_0','Rk') -> 'Rk'
    ('Player','Player') -> 'Player'
    ('Per 90 Minutes','Gls') -> 'Per 90 Minutes_Gls'
    """
    flat = []
    if isinstance(columns, pd.MultiIndex):
        for tup in columns:
            parts = [str(p).strip() for p in tup if p and p != "None"]
            parts = [p for p in parts if not p.startswith("Unnamed")]
            name = "_".join(parts) if parts else "_".join([p for p in map(str, tup) if p])
            name = re.sub(r"\s+", " ", name).strip("_ ").strip()
            flat.append(name if name else "col")
    else:
        flat = [str(c).strip() for c in columns]
    # de-duplicate while preserving order
    seen = {}
    deduped = []
    for c in flat:
        if c not in seen:
            seen[c] = 0
            deduped.append(c)
        else:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
    return deduped

def clean_standard_table(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten columns
    df = df.copy()
    df.columns = flatten_columns(df.columns)

    # Drop header-repeats and summary rows if present
    # Typical column names include: 'Rk','Player','Nation','Pos','Age','MP','Starts','Min',...
    # Keep only player rows: drop rows where 'Player' is null or placeholder
    if "Player" in df.columns:
        df = df[df["Player"].notna()]
        bad = {"Player", "Squad Total", "Opponent Total", "Team Total", "Nation"}
        df = df[~df["Player"].astype(str).isin(bad)]
        # FBref sometimes adds repeated header rows (e.g., 'Rk' == 'Rk')
        if "Rk" in df.columns:
            df = df[df["Rk"].astype(str).str.isnumeric()]

    # Add context columns if useful
    df.insert(0, "team", "Arsenal")
    df.insert(1, "season", "2023-24")

    # Reset index
    return df.reset_index(drop=True)

def main():
    print(f"Fetching: {FBREF_URL}")
    resp = http_get(FBREF_URL, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "lxml")

    commented = find_commented_html_with_id(soup, TABLE_DIV_ID)

    # pandas can parse an HTML snippet directly
    tables = pd.read_html(io.StringIO(commented), flavor="lxml")
    if not tables:
        raise RuntimeError("pandas.read_html found no tables inside the commented block.")
    # The first table in this block is the Standard Stats table
    raw = tables[0]

    df = clean_standard_table(raw)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    # Optional: show first few rows/cols
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(10))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
