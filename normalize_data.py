# pip install pandas langchain faiss-cpu sentence-transformers streamlit
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1) Load & normalize
df1 = pd.read_csv("arsenal_2023_24_fbref_standard.csv")
df2 = pd.read_csv("arsenal_2024_25_fbref_standard.csv")

# rename a few long columns for convenience (adjust to your exact headers)
rename = {
    "Player":"player",
    "Pos":"pos",
    "Nation":"nation",
    "team":"team",
    "season":"season",
    "Playing Time_Min":"min",
    "Performance_Gls":"gls",
    "Performance_Ast":"ast",
    "Expected_xG":"xg",
    "Expected_npxG":"npxg",
    "Expected_xAG":"xag",
    "Progression_PrgC":"prgc",
    "Progression_PrgP":"prgp",
    "Per 90 Minutes_G+A":"ga_per90",
}
df1 = df1.rename(columns=rename)
df2 = df2.rename(columns=rename)


# add a couple computed fields safely
for col in ["min","gls","ast","xg","npxg","xag","prgc","prgp","ga_per90"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["ga"] = df["gls"].fillna(0) + df["ast"].fillna(0)
df["ga_per90_calc"] = (df["ga"] / (df["min"]/90)).round(2)

# 2) Create concise stat-card text per player
def stat_card(row):
    bits = []
    bits.append(f"{row['player']} — {row.get('pos','?')} | {row['team']} {row['season']}")
    if pd.notna(row["min"]):  bits.append(f"Minutes: {int(row['min'])}")
    if pd.notna(row["gls"]):  bits.append(f"Gls: {int(row['gls'])}")
    if pd.notna(row["ast"]):  bits.append(f"Ast: {int(row['ast'])}")
    if pd.notna(row["xg"]):   bits.append(f"xG: {row['xg']:.2f}")
    if pd.notna(row["xag"]):  bits.append(f"xAG: {row['xag']:.2f}")
    if pd.notna(row["npxg"]): bits.append(f"npxG: {row['npxg']:.2f}")
    if pd.notna(row["prgc"]): bits.append(f"Prog carries: {int(row['prgc'])}")
    if pd.notna(row["prgp"]): bits.append(f"Prog passes: {int(row['prgp'])}")
    # use your own preference for GA/90 source
    ga90 = row["ga_per90"] if pd.notna(row.get("ga_per90")) else row["ga_per90_calc"]
    if pd.notna(ga90):        bits.append(f"G+A/90: {ga90:.2f}")
    return " | ".join(bits)

docs = []
for _, r in df.iterrows():
    txt = stat_card(r)
    meta = r.to_dict()
    docs.append(Document(page_content=txt, metadata=meta))

# 3) Build FAISS vector store
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.from_documents(docs, emb)

# Save for reuse
vs.save_local("arsenal_fbref_2023_24_faiss")
