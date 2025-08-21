import os
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama  # local LLM

# ---- Load FAISS index (built from your profiles/cards) ----
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 8})

def format_docs(docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(docs, 1):
        chunks.append(f"[CARD {i}] {d.page_content}")
    return "\n\n".join(chunks)

def season_filter(docs: List[Document], question: str) -> List[Document]:
    q = (question or "").lower()
    if any(k in q for k in ["2023", "23-24", "2023/24"]):
        ds = [d for d in docs if "2023-24" in d.page_content or "2023/24" in d.page_content]
        return ds or docs
    if any(k in q for k in ["2024", "24-25", "2024/25", "2024-25"]):
        ds = [d for d in docs if "2024-25" in d.page_content or "2024/25" in d.page_content]
        return ds or docs
    return docs

def fetch_docs_with_query(q: str) -> List[Document]:
    # use .invoke() to avoid deprecation
    docs = retriever.invoke(q)
    for d in docs:
        d.metadata["__query__"] = q
    return season_filter(docs, q)

SYSTEM = (
    "You are an analyst answering questions about Arsenal player statistics. "
    "Use ONLY the numbers in the provided cards. If computing ratios, show the math briefly. "
    "If data isn't present, say so. Prefer the season specified; if user says 'last season', use the most recent."
)

USER_TMPL = """Question:
{question}

Context cards:
{context}

Write a concise answer (4–7 sentences). Include player names with key figures and season labels when relevant.
At the end, list the cards you used like: Sources: [CARD 1], [CARD 3]
"""

prompt = PromptTemplate.from_template(USER_TMPL)

# ---- Local LLM via Ollama ----
# Ensure you've pulled a model: `ollama pull llama3.1:8b`
llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"), temperature=0.2)

chain = (
    RunnableMap(
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(lambda q: format_docs(fetch_docs_with_query(q))),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    tests = [
        "Who led Arsenal in xG in 2023-24, and what was their GA/90?",
        "Compare Saka and Martinelli in GA and progressive passes across 23-24 and 24-25.",
        "Top three Arsenal players by progressive passes in 2023-24.",
        "Who had the most minutes last season?",
    ]
    for q in tests:
        print("\nQ:", q)
        print(chain.invoke(q))
