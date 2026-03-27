"""
rag_engine.py — Mini RAG pipeline using sentence-transformers + SQLite
Supports: Google Gemini API (free) · Anthropic Claude · Ollama local · Extractive fallback
"""

import os
import json
import sqlite3
import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def get_db(db_path: str = "rag_store.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT NOT NULL,
            chunk_text  TEXT NOT NULL,
            chunk_hash  TEXT UNIQUE NOT NULL,
            embedding   BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash  TEXT PRIMARY KEY,
            answer      TEXT NOT NULL,
            sources     TEXT NOT NULL,
            created_at  INTEGER DEFAULT (strftime('%s','now'))
        );
    """)
    conn.commit()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    sentences = text.replace("\n\n", ". ").replace("\n", " ").split(". ")
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        while len(sent) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sent[:chunk_size])
            sent = sent[chunk_size - overlap:]
        if len(current) + len(sent) + 2 > chunk_size and current:
            chunks.append(current.strip())
            words = current.split()
            current = " ".join(words[-(overlap // 5):]) + " " + sent
        else:
            current = (current + ". " + sent).strip() if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_knowledge_base(kb_dir: str, conn: sqlite3.Connection):
    embedder = _get_embedder()
    kb_path = Path(kb_dir)
    files = list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))
    if not files:
        logger.warning(f"No .md/.txt files found in {kb_dir}")
        return

    new_chunks = 0
    for fpath in files:
        text = fpath.read_text(encoding="utf-8")
        source = fpath.name
        for chunk in chunk_text(text):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            if conn.execute("SELECT id FROM chunks WHERE chunk_hash = ?", (chunk_hash,)).fetchone():
                continue
            embedding = embedder.encode(chunk, normalize_embeddings=True)
            conn.execute(
                "INSERT INTO chunks (source, chunk_text, chunk_hash, embedding) VALUES (?,?,?,?)",
                (source, chunk, chunk_hash, embedding.tobytes()),
            )
            new_chunks += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    logger.info(f"KB loaded: {new_chunks} new chunks added, {total} total chunks in DB")


def cosine_similarity(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve_top_k(query: str, conn: sqlite3.Connection, top_k: int = 3) -> list:
    embedder = _get_embedder()
    q_emb = embedder.encode(query, normalize_embeddings=True)
    rows = conn.execute("SELECT source, chunk_text, embedding FROM chunks").fetchall()
    if not rows:
        return []
    scored = []
    for row in rows:
        doc_emb = np.frombuffer(row["embedding"], dtype=np.float32)
        scored.append({"source": row["source"], "text": row["chunk_text"], "score": cosine_similarity(q_emb, doc_emb)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [c for c in scored[:top_k] if c["score"] > 0.35]


def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(f"[{i}] (from {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have information about that in my knowledge base."
Be concise, clear, and friendly. Do not invent facts.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""


def call_llm(prompt: str):
    """
    Priority: Gemini (free) -> Anthropic -> Ollama -> None
    """
    import requests

    # 1. Google Gemini (free tier)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

    # 2. Anthropic Claude
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Anthropic failed: {e}")

    # 3. Ollama local
    try:
        resp = requests.post(
            f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/generate",
            json={"model": os.getenv("OLLAMA_MODEL", "phi3"), "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama failed: {e}")

    return None


def answer_query(query: str, conn: sqlite3.Connection, top_k: int = 3, use_cache: bool = True) -> dict:
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()

    if use_cache:
        cached = conn.execute("SELECT answer, sources FROM query_cache WHERE query_hash = ?", (query_hash,)).fetchone()
        if cached:
            return {"answer": cached["answer"], "sources": json.loads(cached["sources"]), "cached": True}

    chunks = retrieve_top_k(query, conn, top_k)
    if not chunks:
        return {"answer": "❌ The knowledge base is empty. Please ask an admin to load documents.", "sources": [], "cached": False}

    answer = call_llm(build_prompt(query, chunks))

    if answer is None:
        snippets = "\n\n".join(f"📄 *{c['source']}* (score: {c['score']:.2f})\n{c['text'][:350]}" for c in chunks)
        answer = f"⚠️ No LLM configured — showing top matching snippets:\n\n{snippets}"

    sources = list({c["source"] for c in chunks})
    conn.execute(
        "INSERT OR REPLACE INTO query_cache (query_hash, answer, sources) VALUES (?,?,?)",
        (query_hash, answer, json.dumps(sources)),
    )
    conn.commit()
    return {"answer": answer, "sources": sources, "cached": False}
