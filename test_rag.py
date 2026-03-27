#!/usr/bin/env python3
"""
test_rag.py — CLI tool to test the RAG engine without Telegram
Usage: python test_rag.py "your question here"
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import get_db, init_db, load_knowledge_base, answer_query, retrieve_top_k

KB_DIR = os.getenv("KB_DIR", str(Path(__file__).parent / "knowledge_base"))
DB_PATH = os.getenv("DB_PATH", "test_rag.db")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_rag.py 'your question'")
        print("       python test_rag.py --chunks 'your question'  (show raw chunks)")
        sys.exit(1)

    show_chunks = "--chunks" in sys.argv
    query = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))

    print(f"🔧 Initializing DB: {DB_PATH}")
    conn = get_db(DB_PATH)
    init_db(conn)

    print(f"📚 Loading knowledge base from: {KB_DIR}")
    load_knowledge_base(KB_DIR, conn)

    if show_chunks:
        print(f"\n🔍 Top chunks for: '{query}'\n")
        chunks = retrieve_top_k(query, conn, top_k=3)
        for i, c in enumerate(chunks, 1):
            print(f"[{i}] Score: {c['score']:.4f} | Source: {c['source']}")
            print(f"    {c['text'][:300]}\n")
        return

    print(f"\n❓ Query: {query}\n")
    result = answer_query(query, conn)

    print(f"💬 Answer {'(cached ⚡)' if result['cached'] else ''}:")
    print(result["answer"])
    print(f"\n📚 Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
