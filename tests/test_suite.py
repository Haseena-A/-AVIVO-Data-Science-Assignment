"""
tests/test_suite.py — Unit and integration tests for the GenAI bot.

Run:
    pip install pytest
    pytest tests/test_suite.py -v
"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine import (
    chunk_text,
    get_db,
    init_db,
    load_knowledge_base,
    answer_query,
    retrieve_top_k,
    cosine_similarity,
)
from history import HistoryManager
from vision_engine import format_vision_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Temporary in-memory SQLite connection for each test."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def kb_dir(tmp_path):
    """Temp knowledge base directory with two small docs."""
    (tmp_path / "doc1.md").write_text(
        "# Photosynthesis\n\n"
        "Photosynthesis is the process by which plants convert sunlight into glucose. "
        "It occurs in the chloroplasts and requires water and carbon dioxide. "
        "The byproduct is oxygen, which is released into the atmosphere."
    )
    (tmp_path / "doc2.md").write_text(
        "# Mitosis\n\n"
        "Mitosis is cell division that produces two identical daughter cells. "
        "It has four phases: prophase, metaphase, anaphase, and telophase. "
        "The result is two cells with the same number of chromosomes as the parent."
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world. This is a short sentence."
        chunks = chunk_text(text, chunk_size=400)
        assert len(chunks) >= 1
        assert any("Hello world" in c or "short sentence" in c for c in chunks)

    def test_long_text_multiple_chunks(self):
        # Generate text larger than one chunk
        text = ". ".join([f"Sentence number {i} with some padding words here" for i in range(30)])
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        assert len(chunks) > 1

    def test_chunks_non_empty(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        chunks = chunk_text(text, chunk_size=400)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=400)
        assert chunks == []

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Each chunk must not exceed chunk_size (hard limit after fix)
        for c in chunks:
            assert len(c) <= 100


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        import numpy as np
        v = np.array([1.0, 0.5, -0.3], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_safe(self):
        import numpy as np
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        # Should not raise ZeroDivisionError
        result = cosine_similarity(a, b)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Database init
# ---------------------------------------------------------------------------

class TestDatabaseInit:
    def test_tables_created(self, tmp_db):
        tables = {
            row[0]
            for row in tmp_db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "chunks" in tables
        assert "query_cache" in tables

    def test_chunks_schema(self, tmp_db):
        cols = {
            row[1]
            for row in tmp_db.execute("PRAGMA table_info(chunks)").fetchall()
        }
        assert {"id", "source", "chunk_text", "chunk_hash", "embedding"}.issubset(cols)

    def test_query_cache_schema(self, tmp_db):
        cols = {
            row[1]
            for row in tmp_db.execute("PRAGMA table_info(query_cache)").fetchall()
        }
        assert {"query_hash", "answer", "sources"}.issubset(cols)


# ---------------------------------------------------------------------------
# Knowledge base loading
# ---------------------------------------------------------------------------

class TestLoadKnowledgeBase:
    def test_chunks_inserted(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        count = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count > 0

    def test_sources_tracked(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        sources = {
            row[0]
            for row in tmp_db.execute("SELECT DISTINCT source FROM chunks").fetchall()
        }
        assert "doc1.md" in sources
        assert "doc2.md" in sources

    def test_idempotent(self, tmp_db, kb_dir):
        """Loading the same KB twice should not duplicate chunks."""
        load_knowledge_base(kb_dir, tmp_db)
        count1 = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        load_knowledge_base(kb_dir, tmp_db)
        count2 = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count1 == count2

    def test_empty_dir(self, tmp_db, tmp_path):
        """Empty directory should not raise."""
        load_knowledge_base(str(tmp_path), tmp_db)
        count = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_retrieve_returns_results(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        results = retrieve_top_k("What is photosynthesis?", tmp_db, top_k=2)
        assert len(results) <= 2
        assert all("text" in r and "source" in r and "score" in r for r in results)

    def test_retrieve_scores_in_range(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        results = retrieve_top_k("plant cell biology", tmp_db, top_k=3)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_relevant_doc_ranked_higher(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        results = retrieve_top_k("photosynthesis chloroplasts sunlight", tmp_db, top_k=3)
        top_sources = [r["source"] for r in results[:2]]
        assert "doc1.md" in top_sources

    def test_empty_db_returns_empty(self, tmp_db):
        results = retrieve_top_k("anything", tmp_db, top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# answer_query (mocked LLM)
# ---------------------------------------------------------------------------

class TestAnswerQuery:
    def test_returns_dict_structure(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        with patch("rag_engine.call_llm", return_value="Mocked answer."):
            # Patch at module level
            import rag_engine
            original = rag_engine.call_llm
            rag_engine.call_llm = lambda p: "Mocked answer."
            try:
                result = answer_query("What is mitosis?", tmp_db)
            finally:
                rag_engine.call_llm = original

        assert "answer" in result
        assert "sources" in result
        assert "cached" in result

    def test_caching_works(self, tmp_db, kb_dir):
        load_knowledge_base(kb_dir, tmp_db)
        import rag_engine
        call_count = [0]

        original_llm = rag_engine.call_llm

        def counting_llm(p):
            call_count[0] += 1
            return "Answer from LLM."

        rag_engine.call_llm = counting_llm
        try:
            r1 = answer_query("What is photosynthesis?", tmp_db, use_cache=True)
            r2 = answer_query("What is photosynthesis?", tmp_db, use_cache=True)
        finally:
            rag_engine.call_llm = original_llm

        assert r2["cached"] is True
        assert r1["answer"] == r2["answer"]

    def test_empty_db_returns_error_message(self, tmp_db):
        result = answer_query("anything", tmp_db)
        assert "empty" in result["answer"].lower() or "❌" in result["answer"]
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# HistoryManager
# ---------------------------------------------------------------------------

class TestHistoryManager:
    def test_add_and_get(self, tmp_db):
        hm = HistoryManager(tmp_db)
        hm.add(1, "user", "Hello")
        hm.add(1, "assistant", "Hi there!")
        history = hm.get(1)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_max_history_respected(self, tmp_db):
        hm = HistoryManager(tmp_db)
        for i in range(20):
            hm.add(1, "user" if i % 2 == 0 else "assistant", f"Message {i}")
        history = hm.get(1)
        from history import MAX_HISTORY
        assert len(history) <= MAX_HISTORY * 2

    def test_clear(self, tmp_db):
        hm = HistoryManager(tmp_db)
        hm.add(1, "user", "Test")
        hm.clear(1)
        assert hm.get(1) == []

    def test_isolated_per_user(self, tmp_db):
        hm = HistoryManager(tmp_db)
        hm.add(1, "user", "User 1 message")
        hm.add(2, "user", "User 2 message")
        assert len(hm.get(1)) == 1
        assert len(hm.get(2)) == 1

    def test_get_last_context_empty(self, tmp_db):
        hm = HistoryManager(tmp_db)
        assert hm.get_last_context(999) == ""

    def test_get_last_context_returns_last_bot_msg(self, tmp_db):
        hm = HistoryManager(tmp_db)
        hm.add(1, "user", "Question")
        hm.add(1, "assistant", "First answer")
        hm.add(1, "user", "Follow up")
        hm.add(1, "assistant", "Second answer")
        assert hm.get_last_context(1) == "Second answer"

    def test_format_for_display_empty(self, tmp_db):
        hm = HistoryManager(tmp_db)
        result = hm.format_for_display(999)
        assert result == "No history yet."

    def test_format_for_display_nonempty(self, tmp_db):
        hm = HistoryManager(tmp_db)
        hm.add(1, "user", "Hello")
        hm.add(1, "assistant", "World")
        result = hm.format_for_display(1)
        assert "Hello" in result
        assert "World" in result


# ---------------------------------------------------------------------------
# format_vision_response
# ---------------------------------------------------------------------------

class TestFormatVisionResponse:
    def test_error_response(self):
        result = format_vision_response({"error": "Model not found"})
        assert "❌" in result
        assert "Model not found" in result

    def test_full_response(self):
        result = format_vision_response({
            "caption": "A cat sitting on a mat",
            "tags": ["cat", "indoor", "pet"],
            "details": "A tabby cat is resting.",
            "model": "Claude Vision API",
        })
        assert "A cat sitting on a mat" in result
        assert "#cat" in result
        assert "#indoor" in result
        assert "Claude Vision API" in result

    def test_missing_tags(self):
        result = format_vision_response({
            "caption": "Something",
            "tags": [],
            "model": "BLIP",
        })
        assert "No tags" in result

    def test_missing_details(self):
        result = format_vision_response({
            "caption": "A dog",
            "tags": ["dog"],
            "model": "BLIP",
        })
        # Should not crash; details section is optional
        assert "A dog" in result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
