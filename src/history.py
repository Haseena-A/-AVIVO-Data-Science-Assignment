"""
history.py — Per-user message history (last N interactions in memory + SQLite)
"""

import sqlite3
import json
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

MAX_HISTORY = 3  # Keep last N interactions per user


class HistoryManager:
    """
    Dual-layer history:
    - In-memory dict for fast access during session
    - SQLite for persistence across restarts
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._cache: dict[int, list[dict]] = defaultdict(list)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                ts          INTEGER DEFAULT (strftime('%s','now'))
            )
        """)
        self.conn.commit()

    def add(self, user_id: int, role: str, content: str):
        """Add a new message to history."""
        entry = {"role": role, "content": content, "ts": int(time.time())}
        self._cache[user_id].append(entry)
        # Keep only last MAX_HISTORY
        self._cache[user_id] = self._cache[user_id][-MAX_HISTORY * 2:]

        self.conn.execute(
            "INSERT INTO user_history (user_id, role, content) VALUES (?,?,?)",
            (user_id, role, content),
        )
        # Prune old DB rows for this user
        self.conn.execute("""
            DELETE FROM user_history
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM user_history
                WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
            )
        """, (user_id, user_id, MAX_HISTORY * 2))
        self.conn.commit()

    def get(self, user_id: int) -> list[dict]:
        """Return last N interactions for user (from cache or DB)."""
        if user_id in self._cache and self._cache[user_id]:
            return self._cache[user_id]

        rows = self.conn.execute("""
            SELECT role, content FROM user_history
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
        """, (user_id, MAX_HISTORY * 2)).fetchall()

        history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        self._cache[user_id] = history
        return history

    def clear(self, user_id: int):
        """Clear history for a specific user."""
        self._cache.pop(user_id, None)
        self.conn.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def format_for_display(self, user_id: int) -> str:
        """Return a readable history string."""
        history = self.get(user_id)
        if not history:
            return "No history yet."

        lines = []
        for entry in history[-MAX_HISTORY * 2:]:
            icon = "👤" if entry["role"] == "user" else "🤖"
            text = entry["content"][:200] + ("..." if len(entry["content"]) > 200 else "")
            lines.append(f"{icon} {text}")
        return "\n\n".join(lines)

    def get_last_context(self, user_id: int) -> str:
        """Get the last bot response for /summarize."""
        history = self.get(user_id)
        bot_msgs = [e for e in history if e["role"] == "assistant"]
        return bot_msgs[-1]["content"] if bot_msgs else ""
