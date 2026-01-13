import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _chunked(values, size):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class SQLiteLabelCache:
    """SQLite-backed label cache for id -> (label_en, description_en)."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                id TEXT PRIMARY KEY,
                label_en TEXT,
                description_en TEXT,
                updated_at TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS missing (
                id TEXT PRIMARY KEY,
                ts TEXT
            )
            """
        )
        self._conn.commit()

    def get_many(self, ids):
        if not ids:
            return {}
        results = {}
        with self._lock:
            for batch in _chunked(ids, 500):
                placeholders = ",".join("?" for _ in batch)
                query = f"SELECT id, label_en, description_en FROM labels WHERE id IN ({placeholders})"
                cursor = self._conn.execute(query, batch)
                for row in cursor.fetchall():
                    results[row[0]] = (row[1], row[2])
        return results

    def put_many(self, rows):
        if not rows:
            return
        now = _utc_now_iso()
        payload = [(key, value[0], value[1], now) for key, value in rows.items()]
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO labels (id, label_en, description_en, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label_en=excluded.label_en,
                    description_en=excluded.description_en,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            self._conn.commit()

    def mark_missing(self, ids):
        if not ids:
            return
        now = _utc_now_iso()
        payload = [(entity_id, now) for entity_id in ids]
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO missing (id, ts)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET ts=excluded.ts
                """,
                payload,
            )
            self._conn.commit()

    def get_missing(self, ids):
        if not ids:
            return set()
        results = set()
        with self._lock:
            for batch in _chunked(ids, 500):
                placeholders = ",".join("?" for _ in batch)
                query = f"SELECT id FROM missing WHERE id IN ({placeholders})"
                cursor = self._conn.execute(query, batch)
                for row in cursor.fetchall():
                    results.add(row[0])
        return results

    def is_missing(self, entity_id):
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM missing WHERE id = ?", (entity_id,)).fetchone()
        return row is not None


class SQLiteSnapshotCache:
    """SQLite-backed snapshot cache keyed by qid:revision_id."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                key TEXT PRIMARY KEY,
                status INTEGER,
                payload BLOB,
                content_type TEXT,
                ts TEXT
            )
            """
        )
        self._conn.commit()

    def get(self, key):
        with self._lock:
            row = self._conn.execute(
                "SELECT status, payload, content_type, ts FROM snapshots WHERE key = ?",
                (key,),
            ).fetchone()
        return row

    def put(self, key, status, payload, content_type):
        now = _utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO snapshots (key, status, payload, content_type, ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    content_type=excluded.content_type,
                    ts=excluded.ts
                """,
                (key, status, payload, content_type, now),
            )
            self._conn.commit()

    def delete(self, key):
        with self._lock:
            self._conn.execute("DELETE FROM snapshots WHERE key = ?", (key,))
            self._conn.commit()
