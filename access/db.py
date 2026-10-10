"""Local SQLite store: studies, runs, findings, warnings and the audit log (R10).

The audit log is append-only: every change a person makes (and every automatic draft) is
recorded as what changed, from what, to what, by whom, when. Nothing is ever deleted from
it, and rows are never updated.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY, uid TEXT UNIQUE NOT NULL, pseudonym TEXT, folder TEXT NOT NULL, imported_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'importing', progress TEXT, laterality TEXT, note_side TEXT, doctor_side TEXT,
    override_reason TEXT, segment_status TEXT NOT NULL DEFAULT '{}', danger TEXT NOT NULL DEFAULT '{}',
    review_started_at REAL, signed_by TEXT, signed_at REAL, report_json TEXT, report_pdf TEXT, measured INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, study_id INTEGER NOT NULL REFERENCES studies(id), idx INTEGER NOT NULL, path TEXT NOT NULL,
    info TEXT NOT NULL, best TEXT, calibration TEXT, cache_dir TEXT, draft_notes TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY, study_id INTEGER NOT NULL, run_id INTEGER NOT NULL, segment TEXT, status TEXT NOT NULL DEFAULT 'draft',
    data TEXT NOT NULL, inputs TEXT NOT NULL, created_by TEXT NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS warnings (
    id INTEGER PRIMARY KEY, study_id INTEGER NOT NULL, run_id INTEGER, kind TEXT NOT NULL, message TEXT NOT NULL,
    data TEXT, acknowledged_by TEXT, acknowledged_at REAL, response TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, username TEXT NOT NULL, study_id INTEGER, entity TEXT NOT NULL,
    entity_id TEXT, field TEXT, old TEXT, new TEXT
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""

JSON_COLUMNS = {"progress", "laterality", "segment_status", "danger", "info", "best", "calibration", "data", "inputs", "draft_notes"}


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as con:
            con.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    @staticmethod
    def _decode(row) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for k in JSON_COLUMNS & d.keys():
            if d[k] is not None:
                d[k] = json.loads(d[k])
        return d

    def query(self, sql: str, args=()) -> list[dict]:
        with self.connect() as con:
            return [self._decode(r) for r in con.execute(sql, args).fetchall()]

    def one(self, sql: str, args=()) -> dict | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def execute(self, sql: str, args=()) -> int:
        with self._lock, self.connect() as con:
            cur = con.execute(sql, [json.dumps(a) if isinstance(a, (dict, list)) else a for a in args])
            return cur.lastrowid

    # ---- audit ----
    def audit(self, username: str, study_id, entity: str, entity_id, field: str | None, old, new) -> None:
        enc = lambda v: v if v is None or isinstance(v, str) else json.dumps(v)  # noqa: E731
        self.execute("INSERT INTO audit (ts, username, study_id, entity, entity_id, field, old, new) VALUES (?,?,?,?,?,?,?,?)",
                     (time.time(), username, study_id, entity, None if entity_id is None else str(entity_id), field, enc(old), enc(new)))

    EDITABLE = {
        "studies": {"note_side", "doctor_side", "override_reason", "segment_status", "danger", "laterality", "review_started_at",
                    "state", "measured", "signed_by", "signed_at", "report_json", "report_pdf"},
        "runs": {"best", "calibration", "draft_notes"},
        "findings": {"segment", "status", "data", "inputs"},
        "warnings": {"acknowledged_by", "acknowledged_at", "response"},
    }

    def update_field(self, table: str, row_id: int, field: str, value, username: str, study_id) -> None:
        """Change one column and log it: the only way user edits reach the database."""
        if field not in self.EDITABLE.get(table, ()):
            raise ValueError(f"{table}.{field} is not editable")
        old = self.one(f"SELECT {field} FROM {table} WHERE id = ?", (row_id,))
        old = None if old is None else old[field]
        self.execute(f"UPDATE {table} SET {field} = ? WHERE id = ?", (value, row_id))
        self.audit(username, study_id, table, row_id, field, old, value)
