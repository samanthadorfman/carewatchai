"""SQLite incident logger."""
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from config import INCIDENTS_DB


INCIDENTS_DB.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(INCIDENTS_DB)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                camera_id   TEXT    NOT NULL,
                track_id    INTEGER NOT NULL,
                event_type  TEXT    NOT NULL,   -- fall | inactivity | wandering | restricted_zone
                zone        TEXT,
                confidence  REAL,
                ack_time    REAL,               -- NULL until acknowledged
                disposition TEXT,              -- false_alarm | assist | clinical
                clip_path   TEXT
            )
        """)


@dataclass
class Incident:
    camera_id: str
    track_id: int
    event_type: str
    zone: str = "default"
    confidence: float = 1.0
    clip_path: str | None = None

    def save(self) -> int:
        init_db()
        with _conn() as con:
            cur = con.execute(
                """INSERT INTO incidents
                   (timestamp, camera_id, track_id, event_type, zone, confidence, clip_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), self.camera_id, self.track_id,
                 self.event_type, self.zone, self.confidence, self.clip_path),
            )
            return cur.lastrowid


def acknowledge(incident_id: int, disposition: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE incidents SET ack_time=?, disposition=? WHERE id=?",
            (time.time(), disposition, incident_id),
        )


def recent_incidents(limit: int = 50) -> list[sqlite3.Row]:
    init_db()
    with _conn() as con:
        return con.execute(
            "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
