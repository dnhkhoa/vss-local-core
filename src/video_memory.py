from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class VideoMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def index_events(self, events_data: dict) -> None:
        video_path = str(events_data.get("video_path") or "")
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            video_id = self._upsert_video(events_data, video_path, now)
            self._delete_video_children(video_id)

            segment_pk_by_id: dict[str, int] = {}
            for segment in events_data.get("segments") or []:
                segment_pk = self._insert_segment(video_id, video_path, segment)
                segment_pk_by_id[str(segment.get("segment_id"))] = segment_pk
                self._insert_segment_children(segment_pk, segment)

            self._insert_tracks(video_id, segment_pk_by_id, events_data.get("tracking_summary") or {})

    def stats(self) -> dict:
        return {
            "db_path": str(self.db_path),
            "videos": self._count("videos"),
            "segments": self._count("segments"),
            "frames": self._count("frames"),
            "people": self._count("people"),
            "objects": self._count("objects"),
            "tracks": self._count("tracks"),
        }

    def get_events_by_fingerprint(self, video_path: str, fingerprint: str) -> dict | None:
        row = self.connection.execute(
            """
            SELECT raw_json
            FROM videos
            WHERE path = ? AND fingerprint = ?
            """,
            (video_path, fingerprint),
        ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["raw_json"])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def search_segments(self, query: str, limit: int = 8) -> list[dict]:
        tokens = [token for token in re_split_query(query) if token.strip()]
        normalized_query = " OR ".join(tokens)
        if not normalized_query:
            return []

        rows = self.connection.execute(
            """
            SELECT s.*
            FROM segment_fts f
            JOIN segments s ON s.id = f.segment_pk
            WHERE segment_fts MATCH ?
            ORDER BY bm25(segment_fts)
            LIMIT ?
            """,
            (normalized_query, limit),
        ).fetchall()
        if not rows:
            rows = self.connection.execute(
                """
                SELECT *
                FROM segments
                WHERE searchable_text LIKE ? OR summary LIKE ?
                ORDER BY start_time
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        return [self._segment_row_to_dict(row) for row in rows]

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    segment_seconds REAL,
                    frames_per_segment INTEGER,
                    fingerprint TEXT,
                    analyzed_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                    segment_id TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    scene TEXT,
                    location TEXT,
                    summary TEXT,
                    person_count INTEGER,
                    risk_level TEXT,
                    searchable_text TEXT,
                    raw_json TEXT NOT NULL,
                    UNIQUE(video_id, segment_id)
                );

                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    path TEXT,
                    timestamp TEXT,
                    timestamp_sec REAL
                );

                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    person_label TEXT,
                    clothing_colors TEXT,
                    position TEXT
                );

                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    name TEXT,
                    count INTEGER,
                    position TEXT,
                    attributes TEXT
                );

                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    activity TEXT
                );

                CREATE TABLE IF NOT EXISTS visible_text (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    text TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    event_type TEXT,
                    description TEXT,
                    approx_time TEXT,
                    risk_level TEXT
                );

                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                    track_label TEXT,
                    first_time TEXT,
                    last_time TEXT,
                    observation_count INTEGER
                );

                CREATE TABLE IF NOT EXISTS track_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                    segment_id INTEGER REFERENCES segments(id) ON DELETE SET NULL,
                    frame_path TEXT,
                    timestamp TEXT,
                    bbox_json TEXT,
                    confidence REAL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5(
                    segment_pk UNINDEXED,
                    video_path UNINDEXED,
                    content
                );

                CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
                CREATE INDEX IF NOT EXISTS idx_frames_segment ON frames(segment_id);
                CREATE INDEX IF NOT EXISTS idx_tracks_video ON tracks(video_id);
                """
            )
            self._ensure_column("videos", "fingerprint", "TEXT")
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_fingerprint ON videos(path, fingerprint)"
            )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            try:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

    def _upsert_video(self, events_data: dict, video_path: str, analyzed_at: str) -> int:
        existing = self.connection.execute(
            "SELECT id FROM videos WHERE path = ?",
            (video_path,),
        ).fetchone()
        raw_json = json.dumps(events_data, ensure_ascii=False)
        fingerprint = (events_data.get("ingestion") or {}).get("fingerprint")
        if existing:
            video_id = int(existing["id"])
            self.connection.execute(
                """
                UPDATE videos
                SET segment_seconds = ?, frames_per_segment = ?, fingerprint = ?, analyzed_at = ?, raw_json = ?
                WHERE id = ?
                """,
                (
                    events_data.get("segment_seconds"),
                    events_data.get("frames_per_segment"),
                    fingerprint,
                    analyzed_at,
                    raw_json,
                    video_id,
                ),
            )
            return video_id

        cursor = self.connection.execute(
            """
            INSERT INTO videos(path, segment_seconds, frames_per_segment, fingerprint, analyzed_at, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                video_path,
                events_data.get("segment_seconds"),
                events_data.get("frames_per_segment"),
                fingerprint,
                analyzed_at,
                raw_json,
            ),
        )
        return int(cursor.lastrowid)

    def _delete_video_children(self, video_id: int) -> None:
        segment_ids = [row["id"] for row in self.connection.execute("SELECT id FROM segments WHERE video_id = ?", (video_id,))]
        if segment_ids:
            placeholders = ",".join("?" for _ in segment_ids)
            self.connection.execute(f"DELETE FROM segment_fts WHERE segment_pk IN ({placeholders})", segment_ids)
        self.connection.execute("DELETE FROM tracks WHERE video_id = ?", (video_id,))
        self.connection.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))

    def _insert_segment(self, video_id: int, video_path: str, segment: dict) -> int:
        searchable_text = str(segment.get("searchable_text") or _build_searchable_text(segment))
        cursor = self.connection.execute(
            """
            INSERT INTO segments(
                video_id, segment_id, start_time, end_time, scene, location, summary,
                person_count, risk_level, searchable_text, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                segment.get("segment_id"),
                segment.get("start_time"),
                segment.get("end_time"),
                segment.get("scene"),
                segment.get("location"),
                segment.get("summary"),
                segment.get("person_count"),
                segment.get("risk_level"),
                searchable_text,
                json.dumps(segment, ensure_ascii=False),
            ),
        )
        segment_pk = int(cursor.lastrowid)
        self.connection.execute(
            "INSERT INTO segment_fts(segment_pk, video_path, content) VALUES (?, ?, ?)",
            (segment_pk, video_path, searchable_text),
        )
        return segment_pk

    def _insert_segment_children(self, segment_pk: int, segment: dict) -> None:
        for frame in segment.get("sampled_frames") or []:
            self.connection.execute(
                "INSERT INTO frames(segment_id, path, timestamp, timestamp_sec) VALUES (?, ?, ?, ?)",
                (segment_pk, frame.get("path"), frame.get("timestamp"), frame.get("timestamp_sec")),
            )
        for person in segment.get("people") or []:
            self.connection.execute(
                "INSERT INTO people(segment_id, person_label, clothing_colors, position) VALUES (?, ?, ?, ?)",
                (
                    segment_pk,
                    person.get("person_label"),
                    json.dumps(person.get("clothing_colors") or [], ensure_ascii=False),
                    person.get("position"),
                ),
            )
        for obj in segment.get("objects") or []:
            self.connection.execute(
                "INSERT INTO objects(segment_id, name, count, position, attributes) VALUES (?, ?, ?, ?, ?)",
                (
                    segment_pk,
                    obj.get("name"),
                    obj.get("count"),
                    obj.get("position"),
                    json.dumps(obj.get("attributes") or [], ensure_ascii=False),
                ),
            )
        for activity in segment.get("activities") or []:
            self.connection.execute(
                "INSERT INTO activities(segment_id, activity) VALUES (?, ?)",
                (segment_pk, activity),
            )
        for text in segment.get("visible_text") or []:
            self.connection.execute(
                "INSERT INTO visible_text(segment_id, text) VALUES (?, ?)",
                (segment_pk, text),
            )
        for event in segment.get("events") or []:
            self.connection.execute(
                """
                INSERT INTO events(segment_id, event_type, description, approx_time, risk_level)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    segment_pk,
                    event.get("event_type"),
                    event.get("description"),
                    event.get("approx_time"),
                    event.get("risk_level"),
                ),
            )

    def _insert_tracks(self, video_id: int, segment_pk_by_id: dict[str, int], tracking_summary: dict) -> None:
        for track in tracking_summary.get("tracks") or []:
            cursor = self.connection.execute(
                """
                INSERT INTO tracks(video_id, track_label, first_time, last_time, observation_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    track.get("track_label"),
                    track.get("first_time"),
                    track.get("last_time"),
                    track.get("observation_count"),
                ),
            )
            track_pk = int(cursor.lastrowid)
            for observation in track.get("observations") or []:
                self.connection.execute(
                    """
                    INSERT INTO track_observations(
                        track_id, segment_id, frame_path, timestamp, bbox_json, confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        track_pk,
                        segment_pk_by_id.get(str(observation.get("segment_id"))),
                        observation.get("frame_path"),
                        observation.get("timestamp"),
                        json.dumps(observation.get("bbox") or [], ensure_ascii=False),
                        observation.get("confidence"),
                    ),
                )

    def _segment_row_to_dict(self, row: sqlite3.Row) -> dict:
        segment = json.loads(row["raw_json"])
        segment["db_segment_id"] = row["id"]
        return segment

    def _count(self, table: str) -> int:
        return int(self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])


def index_events_to_memory(db_path: str, events_data: dict) -> dict:
    store = VideoMemoryStore(db_path)
    try:
        store.index_events(events_data)
        return store.stats()
    finally:
        store.close()


def _build_searchable_text(segment: dict) -> str:
    parts = [
        segment.get("scene"),
        segment.get("location"),
        segment.get("summary"),
        " ".join(segment.get("activities") or []),
        " ".join(segment.get("visible_text") or []),
    ]
    for person in segment.get("people") or []:
        parts.append(person.get("position"))
        parts.extend(person.get("clothing_colors") or [])
    for obj in segment.get("objects") or []:
        parts.append(obj.get("name"))
        parts.append(obj.get("position"))
        parts.extend(obj.get("attributes") or [])
    for event in segment.get("events") or []:
        parts.append(event.get("event_type"))
        parts.append(event.get("description"))
    return " ".join(str(part) for part in parts if part)


def re_split_query(query: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z0-9_]+", query)
