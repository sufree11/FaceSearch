"""SQLite storage for images, faces and their embeddings.

Vectors are stored as float32 blobs and searched in memory with numpy. For a
few thousand photos this is instant; swap in Chroma/Qdrant/pgvector if the
archive grows to hundreds of thousands of images.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id               INTEGER PRIMARY KEY,
    path             TEXT UNIQUE NOT NULL,   -- relative to data/raw
    folder           TEXT,                   -- the folder it came from (source label)
    phash            TEXT,
    width            INTEGER,
    height           INTEGER,
    date             TEXT,                   -- best known date, ISO-ish ("2018-05-10" or "2018")
    year             INTEGER,
    date_source      TEXT,                   -- exif | source | none
    title            TEXT,
    original_caption TEXT,                   -- caption from the source page
    description      TEXT,                   -- AI-generated description
    author           TEXT,
    license          TEXT,
    source_url       TEXT,                   -- file page on the web
    camera           TEXT,
    gps              TEXT,
    doc_text         TEXT,                   -- the text that was embedded
    embedding        BLOB,
    indexed_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    person_id   TEXT,                        -- NULL = unknown face
    match_score REAL,
    det_score   REAL,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS faces_image ON faces(image_id);
"""


@contextmanager
def connect():
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    _migrate(con)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _migrate(con):
    """Add columns introduced after the first version, so old databases keep working."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    if "desc_source" not in cols:
        # 'ai' = written by the vision model, 'caption' = Commons caption, AI description still to come
        con.execute("ALTER TABLE images ADD COLUMN desc_source TEXT")


def to_blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


def indexed_paths() -> set[str]:
    with connect() as con:
        return {r[0] for r in con.execute("SELECT path FROM images")}


def known_hashes() -> dict[str, str]:
    with connect() as con:
        return {r[0]: r[1] for r in con.execute("SELECT phash, path FROM images WHERE phash IS NOT NULL")}


def save_image(rec: dict, faces: list) -> int:
    cols = ["path", "folder", "phash", "width", "height", "date", "year", "date_source",
            "title", "original_caption", "description", "author", "license", "source_url",
            "camera", "gps", "doc_text", "desc_source"]
    with connect() as con:
        con.execute("DELETE FROM images WHERE path = ?", (rec["path"],))
        cur = con.execute(
            f"INSERT INTO images ({', '.join(cols)}, embedding) VALUES ({', '.join('?' * (len(cols) + 1))})",
            [rec.get(c) for c in cols] + [to_blob(rec.get("embedding"))])
        image_id = cur.lastrowid
        con.executemany(
            "INSERT INTO faces (image_id, person_id, match_score, det_score, x, y, w, h, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(image_id, f.person_id, f.match_score, f.score, *f.box, to_blob(f.embedding)) for f in faces])
        return image_id


def update_description(image_id: int, description: str, doc_text: str, embedding, source: str = "ai"):
    with connect() as con:
        con.execute("UPDATE images SET description = ?, doc_text = ?, embedding = COALESCE(?, embedding), "
                    "desc_source = ? WHERE id = ?",
                    (description, doc_text, to_blob(embedding), source, image_id))


def image_record(row: sqlite3.Row, faces: list[sqlite3.Row], names: dict[str, str]) -> dict:
    """Shape an image row for the API / frontend."""
    return {
        "id": row["id"],
        "url": "/media/" + row["path"].replace("\\", "/"),
        "width": row["width"],
        "height": row["height"],
        "date": row["date"],
        "year": row["year"],
        "date_source": row["date_source"],
        "title": row["title"],
        "description": row["description"],
        "original_caption": row["original_caption"],
        "author": row["author"],
        "license": row["license"],
        "source_url": row["source_url"],
        "camera": row["camera"],
        "gps": json.loads(row["gps"]) if row["gps"] else None,
        "folder": row["folder"],
        "desc_source": row["desc_source"] if "desc_source" in row.keys() else None,
        "faces": [{
            "person_id": f["person_id"],
            "name": names.get(f["person_id"]) if f["person_id"] else None,
            "score": round(f["match_score"] or 0, 3),
            "box": [f["x"], f["y"], f["w"], f["h"]],
        } for f in faces],
    }
