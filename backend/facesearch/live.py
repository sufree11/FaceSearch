"""Live fetch: when the archive has too few photos for a question, get more from
Wikimedia Commons on the spot.

  catalog (made in advance)  ->  pick files for the person + year not downloaded yet
  -> download a few in parallel -> face check (YuNet + SFace) -> save with the
  Commons caption as a temporary description -> embed -> searchable

Writing the AI description takes the vision model tens of seconds per photo on a
CPU, so that is done afterwards by the Describer thread below.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import imagehash
import requests
from PIL import Image

from . import config, llm, parse, store
from .commons import load_catalog
from .faces import load_people, read_image
from .indexer import build_doc_text, identified
from .metadata import normalise_date, read_exif

_lock = threading.Lock()          # one live fetch at a time
_failed: set[str] = set()         # download links that are gone (404 etc.), skipped this session
MAX_BYTES = 15 * 1024 * 1024


def _year_ok(year: int | None, parsed: dict) -> bool:
    y0, y1, years = parsed.get("year_from"), parsed.get("year_to"), parsed.get("years")
    if not (y0 or y1 or years):
        return True
    if year is None:
        return False
    if years:
        return year in years
    return not (y0 and year < y0 or y1 and year > y1)


def candidates(person: dict, parsed: dict) -> list[dict]:
    """Catalog files for this person and year that are not downloaded yet.
    Files whose title or caption names the person come first (more likely to show them)."""
    cat = load_catalog(person["id"])
    if not cat:
        return []
    folder = config.RAW_DIR / person["id"]
    have = {p.name for p in folder.iterdir()} if folder.exists() else set()
    keys = {t for n in [person["name"], *person.get("aliases", [])]
            for t in parse.norm(n).split() if len(t) > 3 and t not in parse.TITLES}
    out = []
    for f in cat["files"]:
        if f["file"] in have or f["download_url"] in _failed or not _year_ok(f.get("year"), parsed):
            continue
        text = parse.norm(f"{f.get('title', '')} {f.get('original_caption', '')}")
        out.append((0 if keys & set(text.split()) else 1, f))
    out.sort(key=lambda x: x[0])
    return [f for _, f in out]


def has_catalog(person_id: str) -> bool:
    return load_catalog(person_id) is not None


def _download(session: requests.Session, url: str) -> tuple[bytes | None, bool]:
    """(photo bytes or None, True if the file is permanently unusable)."""
    try:
        r = session.get(url, timeout=(5, 20))
    except requests.RequestException:
        return None, False  # offline / slow: may work next time
    if r.status_code in (403, 404, 410):
        return None, True
    if not r.ok:
        return None, False
    if not r.headers.get("Content-Type", "image").startswith("image") or len(r.content) > MAX_BYTES:
        return None, True
    return r.content, False


def _analyse(engine, gallery, people: dict, path, folder: str, src: dict, hashes: dict):
    """Faces + metadata for one downloaded photo. Returns (record, faces) or None if it's a duplicate."""
    rel = path.relative_to(config.RAW_DIR).as_posix()
    with Image.open(path) as im:
        ph = imagehash.phash(im)
    if any(h - ph <= 4 for h in hashes):
        return None
    img = read_image(path)
    if img is None:
        return None
    faces = engine.analyze(img)
    for f in faces:
        f.person_id, f.match_score = gallery.identify(f.embedding)
    names = list(dict.fromkeys(people[f.person_id] for f in faces if f.person_id))
    exif = read_exif(path)
    if exif.get("exif_date"):
        date, year = normalise_date(exif["exif_date"])
        date_source = "exif"
    else:
        date, year = src.get("norm_date"), src.get("year")
        date_source = "source" if date else "none"
    caption = src.get("original_caption") or ""
    rec = {
        "path": rel, "folder": folder, "phash": str(ph),
        "width": exif.get("width"), "height": exif.get("height"),
        "date": date, "year": year, "date_source": date_source,
        "title": src.get("title") or path.stem.replace("_", " "),
        "original_caption": caption, "author": src.get("author"), "license": src.get("license"),
        "source_url": src.get("source_url"), "camera": exif.get("camera"),
        "gps": json.dumps(exif["gps"]) if exif.get("gps") else None,
        # temporary: the Describer replaces this with an AI description later
        "description": caption or src.get("title") or "", "desc_source": "caption",
    }
    rec["doc_text"] = build_doc_text(names, date, rec)
    hashes[ph] = rel
    return rec, faces


def fetch(svc, person_id: str, parsed: dict, need: int, log=print) -> dict:
    """Download, face-check and save up to `need` new photos of one person.

    Returns {"added": photos showing the person, "checked": photos downloaded,
             "new_ids": image ids saved, "reason": why nothing was added (or None)}.
    Never raises: on any network problem the search just uses the archive.
    """
    result = {"added": 0, "checked": 0, "new_ids": [], "reason": None}
    network_errors = 0
    person = next((p for p in load_people() if p["id"] == person_id), None)
    if not person:
        result["reason"] = "not in people.json"
        return result
    with _lock:
        cands = candidates(person, parsed)[: config.LIVE_MAX_CANDIDATES]
        if not cands:
            result["reason"] = ("no catalog yet - run: python cli.py catalog" if not has_catalog(person_id)
                                else "no more Commons photos for this search")
            return result

        folder = config.RAW_DIR / person_id
        folder.mkdir(parents=True, exist_ok=True)
        meta_file = folder / "metadata.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        engine, gallery = svc.get_engine(), svc.gallery
        names = {p["id"]: p["name"] for p in load_people()}
        hashes = {imagehash.hex_to_hash(h): p for h, p in store.known_hashes().items()}
        session = requests.Session()
        session.headers["User-Agent"] = config.WIKI_USER_AGENT
        deadline = time.time() + config.LIVE_TIME_BUDGET
        new = []

        pool = ThreadPoolExecutor(config.LIVE_WORKERS)
        try:
            futures = {pool.submit(_download, session, c["download_url"]): c for c in cands}
            for fut in as_completed(futures):
                if result["added"] >= need or time.time() > deadline:
                    break
                c, (data, gone) = futures[fut], fut.result()
                if data is None:
                    if gone:
                        _failed.add(c["download_url"])
                    else:
                        network_errors += 1
                    continue
                path = folder / c["file"]
                path.write_bytes(data)
                meta[c["file"]] = {k: v for k, v in c.items()
                                   if k not in ("download_url", "file", "norm_date", "year")} | {
                    "source": "Wikimedia Commons (live)"}
                result["checked"] += 1
                try:
                    out = _analyse(engine, gallery, names, path, person_id, c, hashes)
                except Exception as e:  # unreadable image etc.
                    log(f"live: {c['file']}: {e}")
                    continue
                if out is None:
                    continue  # duplicate of a photo we already have
                new.append(out)
                if any(f.person_id == person_id for f in out[1]):
                    result["added"] += 1
        except Exception as e:
            log(f"live fetch stopped: {e}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        if new:
            try:  # one batch, fast; if Ollama is down they're still found by face
                vecs = llm.embed([rec["doc_text"] for rec, _ in new])
            except llm.OllamaError:
                vecs = [None] * len(new)
            for (rec, faces), vec in zip(new, vecs):
                rec["embedding"] = vec
                result["new_ids"].append(store.save_image(rec, faces))
        if not result["checked"]:
            result["reason"] = ("could not download from Wikimedia Commons (offline?)" if network_errors
                                else "no more Commons photos for this search")
        elif not result["added"]:
            result["reason"] = "no new photos passed the face check"
        log(f"live: {person['name']}: downloaded {result['checked']}, "
            f"{result['added']} show them, {len(new)} saved")
        return result


# --------------------------------------------------------------------------- background
def describe_one(image_id: int, people: dict) -> bool:
    """Replace a temporary Commons caption with an AI description. False if nothing to do."""
    with store.connect() as con:
        row = con.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        faces = con.execute("SELECT * FROM faces WHERE image_id = ? ORDER BY w * h DESC",
                            (image_id,)).fetchall()
    if not row or row["desc_source"] != "caption":
        return False
    path = config.RAW_DIR / row["path"]
    if not path.exists():
        return False
    known = [(people[f["person_id"]], (f["x"], f["y"], f["w"], f["h"]))
             for f in faces if f["person_id"] in people]
    if not known:
        return False  # nobody recognised: not worth minutes of CPU; keeps the Commons caption
    width = row["width"]
    if not width:
        with Image.open(path) as im:
            width = im.width
    who = identified(known, width, len(faces))
    desc = llm.describe_image(path, who, row["original_caption"] or "")
    rec = {"description": desc, "original_caption": row["original_caption"], "title": row["title"]}
    doc = build_doc_text([n for n, _ in who], row["date"], rec)
    store.update_description(image_id, desc, doc, llm.embed([doc])[0])
    return True


class Describer(threading.Thread):
    """Writes AI descriptions for live-fetched photos in the background, one at a time,
    waiting while someone is chatting so it doesn't slow the answer down."""

    def __init__(self, on_change):
        super().__init__(daemon=True, name="describer")
        self.q: queue.Queue[int] = queue.Queue()
        self.queued: set[int] = set()
        self.on_change = on_change
        self.last_chat = 0.0

    def enqueue(self, ids):
        for i in ids:
            if i not in self.queued:
                self.queued.add(i)
                self.q.put(i)

    def enqueue_pending(self):
        """Photos with a recognised person still showing a Commons caption (e.g. after a restart)."""
        with store.connect() as con:
            ids = [r[0] for r in con.execute(
                "SELECT DISTINCT i.id FROM images i JOIN faces f ON f.image_id = i.id "
                "WHERE i.desc_source = 'caption' AND f.person_id IS NOT NULL ORDER BY i.id")]
        self.enqueue(ids)

    def run(self):
        people = {p["id"]: p["name"] for p in load_people()}
        while True:
            image_id = self.q.get()
            while time.time() - self.last_chat < 15:  # let chat answers go first
                time.sleep(2)
            try:
                if describe_one(image_id, people):
                    self.on_change()
                self.queued.discard(image_id)
            except llm.OllamaError:
                time.sleep(60)          # Ollama busy or closed: try again later
                self.q.put(image_id)
            except Exception:
                self.queued.discard(image_id)
