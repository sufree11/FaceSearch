"""Indexing pipeline: gallery building, per-image processing, relabelling, evaluation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from . import config, llm, store
from .faces import FaceEngine, Gallery, load_people, read_image
from .metadata import load_source_meta, normalise_date, read_exif

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
GALLERY_FILE = config.DATA_DIR / "gallery.npz"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.exists() else []


# --------------------------------------------------------------------------- gallery
def build_gallery(engine: FaceEngine, per_person: int = config.GALLERY_PER_PERSON, log=print) -> Gallery:
    """Reference faces for each person.

    Uses data/gallery/<person-id>/ if you have put photos there. Otherwise it picks
    photos with exactly one clear face from data/raw/<person-id>/ and copies them
    into the gallery folder so you can check them and delete wrong ones.
    """
    gallery = Gallery()
    for p in load_people():
        gdir = config.GALLERY_DIR / p["id"]
        gdir.mkdir(parents=True, exist_ok=True)
        if not list_images(gdir):
            picked = 0
            for img_path in list_images(config.RAW_DIR / p["id"]):
                img = read_image(img_path)
                if img is None:
                    continue
                faces = engine.analyze(img, min_size=80)
                if len(faces) == 1 and faces[0].score >= 0.9:
                    shutil.copy2(img_path, gdir / img_path.name)
                    picked += 1
                    if picked >= per_person:
                        break
            log(f"  {p['name']}: auto-picked {picked} single-face photos into {gdir}")

        embs, names = [], []
        for img_path in list_images(gdir):
            img = read_image(img_path)
            faces = engine.analyze(img, min_size=40) if img is not None else []
            if faces:
                embs.append(faces[0].embedding)  # largest face
                names.append(img_path.name)
        if not embs:
            log(f"  {p['name']}: no usable reference faces - skipped")
            continue
        E = np.stack(embs)
        # drop outliers: a reference face that doesn't look like the others is
        # probably someone else (common when photos are auto-picked)
        if len(E) >= 3:
            sim = E @ E.T
            mean_sim = (sim.sum(1) - 1) / (len(E) - 1)
            keep = mean_sim >= max(0.25, float(np.median(mean_sim)) - 0.2)
            for n, k, s in zip(names, keep, mean_sim):
                if not k:
                    log(f"    ! {n} looks different from the other references "
                        f"(avg sim {s:.2f}) - ignored. Delete it from the gallery folder.")
            E = E[keep]
        gallery.people[p["id"]] = E
        log(f"  {p['name']}: {len(E)} reference faces")
    gallery.save(GALLERY_FILE)
    return gallery


def load_gallery() -> Gallery:
    return Gallery.load(GALLERY_FILE)


# --------------------------------------------------------------------------- indexing
def position(box, width: int, n_faces: int) -> str:
    """Where a face is in the frame, so the vision model can tell people apart."""
    if n_faces <= 1 or not width:
        return ""
    cx = (box[0] + box[2] / 2) / width
    return "on the left" if cx < 0.4 else "on the right" if cx > 0.6 else "in the centre"


def identified(faces_with_names: list[tuple[str, tuple]], width: int, n_faces: int) -> list[tuple[str, str]]:
    """[(name, box)] -> [(name, where)], one entry per person."""
    seen, out = set(), []
    for name, box in faces_with_names:
        if name not in seen:
            seen.add(name)
            out.append((name, position(box, width, n_faces)))
    return out


def build_doc_text(names: list[str], date: str | None, rec: dict) -> str:
    """The text that represents an image in the vector index."""
    parts = []
    if names:
        parts.append("People: " + ", ".join(names) + ".")
    if date:
        parts.append(f"Date: {date}.")
    if rec.get("description"):
        parts.append("Description: " + rec["description"])
    if rec.get("original_caption"):
        parts.append("Source caption: " + rec["original_caption"][:500])
    if rec.get("title"):
        parts.append("Title: " + rec["title"])
    return " ".join(parts)


def index_images(engine: FaceEngine, gallery: Gallery, *, force: bool = False,
                 captions: bool = True, limit: int | None = None, log=print) -> int:
    people = {p["id"]: p["name"] for p in load_people()}
    done = set() if force else store.indexed_paths()
    hashes = {imagehash.hex_to_hash(h): p for h, p in store.known_hashes().items()}
    todo = []
    for folder in sorted(d for d in config.RAW_DIR.iterdir() if d.is_dir()):
        meta = load_source_meta(folder)
        per_folder = 0
        for img_path in list_images(folder):
            rel = img_path.relative_to(config.RAW_DIR).as_posix()
            if rel not in done:
                if limit and per_folder >= limit:  # --limit is per person/folder
                    break
                todo.append((img_path, rel, folder.name, meta.get(img_path.name, {})))
                per_folder += 1
    log(f"{len(todo)} images to index")

    count = 0
    for i, (img_path, rel, folder, src) in enumerate(todo, 1):
        try:
            with Image.open(img_path) as im:
                ph = imagehash.phash(im)
        except Exception as e:
            log(f"[{i}/{len(todo)}] {rel}: unreadable ({e})")
            continue
        dup = next((p for h, p in hashes.items() if h - ph <= 4 and p != rel), None)
        if dup:
            log(f"[{i}/{len(todo)}] {rel}: duplicate of {dup} - skipped")
            continue
        img = read_image(img_path)
        if img is None:
            continue

        # 1. faces
        faces = engine.analyze(img)
        for f in faces:
            f.person_id, f.match_score = gallery.identify(f.embedding)
        names = list(dict.fromkeys(people[f.person_id] for f in faces if f.person_id))

        # 2. metadata
        exif = read_exif(img_path)
        if exif.get("exif_date"):
            date, year = normalise_date(exif["exif_date"])
            date_source = "exif"
        else:
            date, year = normalise_date(src.get("date"))
            date_source = "source" if date else "none"
        rec = {
            "path": rel, "folder": folder, "phash": str(ph),
            "width": exif.get("width"), "height": exif.get("height"),
            "date": date, "year": year, "date_source": date_source,
            "title": src.get("title") or img_path.stem.replace("_", " "),
            "original_caption": src.get("original_caption"),
            "author": src.get("author"), "license": src.get("license"),
            "source_url": src.get("source_url"), "camera": exif.get("camera"),
            "gps": json.dumps(exif["gps"]) if exif.get("gps") else None,
        }

        # 3. description from the local vision model
        if captions:
            who = identified([(people[f.person_id], f.box) for f in faces if f.person_id],
                             img.shape[1], len(faces))
            try:
                rec["description"] = llm.describe_image(img_path, who, rec["original_caption"] or "")
            except llm.OllamaError as e:
                if "longer than" not in str(e):
                    raise  # Ollama not running / model missing: stop
                log(f"[{i}/{len(todo)}] {rel}: description timed out - using the source caption "
                    "(fix later with: python cli.py redescribe --all)")
                rec["description"] = rec["original_caption"] or ""
        else:
            rec["description"] = rec["original_caption"] or ""

        # 4. embed for semantic search
        rec["doc_text"] = build_doc_text(names, date, rec)
        rec["embedding"] = llm.embed([rec["doc_text"]])[0]

        store.save_image(rec, faces)
        hashes[ph] = rel
        count += 1
        who = ", ".join(names) or ("unknown face(s)" if faces else "no faces")
        log(f"[{i}/{len(todo)}] {rel}: {len(faces)} face(s) - {who} - {date or 'no date'}")
    return count


PENDING_EMBED = config.DATA_DIR / "redescribe_pending.json"


def redescribe(all_images: bool = False, force: bool = False, log=print) -> int:
    """Regenerate descriptions + embeddings for indexed photos (faces are kept).

    - Only photos with a recognised person are redone (--all adds the rest).
    - Photos whose description already names everyone recognised are skipped,
      so running it again after a crash carries on where it stopped (--force redoes all).
    - A photo that times out is skipped; run the command again to retry it.
    - All descriptions are written first, then embedded together at the end,
      so Ollama doesn't swap between the two models for every photo.
    """
    people = {p["id"]: p["name"] for p in load_people()}
    with store.connect() as con:
        rows = con.execute("SELECT * FROM images ORDER BY id").fetchall()
        faces = {}
        for f in con.execute("SELECT * FROM faces ORDER BY w * h DESC").fetchall():
            faces.setdefault(f["image_id"], []).append(f)

    pending = set(json.loads(PENDING_EMBED.read_text())) if PENDING_EMBED.exists() else set()
    todo, already = [], 0
    for row in rows:
        fs = faces.get(row["id"], [])
        known = [(people[f["person_id"]], (f["x"], f["y"], f["w"], f["h"]))
                 for f in fs if f["person_id"] in people]
        if not known and not all_images:
            continue
        desc = row["description"] or ""
        if not force and row["id"] not in pending:
            if known and all(llm._mentioned(n, desc) for n, _ in known):
                already += 1
                continue
            if not known and desc:
                already += 1
                continue
        if row["id"] in pending:
            continue  # described in an earlier run, only needs embedding
        todo.append((row, fs, known))
    log(f"{len(todo)} photos to describe, {already} already fine"
        + (f", {len(pending)} left to embed from last run" if pending else ""))

    # 1) descriptions (vision model only)
    done, failed, fails_in_a_row = 0, [], 0
    for i, (row, fs, known) in enumerate(todo, 1):
        path = config.RAW_DIR / row["path"]
        if not path.exists():
            continue
        width = row["width"]
        if not width:
            with Image.open(path) as im:
                width = im.width
        who = identified(known, width, len(fs))
        try:
            desc = llm.describe_image(path, who, row["original_caption"] or "")
        except llm.OllamaError as e:
            failed.append(row["path"])
            fails_in_a_row += 1
            log(f"[{i}/{len(todo)}] {row['path']}: SKIPPED ({e})")
            if fails_in_a_row >= 3:
                log("\nOllama failed 3 photos in a row - it may be stuck. Restart the Ollama app "
                    "and run this command again; finished photos are kept.")
                break
            continue
        fails_in_a_row = 0
        rec = {"description": desc, "original_caption": row["original_caption"], "title": row["title"]}
        doc = build_doc_text([n for n, _ in who], row["date"], rec)
        with store.connect() as con:  # old embedding stays until step 2 replaces it
            con.execute("UPDATE images SET description = ?, doc_text = ? WHERE id = ?",
                        (desc, doc, row["id"]))
        pending.add(row["id"])
        PENDING_EMBED.write_text(json.dumps(sorted(pending)))
        done += 1
        log(f"[{i}/{len(todo)}] {row['path']}: {desc[:90]}...")

    # 2) embeddings for everything described (embedding model only, in batches)
    if pending:
        log(f"\nEmbedding {len(pending)} descriptions...")
        ids = sorted(pending)
        with store.connect() as con:
            docs = {r["id"]: r["doc_text"] for r in con.execute(
                f"SELECT id, doc_text FROM images WHERE id IN ({','.join('?' * len(ids))})", ids)}
        for k in range(0, len(ids), 16):
            batch = [i for i in ids[k:k + 16] if docs.get(i)]
            vecs = llm.embed([docs[i] for i in batch])
            with store.connect() as con:
                con.executemany("UPDATE images SET embedding = ? WHERE id = ?",
                                [(store.to_blob(v), i) for v, i in zip(vecs, batch)])
            pending -= set(batch)
            PENDING_EMBED.write_text(json.dumps(sorted(pending)))
        PENDING_EMBED.unlink(missing_ok=True)

    if failed:
        log(f"\n{len(failed)} photo(s) skipped because Ollama was too slow. Run the command again to retry them.")
    return done


def relabel(gallery: Gallery, log=print) -> int:
    """Re-run identification on stored face embeddings (after editing the gallery)."""
    changed = 0
    with store.connect() as con:
        for row in con.execute("SELECT id, person_id, embedding FROM faces").fetchall():
            pid, score = gallery.identify(store.from_blob(row["embedding"]))
            if pid != row["person_id"]:
                changed += 1
            con.execute("UPDATE faces SET person_id = ?, match_score = ? WHERE id = ?",
                        (pid, score, row["id"]))
    log(f"{changed} face labels changed")
    return changed


# --------------------------------------------------------------------------- evaluation
def evaluate(engine: FaceEngine, gallery: Gallery, log=print) -> dict:
    """Face recognition accuracy on photos that are NOT in the gallery.

    Ground truth: a photo in data/raw/<person-id>/ with exactly one face is
    assumed to show that person. Source categories are not perfect, so treat
    the numbers as an estimate and look at the misses by hand.
    """
    people = {p["id"]: p["name"] for p in load_people()}
    rows, total = [], {"n": 0, "correct": 0, "unknown": 0, "wrong": 0}
    for pid, name in people.items():
        refs = {p.name for p in list_images(config.GALLERY_DIR / pid)}
        stats = {"n": 0, "correct": 0, "unknown": 0, "wrong": 0}
        for img_path in list_images(config.RAW_DIR / pid):
            if img_path.name in refs:
                continue
            img = read_image(img_path)
            faces = engine.analyze(img, min_size=60) if img is not None else []
            if len(faces) != 1:
                continue
            got, _ = gallery.identify(faces[0].embedding)
            stats["n"] += 1
            stats["correct" if got == pid else "unknown" if got is None else "wrong"] += 1
        if stats["n"]:
            rows.append((name, stats))
            for k in total:
                total[k] += stats[k]
    log(f"\n{'Person':28} {'test':>5} {'correct':>8} {'unknown':>8} {'wrong':>6}")
    for name, s in rows + [("TOTAL", total)]:
        acc = 100 * s["correct"] / s["n"] if s["n"] else 0
        log(f"{name:28} {s['n']:5d} {acc:7.1f}% {s['unknown']:8d} {s['wrong']:6d}")
    log(f"\nThreshold: {config.FACE_MATCH_THRESHOLD}. More 'unknown' -> lower it; "
        "more 'wrong' -> raise it (FS_FACE_THRESHOLD).")
    return {"people": dict(rows), "total": total}
