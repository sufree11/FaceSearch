"""Learn a new public figure on the fly.

  Wikidata (who is it?)  ->  Commons catalog + portrait  ->  download ~12 photos
  -> reference faces: clear single-face photos that match the Wikidata portrait
  -> add to data/people_auto.json + gallery  ->  index the photos  ->  relabel the archive

After this the person works exactly like the ones in people.json, including live search.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from . import config, live, llm, parse, store, wikidata
from .commons import Commons, WikimediaBusy, build_catalog, safe_name
from .faces import load_people, read_image
from .indexer import GALLERY_FILE, relabel
from .metadata import normalise_date


class OnboardError(Exception):
    pass


class LearnerBusy(Exception):
    """Another person is being learned right now (one at a time)."""


_lock = threading.Lock()  # learn one person at a time


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", parse.norm(name)).strip("-") or "person"


def _save_person(person: dict):
    f = config.PEOPLE_AUTO_FILE
    people = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    people = [p for p in people if p["id"] != person["id"]] + [person]
    f.write_text(json.dumps(people, indent=2, ensure_ascii=False), encoding="utf-8")


def learn(svc, wd: dict, log=print, patient: bool = False) -> str:
    """Make `wd` (from wikidata.lookup) a known person. Returns the person id.
    Not patient (someone is waiting in the chat): raises WikimediaBusy when Wikimedia is
    rate-limiting us and LearnerBusy when another person is being learned."""
    if not _lock.acquire(timeout=-1 if patient else 2):  # -1: wait as long as it takes
        raise LearnerBusy()
    try:
        known = next((p for p in load_people() if p.get("qid") == wd["qid"]), None)
        if known:  # learned by another request meanwhile
            return known["id"]
        return _learn(svc, wd, log, patient)
    finally:
        _lock.release()


# ------------------------------------------------------------------ learning in the background
# Under Wikimedia's rate limits a new person can take a few minutes (about 6-10 API requests,
# and only a few may go through per minute). Rather than make someone wait, the chat hands the
# job to this one background thread, which waits as long as Wikimedia asks; asking again later
# finds the person ready.
_jobs: queue.Queue = queue.Queue()
_pending: dict[str, dict] = {}  # key -> {"label", "since", "confirmed": found on Wikidata yet}
_failed: dict[str, str] = {}    # key -> why the last background attempt failed
_done: dict[str, str] = {}      # key -> person id learned in the background (for status())
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def job_key(name: str) -> str:
    return parse.norm(name)


def pending(key: str) -> dict | None:
    return _pending.get(key)


def status(key: str) -> dict:
    """What the website polls while someone is learned in the background:
    {"state": "pending", "label", "minutes"} | {"state": "done"} | {"state": "failed"} |
    {"state": "unknown"} (no such job, e.g. the server was restarted). Reading it changes nothing:
    the next question still reports a failure (take_failure) or finds the person."""
    job = _pending.get(key)
    if job:
        return {"state": "pending", "label": job["label"], "confirmed": job["confirmed"],
                "minutes": int((time.time() - job["since"]) // 60)}
    if key in _failed:
        return {"state": "failed"}
    if key in _done:
        return {"state": "done"}
    return {"state": "unknown"}


def take_failure(key: str) -> str | None:
    """Why learning `key` in the background failed (reported once, so the next ask retries)."""
    return _failed.pop(key, None)


def learn_later(svc, key: str, label: str, wd: dict | None = None, names: tuple = (),
                hints=(), years=None, log=print) -> bool:
    """Queue a person to be learned in the background: either `wd` (already found on Wikidata)
    or the `names` to look up first. Returns False if they're already queued."""
    global _worker
    with _worker_lock:
        if key in _pending:
            return False
        _pending[key] = {"label": label, "since": time.time(), "confirmed": wd is not None}
        _failed.pop(key, None)
        _done.pop(key, None)
        _jobs.put({"key": key, "wd": wd, "names": names, "hints": hints, "years": years})
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work, args=(svc, log), daemon=True, name="learner")
            _worker.start()
    return True


def _work(svc, log):
    while True:
        job = _jobs.get()
        key = job["key"]
        try:
            wd = job["wd"]
            for name in job["names"]:
                wd = wd or wikidata.lookup(name, job["hints"], job["years"], patient=True)
            if not wd:
                _failed[key] = "not on Wikidata"
            else:
                _pending[key].update(label=wd["name"], confirmed=True)  # "Robert Downey Jr."
                _done[key] = learn(svc, wd, log=log, patient=True)
        except OnboardError as e:
            _failed[key] = str(e)
        except WikimediaBusy:
            mins = max(1, round((time.time() - _pending[key]["since"]) / 60))
            _failed[key] = (f"Wikimedia refused almost every request for {mins} minutes, so I gave up "
                            f"for now. Ask again to try once more")
        except requests.RequestException as e:
            _failed[key] = f"could not reach Wikimedia ({e})"
        except Exception as e:  # never let the worker die
            _failed[key] = f"unexpected error: {e}"
            log(f"background learning of {key} failed: {e!r}")
        finally:
            _pending.pop(key, None)


def _cleanup(pid: str):
    """Remove photos and catalog left by a failed attempt to learn `pid` (only files this module
    downloaded, recognisable by their metadata), so they don't end up in the archive."""
    from .commons import catalog_path
    folder = config.RAW_DIR / pid
    meta_file = folder / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for name, m in list(meta.items()):
            if m.get("source") == "Wikimedia Commons (learned)":
                (folder / name).unlink(missing_ok=True)
                meta.pop(name)
        if meta:
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            meta_file.unlink()
            if not any(folder.iterdir()):
                folder.rmdir()
    catalog_path(pid).unlink(missing_ok=True)


def _learn(svc, wd: dict, log=print, patient: bool = False) -> str:
    pid = slug(wd["name"])
    if any(p["id"] == pid for p in load_people()):
        pid = f"{pid}-{wd['qid'].lower()}"
    _cleanup(pid)  # leftovers from an earlier failed attempt (e.g. a namesake)
    try:
        return _learn_person(svc, wd, pid, log, patient)
    except (OnboardError, WikimediaBusy):
        _cleanup(pid)
        raise


def _learn_person(svc, wd: dict, pid: str, log=print, patient: bool = False) -> str:
    person = {"id": pid, "name": wd["name"],
              "aliases": [a for a in wd["aliases"] if parse.norm(a) != parse.norm(wd["name"])],
              "commons_category": wd.get("commons_category"),  # None: catalog searches by name
              "qid": wd["qid"], "description": wd.get("description", ""), "source": "wikidata"}

    try:
        cat = build_catalog(person, limit=config.AUTO_MAX_FILES, log=log, patient=patient,
                            max_cats=config.AUTO_MAX_CATEGORIES)
        portrait = None
        if wd.get("image"):
            title = wd["image"].replace("_", " ")
            portrait = next((f for f in cat["files"] if f["title"] == title), None)  # often already there
            if not portrait:
                info = Commons(log=log, patient=patient).file_info([f"File:{wd['image']}"], 1024)
                if info:
                    d, y = normalise_date(info[0].get("date"))
                    portrait = info[0] | {"file": safe_name(info[0]["title"]), "norm_date": d, "year": y}
    except WikimediaBusy:
        raise  # the caller decides: tell the user, or learn in the background
    except requests.HTTPError as e:  # 429/503 never get here: Commons raises WikimediaBusy for those
        code = e.response.status_code if e.response is not None else "?"
        raise OnboardError(f"Wikimedia Commons answered with an error (HTTP {code})") from e
    except requests.RequestException as e:
        raise OnboardError(f"could not reach Wikimedia Commons ({e.__class__.__name__})") from e

    # portrait first, then likely single-person photos: titles that name the person and don't
    # sound like group shots ("with", "family", "summit"...), portraits and crops first
    keys = {t for t in parse.norm(wd["name"]).split() if len(t) > 3}
    group = {"with", "and", "dan", "bersama", "family", "keluarga", "cabinet", "summit", "meeting",
             "leaders", "group", "delegation", "team", "crowd", "wife", "husband", "signs", "visit"}
    good = {"portrait", "cropped", "crop", "headshot", "official", "face"}

    def pick_order(f):
        words = set(parse.norm(f["title"]).split())
        return (0 if keys & words else 1, 1 if group & words else 0, 0 if good & words else 1)
    ranked = sorted(cat["files"], key=pick_order)
    picks = ([portrait] if portrait else []) + [f for f in ranked if not portrait or f["file"] != portrait["file"]]
    picks = picks[: config.AUTO_MAX_DOWNLOADS]
    if not picks:
        raise OnboardError(f"no photos of {wd['name']} on Wikimedia Commons")

    # Download in rounds (photo downloads aren't rate-limited like the API) until there are
    # enough clear faces of the person: actors' and athletes' Commons photos are mostly red
    # carpets, panels and podiums, so the first dozen often isn't enough.
    folder = config.RAW_DIR / pid
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = config.WIKI_USER_AGENT
    engine = svc.get_engine()
    downloaded, meta, cands, faces_by_file = [], {}, [], {}
    anchor, refs = None, []
    for start in range(0, len(picks), config.AUTO_DOWNLOADS):
        batch = picks[start:start + config.AUTO_DOWNLOADS]
        with ThreadPoolExecutor(config.LIVE_WORKERS) as pool:
            results = list(pool.map(lambda f: live._download(session, f["download_url"]), batch))
        for f, (data, _gone) in zip(batch, results):
            if data is None:
                continue
            (folder / f["file"]).write_bytes(data)
            meta[f["file"]] = {k: v for k, v in f.items() if k not in ("download_url", "file", "norm_date", "year")} | {
                "source": "Wikimedia Commons (learned)"}
            downloaded.append(f)
            img = read_image(folder / f["file"])
            fs = faces_by_file[f["file"]] = engine.analyze(img) if img is not None else []
            main = _main_face(fs)
            if portrait and f["file"] == portrait["file"] and fs and fs[0].score >= 0.85:
                anchor = fs[0].embedding  # biggest face in the Wikidata portrait
            if main is not None:
                cands.append((f["file"], main.embedding))
        refs = _choose_refs(cands, anchor)
        if len(refs) >= REF_TARGET:
            break
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if not downloaded:
        raise OnboardError("could not download photos from Wikimedia Commons")
    if len(refs) < 2:
        raise OnboardError(
            f"not enough clear photos of {wd['name']}'s face on Commons to learn it (checked "
            f"{len(downloaded)} photos: {len(cands)} with one clear main face, and no 2 of them "
            f"looked like the same person{' as the Wikidata portrait' if anchor is not None else ''})")
    refs = refs[: config.GALLERY_PER_PERSON]

    gdir = config.GALLERY_DIR / pid
    gdir.mkdir(parents=True, exist_ok=True)
    for name, _ in refs:
        shutil.copy2(folder / name, gdir / name)
    with svc.lock:
        svc.gallery.people[pid] = np.stack([e for _, e in refs])
        svc.gallery.save(GALLERY_FILE)
    _save_person(person)
    log(f"learned {wd['name']} from {len(refs)} reference faces ({len(downloaded)} photos downloaded)")

    # index the downloaded photos that show the person (the rest are deleted: up to 36 were
    # fetched just to find good reference faces), then re-check every face already in the archive
    gallery_emb = svc.gallery.people[pid]
    keep, drop = [], []
    for f in downloaded:
        shows = any(float((gallery_emb @ face.embedding).max()) >= config.FACE_MATCH_THRESHOLD
                    for face in faces_by_file.get(f["file"], []))
        (keep if shows else drop).append(f)
    for f in drop:
        (folder / f["file"]).unlink(missing_ok=True)
        meta.pop(f["file"], None)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    new_ids = ingest(svc, pid, [(folder / f["file"], f) for f in keep])
    relabel(svc.gallery, log=lambda *a: None)
    svc.reload()
    if svc.describer:
        svc.describer.enqueue(new_ids)
    return pid


REF_TARGET = 5  # stop downloading once this many reference faces are found


def _main_face(faces):
    """The one clear face of a photo: the only face, or the biggest when it's at least 3x the
    area of the next (a photographer or fan in the background doesn't spoil a good photo).
    `faces` come biggest first."""
    if not faces:
        return None
    f = faces[0]
    if f.score < 0.85 or min(f.box[2:]) < 70:
        return None
    if len(faces) > 1 and faces[1].box[2] * faces[1].box[3] * 3 > f.box[2] * f.box[3]:
        return None
    return f


def _choose_refs(cands: list, anchor) -> list:
    """Reference faces among `cands` [(file, embedding)]: the faces like the Wikidata portrait;
    if the portrait doesn't match (old, side-on, sunglasses...) or there is none, the biggest group
    of faces that look alike - in photos tagged as depicting someone, that's the face that keeps
    coming back."""
    if not cands:
        return []
    E = np.stack([e for _, e in cands])
    if anchor is not None:
        refs = [c for c, s in zip(cands, E @ anchor) if s >= 0.35]
        if len(refs) >= 2:
            return refs
    S = E @ E.T
    best = int((S >= 0.40).sum(1).argmax())
    group = [c for c, s in zip(cands, S[best]) if s >= 0.40]
    return group if len(group) >= (3 if len(cands) >= 4 else 2) else []


def ingest(svc, person_id: str, items) -> list[int]:
    """Analyse + save photos already on disk (Commons caption as temporary description)."""
    import imagehash
    names = {p["id"]: p["name"] for p in load_people()}
    hashes = {imagehash.hex_to_hash(h): p for h, p in store.known_hashes().items()}
    new = []
    for path, src in items:
        try:
            out = live._analyse(svc.get_engine(), svc.gallery, names, path, person_id, src, hashes)
        except Exception:
            continue
        if out:
            new.append(out)
    if not new:
        return []
    try:
        vecs = llm.embed([rec["doc_text"] for rec, _ in new])
    except llm.OllamaError:
        vecs = [None] * len(new)
    ids = []
    for (rec, faces), vec in zip(new, vecs):
        rec["embedding"] = vec
        ids.append(store.save_image(rec, faces))
    return ids


def forget(pid: str, delete_photos: bool = False, log=print) -> bool:
    """Remove a person learned from Wikidata (e.g. the wrong namesake).
    Their reference faces go; their photos stay as unidentified unless delete_photos."""
    from .commons import catalog_path
    from .faces import Gallery
    f = config.PEOPLE_AUTO_FILE
    people = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    person = next((p for p in people if p["id"] == pid), None)
    if not person:
        log(f"'{pid}' is not a learned person (people.json entries are edited by hand).")
        return False
    f.write_text(json.dumps([p for p in people if p["id"] != pid], indent=2, ensure_ascii=False),
                 encoding="utf-8")
    shutil.rmtree(config.GALLERY_DIR / pid, ignore_errors=True)
    gallery = Gallery.load(GALLERY_FILE)
    gallery.people.pop(pid, None)
    gallery.save(GALLERY_FILE)
    catalog_path(pid).unlink(missing_ok=True)
    if delete_photos:
        with store.connect() as con:
            n = con.execute("DELETE FROM images WHERE folder = ?", (pid,)).rowcount
        shutil.rmtree(config.RAW_DIR / pid, ignore_errors=True)
        log(f"  deleted {n} photos")
    changed = relabel(gallery, log=lambda *a: None)
    log(f"Forgot {person['name']} ({person.get('description', '')}); {changed} faces re-checked.")
    return True
