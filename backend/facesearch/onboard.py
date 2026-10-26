"""Learn a new public figure on the fly.

  Wikidata (who is it?)  ->  Commons catalog + portrait  ->  download ~12 photos
  -> reference faces: clear single-face photos that match the Wikidata portrait
  -> add to data/people_auto.json + gallery  ->  index the photos  ->  relabel the archive

After this the person works exactly like the ones in people.json, including live search.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from . import config, live, llm, parse, store
from .commons import Commons, build_catalog, safe_name
from .faces import load_people, read_image
from .indexer import GALLERY_FILE, relabel
from .metadata import normalise_date


class OnboardError(Exception):
    pass


_lock = threading.Lock()  # learn one person at a time


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", parse.norm(name)).strip("-") or "person"


def _save_person(person: dict):
    f = config.PEOPLE_AUTO_FILE
    people = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    people = [p for p in people if p["id"] != person["id"]] + [person]
    f.write_text(json.dumps(people, indent=2, ensure_ascii=False), encoding="utf-8")


def learn(svc, wd: dict, log=print) -> str:
    """Make `wd` (from wikidata.lookup) a known person. Returns the person id."""
    with _lock:
        known = next((p for p in load_people() if p.get("qid") == wd["qid"]), None)
        if known:  # learned by another request meanwhile
            return known["id"]
        return _learn(svc, wd, log)


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


def _learn(svc, wd: dict, log=print) -> str:
    pid = slug(wd["name"])
    if any(p["id"] == pid for p in load_people()):
        pid = f"{pid}-{wd['qid'].lower()}"
    _cleanup(pid)  # leftovers from an earlier failed attempt (e.g. a namesake)
    try:
        return _learn_person(svc, wd, pid, log)
    except OnboardError:
        _cleanup(pid)
        raise


def _learn_person(svc, wd: dict, pid: str, log=print) -> str:
    person = {"id": pid, "name": wd["name"],
              "aliases": [a for a in wd["aliases"] if parse.norm(a) != parse.norm(wd["name"])],
              "commons_category": wd.get("commons_category"),  # None: catalog searches by name
              "qid": wd["qid"], "description": wd.get("description", ""), "source": "wikidata"}

    try:
        cat = build_catalog(person, limit=config.AUTO_MAX_FILES, log=log, patient=False,
                            max_cats=config.AUTO_MAX_CATEGORIES)
        portrait = None
        if wd.get("image"):
            info = Commons(log=log, patient=False).file_info([f"File:{wd['image']}"], 1024)
            if info:
                d, y = normalise_date(info[0].get("date"))
                portrait = info[0] | {"file": safe_name(info[0]["title"]), "norm_date": d, "year": y}
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code == 429:
            raise OnboardError("Wikimedia Commons is limiting requests right now (HTTP 429). "
                               "Wait a minute and ask again") from e
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
    picks = picks[: config.AUTO_DOWNLOADS]
    if not picks:
        raise OnboardError(f"no photos of {wd['name']} on Wikimedia Commons")

    folder = config.RAW_DIR / pid
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = config.WIKI_USER_AGENT
    with ThreadPoolExecutor(config.LIVE_WORKERS) as pool:
        results = list(pool.map(lambda f: live._download(session, f["download_url"]), picks))
    downloaded, meta = [], {}
    for f, (data, _gone) in zip(picks, results):
        if data is None:
            continue
        (folder / f["file"]).write_bytes(data)
        meta[f["file"]] = {k: v for k, v in f.items() if k not in ("download_url", "file", "norm_date", "year")} | {
            "source": "Wikimedia Commons (learned)"}
        downloaded.append(f)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if not downloaded:
        raise OnboardError("could not download photos from Wikimedia Commons")

    # reference faces: one clear face per photo, matching the Wikidata portrait if we have it
    engine = svc.get_engine()
    faces_by_file = {}
    for f in downloaded:
        img = read_image(folder / f["file"])
        faces_by_file[f["file"]] = engine.analyze(img) if img is not None else []
    anchor = None
    if portrait and faces_by_file.get(portrait["file"]):
        anchor = faces_by_file[portrait["file"]][0].embedding  # biggest face in the portrait
    refs = []
    for f in downloaded:
        fs = faces_by_file[f["file"]]
        if len(fs) == 1 and fs[0].score >= 0.9 and min(fs[0].box[2:]) >= 80:
            if anchor is None or float(fs[0].embedding @ anchor) >= 0.35:
                refs.append((f["file"], fs[0].embedding))
    if anchor is None and len(refs) >= 3:  # no portrait: drop faces unlike the rest
        E = np.stack([e for _, e in refs])
        mean_sim = ((E @ E.T).sum(1) - 1) / (len(E) - 1)
        refs = [r for r, s in zip(refs, mean_sim) if s >= max(0.25, float(np.median(mean_sim)) - 0.2)]
    if len(refs) < 2:
        raise OnboardError(f"not enough clear photos of {wd['name']}'s face on Commons to learn it")
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

    # index the downloaded photos, then re-check every face already in the archive
    new_ids = ingest(svc, pid, [(folder / f["file"], f) for f in downloaded])
    relabel(svc.gallery, log=lambda *a: None)
    svc.reload()
    if svc.describer:
        svc.describer.enqueue(new_ids)
    return pid


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
