"""Query pipeline (the RAG part).

  1. parse    - who is asked about + year filter            (parse.py)
  2. retrieve - filter by recognised faces / year, rank by
                semantic similarity of the question to each
                image's description                          (here)
  3. generate - local LLM writes the reply using ONLY the
                retrieved images' descriptions and metadata  (here)
"""
from __future__ import annotations

import random
import threading
import time
from collections import defaultdict

import numpy as np

from . import config, live, llm, onboard, parse, store, wikidata
from .faces import FaceEngine, Gallery, decode_image, load_people
from .indexer import load_gallery

SYSTEM_PROMPT = """You are FaceSearch, an assistant for a local photo archive.
You are given the photos that the search system found for the user's request, numbered [1], [2], ...
Rules:
- Reply in the same language as the user (Malay or English).
- Write 2-4 short sentences that summarise what the photos show, citing them like [1] or [2][3].
- Use ONLY the information given for each photo. Never invent dates, places or events.
- If a photo has no date, say the date is unknown rather than guessing.
- Do not list every photo; the user can see them. Do not use markdown headings."""


class SearchService:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine: FaceEngine | None = None
        self.gallery: Gallery = Gallery()
        self.describer: live.Describer | None = None
        self.reload()

    def start_background(self):
        """Server only: write AI descriptions for live-fetched photos in the background."""
        if config.BACKGROUND_DESCRIBE and self.describer is None:
            self.describer = live.Describer(on_change=self.reload)
            self.describer.start()
            self.describer.enqueue_pending()

    # ----------------------------------------------------------- loading
    def reload(self):
        with self.lock:
            self.people = load_people()
            self.names = {p["id"]: p["name"] for p in self.people}
            self.gallery = load_gallery()
            with store.connect() as con:
                rows = con.execute("SELECT * FROM images ORDER BY id").fetchall()
                faces = con.execute("SELECT * FROM faces").fetchall()
            self.rows = {r["id"]: r for r in rows}
            self.faces = defaultdict(list)
            for f in faces:
                self.faces[f["image_id"]].append(f)
            self.ids = [r["id"] for r in rows]
            vecs = [store.from_blob(r["embedding"]) for r in rows]
            dim = next((len(v) for v in vecs if v is not None), 0)
            self.emb = (np.stack([v if v is not None else np.zeros(dim, np.float32) for v in vecs])
                        if rows and dim else np.zeros((len(rows), 0), np.float32))
            self.face_rows = [f for f in faces if f["embedding"] is not None]
            self.face_emb = (np.stack([store.from_blob(f["embedding"]) for f in self.face_rows])
                             if self.face_rows else np.zeros((0, 128), np.float32))

    def get_engine(self) -> FaceEngine:
        if self.engine is None:
            self.engine = FaceEngine()
        return self.engine

    def record(self, image_id: int) -> dict | None:
        row = self.rows.get(image_id)
        return store.image_record(row, self.faces[image_id], self.names) if row else None

    def people_summary(self) -> list[dict]:
        counts = defaultdict(set)
        for img_id, fs in self.faces.items():
            for f in fs:
                if f["person_id"]:
                    counts[f["person_id"]].add(img_id)
        out = []
        for p in self.people:
            ids = sorted(counts.get(p["id"], set()))
            # thumbnail: the image where this person's face is biggest
            best = max(ids, default=None, key=lambda i: max(
                (f["w"] * f["h"] for f in self.faces[i] if f["person_id"] == p["id"]), default=0))
            face = next((f for f in self.faces.get(best, []) if f["person_id"] == p["id"]), None)
            out.append({
                "id": p["id"], "name": p["name"], "aliases": p.get("aliases", []),
                "image_count": len(ids), "in_gallery": p["id"] in self.gallery.people,
                "thumb": self.record(best)["url"] if best else None,
                "thumb_box": [face["x"], face["y"], face["w"], face["h"]] if face else None,
                "thumb_size": [self.rows[best]["width"], self.rows[best]["height"]] if best else None,
            })
        return out

    def stats(self) -> dict:
        return {"images": len(self.rows),
                "faces": sum(len(v) for v in self.faces.values()),
                "known_faces": sum(1 for v in self.faces.values() for f in v if f["person_id"]),
                "people_in_gallery": len(self.gallery.people)}

    # ----------------------------------------------------------- retrieval
    def retrieve(self, query: str, parsed: dict, k: int = config.TOP_K) -> list[dict]:
        with self.lock:  # consistent snapshot: the background describer may reload meanwhile
            ids, emb, rows, faces = self.ids, self.emb, self.rows, self.faces
        if not ids:
            return []
        sem = np.zeros(len(ids), np.float32)  # without embeddings it still works by face
        if emb.shape[1]:
            try:
                q = llm.embed([query])[0]
                if len(q) == emb.shape[1]:
                    sem = emb @ q
            except llm.OllamaError:
                if not parsed["people"] and not parsed.get("lookup_error"):
                    raise
        want = set(parsed["people"])
        y0, y1 = parsed["year_from"], parsed["year_to"]
        hits = []
        for idx, img_id in enumerate(ids):
            row = rows[img_id]
            year = row["year"]
            if (y0 or y1) and year is None:
                continue
            if parsed.get("years"):
                if year not in parsed["years"]:
                    continue
            elif y0 and year < y0 or y1 and year > y1:
                continue
            face_score, match = 0.0, None
            if want:
                matched = {f["person_id"]: f["match_score"] for f in faces[img_id]
                           if f["person_id"] in want}
                if want <= matched.keys():          # every requested person is in the photo
                    match = "face"
                    face_score = float(np.mean(list(matched.values())))
                elif (row["folder"] in want and len(want) == 1 and faces[img_id]
                      and not any(f["person_id"] for f in faces[img_id])):
                    # from the person's own folder, a face was found but not confirmed,
                    # and nobody else was recognised (e.g. a side profile)
                    match = "source"                # labelled by the source, face not confirmed
                else:
                    continue
            elif sem[idx] < config.MIN_SEMANTIC_SCORE:
                continue
            score = float(sem[idx]) + (0.5 * face_score if match == "face" else 0)
            hits.append((match == "face", score, img_id, float(sem[idx]), match))
        # confirmed faces first, then by score
        hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
        out = []
        for _, score, img_id, s, match in hits[:k]:
            rec = self.record(img_id)
            rec["relevance"] = round(s, 3)
            rec["match"] = match
            out.append(rec)
        return out

    # ----------------------------------------------------------- generation
    def _context(self, images: list[dict]) -> str:
        lines = []
        for n, im in enumerate(images, 1):
            people = ", ".join(f["name"] for f in im["faces"] if f["name"]) or "no recognised people"
            if im["match"] == "source":
                people += " (face not confirmed; the photo comes from the requested person's source collection)"
            label = "Description (Commons caption)" if im.get("desc_source") == "caption" else "Description"
            lines.append(
                f"[{n}] People: {people}. Date: {im['date'] or 'unknown'}.\n"
                f"    {label}: {im['description'] or 'none'}\n"
                f"    Source caption: {(im['original_caption'] or 'none')[:300]}")
        return "\n".join(lines)

    def _fallback_answer(self, query: str, parsed: dict, images: list[dict]) -> str:
        malay = any(w in parse.norm(query).split() for w in ("bagi", "gambar", "tunjuk", "cari", "tahun", "dan"))
        who = ", ".join(self.names[p] for p in parsed["people"]) or None
        if malay:
            head = f"Saya jumpa {len(images)} gambar" + (f" {who}" if who else "") + "."
        else:
            head = f"I found {len(images)} photo(s)" + (f" of {who}" if who else "") + "."
        dated = [im for im in images if im["date"]]
        if dated:
            ys = sorted({im["year"] for im in dated if im["year"]})
            span = f"{ys[0]}" if len(ys) == 1 else f"{ys[0]}-{ys[-1]}"
            head += (f" Tarikh gambar: {span}." if malay else f" Dates: {span}.")
        return head

    def _parse(self, query: str, history: list[dict] | None) -> dict:
        """Step 1: who + which years. Follow-ups like "yang tahun 2019 pula?" keep the last person."""
        parsed = parse.parse(query, self.people)
        if not parsed["people"] and history:
            last = next((h.get("parsed") for h in reversed(history) if h.get("parsed")), None)
            if last and last.get("people") and len(parse.norm(query).split()) <= 6 \
                    and (parsed["year_from"] or parsed["year_to"] or "pula" in query.lower()):
                parsed["people"] = last["people"]
                parsed["inherited"] = True
        parsed["names"] = [self.names[p] for p in parsed["people"]]
        if not parsed["people"] and config.AUTO_PEOPLE:
            name = parse.candidate_name(query)  # someone not in the list?
            if name:
                parsed["lookup"] = name
                parsed["lookup_titled"] = parse.candidate_name_with_titles(query, name)
                parsed["lookup_hints"] = parse.role_hints(query)
                parsed["lookup_is_name"] = parse.looks_like_name(query, name)
        return parsed

    def _resolve_unknown(self, parsed: dict):
        """Generator: look up an unknown name on Wikidata and learn their face if needed.
        Yields status texts; updates `parsed` in place."""
        name = parsed.get("lookup")
        if not name:
            return
        yield f"Looking up “{name}” on Wikidata…"
        try:
            hints, years = parsed.get("lookup_hints", []), (parsed.get("year_from"), parsed.get("year_to"))
            wd = None
            if parsed.get("lookup_titled"):  # "tunku abdul rahman" before plain "abdul rahman"
                wd = wikidata.lookup(parsed["lookup_titled"], hints, years)
            wd = wd or wikidata.lookup(name, hints, years)
        except Exception:
            parsed["lookup_error"] = "could not reach Wikidata"
            return
        if not wd:
            parsed["lookup_error"] = "not on Wikidata"
            return
        known = next((p for p in self.people if p.get("qid") == wd["qid"]), None)
        if known:
            pid = known["id"]
        else:
            desc = f" ({wd['description']})" if wd.get("description") else ""
            yield f"Found {wd['name']}{desc}. Learning their face from Wikimedia Commons…"
            try:
                pid = onboard.learn(self, wd)
            except onboard.OnboardError as e:
                parsed["lookup_error"] = f"{e} (tried: {wd['name']}{desc})"
                return
            parsed["new_person"] = wd["name"]
        parsed["people"] = [pid]
        parsed["names"] = [self.names.get(pid, wd["name"])]

    @staticmethod
    def _unknown_name(parsed: dict) -> bool:
        """A clearly named person ("Anne Hathaway") we couldn't find or learn: say so instead of
        showing unrelated photos from a meaning-based search."""
        return bool(parsed.get("lookup_error") and parsed.get("lookup_is_name") and not parsed["people"])

    def _no_result(self, parsed: dict, images: list[dict]) -> str | None:
        if images:
            return None
        if parsed.get("lookup_error") == "not on Wikidata" and not parsed["people"]:
            return (f"Maaf, tiada gambar yang sepadan dalam arkib, dan tiada tokoh awam bernama "
                    f"“{parsed['lookup']}” di Wikidata.\nSorry, no matching photos in the archive, and no "
                    f"public figure called “{parsed['lookup']}” on Wikidata.")
        if parsed.get("lookup_error") and not parsed["people"]:
            return (f"Maaf, saya tidak dapat mencari “{parsed['lookup']}”: {parsed['lookup_error']}.\n"
                    f"Sorry, I couldn't search for “{parsed['lookup']}”: {parsed['lookup_error']}.")
        who = " / ".join(parsed["names"])
        if parsed["people"] and not any(p in self.gallery.people for p in parsed["people"]):
            return (f"{who} has no reference faces yet. Add photos to data/gallery/"
                    f"{parsed['people'][0]}/ and run `python cli.py gallery` then `python cli.py relabel`.")
        if parsed["people"] and (parsed["year_from"] or parsed["year_to"]):
            return self._no_year_message(parsed)
        return ("Maaf, tiada gambar yang sepadan dalam arkib. / "
                "Sorry, no matching photos in the archive.")

    def _live_person(self, parsed: dict, images: list[dict]) -> str | None:
        """Person to fetch live photos for, or None: one known person, fewer than TOP_K
        confirmed photos, reference faces to check against, and new files in the catalog."""
        if not config.LIVE_FETCH or len(parsed["people"]) != 1:
            return None
        pid = parsed["people"][0]
        if pid not in self.gallery.people:
            return None
        if sum(1 for im in images if im["match"] == "face") >= config.TOP_K:
            return None
        person = next((p for p in self.people if p["id"] == pid), None)
        return pid if person and live.candidates(person, parsed) else None

    def _live(self, query: str, parsed: dict, images: list[dict], pid: str):
        need = config.TOP_K - sum(1 for im in images if im["match"] == "face")
        info = live.fetch(self, pid, parsed, need)
        if info["new_ids"]:
            self.reload()
            if self.describer:
                self.describer.enqueue(info["new_ids"])
            images = self.retrieve(query, parsed)
        new = set(info["new_ids"])
        for im in images:
            im["live"] = im["id"] in new
        return images, {"added": info["added"], "checked": info["checked"], "reason": info["reason"]}

    def _touch(self):
        if self.describer:
            self.describer.last_chat = time.time()

    def _no_year_message(self, parsed: dict) -> str:
        """No photo in the asked years: say which years this person does have."""
        want = set(parsed["people"])
        years = sorted({self.rows[i]["year"] for i, fs in self.faces.items()
                        if self.rows[i]["year"] and want <= {f["person_id"] for f in fs}})
        if parsed.get("years"):
            asked = ", ".join(map(str, parsed["years"]))
        elif parsed["year_from"] == parsed["year_to"]:
            asked = str(parsed["year_from"])
        else:
            asked = f"{parsed['year_from'] or '…'}–{parsed['year_to'] or '…'}"
        who = " & ".join(parsed["names"])
        if not years:
            return f"Tiada gambar {who} yang bertarikh. / No dated photos of {who}."
        have = ", ".join(map(str, years))
        return (f"Tiada gambar {who} untuk tahun {asked}. Tahun yang ada: {have}.\n"
                f"No photos of {who} from {asked}. Years available: {have}.")

    def _messages(self, query: str, images: list[dict]) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Photos found:\n{self._context(images)}\n\nUser request: {query}"}]

    def chat(self, query: str, history: list[dict] | None = None) -> dict:
        self._touch()
        t0 = time.time()
        parsed = self._parse(query, history)
        for _ in self._resolve_unknown(parsed):
            pass
        images = [] if self._unknown_name(parsed) else self.retrieve(query, parsed)
        live_info = None
        pid = self._live_person(parsed, images)
        if pid:
            images, live_info = self._live(query, parsed, images, pid)
        t_retrieve = round(time.time() - t0, 2)
        no_result = self._no_result(parsed, images)
        if no_result:
            return {"answer": no_result, "images": [], "parsed": parsed, "live": live_info,
                    "timing": {"retrieve": t_retrieve, "generate": 0}, "llm": False}
        t1 = time.time()
        used_llm = True
        try:
            answer = llm.chat(self._messages(query, images))
        except llm.OllamaError:
            answer, used_llm = self._fallback_answer(query, parsed, images), False
        self._touch()
        return {"answer": answer, "images": images, "parsed": parsed, "llm": used_llm, "live": live_info,
                "timing": {"retrieve": t_retrieve, "generate": round(time.time() - t1, 2)}}

    def chat_stream(self, query: str, history: list[dict] | None = None):
        """Same as chat(), but as events: the photos first, then the reply word by word.

        {"type": "results", images, parsed, timing, live}  -> show the photo cards straight away
        {"type": "status", "text": "..."}                    -> e.g. "Fetching more photos..."
        {"type": "results", ...}                             -> again, with the live photos added
        {"type": "token", "text": "..."}                     -> append to the reply (repeated)
        {"type": "done", answer, llm, timing}                -> final full reply
        """
        self._touch()
        t0 = time.time()
        parsed = self._parse(query, history)
        for text in self._resolve_unknown(parsed):
            yield {"type": "status", "text": text}
        images = [] if self._unknown_name(parsed) else self.retrieve(query, parsed)
        live_info = None
        pid = self._live_person(parsed, images)
        if pid:
            # show what the archive has now, then look for more
            yield {"type": "results", "images": images, "parsed": parsed, "live": {"pending": True},
                   "timing": {"retrieve": round(time.time() - t0, 2), "generate": 0}}
            yield {"type": "status", "text": f"Fetching more photos of {self.names[pid]} from Wikimedia Commons…"}
            images, live_info = self._live(query, parsed, images, pid)
        t_retrieve = round(time.time() - t0, 2)
        yield {"type": "results", "images": images, "parsed": parsed, "live": live_info,
               "timing": {"retrieve": t_retrieve, "generate": 0}}
        no_result = self._no_result(parsed, images)
        if no_result:
            yield {"type": "done", "answer": no_result, "llm": False,
                   "timing": {"retrieve": t_retrieve, "generate": 0}}
            return
        t1 = time.time()
        parts: list[str] = []
        used_llm = True
        try:
            for piece in llm.chat_stream(self._messages(query, images)):
                parts.append(piece)
                yield {"type": "token", "text": piece}
            answer = "".join(parts).strip()
        except llm.OllamaError:
            if parts:  # connection dropped half way: keep what we have
                answer = "".join(parts).strip()
            else:
                answer, used_llm = self._fallback_answer(query, parsed, images), False
        self._touch()
        yield {"type": "done", "answer": answer, "llm": used_llm,
               "timing": {"retrieve": t_retrieve, "generate": round(time.time() - t1, 2)}}

    # ----------------------------------------------------------- browse
    UNIDENTIFIED = "_unidentified"
    UNDATED = "none"

    def browse(self, person: str = "", year: str = "", sort: str = "newest",
               offset: int = 0, limit: int = 60) -> dict:
        """Every indexed photo, filtered by person and year, with counts for the filters."""
        def people_in(img_id):
            return {f["person_id"] for f in self.faces.get(img_id, []) if f["person_id"]}

        all_ids = list(self.rows)
        # person filter
        if person == self.UNIDENTIFIED:
            by_person = [i for i in all_ids if not people_in(i)]
        elif person:
            by_person = [i for i in all_ids if person in people_in(i)]
        else:
            by_person = all_ids
        # counts for the year menu (within the chosen person)
        year_counts: dict = defaultdict(int)
        for i in by_person:
            year_counts[self.rows[i]["year"] or self.UNDATED] += 1
        # year filter
        if year == self.UNDATED:
            ids = [i for i in by_person if not self.rows[i]["year"]]
        elif year:
            ids = [i for i in by_person if self.rows[i]["year"] == int(year)]
        else:
            ids = by_person
        # counts for the person chips (within the chosen year)
        person_counts: dict = defaultdict(int)
        in_year = all_ids if not year else [
            i for i in all_ids if (not self.rows[i]["year"] if year == self.UNDATED
                                   else self.rows[i]["year"] == int(year))]
        for i in in_year:
            ps = people_in(i)
            for p in ps:
                person_counts[p] += 1
            if not ps:
                person_counts[self.UNIDENTIFIED] += 1

        if sort == "added":
            ids.sort(reverse=True)
        else:  # by date; undated photos always last
            dated = sorted((i for i in ids if self.rows[i]["date"]),
                           key=lambda i: self.rows[i]["date"], reverse=(sort != "oldest"))
            ids = dated + [i for i in ids if not self.rows[i]["date"]]

        return {
            "total": len(ids),
            "items": [self.record(i) for i in ids[offset:offset + limit]],
            "people": [{"id": p["id"], "name": p["name"], "count": person_counts.get(p["id"], 0)}
                       for p in self.people]
                      + [{"id": self.UNIDENTIFIED, "name": "Unidentified",
                          "count": person_counts.get(self.UNIDENTIFIED, 0)}],
            "years": sorted(({"year": y, "count": n} for y, n in year_counts.items() if y != self.UNDATED),
                            key=lambda x: -x["year"])
                     + ([{"year": self.UNDATED, "count": year_counts[self.UNDATED]}]
                        if year_counts.get(self.UNDATED) else []),
            "all_count": len(all_ids),
        }

    # ----------------------------------------------------------- face search
    def face_search(self, raw: bytes, k: int = 12) -> dict:
        img = decode_image(raw)
        if img is None:
            raise ValueError("Could not read that image.")
        faces = self.get_engine().analyze(img, min_size=24)
        h, w = img.shape[:2]
        found = []
        for f in faces:
            pid, score = self.gallery.identify(f.embedding)
            found.append({"box": list(f.box), "person_id": pid,
                          "name": self.names.get(pid) if pid else None, "score": round(score, 3)})
        images = []
        if faces and len(self.face_rows):
            # search by the largest face: nearest faces anywhere in the archive,
            # so it also works for people who are not in people.json
            sims = self.face_emb @ faces[0].embedding
            best: dict[int, float] = {}
            for i in np.argsort(-sims):
                if sims[i] < config.FACE_MATCH_THRESHOLD or len(best) >= k:
                    break
                img_id = self.face_rows[i]["image_id"]
                best.setdefault(img_id, float(sims[i]))
            for img_id, s in best.items():
                rec = self.record(img_id)
                rec["relevance"], rec["match"] = round(s, 3), "face"
                images.append(rec)
        return {"faces": found, "size": [w, h], "images": images}

    def random_query(self) -> str:
        known = [p for p in self.people_summary() if p["image_count"]]
        p = random.choice(known or self.people)
        return random.choice([f"Bagi gambar {p['name']}", f"Show me photos of {p['name']}"])
