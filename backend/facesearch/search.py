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

from . import config, llm, parse, store
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
        self.reload()

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
            self.ids = [r["id"] for r in rows if r["embedding"] is not None]
            self.emb = (np.stack([store.from_blob(self.rows[i]["embedding"]) for i in self.ids])
                        if self.ids else np.zeros((0, 1), np.float32))
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
        if not self.ids:
            return []
        try:
            sem = self.emb @ llm.embed([query])[0]
        except llm.OllamaError:
            if not parsed["people"]:
                raise
            sem = np.zeros(len(self.ids), np.float32)  # still works by face alone
        want = set(parsed["people"])
        y0, y1 = parsed["year_from"], parsed["year_to"]
        hits = []
        for idx, img_id in enumerate(self.ids):
            row = self.rows[img_id]
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
                matched = {f["person_id"]: f["match_score"] for f in self.faces[img_id]
                           if f["person_id"] in want}
                if want <= matched.keys():          # every requested person is in the photo
                    match = "face"
                    face_score = float(np.mean(list(matched.values())))
                elif row["folder"] in want and len(want) == 1:
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
            lines.append(
                f"[{n}] People: {people}. Date: {im['date'] or 'unknown'}.\n"
                f"    Description: {im['description'] or 'none'}\n"
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

    def _prepare(self, query: str, history: list[dict] | None):
        """Steps 1-2 (parse + retrieve). Returns (parsed, images, seconds, answer-if-nothing-found)."""
        t0 = time.time()
        parsed = parse.parse(query, self.people)
        # follow-ups like "yang tahun 2019 pula?" keep the person from the last turn
        if not parsed["people"] and history:
            last = next((h.get("parsed") for h in reversed(history) if h.get("parsed")), None)
            if last and last.get("people") and len(parse.norm(query).split()) <= 6 \
                    and (parsed["year_from"] or parsed["year_to"] or "pula" in query.lower()):
                parsed["people"] = last["people"]
                parsed["inherited"] = True
        parsed["names"] = [self.names[p] for p in parsed["people"]]

        images = self.retrieve(query, parsed)
        no_result = None
        if not images:
            who = " / ".join(parsed["names"])
            if parsed["people"] and not any(p in self.gallery.people for p in parsed["people"]):
                no_result = (f"{who} has no reference faces yet. Add photos to data/gallery/"
                             f"{parsed['people'][0]}/ and run `python cli.py gallery` then `python cli.py relabel`.")
            elif parsed["people"] and (parsed["year_from"] or parsed["year_to"]):
                no_result = self._no_year_message(parsed)
            else:
                no_result = ("Maaf, tiada gambar yang sepadan dalam arkib. / "
                             "Sorry, no matching photos in the archive.")
        return parsed, images, round(time.time() - t0, 2), no_result

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
        parsed, images, t_retrieve, no_result = self._prepare(query, history)
        if no_result:
            return {"answer": no_result, "images": [], "parsed": parsed,
                    "timing": {"retrieve": t_retrieve, "generate": 0}, "llm": False}
        t1 = time.time()
        used_llm = True
        try:
            answer = llm.chat(self._messages(query, images))
        except llm.OllamaError:
            answer, used_llm = self._fallback_answer(query, parsed, images), False
        return {"answer": answer, "images": images, "parsed": parsed, "llm": used_llm,
                "timing": {"retrieve": t_retrieve, "generate": round(time.time() - t1, 2)}}

    def chat_stream(self, query: str, history: list[dict] | None = None):
        """Same as chat(), but as events: the photos first, then the reply word by word.

        {"type": "results", images, parsed, timing}  -> show the photo cards straight away
        {"type": "token", "text": "..."}             -> append to the reply (repeated)
        {"type": "done", answer, llm, timing}        -> final full reply
        """
        parsed, images, t_retrieve, no_result = self._prepare(query, history)
        yield {"type": "results", "images": images, "parsed": parsed,
               "timing": {"retrieve": t_retrieve, "generate": 0}}
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
