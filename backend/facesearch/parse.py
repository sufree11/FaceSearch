"""Understand a chat query: who is being asked about, and which years.

This is rule-based on purpose: it is fast, predictable and easy to explain,
and small local LLMs are unreliable at structured extraction. It handles
Malay and English, titles (Tun, Datuk, Dr...) and small spelling mistakes.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

TITLES = {
    "tun", "tan", "sri", "seri", "dato", "datuk", "datin", "dr", "tunku", "tengku", "haji",
    "hajah", "yb", "yab", "yang", "berhormat", "amat", "perdana", "menteri", "pm", "encik",
    "puan", "cik", "mr", "mrs", "ms", "sir", "prof", "bin", "binti", "bt", "bte",
}
STOP = {
    "bagi", "beri", "berikan", "tunjuk", "tunjukkan", "cari", "carikan", "gambar", "foto",
    "photo", "photos", "picture", "pictures", "image", "images", "show", "me", "find", "of",
    "the", "a", "an", "and", "dan", "dengan", "with", "in", "at", "pada", "di", "ke", "saya",
    "nak", "mahu", "please", "tolong", "sila", "yang", "ada", "semua", "all", "some",
    "tahun", "year", "years", "from", "dari", "daripada", "to", "hingga", "sampai",
    "sebelum", "before", "selepas", "after", "since", "sejak", "antara", "between", "lepas",
}
FUZZY_MIN = 0.84


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"['’`]", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _key_tokens(people: list[dict]) -> dict[str, set[str]]:
    """Distinctive name tokens per person (titles removed)."""
    out = {}
    for p in people:
        toks = set()
        for n in [p["name"], *p.get("aliases", [])]:
            toks |= {t for t in norm(n).split() if t not in TITLES and len(t) >= 3}
        out[p["id"]] = toks
    return out


def find_people(query: str, people: list[dict]) -> list[str]:
    q = f" {norm(query)} "
    hits: dict[str, float] = {}
    # 1) exact alias / name phrase (longest wins)
    for p in people:
        for n in [p["name"], *p.get("aliases", [])]:
            phrase = norm(n)
            if phrase and f" {phrase} " in q:
                hits[p["id"]] = max(hits.get(p["id"], 0), 1.0 + len(phrase) / 100)
    # 2) fuzzy match on distinctive tokens ("mahatir" -> "mahathir")
    qtoks = [t for t in q.split() if len(t) >= 4 and t not in STOP and t not in TITLES]
    for pid, keys in _key_tokens(people).items():
        if pid in hits:
            continue
        for t in qtoks:
            for k in keys:
                if len(k) >= 4 and SequenceMatcher(None, t, k).ratio() >= FUZZY_MIN:
                    hits[pid] = max(hits.get(pid, 0), SequenceMatcher(None, t, k).ratio())
    return [pid for pid, _ in sorted(hits.items(), key=lambda kv: -kv[1])]


YEAR = r"(1[89]\d{2}|20\d{2})"


def find_years(query: str, this_year: int | None = None) -> dict:
    """Year filter from the question.

    "2015", "dari tahun 2015", "from 2015", "pada 2015"  -> only 2015
    "2013-2015", "antara 2013 dan 2015", "2013 to 2015"   -> 2013..2015
    "2015 dan 2018", "2015 and 2018", "2015, 2018"        -> 2015 or 2018 (not the years between)
    "sejak/since 2015"  -> 2015 onwards      "selepas/after 2015" -> 2016 onwards
    "sebelum/before 2015" -> up to 2014      "hingga/until 2015"  -> up to 2015
    "1990an", "1990s"   -> 1990..1999        "tahun lepas", "last year", "tahun ini"
    """
    import datetime
    this_year = this_year or datetime.date.today().year
    out = {"year_from": None, "year_to": None, "years": None}
    q = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode().lower()
    q = re.sub(r"[^a-z0-9\-]+", " ", q.replace("–", "-"))

    def rng(a, b):
        out["year_from"], out["year_to"] = a, b
        return out

    m = re.search(rf"\b{YEAR}\s*(?:-|to|till|until|hingga|sampai)\s*{YEAR}\b", q) or \
        re.search(rf"(?:antara|between)\s+(?:tahun\s+)?{YEAR}\s+(?:dan|and|hingga|to)\s+{YEAR}\b", q)
    if m:
        return rng(*sorted((int(m[1]), int(m[2]))))
    m = re.search(rf"\b{YEAR}\s*(?:s|an)\b", q)  # 1990s / 1990an
    if m and m[1].endswith("0"):
        return rng(int(m[1]), int(m[1]) + 9)
    m = re.search(rf"\b(sejak|since|selepas|after|lepas)\s+(?:tahun\s+)?{YEAR}\b", q)
    if m:
        y = int(m[2])
        return rng(y if m[1] in ("sejak", "since") else y + 1, None)
    m = re.search(rf"\b(sebelum|before|until|till|hingga|sampai)\s+(?:tahun\s+)?{YEAR}\b", q)
    if m:
        y = int(m[2])
        return rng(None, y - 1 if m[1] in ("sebelum", "before") else y)
    if re.search(r"\b(tahun lepas|last year)\b", q):
        return rng(this_year - 1, this_year - 1)
    if re.search(r"\b(tahun ini|this year)\b", q):
        return rng(this_year, this_year)
    years = sorted({int(y) for y in re.findall(rf"\b{YEAR}\b", q)})
    if len(years) == 1:
        return rng(years[0], years[0])
    if years:  # several separate years: match any of them, not the years in between
        out["years"] = years
        return rng(years[0], years[-1])
    return out


def parse(query: str, people: list[dict]) -> dict:
    return {"people": find_people(query, people), **find_years(query)}
