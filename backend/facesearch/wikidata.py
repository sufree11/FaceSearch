"""Find a public figure on Wikidata by name.

Wikidata knows every notable person: their name and aliases, a short description,
a portrait (property P18) and their Wikimedia Commons category (property P373).
Only humans (P31 = Q5) with a Wikipedia article are accepted, i.e. public figures.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from . import config, parse
from .commons import Commons

_cache: dict[str, dict | None] = {}


def _claim(entity: dict, prop: str):
    for c in entity.get("claims", {}).get(prop, []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if v is not None:
            return v
    return None


def _names_match(query: str, names: list[str]) -> bool:
    """Every word the user typed must (roughly) appear in one of the person's names,
    so "people party" never turns into some person called 'Party'."""
    qtoks = [t for t in parse.norm(query).split() if t not in parse.TITLES]
    if not qtoks:
        return False
    toks = {t for n in names for t in parse.norm(n).split()}
    return all(any(t == k or (len(t) >= 4 and SequenceMatcher(None, t, k).ratio() >= 0.84) for k in toks)
               for t in qtoks)


def _year(entity: dict, prop: str) -> int | None:
    """Year from a Wikidata date claim such as P569 (born) / P570 (died): '+1556-00-00T...' -> 1556."""
    v = _claim(entity, prop)
    if isinstance(v, dict) and isinstance(v.get("time"), str):
        m = re.match(r"([+-])0*(\d+)-", v["time"])
        if m:
            return int(m.group(2)) * (-1 if m.group(1) == "-" else 1)
    return None


def _commons_category(entity: dict) -> str | None:
    """The person's own Commons category: the Commons sitelink (current practice), else P373."""
    title = entity.get("sitelinks", {}).get("commonswiki", {}).get("title", "")
    if title.startswith("Category:"):
        return title.removeprefix("Category:")
    return _claim(entity, "P373")


def lookup(name: str, hints: list[str] | tuple = (), years: tuple | None = None) -> dict | None:
    """{qid, name, description, aliases, commons_category, image} or None. Cached per session.

    Among people with the same name it prefers, in order: a description that fits `hints`
    (e.g. "actress"), being alive in the asked `years` (year_from, year_to), having a portrait,
    and being the most widely known (Wikipedia articles in the most languages)."""
    y0, y1 = years or (None, None)
    key = f"{parse.norm(name)}|{','.join(hints)}|{y0}-{y1}"
    if key in _cache:
        return _cache[key]
    wd = Commons(api=config.WIKIDATA_API, delay=0.5, log=lambda *a: None, patient=False)
    ids = []
    for lang in ("en", "ms"):
        res = wd.request(action="wbsearchentities", search=name, language=lang, uselang="en",
                         type="item", limit=7)
        ids += [r["id"] for r in res.get("search", []) if r["id"] not in ids]
        if ids:
            break
    found, matches = None, []
    if ids:
        ents = wd.request(action="wbgetentities", ids="|".join(ids),
                          props="labels|aliases|descriptions|claims|sitelinks", languages="en|ms|mul")
        for rank, qid in enumerate(ids):
            e = ents.get("entities", {}).get(qid, {})
            humans = [c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                      for c in e.get("claims", {}).get("P31", [])]
            if "Q5" not in humans:
                continue
            sitelinks = e.get("sitelinks", {})
            wikis = [k for k in sitelinks if k.endswith("wiki") and k not in ("commonswiki", "specieswiki")]
            if not wikis:
                continue  # no Wikipedia article: not a public figure
            labels = e.get("labels", {})
            # many people now only have a "mul" (multilingual) label, e.g. Anne Hathaway the actress
            label = (labels.get("en") or labels.get("mul") or labels.get("ms") or {}).get("value")
            if not label:
                continue
            aliases = [a["value"] for lang in ("en", "mul", "ms") for a in e.get("aliases", {}).get(lang, [])]
            if not _names_match(name, [label, *aliases]):
                continue
            born, died = _year(e, "P569"), _year(e, "P570")
            if y1 and born and born > y1 or y0 and died and died < y0:
                continue  # can't be in a photo from the asked years
            aliases = [a for a in dict.fromkeys(aliases) if 2 < len(a) <= 40 and re.search(r"[A-Za-z]", a)]
            desc = (e.get("descriptions", {}).get("en") or {}).get("value", "")
            matches.append({
                "qid": qid,
                "name": label,
                "description": desc,
                "aliases": aliases[:10],
                "commons_category": _commons_category(e),
                "image": _claim(e, "P18"),
                "_score": (
                    bool(hints) and any(h in desc.lower() for h in hints),  # fits "actress" etc.
                    _claim(e, "P18") is not None,                            # has a portrait to learn from
                    len(wikis),                                              # how widely known
                    -rank,                                                   # Wikidata's own ranking
                ),
            })
        if matches:
            found = max(matches, key=lambda m: m["_score"])
            found = {k: v for k, v in found.items() if k != "_score"}
    _cache[key] = found
    return found
