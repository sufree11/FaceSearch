"""Download photos and their metadata from Wikimedia Commons.

Commons is a good source for this project: photos of public figures are sorted
into categories, every file has structured metadata (date, author, license,
description) and the images are freely licensed.
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import requests

from . import config

API = config.COMMONS_API
OK_MIME = {"image/jpeg", "image/png", "image/webp"}
TAG_RE = re.compile(r"<[^>]+>")


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", value))).strip()


CAT_REDIRECT_RE = re.compile(
    r"\{\{\s*(?:Category redirect|Cat redirect|Categoryredirect|Seecat)\s*\|\s*(?:Category:)?([^|}]+)",
    re.IGNORECASE)


class Commons:
    """Polite Wikimedia Commons API client.

    - POST instead of GET, so long lists of file names don't hit "414 URI Too Long"
    - at most ~1 request per second
    - on "429 Too Many Requests" waits as long as Wikimedia asks (Retry-After), up to 6 tries
    """

    def __init__(self, delay: float = 1.0, log=print, api: str | None = None, patient: bool = True):
        """patient=True (commands): wait out throttling for minutes.
        patient=False (someone waiting in the chat): give up after a few seconds."""
        self.s = requests.Session()
        self.s.headers["User-Agent"] = config.WIKI_USER_AGENT
        self.delay, self.log, self._last = delay, log, 0.0
        self.api = api or API
        self.tries, self.max_pause = (6, 120) if patient else (3, 10)
        self.timeout = 60 if patient else 15

    def _get(self, **params) -> dict:
        return self.request(action="query", formatversion=2, **params)

    def request(self, **params) -> dict:
        """Any MediaWiki API call (also used for Wikidata)."""
        params |= {"format": "json"}
        r = None
        for attempt in range(self.tries):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.s.post(self.api, data=params, timeout=self.timeout)
            if r.status_code == 429 or r.status_code >= 500:
                retry = r.headers.get("Retry-After", "")
                pause = min(int(retry) if retry.isdigit() else 15 * (attempt + 1), self.max_pause)
                self.log(f"  Wikimedia is busy ({r.status_code}), waiting {pause}s...")
                time.sleep(pause)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def resolve_category(self, names: list[str]) -> str | None:
        """First of `names` that is a non-empty Commons category, following category redirects."""
        for name in dict.fromkeys(n for n in names if n):
            title = f"Category:{name}"
            data = self._get(titles=title, prop="categoryinfo|revisions", rvprop="content", rvslots="main")
            page = (data.get("query", {}).get("pages") or [{}])[0]
            if page.get("missing"):
                continue
            rev = (page.get("revisions") or [{}])[0]
            content = rev.get("slots", {}).get("main", {}).get("content", "") or rev.get("content", "")
            m = CAT_REDIRECT_RE.search(content)
            if m:
                target = m.group(1).strip()
                self.log(f"  category '{name}' redirects to '{target}'")
                return target
            info = page.get("categoryinfo", {})
            if info.get("files", 0) or info.get("subcats", 0):
                return name
        return None

    def category_files(self, category: str, depth: int = 1, limit: int = 200,
                       max_cats: int | None = None) -> list[str]:
        """File titles in a category, including sub-categories up to `depth`
        (visiting at most `max_cats` categories, to keep the number of requests down)."""
        seen_cats, files, queue = set(), [], [(f"Category:{category}", 0)]
        while queue and len(files) < limit:
            cat, d = queue.pop(0)
            if cat in seen_cats:
                continue
            if max_cats and len(seen_cats) >= max_cats:
                break
            seen_cats.add(cat)
            cont = {}
            while len(files) < limit:
                data = self._get(list="categorymembers", cmtitle=cat, cmlimit=500,
                                 cmtype="file|subcat", **cont)
                for m in data.get("query", {}).get("categorymembers", []):
                    if m["ns"] == 6 and m["title"] not in files:
                        files.append(m["title"])
                    elif m["ns"] == 14 and d < depth:
                        queue.append((m["title"], d + 1))
                if "continue" not in data:
                    break
                cont = data["continue"]
        return files[:limit]

    def depicts_files(self, qid: str, limit: int = 150) -> list[str]:
        """Photos whose Commons structured data says they depict this Wikidata person (P180).
        Far more complete than categories: famous people's photos sit deep in sub-categories."""
        titles, offset = [], 0
        while len(titles) < limit:
            data = self._get(list="search", srsearch=f"haswbstatement:P180={qid} filetype:bitmap",
                             srnamespace=6, srlimit=min(50, limit - len(titles)), sroffset=offset)
            batch = [m["title"] for m in data.get("query", {}).get("search", [])]
            titles += [t for t in batch if t not in titles]
            if "continue" not in data or not batch:
                break
            offset = data["continue"].get("sroffset", offset + len(batch))
        return titles[:limit]

    def search_files(self, query: str, limit: int = 100) -> list[str]:
        data = self._get(list="search", srsearch=query, srnamespace=6, srlimit=min(limit, 500))
        return [m["title"] for m in data.get("query", {}).get("search", [])]

    def file_info(self, titles: list[str], width: int) -> list[dict]:
        out = []
        for i in range(0, len(titles), 50):
            data = self._get(titles="|".join(titles[i:i + 50]), prop="imageinfo",
                             iiprop="url|size|mime|extmetadata", iiurlwidth=width)
            for page in data.get("query", {}).get("pages", []):
                info = (page.get("imageinfo") or [None])[0]
                if not info or info.get("mime") not in OK_MIME:
                    continue
                meta = info.get("extmetadata", {})
                val = lambda k: _clean(meta.get(k, {}).get("value"))  # noqa: E731
                out.append({
                    "title": page["title"].removeprefix("File:"),
                    "download_url": info.get("thumburl") or info["url"],
                    "source_url": info.get("descriptionurl"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "date": val("DateTimeOriginal") or val("DateTime"),
                    "original_caption": val("ImageDescription") or val("ObjectName"),
                    "author": val("Artist"),
                    "license": val("LicenseShortName"),
                    "categories": val("Categories"),
                })
        return out


def safe_name(title: str) -> str:
    """File name for a Commons title. Long titles are shortened with a short code from the full
    title, so "... New Delhi (1).jpg" and "... New Delhi (2).jpg" don't end up with the same name."""
    import hashlib
    stem, _, ext = title.rpartition(".")
    clean = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    if len(clean) > 80:
        clean = clean[:70].rstrip("_") + "_" + hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    ext = ext.lower() if ext.lower() in ("jpg", "jpeg", "png", "webp") else "jpg"
    return f"{clean}.{ext}"


def download_person(person: dict, limit: int, width: int = 1024, depth: int = 1,
                    min_side: int = 300, log=print) -> int:
    """Download up to `limit` images for one person into data/raw/<id>/."""
    c = Commons()
    folder: Path = config.RAW_DIR / person["id"]
    folder.mkdir(parents=True, exist_ok=True)
    meta_file = folder / "metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    names = [person.get("commons_category"), person["name"],
             *[a for a in person.get("aliases", []) if " " in a]]
    category = c.resolve_category(names) or person.get("commons_category") or person["name"]
    titles = c.category_files(category, depth=depth, limit=limit * 2)
    if not titles:
        log(f"  no category '{category}', falling back to search")
        titles = c.search_files(person["name"], limit=limit * 2)

    infos = [i for i in c.file_info(titles, width)
             if min(i["width"] or 0, i["height"] or 0) >= min_side]
    got = 0
    for info in infos:
        if got >= limit:
            break
        name = safe_name(info["title"])
        if name in meta and (folder / name).exists():
            got += 1
            continue
        try:
            r = c.s.get(info["download_url"], timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            log(f"  skip {name}: {e}")
            continue
        (folder / name).write_bytes(r.content)
        meta[name] = {k: v for k, v in info.items() if k != "download_url"} | {
            "source": "Wikimedia Commons", "category": category}
        got += 1
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.3)  # be polite to Wikimedia
    return got


# --------------------------------------------------------------------------- catalog
def catalog_path(person_id: str) -> Path:
    return config.CATALOG_DIR / f"{person_id}.json"


def build_catalog(person: dict, limit: int = 2000, width: int = 1024, depth: int = 1,
                  min_side: int = 300, log=print, patient: bool = True, max_cats: int | None = None) -> dict:
    """Record every Commons file for a person (title, download link, date, caption, license)
    without downloading the photos. Live search uses it to know instantly which files
    exist for a given year."""
    from datetime import datetime, timezone

    from .metadata import normalise_date
    c = Commons(log=log, patient=patient)
    if person.get("qid") and person.get("commons_category"):
        # exact category from Wikidata: never fall back to the bare name, which may belong to a namesake
        names = [person["commons_category"]]
    else:
        names = [person.get("commons_category"), person["name"],
                 *[a for a in person.get("aliases", []) if " " in a]]
    category = c.resolve_category(names) or person.get("commons_category") or person["name"]
    titles = []
    if person.get("qid"):  # photos tagged as depicting this person, then the category
        titles = c.depicts_files(person["qid"], limit=limit)
        log(f"  {len(titles)} photos tagged as depicting {person['name']}")
    titles += [t for t in c.category_files(category, depth=depth, limit=max(0, limit - len(titles)),
                                           max_cats=max_cats) if t not in titles]
    if not titles:
        log(f"  no category '{category}', falling back to search")
        titles = c.search_files(person["name"], limit=min(limit, 500))
    files = []
    for info in c.file_info(titles, width):
        if min(info["width"] or 0, info["height"] or 0) < min_side:
            continue
        date, year = normalise_date(info.get("date"))
        files.append(info | {"file": safe_name(info["title"]), "norm_date": date, "year": year})
    cat = {"person": person["id"], "category": category,
           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files}
    catalog_path(person["id"]).write_text(json.dumps(cat, indent=1, ensure_ascii=False), encoding="utf-8")
    return cat


_catalog_cache: dict[str, tuple[float, dict]] = {}


def load_catalog(person_id: str) -> dict | None:
    """Catalog from disk, cached in memory until the file changes."""
    p = catalog_path(person_id)
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    hit = _catalog_cache.get(person_id)
    if hit and hit[0] == mtime:
        return hit[1]
    cat = json.loads(p.read_text(encoding="utf-8"))
    _catalog_cache[person_id] = (mtime, cat)
    return cat
