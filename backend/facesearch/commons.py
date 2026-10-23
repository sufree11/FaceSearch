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

API = "https://commons.wikimedia.org/w/api.php"
OK_MIME = {"image/jpeg", "image/png", "image/webp"}
TAG_RE = re.compile(r"<[^>]+>")


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", value))).strip()


class Commons:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = config.WIKI_USER_AGENT

    def _get(self, **params) -> dict:
        params |= {"action": "query", "format": "json", "formatversion": 2}
        for attempt in range(4):
            r = self.s.get(API, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def category_files(self, category: str, depth: int = 1, limit: int = 200) -> list[str]:
        """File titles in a category, including sub-categories up to `depth`."""
        seen_cats, files, queue = set(), [], [(f"Category:{category}", 0)]
        while queue and len(files) < limit:
            cat, d = queue.pop(0)
            if cat in seen_cats:
                continue
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
    stem, _, ext = title.rpartition(".")
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")[:80]
    ext = ext.lower() if ext.lower() in ("jpg", "jpeg", "png", "webp") else "jpg"
    return f"{stem}.{ext}"


def download_person(person: dict, limit: int, width: int = 1024, depth: int = 1,
                    min_side: int = 300, log=print) -> int:
    """Download up to `limit` images for one person into data/raw/<id>/."""
    c = Commons()
    folder: Path = config.RAW_DIR / person["id"]
    folder.mkdir(parents=True, exist_ok=True)
    meta_file = folder / "metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    category = person.get("commons_category") or person["name"]
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
