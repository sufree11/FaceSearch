"""Metadata: EXIF from the file itself, plus the source info saved at download time.

Most photos from the internet have had their EXIF stripped, so the date usually
comes from the source page (Wikimedia Commons gives a structured date).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import ExifTags, Image

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
ISO_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})[-:/.](\d{1,2})[-:/.](\d{1,2})\b")


MONTHS = {m: i for i, names in enumerate([
    ("jan", "january", "januari"), ("feb", "february", "februari"), ("mar", "march", "mac"),
    ("apr", "april"), ("may", "mei"), ("jun", "june"), ("jul", "july", "julai"),
    ("aug", "august", "ogos"), ("sep", "sept", "september"), ("oct", "october", "oktober"),
    ("nov", "november"), ("dec", "december", "disember")], 1) for m in names}
TEXT_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(1[89]\d{2}|20\d{2})\b")
MONTH_FIRST_RE = re.compile(r"\b([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(1[89]\d{2}|20\d{2})\b")


def normalise_date(text: str | None) -> tuple[str | None, int | None]:
    """'2018:05:10 12:00:00' -> ('2018-05-10', 2018); 'Taken on 4 June 2007' -> ('2007-06-04', 2007);
    'circa 1985' -> ('1985', 1985)."""
    if not text:
        return None, None
    m = ISO_RE.search(text)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}", y
    for rx, di, mi in ((TEXT_DATE_RE, 1, 2), (MONTH_FIRST_RE, 2, 1)):
        m = rx.search(text)
        if m and m[mi].lower() in MONTHS and 1 <= int(m[di]) <= 31:
            y = int(m[3])
            return f"{y:04d}-{MONTHS[m[mi].lower()]:02d}-{int(m[di]):02d}", y
    m = YEAR_RE.search(text)
    if m:
        return m[1], int(m[1])
    return None, None


def _gps_to_deg(value, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in value)
        deg = d + m / 60 + s / 3600
        return -deg if ref in ("S", "W") else deg
    except Exception:
        return None


def read_exif(path: Path) -> dict:
    out: dict = {}
    try:
        with Image.open(path) as img:
            out["width"], out["height"] = img.size
            exif = img.getexif()
            if not exif:
                return out
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            sub = exif.get_ifd(ExifTags.IFD.Exif)
            tags.update({ExifTags.TAGS.get(k, k): v for k, v in sub.items()})
            date = tags.get("DateTimeOriginal") or tags.get("DateTime")
            if date:
                out["exif_date"] = str(date)
            make, model = (str(tags.get(k) or "").strip(" \x00") for k in ("Make", "Model"))
            cam = model if make and model.lower().startswith(make.lower()) else f"{make} {model}".strip()
            if cam:
                out["camera"] = cam
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if gps and 2 in gps and 4 in gps:
                lat, lon = _gps_to_deg(gps[2], gps.get(1)), _gps_to_deg(gps[4], gps.get(3))
                if lat is not None and lon is not None:
                    out["gps"] = {"lat": round(lat, 6), "lon": round(lon, 6)}
    except Exception:
        pass
    return out


def load_source_meta(folder: Path) -> dict:
    """metadata.json written by the downloader: {filename: {...}}."""
    f = folder / "metadata.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}
