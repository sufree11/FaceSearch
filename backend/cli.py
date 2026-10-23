"""FaceSearch command line.

  python cli.py models                     download the face models (once)
  python cli.py download --per-person 40   photos + metadata from Wikimedia Commons
  python cli.py gallery                    build reference faces per person
  python cli.py index                      faces -> descriptions -> embeddings -> SQLite
  python cli.py relabel                    re-identify faces after editing the gallery
  python cli.py redescribe                 regenerate AI descriptions (keeps faces)
  python cli.py evaluate                   face recognition accuracy
  python cli.py ask "Bagi gambar Tun Mahathir"   test a query without the website
"""
from __future__ import annotations

import argparse
import json
import sys

import requests

from facesearch import config


def cmd_models(_):
    for url, path in ((config.YUNET_URL, config.YUNET_FILE), (config.SFACE_URL, config.SFACE_FILE)):
        if path.exists() and path.stat().st_size > 100_000:
            print(f"ok  {path.name}")
            continue
        print(f"downloading {path.name} ...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"ok  {path.name} ({len(r.content) // 1024} KB)")


def cmd_download(a):
    from facesearch.commons import download_person
    from facesearch.faces import load_people
    people = load_people()
    if a.only:
        people = [p for p in people if p["id"] in a.only]
    for p in people:
        print(f"{p['name']} ...")
        n = download_person(p, limit=a.per_person, width=a.width, depth=a.depth)
        print(f"  {n} images in data/raw/{p['id']}/")


def cmd_gallery(a):
    from facesearch.faces import FaceEngine
    from facesearch.indexer import build_gallery
    build_gallery(FaceEngine(), per_person=a.per_person)
    print("\nCheck the photos in data/gallery/<person>/ - delete any that show the wrong person,"
          " then run this command again.")


def cmd_index(a):
    from facesearch.faces import FaceEngine
    from facesearch.indexer import index_images, load_gallery
    gallery = load_gallery()
    if not gallery.people:
        sys.exit("No gallery yet. Run:  python cli.py gallery")
    n = index_images(FaceEngine(), gallery, force=a.force, captions=not a.no_captions, limit=a.limit)
    print(f"\nIndexed {n} images.")
    try:
        requests.post("http://127.0.0.1:8000/api/reload", timeout=5)
        print("Running server reloaded.")
    except requests.RequestException:
        pass


def cmd_relabel(_):
    from facesearch.indexer import load_gallery, relabel
    relabel(load_gallery())


def cmd_redescribe(a):
    from facesearch.indexer import redescribe
    n = redescribe(all_images=a.all, force=a.force)
    print(f"\nRe-described {n} images.")
    try:
        requests.post("http://127.0.0.1:8000/api/reload", timeout=5)
        print("Running server reloaded.")
    except requests.RequestException:
        pass


def cmd_evaluate(_):
    from facesearch.faces import FaceEngine
    from facesearch.indexer import evaluate, load_gallery
    evaluate(FaceEngine(), load_gallery())


def cmd_ask(a):
    from facesearch.search import SearchService
    res = SearchService().chat(a.query)
    print(json.dumps(res["parsed"], ensure_ascii=False))
    print("\n" + res["answer"] + "\n")
    for n, im in enumerate(res["images"], 1):
        who = ", ".join(f["name"] for f in im["faces"] if f["name"]) or "-"
        print(f"[{n}] {im['url']}  {im['date'] or 'no date'}  {who}  ({im['match']}, {im['relevance']})")


def main():
    ap = argparse.ArgumentParser(description="FaceSearch pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    d = sub.add_parser("download")
    d.add_argument("--per-person", type=int, default=40)
    d.add_argument("--width", type=int, default=1024, help="download size in px")
    d.add_argument("--depth", type=int, default=1, help="sub-category depth")
    d.add_argument("--only", nargs="*", help="person ids from people.json")
    d.set_defaults(fn=cmd_download)
    g = sub.add_parser("gallery")
    g.add_argument("--per-person", type=int, default=config.GALLERY_PER_PERSON)
    g.set_defaults(fn=cmd_gallery)
    i = sub.add_parser("index")
    i.add_argument("--force", action="store_true", help="re-index everything")
    i.add_argument("--no-captions", action="store_true", help="skip the vision model (fast test)")
    i.add_argument("--limit", type=int, help="max new images per person/folder")
    i.set_defaults(fn=cmd_index)
    sub.add_parser("relabel").set_defaults(fn=cmd_relabel)
    r = sub.add_parser("redescribe", help="regenerate descriptions, keep faces")
    r.add_argument("--all", action="store_true", help="also photos with no recognised person")
    r.add_argument("--force", action="store_true", help="redo photos that already look fine")
    r.set_defaults(fn=cmd_redescribe)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    q = sub.add_parser("ask")
    q.add_argument("query")
    q.set_defaults(fn=cmd_ask)
    a = ap.parse_args()
    from facesearch.llm import OllamaError
    try:
        a.fn(a)
    except OllamaError as e:
        sys.exit(f"\n{e}")


if __name__ == "__main__":
    main()
