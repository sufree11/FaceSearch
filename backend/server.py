"""FaceSearch API.  Run:  uvicorn server:app --reload --port 8000"""
from __future__ import annotations

import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from facesearch import config, llm
from facesearch.search import SearchService

app = FastAPI(title="FaceSearch", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
svc = SearchService()


@app.on_event("startup")
def start_background():
    svc.start_background()  # AI descriptions for live-fetched photos


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/api/health")
def health():
    return {"ollama": llm.status(), "index": svc.stats(),
            "face_models": config.YUNET_FILE.exists() and config.SFACE_FILE.exists()}


@app.get("/api/people")
def people():
    return svc.people_summary()


@app.post("/api/chat")
def chat(body: ChatIn):
    q = body.message.strip()
    if not q:
        raise HTTPException(400, "Empty message")
    try:
        return svc.chat(q, body.history[-6:])
    except llm.OllamaError as e:
        raise HTTPException(503, str(e))


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn):
    """Like /api/chat, but streams newline-delimited JSON events (photos first, then words)."""
    q = body.message.strip()
    if not q:
        raise HTTPException(400, "Empty message")

    def events():
        try:
            for ev in svc.chat_stream(q, body.history[-6:]):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except llm.OllamaError as e:  # e.g. the embedding model is missing
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/face-search")
async def face_search(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image is larger than 15 MB")
    try:
        return svc.face_search(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@app.get("/api/images")
def images(person: str = "", year: str = "", sort: str = "newest", offset: int = 0, limit: int = 60):
    """All indexed photos. person: id or "_unidentified"; year: e.g. "2015" or "none"."""
    if year and year != "none" and not year.isdigit():
        raise HTTPException(400, "year must be a number or 'none'")
    return svc.browse(person, year, sort, max(offset, 0), min(max(limit, 1), 200))


@app.get("/api/images/{image_id}")
def image(image_id: int):
    rec = svc.record(image_id)
    if not rec:
        raise HTTPException(404, "Not found")
    return rec


@app.get("/api/random")
def random_query():
    return {"query": svc.random_query()}


@app.post("/api/reload")
def reload():
    """Call after indexing new photos so the running server picks them up."""
    svc.reload()
    return svc.stats()


app.mount("/media", StaticFiles(directory=config.RAW_DIR), name="media")


if __name__ == "__main__":  # python server.py
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
