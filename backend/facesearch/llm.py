"""Thin client for the local Ollama server (vision captions, chat, embeddings)."""
from __future__ import annotations

import base64
import io

import numpy as np
import requests
from PIL import Image

from . import config


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int | None = None) -> dict:
    try:
        r = requests.post(f"{config.OLLAMA_URL}{path}", json=payload,
                          timeout=timeout or config.OLLAMA_TIMEOUT)
    except requests.Timeout as e:
        raise OllamaError(f"Ollama took longer than {timeout or config.OLLAMA_TIMEOUT}s to answer.") from e
    except requests.ConnectionError as e:
        raise OllamaError("Ollama is not running. Start the Ollama app, then try again.") from e
    if r.status_code == 404 and "not found" in r.text:
        raise OllamaError(f"Model '{payload.get('model')}' is not pulled. "
                          f"Run:  ollama pull {payload.get('model')}")
    if not r.ok:
        raise OllamaError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()


def status() -> dict:
    """Which of the configured models are available."""
    wanted = {"vision": config.VISION_MODEL, "chat": config.CHAT_MODEL, "embed": config.EMBED_MODEL}
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).json()
    except Exception:
        return {"running": False, "models": {k: {"name": v, "ready": False} for k, v in wanted.items()}}
    have = {m["name"] for m in tags.get("models", [])}
    have |= {n.removesuffix(":latest") for n in have}
    return {"running": True,
            "models": {k: {"name": v, "ready": v in have} for k, v in wanted.items()}}


def _image_b64(path, max_side: int = 896) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _mentioned(name: str, text: str) -> bool:
    low = text.lower()
    return any(part.lower() in low for part in name.split() if len(part) > 3)


def describe_image(path, people: list, context: str = "") -> str:
    """Ask the vision model for a factual, searchable description.

    `people` is a list of names, or (name, where-in-photo) tuples from face recognition.
    Small vision models often ignore names and write "a man in a suit", so the prompt
    insists on them and any name still missing is added in front of the description.
    """
    items = [p if isinstance(p, tuple) else (p, "") for p in people]
    who = ""
    if items:
        listed = "; ".join(f"{n} ({w})" if w else n for n, w in items)
        first = " and ".join(n for n, _ in items)
        who = (f"The people in this photo have already been identified: {listed}. "
               f"You MUST refer to them by these names. Start your first sentence with \"{first}\". "
               "Never call them 'a man', 'a woman', 'a person' or 'an individual'. "
               "Other people in the photo are unidentified; do not name them. ")
    hint = f"Background from the photo's source: {context[:400]} " if context else ""
    prompt = (
        f"{who}{hint}\n"
        "Describe this photograph in 2-4 factual sentences for a searchable photo archive. "
        "Say what the people are doing, their clothing, the setting, objects, any visible text, "
        "and whether it is indoor or outdoor, a portrait or a group photo. "
        "Do not guess names that were not given to you. Do not mention face recognition."
    )
    out = _post("/api/generate", {
        "model": config.VISION_MODEL,
        "prompt": prompt,
        "images": [_image_b64(path)],
        "stream": False,
        "options": {"temperature": 0.2},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    })
    text = out.get("response", "").strip()
    missing = [(n, w) for n, w in items if not _mentioned(n, text)]
    if missing:
        prefix = "; ".join(f"{n} ({w})" if w else n for n, w in missing)
        text = f"Shows {prefix}. {text}"
    return text


def embed(texts: list[str]) -> np.ndarray:
    """Unit-normalised embeddings, shape (n, d)."""
    out = _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts,
                               "keep_alive": config.OLLAMA_KEEP_ALIVE})
    vecs = np.asarray(out["embeddings"], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def chat(messages: list[dict], temperature: float = 0.3) -> str:
    out = _post("/api/chat", {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }, timeout=120)
    return out["message"]["content"].strip()


def chat_stream(messages: list[dict], temperature: float = 0.3):
    """Yield the reply a few characters at a time as Ollama generates it."""
    import json
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
            "model": config.CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }, stream=True, timeout=(5, 120))
    except requests.ConnectionError as e:
        raise OllamaError("Ollama is not running. Start the Ollama app, then try again.") from e
    if not r.ok:
        raise OllamaError(f"Ollama error {r.status_code}: {r.text[:300]}")
    with r:
        for line in r.iter_lines(chunk_size=None):  # pass words on as soon as they arrive
            if not line:
                continue
            part = json.loads(line)
            if part.get("error"):
                raise OllamaError(part["error"])
            text = part.get("message", {}).get("content", "")
            if text:
                yield text
            if part.get("done"):
                break
