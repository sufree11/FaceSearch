"""Face detection (YuNet) and face recognition (SFace) with OpenCV.

Both models are small ONNX files that run on CPU through opencv-python, so
nothing needs compiling on Windows.

  detect -> align/crop each face -> 128-d embedding -> compare with the
  reference gallery using cosine similarity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config

try:  # silence OpenCV's backend warnings
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

MAX_DETECT_SIDE = 1280  # downscale big photos before detection (faster, same result)


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h in original image pixels
    score: float                    # detector confidence
    embedding: np.ndarray           # unit-length 128-d vector
    person_id: str | None = None
    match_score: float = 0.0


@dataclass
class Gallery:
    """Reference embeddings for each known person."""
    people: dict[str, np.ndarray] = field(default_factory=dict)  # id -> (n, 128)

    def identify(self, emb: np.ndarray, threshold: float | None = None):
        """Return (person_id or None, best score)."""
        threshold = config.FACE_MATCH_THRESHOLD if threshold is None else threshold
        best_id, best = None, -1.0
        for pid, refs in self.people.items():
            sims = refs @ emb
            # mean of the top 3 matches is steadier than a single best match
            k = min(3, len(sims))
            score = float(np.sort(sims)[-k:].mean())
            if score > best:
                best_id, best = pid, score
        return (best_id if best >= threshold else None), best

    def save(self, path: Path):
        np.savez_compressed(path, **self.people)

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        if not path.exists():
            return cls()
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})


class FaceEngine:
    def __init__(self):
        for f in (config.YUNET_FILE, config.SFACE_FILE):
            if not f.exists():
                raise FileNotFoundError(
                    f"Missing model {f.name}. Run:  python cli.py models")
        self.detector = cv2.FaceDetectorYN.create(
            str(config.YUNET_FILE), "", (320, 320),
            config.FACE_DETECT_SCORE, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(config.SFACE_FILE), "")

    def analyze(self, image: np.ndarray, min_size: int | None = None) -> list[Face]:
        """Detect every face in a BGR image and embed it."""
        min_size = config.FACE_MIN_SIZE if min_size is None else min_size
        h, w = image.shape[:2]
        scale = min(1.0, MAX_DETECT_SIDE / max(h, w))
        small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.detector.detect(small)
        faces: list[Face] = []
        if dets is None:
            return faces
        for det in dets:
            det = det.copy()
            det[:14] /= scale  # back to original-resolution coordinates
            x, y, fw, fh = (int(round(v)) for v in det[:4])
            if min(fw, fh) < min_size:
                continue
            aligned = self.recognizer.alignCrop(image, det)
            emb = self.recognizer.feature(aligned).flatten().astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            faces.append(Face((max(x, 0), max(y, 0), fw, fh), float(det[14]), emb))
        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)  # biggest first
        return faces


def read_image(path: str | Path) -> np.ndarray | None:
    """cv2.imread can't open non-ASCII Windows paths; this can."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_image(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def load_people() -> list[dict]:
    return json.loads(config.PEOPLE_FILE.read_text(encoding="utf-8"))
