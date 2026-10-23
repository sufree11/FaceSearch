"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("FS_DATA_DIR", BACKEND_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"          # downloaded / your own photos, one folder per person
GALLERY_DIR = DATA_DIR / "gallery"  # reference faces, one folder per person
DB_PATH = DATA_DIR / "index.db"
MODELS_DIR = Path(os.getenv("FS_MODELS_DIR", BACKEND_DIR / "models"))
PEOPLE_FILE = Path(os.getenv("FS_PEOPLE_FILE", BACKEND_DIR / "people.json"))

# --- Face models (OpenCV Zoo, run on CPU, no compiler needed) ---
YUNET_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")

FACE_DETECT_SCORE = float(os.getenv("FS_FACE_DETECT_SCORE", 0.80))
FACE_MIN_SIZE = int(os.getenv("FS_FACE_MIN_SIZE", 40))         # px; ignore tiny background faces
FACE_MATCH_THRESHOLD = float(os.getenv("FS_FACE_THRESHOLD", 0.38))  # SFace cosine; OpenCV suggests 0.363
GALLERY_PER_PERSON = int(os.getenv("FS_GALLERY_PER_PERSON", 8))

# --- Local LLMs served by Ollama (https://ollama.com) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("FS_VISION_MODEL", "qwen2.5vl:3b")   # image descriptions
CHAT_MODEL = os.getenv("FS_CHAT_MODEL", "qwen2.5:3b")         # chat answers
EMBED_MODEL = os.getenv("FS_EMBED_MODEL", "bge-m3")           # multilingual text embeddings
OLLAMA_TIMEOUT = int(os.getenv("FS_OLLAMA_TIMEOUT", 300))

# --- Search ---
TOP_K = int(os.getenv("FS_TOP_K", 6))
MIN_SEMANTIC_SCORE = float(os.getenv("FS_MIN_SEMANTIC", 0.35))  # for searches with no person in them

# Wikimedia asks every API client to identify itself. Put your own contact here.
WIKI_USER_AGENT = os.getenv(
    "FS_WIKI_USER_AGENT",
    "FaceSearch/0.1 (internship learning project; set FS_WIKI_USER_AGENT to add your contact)",
)

for d in (DATA_DIR, RAW_DIR, GALLERY_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
