"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("FS_DATA_DIR", BACKEND_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"          # downloaded / your own photos, one folder per person
CATALOG_DIR = DATA_DIR / "catalog"  # list of every Commons file + date per person (no photos)
GALLERY_DIR = DATA_DIR / "gallery"  # reference faces, one folder per person
DB_PATH = DATA_DIR / "index.db"
MODELS_DIR = Path(os.getenv("FS_MODELS_DIR", BACKEND_DIR / "models"))
PEOPLE_FILE = Path(os.getenv("FS_PEOPLE_FILE", BACKEND_DIR / "people.json"))
PEOPLE_AUTO_FILE = DATA_DIR / "people_auto.json"   # public figures learned on the fly from Wikidata

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
OLLAMA_TIMEOUT = int(os.getenv("FS_OLLAMA_TIMEOUT", 300))     # seconds per photo before skipping it
OLLAMA_KEEP_ALIVE = os.getenv("FS_OLLAMA_KEEP_ALIVE", "30m")   # keep models loaded between photos

# --- Search ---
TOP_K = int(os.getenv("FS_TOP_K", 6))
MIN_SEMANTIC_SCORE = float(os.getenv("FS_MIN_SEMANTIC", 0.35))  # for searches with no person in them

# --- Live fetch: get more photos from Wikimedia Commons when the archive has too few ---
LIVE_FETCH = os.getenv("FS_LIVE_FETCH", "1") != "0"
LIVE_MAX_CANDIDATES = int(os.getenv("FS_LIVE_MAX", 12))        # photos downloaded per live search, at most
LIVE_TIME_BUDGET = float(os.getenv("FS_LIVE_SECONDS", 25))     # stop fetching after this many seconds
LIVE_WORKERS = int(os.getenv("FS_LIVE_WORKERS", 4))            # parallel downloads
BACKGROUND_DESCRIBE = os.getenv("FS_BACKGROUND_DESCRIBE", "1") != "0"  # write AI descriptions later
COMMONS_API = os.getenv("FS_COMMONS_API", "https://commons.wikimedia.org/w/api.php")

# --- Unknown public figures: look them up on Wikidata and learn their face from Commons ---
AUTO_PEOPLE = os.getenv("FS_AUTO_PEOPLE", "1") != "0"
WIKIDATA_API = os.getenv("FS_WIKIDATA_API", "https://www.wikidata.org/w/api.php")
AUTO_MAX_FILES = int(os.getenv("FS_AUTO_MAX_FILES", 150))     # catalog size for a new person
AUTO_MAX_CATEGORIES = int(os.getenv("FS_AUTO_MAX_CATEGORIES", 8))  # sub-categories visited for a new person
AUTO_DOWNLOADS = int(os.getenv("FS_AUTO_DOWNLOADS", 12))      # photos downloaded to learn the face

# Wikimedia asks every API client to identify itself. Put your own contact here.
WIKI_USER_AGENT = os.getenv(
    "FS_WIKI_USER_AGENT",
    "FaceSearch/0.1 (https://github.com/sufree11/FaceSearch; internship learning project)",
)

for d in (DATA_DIR, RAW_DIR, GALLERY_DIR, MODELS_DIR, CATALOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
