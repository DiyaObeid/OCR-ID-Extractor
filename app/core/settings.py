import os
from pathlib import Path

# if Tesseract isn't in PATH, set manually like this:
# TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_EXE = os.environ.get("TESSERACT_EXE", "")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
USERS_JSON = DATA_DIR / "users.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# matching thresholds
MIN_MATCH_COUNT = 15
GOOD_MATCH_RATIO = 0.75
