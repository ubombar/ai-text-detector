"""Settings loaded from .env and thresholds.json."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B")
INSTRUCT_MODEL = os.getenv("INSTRUCT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "desklib/ai-text-detector-v1.01")
GLTR_MODEL = os.getenv("GLTR_MODEL", "gpt2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

MIN_TOKENS = 100
LOW_CONFIDENCE_WORDS = 250

THRESHOLDS_FILE = ROOT / "thresholds.json"
# Rough starting points for the Qwen2.5-3B pair (the papers' values are for
# Falcon-7B). Re-fit them on your own data with `python -m app.calibrate`.
DEFAULT_THRESHOLDS = {
    "binoculars": 0.97,
    "fast_detectgpt": 0.3,
    "classifier": 0.5,
    "gltr": 0.70,
}


def load_thresholds() -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if THRESHOLDS_FILE.exists():
        thresholds.update(json.loads(THRESHOLDS_FILE.read_text()))
    return thresholds

# Scores within this distance of a threshold count as "uncertain".
UNCERTAIN_BAND = {
    "binoculars": 0.02,
    "fast_detectgpt": 0.3,
    "classifier": 0.1,
    "gltr": 0.05,
}
