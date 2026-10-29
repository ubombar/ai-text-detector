"""Runs all detectors on a text and combines them into one report."""
import re
import threading
import time

from . import config
from .detectors.classifier import Classifier
from .detectors.gltr import GLTR
from .detectors.llm_pair import LLMPair

_lock = threading.Lock()
_models: dict = {}

METHOD_INFO = {
    "binoculars": ("Binoculars", "lower_is_ai"),
    "fast_detectgpt": ("Fast-DetectGPT", "higher_is_ai"),
    "classifier": ("DeBERTa classifier", "higher_is_ai"),
    "gltr": ("GLTR (top-10 share)", "higher_is_ai"),
}
VOTING = ("binoculars", "fast_detectgpt", "classifier")


class TextTooShort(ValueError):
    pass


def models() -> dict:
    """Load lazily on first request, then keep in memory."""
    if not _models:
        _models["pair"] = LLMPair()
        _models["classifier"] = Classifier()
        _models["gltr"] = GLTR()
    return _models


def verdict(score: float, threshold: float, direction: str, band: float) -> str:
    if abs(score - threshold) <= band:
        return "uncertain"
    is_ai = score < threshold if direction == "lower_is_ai" else score > threshold
    return "ai" if is_ai else "human"


def sentence_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"[^.!?\n]+[.!?]*", text) if m.group().strip()]


def score_sentences(text, pair_scores, threshold):
    """Binoculars per sentence, reusing the per-token values (no extra model calls)."""
    out = []
    offs = pair_scores.token_offsets
    for start, end in sentence_spans(text):
        idx = [i for i, (s, e) in enumerate(offs) if s >= start and e <= end]
        if len(idx) < 3:
            continue
        s = (pair_scores.log_ppl[idx].mean() / pair_scores.x_ppl[idx].mean()).item()
        out.append({"text": text[start:end].strip(), "score": s,
                    "verdict": verdict(s, threshold, "lower_is_ai", config.UNCERTAIN_BAND["binoculars"])})
    return out


def overall(verdicts: dict) -> dict:
    votes = [verdicts[m] for m in VOTING]
    n_ai, n_human = votes.count("ai"), votes.count("human")
    if n_ai >= 2:
        v = "ai"
        summary = f"{n_ai} of 3 detectors say this looks AI-generated."
    elif n_human >= 2:
        v = "human"
        summary = f"{n_human} of 3 detectors say this looks human-written."
    else:
        v = "inconclusive"
        summary = "The detectors disagree or are near their thresholds."
    return {"verdict": v, "summary": summary}


def analyze(text: str) -> dict:
    text = text.strip()
    t0 = time.time()
    with _lock:  # one analysis at a time; models share the GPU
        m = models()
        n_tokens = m["pair"].count_tokens(text)
        if n_tokens < config.MIN_TOKENS:
            raise TextTooShort(f"Text is too short ({n_tokens} tokens). Need at least {config.MIN_TOKENS}.")
        pair = m["pair"].score(text)
        clf = m["classifier"].score(text)
        gltr = m["gltr"].analyze(text)

    th = config.load_thresholds()
    scores = {
        "binoculars": pair.binoculars,
        "fast_detectgpt": pair.fast_detectgpt,
        "classifier": clf,
        "gltr": gltr["fractions"][0],
    }
    details = {
        "binoculars": "Perplexity relative to cross-perplexity between base and instruct model.",
        "fast_detectgpt": "z-score of the text's log-likelihood vs. what the model would sample.",
        "classifier": "Mean P(AI) over 512-token chunks.",
        "gltr": "Share of tokens in GPT-2's top-10 predictions. Shown for context, not counted in the vote.",
    }
    methods, verdicts = [], {}
    for mid, (name, direction) in METHOD_INFO.items():
        verdicts[mid] = verdict(scores[mid], th[mid], direction, config.UNCERTAIN_BAND[mid])
        methods.append({"id": mid, "name": name, "score": scores[mid], "threshold": th[mid],
                        "direction": direction, "verdict": verdicts[mid], "detail": details[mid]})

    n_words = len(text.split())
    warning = None
    if n_words < config.LOW_CONFIDENCE_WORDS:
        warning = f"Only {n_words} words. Results under {config.LOW_CONFIDENCE_WORDS} words are low confidence."
    if pair.n_tokens >= config.MAX_TOKENS:
        warning = (warning + " " if warning else "") + f"Only the first {config.MAX_TOKENS} tokens were scored."

    return {
        "n_words": n_words,
        "n_tokens": n_tokens,
        "warning": warning,
        "elapsed_s": round(time.time() - t0, 2),
        "overall": overall(verdicts),
        "methods": methods,
        "gltr": gltr,
        "sentences": score_sentences(text, pair, th["binoculars"]),
    }
