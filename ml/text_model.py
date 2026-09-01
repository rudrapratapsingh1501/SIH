"""
Text pipeline: MuRIL (fine-tuned multilingual emotion model) + bilingual
(English/Hindi) keyword rules.

Design choice for this prototype: the keyword-rule scorer ALWAYS runs and
never depends on any downloaded model, so the API works immediately on a
fresh deploy with zero setup. MuRIL is loaded lazily and only used if
- the `transformers` + `torch` packages are installed, AND
- the model can be downloaded/loaded successfully (needs internet access
  to huggingface.co, which some sandboxes/CI environments block).

If MuRIL is unavailable for any reason, the module transparently falls back
to keyword-only scoring and reports which mode was used in the response, so
you always know whether a real model was in the loop.

To use a real fine-tuned MuRIL checkpoint: set the environment variable
MURIL_MODEL_ID to your model's Hugging Face Hub id (or a local path).
Without it, this defaults to a public multilingual sentiment model as a
stand-in so the pipeline is exercised end-to-end even before you have your
own fine-tuned checkpoint.
"""

import os
import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Bilingual (English + Hindi/Hinglish) keyword rules
# ---------------------------------------------------------------------------
# Weighted keyword lists. Scores are on a 0-1 scale contribution per hit,
# capped in aggregate. This is intentionally simple and transparent
# (explainable), matching the "lightweight, explainable" fusion goal.

HIGH_DISTRESS_EN = [
    "kill myself", "suicide", "end my life", "can't take it anymore",
    "hopeless", "no one is listening", "threatened", "assaulted", "raped",
    "beaten", "abuse", "abused", "terrified", "scared for my life",
    "help me please", "emergency", "bleeding", "he will kill me",
    "she will kill me", "trapped", "locked in",
]
MODERATE_DISTRESS_EN = [
    "scared", "afraid", "anxious", "panic", "crying", "shaking", "worried",
    "unsafe", "threat", "harassed", "harassment", "stalking", "stalked",
    "nightmares", "can't sleep", "flashback", "distressed", "overwhelmed",
]

# Hindi / Hinglish (Devanagari + common romanized forms)
HIGH_DISTRESS_HI = [
    "मार डालूंगा", "जान से मार", "आत्महत्या", "मुझे बचाओ", "बलात्कार",
    "मार दिया", "बंधक", "khatam kar dunga", "jaan se maar",
    "bachao mujhe", "atmahatya", "balatkar",
]
MODERATE_DISTRESS_HI = [
    "डर लग रहा है", "परेशान", "चिंतित", "रो रही हूं", "रो रहा हूं",
    "असुरक्षित", "धमकी", "dar lag raha", "pareshan", "chintit",
    "asurakshit", "dhamki",
]

_HIGH_WORDS = [w.lower() for w in HIGH_DISTRESS_EN + HIGH_DISTRESS_HI]
_MOD_WORDS = [w.lower() for w in MODERATE_DISTRESS_EN + MODERATE_DISTRESS_HI]


def keyword_rule_score(text: str) -> Tuple[float, List[str]]:
    """Return (score in [0,1], matched_keywords)."""
    if not text:
        return 0.0, []
    t = text.lower()
    matched = []
    score = 0.0
    for phrase in _HIGH_WORDS:
        if phrase in t:
            matched.append(phrase)
            score += 0.45
    for phrase in _MOD_WORDS:
        if phrase in t:
            matched.append(phrase)
            score += 0.2
    # Simple punctuation/caps intensity signal
    if re.search(r"!!+", text) or (
        sum(1 for c in text if c.isupper()) > max(6, len(text) * 0.3)
    ):
        score += 0.1
        matched.append("[intensity: caps/exclamation]")
    return min(score, 1.0), matched


# ---------------------------------------------------------------------------
# MuRIL model (optional, lazy-loaded)
# ---------------------------------------------------------------------------

_MURIL_PIPELINE = None
_MURIL_LOAD_ATTEMPTED = False
_MURIL_MODEL_ID = os.environ.get(
    "MURIL_MODEL_ID",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # placeholder stand-in
)


def _try_load_muril():
    global _MURIL_PIPELINE, _MURIL_LOAD_ATTEMPTED
    if _MURIL_LOAD_ATTEMPTED:
        return _MURIL_PIPELINE
    _MURIL_LOAD_ATTEMPTED = True
    try:
        from transformers import pipeline  # noqa: WPS433 (lazy import)

        _MURIL_PIPELINE = pipeline(
            "sentiment-analysis", model=_MURIL_MODEL_ID
        )
    except Exception:
        # transformers/torch not installed, no internet to the hub, or the
        # model id is unavailable. Fail silently into keyword-only mode.
        _MURIL_PIPELINE = None
    return _MURIL_PIPELINE


def muril_score(text: str) -> Tuple[float, bool]:
    """Return (score in [0,1], model_used: bool)."""
    clf = _try_load_muril()
    if clf is None or not text:
        return 0.0, False
    try:
        result = clf(text[:512])[0]
        label = str(result.get("label", "")).lower()
        conf = float(result.get("score", 0.0))
        # Map generic sentiment models onto a "negative == distress" proxy.
        # NOTE: this is a stand-in mapping until a real fine-tuned
        # multilingual *emotion* MuRIL checkpoint is swapped in via
        # MURIL_MODEL_ID.
        if "neg" in label or label in {"0", "label_0"}:
            return conf, True
        return 1.0 - conf, True
    except Exception:
        return 0.0, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> Dict:
    kw_score, matched = keyword_rule_score(text)
    ml_score, model_used = muril_score(text)

    if model_used:
        # Blend: keyword rules stay explainable/dominant, model adds signal.
        combined = 0.6 * kw_score + 0.4 * ml_score
    else:
        combined = kw_score

    return {
        "text_stress_score": round(min(combined, 1.0), 4),
        "keyword_score": round(kw_score, 4),
        "model_score": round(ml_score, 4) if model_used else None,
        "matched_keywords": matched,
        "model_used": "MuRIL/transformer" if model_used else "keyword_rules_only",
    }
