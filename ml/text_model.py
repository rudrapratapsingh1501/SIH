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
    "hopeless", "no one is listening", "threatened", "threatening",
    "threatening to kill",
    "assaulted", "assault", "raped", "rape", "beaten", "beat me",
    "beating me", "abuse", "abused", "abusing me", "terrified",
    "scared for my life", "help me please", "emergency", "bleeding",
    "he will kill me", "she will kill me", "he might kill me",
    "she might kill me", "going to kill me", "trapped", "locked in",
    "locked me in", "hit me", "hits me", "hitting me", "punched",
    "punched me", "slapped", "slapped me", "kicked me",
    "strangle", "strangled", "strangling", "strangled me",
    "choke", "choked", "choking", "choked me", "chokehold",
    "molested", "molesting", "attacked", "attacked me", "stabbed",
    "kidnapped", "kidnap", "trafficking", "acid attack", "in danger",
    "life is in danger", "blackmail", "blackmailing", "extort",
    "extorting", "gun", "weapon", "knife to me", "chasing me",
    "broke into my house", "broke in", "break into my house",
    "grabbed my throat", "grabbed her throat", "grabbed his throat",
    "hands around my throat", "hands around my neck", "finish me off",
    "sending me nudes", "sending nudes", "leak them", "leak my photos",
    "leak the photos", "leak my pictures", "revenge porn", "sextortion",
    "expose my photos", "morphed photos",
]
MODERATE_DISTRESS_EN = [
    "scared", "afraid", "anxious", "panic", "crying", "shaking", "worried",
    "unsafe", "not safe", "threat", "harassed", "harassment", "harassing",
    "stalking", "stalked", "following me", "followed me", "nightmares",
    "can't sleep", "flashback", "distressed", "overwhelmed", "hurt me",
    "hurting me", "injured", "injury", "bruise", "bruises",
    "intimidated", "intimidating", "aggressive", "in trouble",
    "robbed", "stole from me", "broke into", "burgled",
    "domestic violence", "violent with me", "violence against me",
]

# Hindi / Hinglish (Devanagari + common romanized forms)
HIGH_DISTRESS_HI = [
    "मार डालूंगा", "जान से मार", "आत्महत्या", "मुझे बचाओ", "बलात्कार",
    "मार दिया", "बंधक", "मारा", "पीटा", "जान को खतरा",
    "khatam kar dunga", "jaan se maar", "bachao mujhe", "atmahatya",
    "balatkar", "maara", "peeta", "jaan ko khatra",
]
MODERATE_DISTRESS_HI = [
    "डर लग रहा है", "परेशान", "चिंतित", "रो रही हूं", "रो रहा हूं",
    "असुरक्षित", "धमकी", "पीछा कर रहा है", "dar lag raha", "pareshan",
    "chintit", "asurakshit", "dhamki", "peecha kar raha",
]

_HIGH_WORDS = sorted({w.lower() for w in HIGH_DISTRESS_EN + HIGH_DISTRESS_HI}, key=len, reverse=True)
_MOD_WORDS = sorted({w.lower() for w in MODERATE_DISTRESS_EN + MODERATE_DISTRESS_HI}, key=len, reverse=True)

# Tiered severity scoring: a *single* high-distress disclosure already
# indicates serious risk and should not need a second exact-phrase hit to
# clear the MODERATE/HIGH thresholds in ml/fusion.py. Each additional
# distinct match nudges the score further, capped at 1.0.
_HIGH_BASE = 0.72
_HIGH_STEP = 0.12
_MOD_BASE = 0.40
_MOD_STEP = 0.12

# Negation cues: if a keyword match is immediately preceded (within a small
# word window) by one of these, the match is almost always describing the
# *absence* of distress ("not scared", "no threat here") and should not
# count toward the score. This is a cheap, explainable guard against the
# most common false-positive pattern in a pure keyword system.
_NEGATIONS = {
    "not", "no", "never", "isn't", "wasn't", "aren't", "weren't",
    "doesn't", "didn't", "won't", "wont", "cannot", "can't", "cant",
    "nothing", "none",
}


def _preceded_by_negation(t: str, match_start: int, window: int = 3) -> bool:
    preceding_words = re.findall(r"[a-z']+", t[:match_start])[-window:]
    return any(w in _NEGATIONS for w in preceding_words)


def _find_matches(t: str, phrases: List[str]) -> List[str]:
    """Word-boundary match (with optional trailing 's' for simple plurals,
    e.g. 'flashback'/'flashbacks', 'bruise'/'bruises') so short entries
    (e.g. 'threat') don't fire on substrings inside unrelated words, while
    still matching multi-word phrases and Hindi/Devanagari text. A phrase
    counts as matched only if at least one occurrence isn't negated."""
    matches = []
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"s?(?!\w)"
        for m in re.finditer(pattern, t, flags=re.UNICODE):
            if not _preceded_by_negation(t, m.start()):
                matches.append(phrase)
                break
    return matches


def keyword_rule_score(text: str) -> Tuple[float, List[str]]:
    """Return (score in [0,1], matched_keywords)."""
    if not text:
        return 0.0, []
    t = text.lower()

    high_matches = _find_matches(t, _HIGH_WORDS)
    mod_matches = _find_matches(t, _MOD_WORDS)
    matched = high_matches + mod_matches

    if high_matches:
        score = _HIGH_BASE + _HIGH_STEP * (len(high_matches) - 1)
    elif mod_matches:
        score = _MOD_BASE + _MOD_STEP * (len(mod_matches) - 1)
    else:
        score = 0.0

    # Simple punctuation/caps intensity signal (small nudge, not a driver)
    if re.search(r"!!+", text) or (
        sum(1 for c in text if c.isupper()) > max(6, len(text) * 0.3)
    ):
        score += 0.05
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
