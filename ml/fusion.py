"""
Fusion: lightweight, trainable, explainable logistic regression combining
- text_score (from MuRIL + keyword rules), and
- prosody_score (from Librosa)
into a single stress/trauma risk probability.

Ships with sensible hand-set default coefficients so the API works
immediately with zero training. Run `train_fusion.py` to fit real
coefficients on labeled data and it will be picked up automatically
(saved to ml/fusion_model.joblib).
"""

import os
from typing import Optional

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fusion_model.joblib")

_model = None
_model_load_attempted = False

# Sensible defaults if no trained model file exists yet: text weighted
# slightly higher than prosody, since keyword/semantic content is a more
# direct signal than heuristic acoustic features in this prototype.
_DEFAULT_COEF = np.array([1.8, 1.2])  # [text_score, prosody_score]
_DEFAULT_INTERCEPT = -1.4


def _load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        import joblib

        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
    except Exception:
        _model = None
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fuse(text_score: float, prosody_score: Optional[float]) -> dict:
    """
    Combine text and prosody scores into a final risk score.
    If prosody_score is None (no audio, or Librosa unavailable), fusion
    falls back to the text score alone.
    """
    if prosody_score is None:
        return {
            "risk_score": round(float(text_score), 4),
            "fusion_mode": "text_only",
        }

    model = _load_model()
    x = np.array([[text_score, prosody_score]])

    if model is not None:
        try:
            proba = float(model.predict_proba(x)[0][1])
            return {"risk_score": round(proba, 4), "fusion_mode": "trained_logreg"}
        except Exception:
            pass  # fall through to default coefficients

    z = float(np.dot(_DEFAULT_COEF, x[0]) + _DEFAULT_INTERCEPT)
    proba = _sigmoid(z)
    return {"risk_score": round(proba, 4), "fusion_mode": "default_logreg_weights"}


def classify(risk_score: float) -> str:
    if risk_score >= 0.70:
        return "HIGH_RISK"
    if risk_score >= 0.40:
        return "MODERATE_RISK"
    return "LOW_RISK"
