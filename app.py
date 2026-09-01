"""
SIH26093 - AI-Based Real-Time Stress and Trauma Assessment Module
for Victims/Complainants Accessing NHAA (14566) and Integrated Portal.

FastAPI backend wiring together:
  Text:   MuRIL (optional/lazy) + bilingual keyword rules  -> ml/text_model.py
  Voice:  faster-whisper (optional/lazy) + Librosa prosody -> ml/voice_model.py
  Fusion: logistic regression                              -> ml/fusion.py

Every heavy ML dependency is optional and lazy-loaded: the API is fully
functional on keyword rules + prosody heuristics alone, and transparently
upgrades itself if transformers/faster-whisper are installed and their
models can be downloaded. Every response reports which mode actually ran,
so nothing is silently faked.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml import fusion, text_model, voice_model

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SIH26093 Stress & Trauma Assessment API",
    description="Real-time text/voice stress & trauma risk assessment for "
    "victims/complainants accessing NHAA (14566) and the integrated portal.",
    version="1.0.0",
)

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=frontend_origin != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

PROBLEM_ID = "SIH26093"
_audit_log: List[Dict[str, Any]] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_payload(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _log(event_type: str, detail: Dict[str, Any]) -> None:
    _audit_log.append(
        {"event_type": event_type, "timestamp": _now_iso(), "detail": detail}
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TextAssessRequest(BaseModel):
    text: str
    complainant_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check() -> Dict[str, Any]:
    return {
        "problem_id": PROBLEM_ID,
        "status": "OPERATIONAL",
        "timestamp": _now_iso(),
    }


@app.post("/api/v1/assess/text", status_code=status.HTTP_201_CREATED)
def assess_text(payload: TextAssessRequest) -> Dict[str, Any]:
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    text_result = text_model.analyze_text(payload.text)
    fusion_result = fusion.fuse(text_result["text_stress_score"], None)
    risk_score = fusion_result["risk_score"]

    record = {
        "ps_id": PROBLEM_ID,
        "complainant_id": payload.complainant_id,
        "input_type": "text",
        "text_analysis": text_result,
        "prosody_analysis": None,
        "risk_score": risk_score,
        "risk_label": fusion.classify(risk_score),
        "fusion_mode": fusion_result["fusion_mode"],
        "timestamp": _now_iso(),
    }
    record["sha256_hash"] = _hash_payload(record)
    _log("TEXT_ASSESSMENT", record)
    return record


@app.post("/api/v1/assess/voice", status_code=status.HTTP_201_CREATED)
async def assess_voice(
    file: UploadFile = File(...), complainant_id: Optional[str] = None
) -> Dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="uploaded audio is empty")

    voice_result = voice_model.analyze_voice(audio_bytes)

    text_result = None
    if voice_result.get("transcript"):
        text_result = text_model.analyze_text(voice_result["transcript"])
        text_score = text_result["text_stress_score"]
    else:
        text_score = 0.0

    prosody_score = voice_result.get("prosody_stress_score")
    fusion_result = fusion.fuse(text_score, prosody_score)
    risk_score = fusion_result["risk_score"]

    record = {
        "ps_id": PROBLEM_ID,
        "complainant_id": complainant_id,
        "input_type": "voice",
        "text_analysis": text_result,
        "prosody_analysis": {
            "prosody_stress_score": prosody_score,
            "features": voice_result.get("features"),
            "available": voice_result.get("available", False),
        },
        "transcript": voice_result.get("transcript"),
        "risk_score": risk_score,
        "risk_label": fusion.classify(risk_score),
        "fusion_mode": fusion_result["fusion_mode"],
        "timestamp": _now_iso(),
    }
    record["sha256_hash"] = _hash_payload(record)
    _log("VOICE_ASSESSMENT", record)
    return record


@app.get("/api/v1/audit/logs")
def audit_logs() -> Dict[str, Any]:
    return {"total_records": len(_audit_log), "records": _audit_log}


@app.get("/api/v1/system/status")
def system_status() -> Dict[str, Any]:
    """Reports which ML components are actually active vs. falling back,
    so the demo is honest about what's really running."""
    muril_active = text_model._try_load_muril() is not None
    whisper_active = voice_model._try_load_whisper() is not None
    fusion_model_active = fusion._load_model() is not None
    return {
        "problem_id": PROBLEM_ID,
        "components": {
            "keyword_rules": "ACTIVE (always on)",
            "muril_text_model": "ACTIVE" if muril_active else "FALLBACK (keyword rules only)",
            "faster_whisper": "ACTIVE" if whisper_active else "UNAVAILABLE (no transcription)",
            "librosa_prosody": "ACTIVE" if voice_model._LIBROSA_AVAILABLE else "UNAVAILABLE",
            "fusion_model": "TRAINED_LOGREG" if fusion_model_active else "DEFAULT_WEIGHTS",
        },
        "total_assessments_logged": len(_audit_log),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
