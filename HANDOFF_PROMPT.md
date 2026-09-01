# Handoff prompt — SIH26093 Stress & Trauma Assessment Prototype

Paste everything below this line into a new chat (with these project files
uploaded) to continue exactly where this session left off.

---

I'm building "SIH26093 — AI-Based Real-Time Stress and Trauma Assessment
Module for Victims/Complainants Accessing NHAA (14566) and Integrated
Portal." I have a working FastAPI + ML prototype already built (files
attached). Please review it and continue from here — don't rebuild from
scratch.

**What's already implemented and tested (all 7 pytest tests pass):**
- `app.py` — FastAPI backend with:
  - `GET /` health check
  - `POST /api/v1/assess/text` — text stress/trauma risk assessment
  - `POST /api/v1/assess/voice` — audio upload → transcription + prosody + risk
  - `GET /api/v1/audit/logs` — in-memory history
  - `GET /api/v1/system/status` — reports which ML components are actually active
- `ml/text_model.py` — bilingual (English + Hindi/Hinglish) keyword-rule
  distress scorer (always active, needs no downloads) + optional lazy-loaded
  MuRIL/transformer sentiment model (currently defaults to a placeholder
  public multilingual sentiment model via `MURIL_MODEL_ID` env var — NOT a
  real fine-tuned MuRIL emotion checkpoint yet).
- `ml/voice_model.py` — optional lazy-loaded faster-whisper transcription
  (`WHISPER_MODEL_SIZE` env var, default "tiny") + Librosa prosodic feature
  extraction (pitch variability, energy variability, speaking-rate proxy)
  that always runs regardless of whether whisper is available.
- `ml/fusion.py` — logistic regression combining text_score + prosody_score
  into a final risk_score/risk_label (LOW/MODERATE/HIGH). Ships with
  hand-set default coefficients; `ml/fusion_model.joblib` has since been
  trained on **synthetic** data via `train_fusion.py`.
- `train_fusion.py` — retrains the fusion model; currently only run on
  synthetic data, not real labeled examples.
- `index.html` — minimal functional UI (not a mockup) with an editable
  backend URL field, text-assessment panel, voice-upload panel, and a
  system-status panel that shows which ML components are really active.
- `Dockerfile` (reads `$PORT` for Railway), `docker-compose.yml` (local
  dev), `requirements.txt` (core deps installed; torch/transformers/
  faster-whisper are commented out — heavy, and need internet access to
  huggingface.co to download models, which the previous sandbox couldn't
  reach to test).
- `test_app.py` — 7 passing tests against the real endpoints (the
  originally uploaded test file tested an unrelated generic
  telemetry/dispatch API, not this project — it was replaced).
- `solution.md` — architecture doc, including an explicit "honesty notes"
  section on what's a real model vs. a placeholder/heuristic.

**Known gaps / what to do next, in priority order:**
1. **Real MuRIL checkpoint**: no fine-tuned MuRIL emotion model exists yet.
   Either fine-tune one on a multilingual emotion/distress dataset, or find
   an existing public checkpoint, and set `MURIL_MODEL_ID` to it.
2. **Real fusion training data**: `train_fusion.py` currently uses
   synthetic data. Needs real labeled (text_score, prosody_score, label)
   examples once available.
3. **faster-whisper not yet tested end-to-end** with a real audio file
   containing speech (only tested with a synthetic sine-wave tone, since
   the build sandbox had no internet access to download Whisper model
   weights from Hugging Face). Verify transcription quality once deployed
   somewhere with full internet access.
4. **Librosa prosody heuristic is hand-built**, not trained/validated —
   consider training a small classifier on labeled prosodic features
   instead of the current weighted-heuristic formula in
   `ml/voice_model.py::prosody_score`.
5. Decide on persistent storage (Postgres/TimescaleDB) to replace the
   in-memory audit log before this goes beyond a demo.
6. Deploy: backend (Dockerfile) → Railway, frontend (`index.html`) →
   Netlify. See `solution.md` for the CORS (`FRONTEND_ORIGIN`) config.

Please pick up from gap #1 unless I say otherwise.
