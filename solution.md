# SIH26093 — Solution Architecture (Prototype)

**Problem Statement:** AI-Based Real-Time Stress and Trauma Assessment
Module for Victims/Complainants Accessing NHAA (14566) and Integrated
Portal.

## Pipeline

```
Text  ──► bilingual keyword rules (EN + HI, always on)
      ──► MuRIL / transformer sentiment model (optional, lazy-loaded)
                                │
Voice ──► faster-whisper transcription (optional, lazy-loaded) ──► text pipeline above
      ──► Librosa prosodic features (pitch, energy, speaking-rate) ──┐
                                                                       ▼
                                              Logistic Regression Fusion
                                                        │
                                          risk_score (0-1) + risk_label
                                     (LOW_RISK / MODERATE_RISK / HIGH_RISK)
```

## Why this design

- **Keyword rules are the always-on baseline.** They need no downloads, no
  GPU, and no internet access, so the API is fully functional the moment
  it's deployed — this matters for a prototype demo on a free-tier host.
- **MuRIL and faster-whisper are optional and lazy-loaded.** If
  `transformers`/`torch`/`faster-whisper` aren't installed, or a model
  can't be downloaded, the pipeline degrades gracefully instead of
  crashing. `GET /api/v1/system/status` reports exactly which components
  are active vs. falling back, so the demo never silently pretends a model
  ran when it didn't.
- **Fusion is a small, explainable logistic regression** over two features
  (`text_score`, `prosody_score`) — matching the "lightweight, trainable,
  explainable" requirement. It ships with reasonable default coefficients
  and can be retrained with `train_fusion.py` on real labeled data.

## Honesty notes for this prototype

- The MuRIL model used by default (`MURIL_MODEL_ID` env var) is a
  **placeholder public multilingual sentiment model**, not a real MuRIL
  checkpoint fine-tuned on emotion/distress data — that fine-tuning
  hasn't been done. Swap in a real fine-tuned checkpoint via the
  `MURIL_MODEL_ID` environment variable when one exists.
- The fusion logistic regression is trained on **synthetic data**
  (`train_fusion.py`) purely to make the pipeline runnable end-to-end.
  Retrain on real labeled distress/non-distress examples before drawing
  any conclusions from its scores.
- The Librosa "prosody stress score" is a **hand-built heuristic**
  (weighted pitch variability + energy variability + speaking rate), not
  a trained model. It's a reasonable placeholder signal for a prototype,
  not a validated clinical or forensic measure.
- This is a **demo/prototype**, not a validated risk-assessment tool. It
  should not be used to make real decisions about anyone's safety.

## Endpoints

- `GET /` — health check
- `POST /api/v1/assess/text` — `{text, complainant_id?}` → risk score/label
- `POST /api/v1/assess/voice` — multipart audio upload → risk score/label
- `GET /api/v1/audit/logs` — in-memory assessment history
- `GET /api/v1/system/status` — which ML components are actually active

## Deployment

- Backend: Dockerized FastAPI app → Railway (reads `$PORT` at runtime).
- Frontend: static `index.html` → Netlify.
- Before pushing the frontend, edit the `API_BASE_URL` constant near the
  bottom of `index.html`'s `<script>` block to your Railway backend URL.
  It's a plain code constant, not a UI field — end users never see or
  need to touch it.
- Set `FRONTEND_ORIGIN` on Railway to the Netlify URL to lock down CORS.
