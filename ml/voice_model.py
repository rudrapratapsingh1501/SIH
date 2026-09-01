"""
Voice pipeline: faster-whisper (speech-to-text) -> feeds transcript into the
text model, + Librosa prosodic features analyzed in parallel ("how it's
said", independent of word content).

Like the text pipeline, faster-whisper is loaded lazily and optional. If it
isn't installed or a model can't be downloaded, transcription is skipped and
`transcript` comes back as None — but prosodic (Librosa) analysis still
runs, since it only needs numpy/scipy on the raw waveform.

Model size is configurable via WHISPER_MODEL_SIZE (default "tiny" — fastest,
smallest download, appropriate for a prototype demo on modest hardware).
"""

import io
import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:
    _LIBROSA_AVAILABLE = False

_WHISPER_MODEL = None
_WHISPER_LOAD_ATTEMPTED = False
_WHISPER_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")


def _try_load_whisper():
    global _WHISPER_MODEL, _WHISPER_LOAD_ATTEMPTED
    if _WHISPER_LOAD_ATTEMPTED:
        return _WHISPER_MODEL
    _WHISPER_LOAD_ATTEMPTED = True
    try:
        from faster_whisper import WhisperModel  # noqa: WPS433

        _WHISPER_MODEL = WhisperModel(
            _WHISPER_SIZE, device="cpu", compute_type="int8"
        )
    except Exception:
        _WHISPER_MODEL = None
    return _WHISPER_MODEL


def transcribe(audio_bytes: bytes) -> Optional[str]:
    model = _try_load_whisper()
    if model is None:
        return None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            segments, _info = model.transcribe(tmp.name, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Librosa prosodic ("how it's said") stress heuristic
# ---------------------------------------------------------------------------

def _load_waveform(audio_bytes: bytes) -> Optional[Tuple[np.ndarray, int]]:
    if not _LIBROSA_AVAILABLE:
        return None
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        if y.size == 0:
            return None
        return y, sr
    except Exception:
        return None


def prosody_score(audio_bytes: bytes) -> Dict:
    """
    Heuristic, explainable prosodic stress score in [0,1] from:
    - pitch (F0) mean/variability — elevated + unstable pitch correlates
      with vocal tension/distress
    - energy (RMS) variability — sharp bursts/tremor
    - speaking-rate proxy via onset density (rushed/erratic speech)
    This is a hand-built heuristic for the prototype, not a trained model —
    it's the "parallel prosodic signal" input that later feeds the fusion
    step alongside the text score.
    """
    loaded = _load_waveform(audio_bytes)
    if loaded is None:
        return {
            "prosody_stress_score": None,
            "features": None,
            "available": False,
        }

    y, sr = loaded
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        f0_mean = float(np.nanmean(f0_voiced)) if f0_voiced.size else 0.0
        f0_std = float(np.nanstd(f0_voiced)) if f0_voiced.size else 0.0

        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_rate = float(np.mean(onset_env))

        # Normalize each signal against loose, hand-picked reference ranges
        # (typical calm conversational speech) and average into one score.
        pitch_variability = min(f0_std / 60.0, 1.0) if f0_mean > 0 else 0.0
        energy_variability = min(rms_std / (rms_mean + 1e-6) / 1.5, 1.0)
        rushed_speech = min(onset_rate / 3.0, 1.0)

        score = float(
            np.clip(
                0.45 * pitch_variability
                + 0.35 * energy_variability
                + 0.20 * rushed_speech,
                0.0,
                1.0,
            )
        )

        return {
            "prosody_stress_score": round(score, 4),
            "features": {
                "pitch_mean_hz": round(f0_mean, 2),
                "pitch_std_hz": round(f0_std, 2),
                "energy_mean": round(rms_mean, 5),
                "energy_std": round(rms_std, 5),
                "onset_rate": round(onset_rate, 4),
            },
            "available": True,
        }
    except Exception:
        return {
            "prosody_stress_score": None,
            "features": None,
            "available": False,
        }


def analyze_voice(audio_bytes: bytes) -> Dict:
    transcript = transcribe(audio_bytes)
    prosody = prosody_score(audio_bytes)
    return {
        "transcript": transcript,
        "transcription_used": transcript is not None,
        **prosody,
    }
