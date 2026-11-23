#!/usr/bin/env python
"""Qwen3-TTS web server.

A small FastAPI app that wraps the same model and options as ``tts_cli.py``
behind a browser UI (``web/index.html``), plus a voice-cloning page backed by
the Qwen3-TTS *Base* model.

Usage:
    python tts_server.py                      # http://127.0.0.1:8000
    python tts_server.py --port 8080 --host 0.0.0.0
    python tts_server.py --dtype float16 --flash-attn
    python tts_server.py --preload-clone      # load the clone model at startup too

Pages:
    /        Preset voices (CustomVoice model, loads at startup)
    /clone   Voice cloning (Base model, loads the first time the page is opened)

Generated clips are written to ``outputs/``; cloned voices (reference audio +
transcript) are stored in ``voices/``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

# Reuse the CLI's constants and helpers so both front-ends stay in sync.
from tts_cli import (
    DEFAULT_MODEL,
    DTYPE_MAP,
    SPEAKER_DESCRIPTIONS,
    SUPPORTED_LANGUAGES,
    resolve_device,
)

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs"
VOICES_DIR = ROOT / "voices"

DEFAULT_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_REF_SECONDS = 60.0

# Settings shared by both models (filled in from the CLI flags in main()).
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"

CONFIG = {"device": "auto", "dtype": "bfloat16", "flash_attn": False,
          "clone_model": DEFAULT_CLONE_MODEL,
          "whisper_model": DEFAULT_WHISPER_MODEL, "whisper_device": "auto"}

# Both models share one GPU, so only one generation runs at a time.
GPU_LOCK = threading.Lock()
# Model loading isn't thread-safe in transformers (concurrent from_pretrained
# calls can leave weights on the "meta" device), so loads run one at a time.
LOAD_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Model state
# --------------------------------------------------------------------------- #
class ModelState:
    """Holds one model plus its load status and generation progress."""

    def __init__(self, warmup: Optional[Callable] = None) -> None:
        self.model = None
        self.status = "idle"  # idle | loading | warming | ready | error
        self.error: Optional[str] = None
        self.device: Optional[str] = None
        self.model_id: Optional[str] = None
        self.dtype: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.busy = False
        self.progress = {"done": 0, "total": 0}
        self._warmup_fn = warmup
        self._load_lock = threading.Lock()

    def info(self) -> dict:
        return {
            "status": self.status,
            "error": self.error,
            "device": self.device,
            "model": self.model_id,
            "dtype": self.dtype,
            "load_seconds": self.load_seconds,
            "busy": self.busy,
            "progress": self.progress,
        }

    def start_loading(self, model_id: str) -> None:
        """Kick off a background load unless one is done or in flight."""
        with self._load_lock:
            if self.status in ("loading", "warming", "ready"):
                return
            self.status = "loading"
            self.error = None
        threading.Thread(target=self._load, args=(model_id,), daemon=True).start()

    def _load(self, model_id: str) -> None:
        from qwen_tts import Qwen3TTSModel

        self.model_id = model_id
        self.dtype = CONFIG["dtype"]
        try:
            self.device = resolve_device(CONFIG["device"])
            print(f"Loading model {model_id} on {self.device} ({self.dtype})...")
            with LOAD_LOCK:
                t0 = time.perf_counter()
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=DTYPE_MAP[self.dtype],
                    attn_implementation="flash_attention_2" if CONFIG["flash_attn"] else None,
                )
                self.load_seconds = time.perf_counter() - t0
            print(f"Model {model_id} loaded in {self.load_seconds:.1f}s.")
            if self._warmup_fn:
                self.status = "warming"
                try:
                    t0 = time.perf_counter()
                    with GPU_LOCK:
                        self._warmup_fn(self.model)
                    print(f"Warm-up done in {time.perf_counter() - t0:.1f}s.")
                except Exception as e:
                    print(f"Warm-up skipped: {e}")
            self.status = "ready"
        except Exception as e:  # surface the failure in the UI
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            print(f"Model load failed: {self.error}")


def _warm_custom(model) -> None:
    # One tiny generation so CUDA kernels / caches are hot before the first
    # real request (the first call is otherwise noticeably slower).
    model.generate_custom_voice(text="Hi.", language="English", speaker="Ryan", max_new_tokens=64)


STATE = ModelState(warmup=_warm_custom)  # preset voices (CustomVoice)
CLONE = ModelState()  # voice cloning (Base)


# --------------------------------------------------------------------------- #
# Text chunking
# --------------------------------------------------------------------------- #
# Sentence boundaries for Latin and CJK punctuation, keeping the punctuation.
_SENT_RE = re.compile(r"(?<=[.!?;:。！？；…])\s+|(?<=[。！？；…])|\n+")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into sentence-aligned chunks of at most ~max_chars.

    Shorter segments are generated in parallel as a batch, which is much
    faster on a GPU than one long autoregressive sequence and avoids the
    quality drift long generations can suffer from.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    sentences = [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        # Hard-wrap a single over-long sentence on whitespace / commas.
        while len(s) > max_chars:
            cut = max(s.rfind(" ", 0, max_chars), s.rfind(",", 0, max_chars), s.rfind("，", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if not s:
            continue
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks or [text]


# --------------------------------------------------------------------------- #
# Shared generation helpers
# --------------------------------------------------------------------------- #
class GenOptions(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    language: str = "Auto"
    max_new_tokens: int = Field(4096, ge=64, le=32768)
    play_on_server: bool = False
    name: Optional[str] = None
    # Speed: long text is split into chunks generated `batch_size` at a time.
    chunk_chars: int = Field(220, ge=0, le=5000)  # 0 = never split
    batch_size: int = Field(4, ge=1, le=16)
    pause_ms: int = Field(200, ge=0, le=3000)  # silence between chunks
    # Sampling (None = model default).
    temperature: Optional[float] = Field(None, gt=0, le=2)
    top_p: Optional[float] = Field(None, gt=0, le=1)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    repetition_penalty: Optional[float] = Field(None, ge=1, le=2)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)


class SynthRequest(GenOptions):
    speaker: str = "Vivian"
    instruct: Optional[str] = None


class CloneSynthRequest(GenOptions):
    voice_id: str


def _sampling_kwargs(req: GenOptions) -> dict:
    return {
        k: v
        for k, v in {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
        }.items()
        if v is not None
    }


def _seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_common(state: ModelState, req: GenOptions) -> str:
    if state.status != "ready":
        raise HTTPException(503, f"Model is not ready (status: {state.status}).")
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported language '{req.language}'.")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text is empty.")
    return text


def _generate_chunked(
    state: ModelState,
    req: GenOptions,
    text: str,
    generate: Callable[[list[str]], tuple[list, int]],
    before: Optional[Callable[[], None]] = None,
) -> tuple[np.ndarray, int, float, int]:
    """Split ``text``, run ``generate`` over batches of chunks under the GPU
    lock, and join the results. Returns (audio, sr, synth_seconds, n_chunks)."""
    chunks = split_text(text, req.chunk_chars)
    with GPU_LOCK:
        state.busy = True
        state.progress = {"done": 0, "total": len(chunks)}
        try:
            if before:
                before()
            _seed(req.seed)
            t0 = time.perf_counter()
            pieces: list[np.ndarray] = []
            sr = 24000
            for i in range(0, len(chunks), req.batch_size):
                wavs, sr = generate(chunks[i : i + req.batch_size])
                pieces.extend(np.asarray(w, dtype=np.float32).flatten() for w in wavs)
                state.progress = {"done": len(pieces), "total": len(chunks)}
            synth_s = time.perf_counter() - t0
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        finally:
            state.busy = False

    if len(pieces) > 1 and req.pause_ms:
        gap = np.zeros(int(sr * req.pause_ms / 1000), dtype=np.float32)
        joined: list[np.ndarray] = []
        for j, p in enumerate(pieces):
            if j:
                joined.append(gap)
            joined.append(p)
        pieces = joined
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return audio, sr, synth_s, len(chunks)


def _save_clip(req: GenOptions, text: str, audio: np.ndarray, sr: int, synth_s: float,
               n_chunks: int, extra: dict) -> dict:
    duration = len(audio) / sr if sr else 0.0
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem_src = (req.name or "").strip() or text
    clip_id = f"{datetime.now():%Y%m%d-%H%M%S}_{_safe_stem(stem_src)}_{uuid.uuid4().hex[:4]}"
    wav_path = OUTPUT_DIR / f"{clip_id}.wav"
    sf.write(wav_path, audio, sr)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "language": req.language,
        **extra,
        "duration": round(duration, 2),
        "synth_seconds": round(synth_s, 2),
        "sample_rate": sr,
        "chunks": n_chunks,
        "seed": req.seed,
    }
    wav_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if req.play_on_server:
        try:
            import sounddevice as sd

            sd.play(audio, samplerate=sr)  # non-blocking
        except Exception as e:
            print(f"Server playback failed: {e}")

    print(f"Synthesized {duration:.2f}s of audio in {synth_s:.2f}s -> {wav_path.name}")
    return _clip_info(wav_path)


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9 _-]+", "", text).strip().replace(" ", "_")
    return stem[:40] or "clip"


def _check_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise HTTPException(400, "Bad id.")
    return value


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clip_info(wav: Path) -> dict:
    meta = _read_json(wav.with_suffix(".json"))
    return {
        "id": wav.stem,
        "url": f"/api/clips/{wav.stem}.wav",
        "created": meta.get("created")
        or datetime.fromtimestamp(wav.stat().st_mtime).isoformat(timespec="seconds"),
        **{k: v for k, v in meta.items() if k != "created"},
    }


# --------------------------------------------------------------------------- #
# App + preset-voice API
# --------------------------------------------------------------------------- #
app = FastAPI(title="Qwen3-TTS")


def _canonical_speaker(name: str) -> str:
    lowered = {k.lower(): k for k in SPEAKER_DESCRIPTIONS}
    if name.lower() not in lowered:
        raise HTTPException(400, f"Unknown speaker '{name}'.")
    return lowered[name.lower()]


@app.get("/api/status")
def status():
    # Top level = preset-voice model (kept for compatibility); `clone` = Base model.
    return {**STATE.info(), "clone": CLONE.info()}


@app.get("/api/options")
def options():
    return {
        "speakers": [
            {"name": k, "description": v} for k, v in SPEAKER_DESCRIPTIONS.items()
        ],
        "languages": SUPPORTED_LANGUAGES,
    }


@app.post("/api/synthesize")
def synthesize(req: SynthRequest):
    # Plain `def` -> FastAPI runs it in a worker thread, so the event loop
    # (and /api/status polling) stays responsive during generation.
    text = _validate_common(STATE, req)
    speaker = _canonical_speaker(req.speaker)
    instruct = (req.instruct or "").strip() or None
    kwargs = _sampling_kwargs(req)

    def gen(batch: list[str]):
        return STATE.model.generate_custom_voice(
            text=batch, language=req.language, speaker=speaker, instruct=instruct,
            max_new_tokens=req.max_new_tokens, **kwargs,
        )

    audio, sr, synth_s, n = _generate_chunked(STATE, req, text, gen)
    return _save_clip(req, text, audio, sr, synth_s, n,
                      {"mode": "preset", "speaker": speaker, "instruct": instruct})


# --------------------------------------------------------------------------- #
# Voice cloning API
# --------------------------------------------------------------------------- #
# Prompt items (reference codes + speaker embedding) are built once per voice
# and cached, so repeat generations skip re-encoding the reference audio.
_PROMPT_CACHE: dict[str, list] = {}


class VoiceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    ref_text: Optional[str] = Field(None, max_length=5000)
    x_vector_only: bool = False
    # Either a base64-encoded audio file (the UI sends 24 kHz mono WAV) ...
    audio_b64: Optional[str] = None
    # ... or the id of an existing generated clip to clone from.
    from_clip: Optional[str] = None
    trim_silence: bool = True


class VoiceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=60)
    ref_text: Optional[str] = Field(None, max_length=5000)
    x_vector_only: Optional[bool] = None


def _load_audio(audio_b64: Optional[str] = None, from_clip: Optional[str] = None,
                voice_id: Optional[str] = None) -> tuple[np.ndarray, int]:
    """Read mono float32 audio from an upload, a history clip, or a saved voice."""
    if voice_id:
        path = _voice_dir(voice_id) / "ref.wav"
    elif from_clip:
        path = OUTPUT_DIR / f"{_check_id(from_clip)}.wav"
        if not path.exists():
            raise HTTPException(404, "Clip not found.")
    elif audio_b64:
        try:
            raw = base64.b64decode(audio_b64.split(",", 1)[-1])
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as e:
            raise HTTPException(400, f"Could not read audio ({e}). Upload a WAV, MP3, FLAC or OGG file.")
        path = None
    else:
        raise HTTPException(400, "Provide reference audio.")
    if path is not None:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


# --------------------------------------------------------------------------- #
# Whisper transcription (fills in the reference transcript for Full clones)
# --------------------------------------------------------------------------- #
class Transcriber:
    """Lazily loads a Whisper ASR pipeline on first use.

    Runs on the GPU when there's enough free VRAM next to the TTS models,
    otherwise on the CPU (slower, but only short reference clips go through it).
    """

    def __init__(self) -> None:
        self.pipe = None
        self.device: Optional[str] = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import pipeline

        want = CONFIG["whisper_device"]
        if want == "auto":
            want = "cpu"
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                # Leave room for any TTS model that hasn't loaded yet (~5 GB each).
                pending = sum(s.status != "ready" for s in (STATE, CLONE))
                if free > (2.5 + 5 * pending) * 1024**3:
                    want = "cuda:0"
        print(f"Loading Whisper {CONFIG['whisper_model']} on {want}...")
        t0 = time.perf_counter()
        with LOAD_LOCK:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=CONFIG["whisper_model"],
                device=want,
                torch_dtype=torch.float16 if want.startswith("cuda") else torch.float32,
            )
        self.device = want
        print(f"Whisper loaded in {time.perf_counter() - t0:.1f}s.")

    def transcribe(self, audio: np.ndarray, sr: int, language: Optional[str]) -> str:
        import librosa

        with self.lock:
            if self.pipe is None:
                self._load()
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        gen = {"task": "transcribe"}
        if language and language != "Auto":
            gen["language"] = language.lower()
        # Share the GPU lock with TTS generation so they don't contend for VRAM.
        lock = GPU_LOCK if self.device and self.device.startswith("cuda") else self.lock
        # Clips over Whisper's 30 s window use its native long-form mode,
        # which needs timestamps enabled.
        extra = {"return_timestamps": True} if len(audio) > 30 * 16000 else {}
        with lock:
            out = self.pipe(
                {"raw": audio.astype(np.float32), "sampling_rate": 16000},
                generate_kwargs=gen,
                **extra,
            )
        return (out.get("text") or "").strip()


TRANSCRIBER = Transcriber()


class TranscribeRequest(BaseModel):
    audio_b64: Optional[str] = None
    from_clip: Optional[str] = None
    voice_id: Optional[str] = None
    language: Optional[str] = None  # one of SUPPORTED_LANGUAGES; Auto = detect


@app.post("/api/transcribe")
def transcribe(req: TranscribeRequest):
    audio, sr = _load_audio(req.audio_b64, req.from_clip, req.voice_id)
    if len(audio) / sr > MAX_REF_SECONDS + 30:
        raise HTTPException(400, "Audio is too long to transcribe here.")
    t0 = time.perf_counter()
    try:
        text = TRANSCRIBER.transcribe(audio, sr, req.language)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {type(e).__name__}: {e}")
    secs = time.perf_counter() - t0
    print(f"Transcribed {len(audio) / sr:.1f}s of audio in {secs:.1f}s on {TRANSCRIBER.device}.")
    return {"text": text, "seconds": round(secs, 2), "device": TRANSCRIBER.device}


def _voice_dir(voice_id: str) -> Path:
    d = VOICES_DIR / _check_id(voice_id)
    if not (d / "ref.wav").exists():
        raise HTTPException(404, "Voice not found.")
    return d


def _voice_info(d: Path) -> dict:
    meta = _read_json(d / "meta.json")
    return {"id": d.name, "ref_url": f"/api/voices/{d.name}/ref.wav", **meta}


@app.post("/api/clone/load")
def clone_load():
    CLONE.start_loading(CONFIG["clone_model"])
    return CLONE.info()


@app.get("/api/voices")
def list_voices():
    if not VOICES_DIR.exists():
        return []
    dirs = [d for d in VOICES_DIR.iterdir() if (d / "ref.wav").exists()]
    dirs.sort(key=lambda d: (d / "ref.wav").stat().st_mtime, reverse=True)
    return [_voice_info(d) for d in dirs]


@app.post("/api/voices")
def create_voice(req: VoiceCreate):
    ref_text = (req.ref_text or "").strip() or None
    x_vec = req.x_vector_only or not ref_text  # no transcript -> embedding-only

    audio, sr = _load_audio(req.audio_b64, req.from_clip)
    if req.from_clip and ref_text is None:
        ref_text = _read_json(OUTPUT_DIR / f"{req.from_clip}.json").get("text")
        x_vec = req.x_vector_only or not ref_text

    if req.trim_silence and len(audio):
        import librosa

        audio, _ = librosa.effects.trim(audio, top_db=40)
    duration = len(audio) / sr
    if duration < 1.0:
        raise HTTPException(400, "Reference audio is too short (need at least ~1 second of speech).")
    if duration > MAX_REF_SECONDS:
        raise HTTPException(400, f"Reference audio is {duration:.0f}s; keep it under {MAX_REF_SECONDS:.0f}s (5-20s works best).")
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak > 0.99 or peak < 0.3:  # normalise very quiet / clipping input
        audio = audio * (0.9 / peak)

    VOICES_DIR.mkdir(exist_ok=True)
    voice_id = f"{_safe_stem(req.name)}_{uuid.uuid4().hex[:6]}"
    d = VOICES_DIR / voice_id
    d.mkdir()
    sf.write(d / "ref.wav", audio.astype(np.float32), sr)
    meta = {
        "name": req.name.strip(),
        "ref_text": ref_text,
        "x_vector_only": bool(x_vec),
        "duration": round(duration, 2),
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": f"clip:{req.from_clip}" if req.from_clip else "upload",
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return _voice_info(d)


@app.patch("/api/voices/{voice_id}")
def update_voice(voice_id: str, req: VoiceUpdate):
    d = _voice_dir(voice_id)
    meta = _read_json(d / "meta.json")
    if req.name is not None:
        meta["name"] = req.name.strip()
    if req.ref_text is not None:
        meta["ref_text"] = req.ref_text.strip() or None
    if req.x_vector_only is not None:
        meta["x_vector_only"] = req.x_vector_only
    if not meta.get("ref_text"):
        meta["x_vector_only"] = True
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _PROMPT_CACHE.pop(voice_id, None)
    return _voice_info(d)


@app.get("/api/voices/{voice_id}/ref.wav")
def voice_ref(voice_id: str):
    return FileResponse(_voice_dir(voice_id) / "ref.wav", media_type="audio/wav")


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    d = _voice_dir(voice_id)
    shutil.rmtree(d)
    _PROMPT_CACHE.pop(voice_id, None)
    return {"deleted": voice_id}


@app.post("/api/clone/synthesize")
def clone_synthesize(req: CloneSynthRequest):
    text = _validate_common(CLONE, req)
    d = _voice_dir(req.voice_id)
    meta = _read_json(d / "meta.json")
    x_vec = bool(meta.get("x_vector_only")) or not meta.get("ref_text")
    kwargs = _sampling_kwargs(req)
    prompt: list = []

    def build_prompt():
        # Runs under the GPU lock: encode the reference once and cache it.
        if req.voice_id not in _PROMPT_CACHE:
            t0 = time.perf_counter()
            _PROMPT_CACHE[req.voice_id] = CLONE.model.create_voice_clone_prompt(
                ref_audio=str(d / "ref.wav"),
                ref_text=None if x_vec else meta["ref_text"],
                x_vector_only_mode=x_vec,
            )
            print(f"Built voice prompt for '{meta.get('name')}' in {time.perf_counter() - t0:.2f}s.")
        prompt[:] = _PROMPT_CACHE[req.voice_id]

    def gen(batch: list[str]):
        return CLONE.model.generate_voice_clone(
            text=batch, language=req.language, voice_clone_prompt=list(prompt),
            max_new_tokens=req.max_new_tokens, **kwargs,
        )

    audio, sr, synth_s, n = _generate_chunked(CLONE, req, text, gen, before=build_prompt)
    return _save_clip(req, text, audio, sr, synth_s, n, {
        "mode": "clone",
        "speaker": meta.get("name", req.voice_id),
        "voice_id": req.voice_id,
        "clone_mode": "x-vector" if x_vec else "icl",
    })


# --------------------------------------------------------------------------- #
# Clips + pages
# --------------------------------------------------------------------------- #
@app.get("/api/clips")
def list_clips():
    if not OUTPUT_DIR.exists():
        return []
    wavs = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_clip_info(w) for w in wavs]


def _clip_path(clip_id: str) -> Path:
    path = OUTPUT_DIR / f"{_check_id(clip_id)}.wav"
    if not path.exists():
        raise HTTPException(404, "Clip not found.")
    return path


@app.get("/api/clips/{clip_id}.wav")
def get_clip(clip_id: str, download: bool = False):
    path = _clip_path(clip_id)
    if download:
        return FileResponse(path, media_type="audio/wav", filename=path.name)
    return FileResponse(path, media_type="audio/wav")


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: str):
    path = _clip_path(clip_id)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": clip_id}


@app.get("/")
@app.get("/clone")
def index():
    # One page app; the path picks the active tab.
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        "<rect width='32' height='32' rx='8' fill='#6d5dfc'/>"
        "<path d='M8 13v6M12 10v12M16 7v18M20 11v10M24 14v4' stroke='white' "
        "stroke-width='2.5' stroke-linecap='round'/></svg>"
    )
    return Response(svg, media_type="image/svg+xml")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-TTS web server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Preset-voice (CustomVoice) model.")
    p.add_argument("--clone-model", default=DEFAULT_CLONE_MODEL,
                   help="Voice-cloning (Base) model, e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base to save VRAM.")
    p.add_argument("--preload-clone", action="store_true",
                   help="Load the clone model at startup instead of on first use.")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                   help="Whisper model used to auto-transcribe reference audio "
                        "(e.g. openai/whisper-small for less memory).")
    p.add_argument("--whisper-device", default="auto",
                   help="Device for Whisper: auto (GPU if >2.5 GB free, else CPU), cuda:0, or cpu.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_MAP))
    p.add_argument("--flash-attn", "-f", action="store_true")
    args = p.parse_args()

    # Keep log lines in order with stderr when output is piped.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Faster matmuls on Ampere+ GPUs with negligible quality impact.
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    CONFIG.update(device=args.device, dtype=args.dtype, flash_attn=args.flash_attn,
                  clone_model=args.clone_model, whisper_model=args.whisper_model,
                  whisper_device=args.whisper_device)
    STATE.start_loading(args.model)
    if args.preload_clone:
        CLONE.start_loading(args.clone_model)

    print(f"UI: http://{args.host}:{args.port}  (voice cloning: /clone)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
