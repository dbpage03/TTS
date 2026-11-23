#!/usr/bin/env python
"""Qwen3-TTS web server.

A small FastAPI app that wraps the same model and options as ``tts_cli.py``
behind a browser UI (``web/index.html``).

Usage:
    python tts_server.py                      # http://127.0.0.1:8000
    python tts_server.py --port 8080 --host 0.0.0.0
    python tts_server.py --dtype float16 --flash-attn

The model loads in the background at startup; the UI shows progress and
enables the Generate button once it's ready. Generated clips are written to
``outputs/`` and listed in the UI's history panel.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

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


# --------------------------------------------------------------------------- #
# Model state
# --------------------------------------------------------------------------- #
class ModelState:
    """Holds the model plus its load status; synthesis is serialized by a lock."""

    def __init__(self) -> None:
        self.model = None
        self.status = "idle"  # idle | loading | ready | error
        self.error: Optional[str] = None
        self.device: Optional[str] = None
        self.model_id: Optional[str] = None
        self.dtype: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.busy = False
        self.progress = {"done": 0, "total": 0}
        self.lock = threading.Lock()

    def load(self, model_id: str, device: str, dtype: str, flash_attn: bool) -> None:
        from qwen_tts import Qwen3TTSModel

        self.status = "loading"
        self.model_id = model_id
        self.dtype = dtype
        try:
            self.device = resolve_device(device)
            print(f"Loading model {model_id} on {self.device} ({dtype})...")
            t0 = time.perf_counter()
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=DTYPE_MAP[dtype],
                attn_implementation="flash_attention_2" if flash_attn else None,
            )
            self.load_seconds = time.perf_counter() - t0
            print(f"Model loaded in {self.load_seconds:.1f}s.")
            self._warmup()
            self.status = "ready"
        except Exception as e:  # surface the failure in the UI
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            print(f"Model load failed: {self.error}")

    def _warmup(self) -> None:
        """Run one tiny generation so CUDA kernels / caches are hot before the
        first real request (the first call is otherwise noticeably slower)."""
        self.status = "warming"
        try:
            t0 = time.perf_counter()
            self.model.generate_custom_voice(
                text="Hi.", language="English", speaker="Ryan", max_new_tokens=64
            )
            print(f"Warm-up done in {time.perf_counter() - t0:.1f}s.")
        except Exception as e:
            print(f"Warm-up skipped: {e}")


STATE = ModelState()


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
# API
# --------------------------------------------------------------------------- #
class SynthRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    speaker: str = "Vivian"
    language: str = "Auto"
    instruct: Optional[str] = None
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


app = FastAPI(title="Qwen3-TTS")


def _canonical_speaker(name: str) -> str:
    lowered = {k.lower(): k for k in SPEAKER_DESCRIPTIONS}
    if name.lower() not in lowered:
        raise HTTPException(400, f"Unknown speaker '{name}'.")
    return lowered[name.lower()]


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9 _-]+", "", text).strip().replace(" ", "_")
    return stem[:40] or "clip"


def _clip_info(wav: Path) -> dict:
    meta_path = wav.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "id": wav.stem,
        "url": f"/api/clips/{wav.stem}.wav",
        "created": meta.get("created")
        or datetime.fromtimestamp(wav.stat().st_mtime).isoformat(timespec="seconds"),
        **{k: v for k, v in meta.items() if k != "created"},
    }


@app.get("/api/status")
def status():
    return {
        "status": STATE.status,
        "error": STATE.error,
        "device": STATE.device,
        "model": STATE.model_id,
        "dtype": STATE.dtype,
        "load_seconds": STATE.load_seconds,
        "busy": STATE.busy,
        "progress": STATE.progress,
    }


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
    if STATE.status != "ready":
        raise HTTPException(503, f"Model is not ready (status: {STATE.status}).")
    speaker = _canonical_speaker(req.speaker)
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported language '{req.language}'.")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text is empty.")
    instruct = (req.instruct or "").strip() or None

    chunks = split_text(text, req.chunk_chars)
    gen_kwargs = {
        k: v
        for k, v in {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
        }.items()
        if v is not None
    }

    with STATE.lock:
        STATE.busy = True
        STATE.progress = {"done": 0, "total": len(chunks)}
        try:
            if req.seed is not None:
                import torch

                torch.manual_seed(req.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(req.seed)
            t0 = time.perf_counter()
            pieces: list[np.ndarray] = []
            sr = 24000
            for i in range(0, len(chunks), req.batch_size):
                batch = chunks[i : i + req.batch_size]
                wavs, sr = STATE.model.generate_custom_voice(
                    text=batch,
                    language=req.language,
                    speaker=speaker,
                    instruct=instruct,
                    max_new_tokens=req.max_new_tokens,
                    **gen_kwargs,
                )
                pieces.extend(np.asarray(w, dtype=np.float32).flatten() for w in wavs)
                STATE.progress = {"done": len(pieces), "total": len(chunks)}
            synth_s = time.perf_counter() - t0
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        finally:
            STATE.busy = False

    if len(pieces) > 1 and req.pause_ms:
        gap = np.zeros(int(sr * req.pause_ms / 1000), dtype=np.float32)
        joined: list[np.ndarray] = []
        for j, p in enumerate(pieces):
            if j:
                joined.append(gap)
            joined.append(p)
        pieces = joined
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    duration = len(audio) / sr if sr else 0.0

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem_src = (req.name or "").strip() or text
    clip_id = f"{datetime.now():%Y%m%d-%H%M%S}_{_safe_stem(stem_src)}_{uuid.uuid4().hex[:4]}"
    wav_path = OUTPUT_DIR / f"{clip_id}.wav"
    sf.write(wav_path, audio, sr)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "speaker": speaker,
        "language": req.language,
        "instruct": instruct,
        "duration": round(duration, 2),
        "synth_seconds": round(synth_s, 2),
        "sample_rate": sr,
        "chunks": len(chunks),
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


@app.get("/api/clips")
def list_clips():
    if not OUTPUT_DIR.exists():
        return []
    wavs = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_clip_info(w) for w in wavs]


def _clip_path(clip_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
        raise HTTPException(400, "Bad clip id.")
    path = OUTPUT_DIR / f"{clip_id}.wav"
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
def index():
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
    p.add_argument("--model", default=DEFAULT_MODEL)
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

    threading.Thread(
        target=STATE.load,
        args=(args.model, args.device, args.dtype, args.flash_attn),
        daemon=True,
    ).start()

    print(f"UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
