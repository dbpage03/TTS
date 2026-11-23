#!/usr/bin/env python
"""Qwen3-TTS quick test CLI.

Loads ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`` via the ``qwen-tts`` package,
synthesizes speech with ``generate_custom_voice``, and plays it straight to the
system speakers via ``sounddevice``.

Usage examples:
    # One-shot: speak the given text and exit.
    python tts_cli.py "Hello, this is a quick test of Qwen TTS."

    # With options: pick a speaker, language, style, and save a WAV.
    python tts_cli.py "Test" --speaker Ryan --language English \
        --instruct "Very happy." --save out.wav

    # Interactive: type-and-speak loop (quit / exit / q to stop).
    python tts_cli.py --interactive

    # Synthesize and save only (no playback).
    python tts_cli.py "Test" --save out.wav --no-play

    # List the supported speakers and languages (instant, no model load).
    python tts_cli.py --list-speakers
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np
import soundfile as sf

import torch

# Note: importing qwen_tts prints two harmless one-time startup warnings:
#   - "flash-attn is not installed" -> we use the manual PyTorch attention path.
#   - "SoX could not be found"      -> our playback path (sd.play/sd.wait)
#                                      does not require SoX.
# Both are cosmetic and do not affect functionality, so we leave them as-is.
from qwen_tts import Qwen3TTSModel

# Default model (CustomVoice 1.7B). Weights auto-download from Hugging Face on
# first run (requires internet; optionally set HF_TOKEN for gated access).
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# The 9 premium CustomVoice timbres (from the model card). This is the
# authoritative speaker set for the CustomVoice model, so ``--list-speakers``
# can run instantly without loading the ~3.8 GB model.
SPEAKER_DESCRIPTIONS = {
    "Vivian": "Bright, slightly edgy young female voice. (Chinese)",
    "Serena": "Warm, gentle young female voice. (Chinese)",
    "Uncle_Fu": "Seasoned male voice with a low, mellow timbre. (Chinese)",
    "Dylan": "Youthful Beijing male voice with a clear, natural timbre. (Beijing Dialect)",
    "Eric": "Lively Chengdu male voice with a slightly husky brightness. (Sichuan Dialect)",
    "Ryan": "Dynamic male voice with strong rhythmic drive. (English)",
    "Aiden": "Sunny American male voice with a clear midrange. (English)",
    "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre. (Japanese)",
    "Sohee": "Warm Korean female voice with rich emotion. (Korean)",
}

# Languages the CustomVoice model supports (plus "Auto" for auto-detection).
SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Map the ``--dtype`` CLI string to a torch dtype.
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts_cli.py",
        description="Quick test CLI for Qwen3-TTS CustomVoice text-to-speech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to synthesize. Omit and use --interactive for a type-and-speak loop.",
    )
    parser.add_argument(
        "--speaker",
        default="Vivian",
        help="Speaker name (e.g. Vivian, Ryan, Aiden, Ono_Anna, Sohee).",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="Target language (e.g. English, Chinese, Japanese) or 'Auto'.",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional natural-language style/tone instruction (e.g. 'Very happy.').",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Also write the synthesized audio to a WAV file at PATH.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Synthesize (and optionally save) but do not play to the speakers.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter an interactive type-and-speak loop (quit / exit / q to stop).",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print the supported speakers and languages, then exit (no model load).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id to load (overrides the default CustomVoice model).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to load the model on (e.g. cuda:0, cpu). 'auto' picks CUDA if available.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Model dtype.",
    )
    parser.add_argument(
        "--flash-attn",
        "-f",
        action="store_true",
        help="Use FlashAttention-2 (requires flash-attn installed and a compatible GPU).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32786,
        help="Maximum number of new codec tokens to generate.",
    )
    return parser


def resolve_device(requested: str) -> str:
    """Resolve the requested device, falling back to CPU if CUDA is unavailable."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    print(
        "Warning: CUDA is not available; falling back to CPU. "
        "Synthesis will be much slower.",
        file=sys.stderr,
    )
    return "cpu"


def validate_speaker(speaker: str) -> None:
    """Fail fast with a helpful message if the speaker name is not recognized."""
    known = set(SPEAKER_DESCRIPTIONS.keys())
    if speaker in known:
        return
    # Case-insensitive match to be forgiving (e.g. "ryan" -> "Ryan").
    lowered = {k.lower(): k for k in known}
    if speaker.lower() in lowered:
        return
    print(
        f"Error: unknown speaker '{speaker}'.\n"
        f"Supported speakers: {', '.join(sorted(known))}",
        file=sys.stderr,
    )
    raise SystemExit(2)


def load_model(args: argparse.Namespace) -> Qwen3TTSModel:
    """Load the Qwen3-TTS CustomVoice model once (reused across the session)."""
    device = resolve_device(args.device)
    print(f"Loading model {args.model} on {device} ({args.dtype})...")
    print("(First run downloads the weights from Hugging Face — this may take a while.)")

    attn_implementation = "flash_attention_2" if args.flash_attn else None

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=DTYPE_MAP[args.dtype],
        attn_implementation=attn_implementation,
    )
    print("Model loaded.")
    return model


def speak(model: Qwen3TTSModel, text: str, args: argparse.Namespace) -> None:
    """Synthesize ``text`` and play it to the speakers (optionally saving a WAV)."""
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
    )
    synth_s = time.perf_counter() - t0

    # Ensure a mono, 1-D float32 waveform for playback.
    audio = np.asarray(wavs[0], dtype=np.float32).flatten()
    duration = len(audio) / sr if sr else 0.0
    print(f"Synthesized {duration:.2f}s of audio in {synth_s:.2f}s.")

    if args.save:
        sf.write(args.save, audio, sr)
        print(f"Saved audio to {args.save}")

    if args.no_play:
        return

    # Import sounddevice lazily so that --list-speakers / --no-play runs never
    # trigger its "SoX could not be found" shell message at import time.
    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover - sounddevice unavailable
        print(f"Warning: sounddevice unavailable ({e}); skipping playback.", file=sys.stderr)
        return

    sd.play(audio, samplerate=sr)
    sd.wait()  # Block until playback finishes so output doesn't overlap.


def list_speakers() -> None:
    """Print the supported speakers (with descriptions) and languages.

    Uses the known CustomVoice speaker set so this runs instantly without
    loading the model.
    """
    print("Supported speakers:")
    for spk, desc in SPEAKER_DESCRIPTIONS.items():
        print(f"  - {spk}: {desc}")

    print("\nSupported languages:")
    print(f"  {', '.join(SUPPORTED_LANGUAGES)}")


def interactive_loop(model: Qwen3TTSModel, args: argparse.Namespace) -> None:
    """Type-and-speak loop until the user quits or hits EOF/Ctrl-C."""
    print("Interactive mode. Type text and press Enter to speak.")
    print("Type 'quit', 'exit', or 'q' (or Ctrl-C) to stop.")
    while True:
        try:
            text = input("text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        try:
            speak(model, text, args)
        except Exception as e:  # Keep the loop alive on a per-line failure.
            print(f"Error: {type(e).__name__}: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Nothing to do? Show a hint.
    if not args.text and not args.interactive and not args.list_speakers:
        parser.print_help()
        print("\nError: provide a text argument, or use --interactive / --list-speakers.")
        return 2

    # --list-speakers is instant and needs no model.
    if args.list_speakers:
        list_speakers()
        return 0

    # Validate the speaker early (before the expensive model load).
    validate_speaker(args.speaker)

    model = load_model(args)

    try:
        if args.text:
            speak(model, args.text, args)
            return 0

        if args.interactive:
            interactive_loop(model, args)
            return 0
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
