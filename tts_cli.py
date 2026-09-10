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

    # List the supported speakers and languages, then exit.
    python tts_cli.py --list-speakers
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np
import soundfile as sf
import sounddevice as sd
import torch

from qwen_tts import Qwen3TTSModel

# Default model (CustomVoice 1.7B). Weights auto-download from Hugging Face on
# first run (requires internet; optionally set HF_TOKEN for gated access).
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Human-readable descriptions for the 9 premium timbres (from the model card).
# Used only to make ``--list-speakers`` output friendlier; the authoritative
# list of names comes from ``model.get_supported_speakers()``.
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
        "--interactive",
        "-i",
        action="store_true",
        help="Enter an interactive type-and-speak loop (quit / exit / q to stop).",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print the supported speakers and languages, then exit.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to load the model on (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Model dtype.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use FlashAttention-2 (requires flash-attn installed and a compatible GPU).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new codec tokens to generate.",
    )
    return parser


def load_model(args: argparse.Namespace) -> Qwen3TTSModel:
    """Load the Qwen3-TTS CustomVoice model once (reused across the session)."""
    print(f"Loading model {DEFAULT_MODEL} on {args.device} ({args.dtype})...")
    print("(First run downloads the weights from Hugging Face — this may take a while.)")

    attn_implementation = "flash_attention_2" if args.flash_attn else None

    model = Qwen3TTSModel.from_pretrained(
        DEFAULT_MODEL,
        device_map=args.device,
        dtype=DTYPE_MAP[args.dtype],
        attn_implementation=attn_implementation,
    )
    print("Model loaded.")
    return model


def speak(model: Qwen3TTSModel, text: str, args: argparse.Namespace) -> None:
    """Synthesize ``text`` and play it to the speakers (optionally saving a WAV)."""
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
    )

    # Ensure a mono, 1-D float32 waveform for playback.
    audio = np.asarray(wavs[0], dtype=np.float32).flatten()

    if args.save:
        sf.write(args.save, audio, sr)
        print(f"Saved audio to {args.save}")

    sd.play(audio, samplerate=sr)
    sd.wait()  # Block until playback finishes so output doesn't overlap.


def list_speakers(model: Qwen3TTSModel) -> None:
    """Print the supported speakers (with descriptions) and languages."""
    speakers = model.get_supported_speakers()
    if isinstance(speakers, dict):
        speaker_list: List[str] = list(speakers.keys())
    else:
        speaker_list = list(speakers)

    print("Supported speakers:")
    for spk in speaker_list:
        desc = SPEAKER_DESCRIPTIONS.get(spk)
        print(f"  - {spk}" + (f": {desc}" if desc else ""))

    languages = model.get_supported_languages()
    if isinstance(languages, (list, tuple)):
        lang_str = ", ".join(str(x) for x in languages)
    else:
        lang_str = str(languages)
    print("\nSupported languages:")
    print(f"  {lang_str}")


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

    model = load_model(args)

    try:
        if args.list_speakers:
            list_speakers(model)
            return 0

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
