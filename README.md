# Qwen3-TTS Quick Test CLI

A small command-line tool that loads **`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`**
and speaks text straight to your speakers. Built on the official `qwen-tts`
package.

## Requirements

- **Python 3.10+** (3.12 recommended)
- **NVIDIA GPU with CUDA** (the model is designed for GPU inference; CPU works
  but is very slow)
- **Internet** on first run (weights auto-download from Hugging Face)
- An audio output device (speakers/headphones)

## Setup

```powershell
# 1. Create and activate an isolated environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

> **Optional (faster, lower VRAM):** install FlashAttention-2 and pass
> `--flash-attn`. This requires a compatible GPU and a C++ toolchain:
> ```powershell
> pip install -U flash-attn --no-build-isolation
> ```

> **Optional (skip the first-run download):** pre-download the weights:
> ```powershell
> pip install -U "huggingface_hub[cli]"
> huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
> ```

## Usage

```powershell
# One-shot: speak the given text and exit.
python tts_cli.py "Hello, this is a quick test of Qwen TTS."

# Pick a speaker, language, and style; also save a WAV.
python tts_cli.py "Test" --speaker Ryan --language English --instruct "Very happy." --save out.wav

# Synthesize and save only (no playback).
python tts_cli.py "Test" --save out.wav --no-play

# Interactive: type-and-speak loop (type 'quit' to stop).
python tts_cli.py --interactive

# List the supported speakers and languages (instant — no model load).
python tts_cli.py --list-speakers
```

## Web UI

A browser front-end with the same model and options, plus a clip history:

```powershell
python tts_server.py            # then open http://127.0.0.1:8000
python tts_server.py --host 0.0.0.0 --port 8080 --dtype float16 --flash-attn
```

- Pick a voice, language and style (with one-click style presets); <kbd>Ctrl</kbd>+<kbd>Enter</kbd> generates.
- Clips are saved to `outputs/` (WAV + JSON metadata) and listed in the History panel
  with play, download, delete and "reuse settings".
- **Faster long text:** text is split at sentence boundaries and the chunks are
  generated in parallel batches (Advanced → chunk size / batch size). In testing
  this was ~2.7× faster than one long generation, with a live progress bar.
- Sampling controls (temperature, top-p, repetition penalty) and a seed for reproducible takes.
- The model loads and warms up in the background at startup; the header shows its status.

### Voice cloning (`/clone`)

The **Voice cloning** tab clones any voice from a short sample using the
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` model. It loads the first time you open the tab
(the first time ever, it downloads ~4 GB). Use `--preload-clone` to load it at
startup, or `--clone-model Qwen/Qwen3-TTS-12Hz-0.6B-Base` to save VRAM.

1. Click **+ New voice**, then upload an audio file (WAV/MP3/FLAC/OGG, drag and drop works)
   or **Record** from your microphone. 5–20 s of clean speech from one speaker works best.
   You can also clone the voice of any clip in History with its copy icon.
2. Name the voice and enter **exactly what is said** in the sample. When you add audio,
   it's transcribed automatically with Whisper (`openai/whisper-large-v3-turbo`, downloaded
   on first use, ~1.6 GB). Use **✨ Transcribe with Whisper** to re-run it, and check the
   result against the audio. Whisper uses the GPU when there's spare VRAM, otherwise the CPU
   (about 10 s for a 20 s clip). Options: `--whisper-model openai/whisper-small`,
   `--whisper-device cpu|cuda:0`.
   An exact transcript gives a *Full*
   clone (the model conditions on the audio + transcript). If you don't have a transcript,
   tick the no-transcript option for a *Quick* clone from the voice's timbre only. It's faster
   but sounds less like the original.
3. Select the voice, type your text, and generate. The same speed and sampling settings
   apply. In Full mode each chunk carries the reference clip, so very small chunk sizes slow
   it down. Keep chunk size around 200+ characters.

Voices are saved in `voices/` (reference WAV + metadata) and can be renamed, re-transcribed,
or deleted from their cards. Only clone voices you have permission to use.

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `text` (positional) | — | Text to synthesize. Omit with `--interactive` for a loop. |
| `--speaker` | `Vivian` | Speaker name (see `--list-speakers`). |
| `--language` | `Auto` | Target language (e.g. `English`, `Chinese`) or `Auto`. |
| `--instruct` | — | Optional natural-language tone/style instruction. |
| `--save PATH` | — | Also write the audio to a WAV file. |
| `--no-play` | off | Synthesize (and optionally save) but don't play to the speakers. |
| `--interactive`, `-i` | off | Enter a type-and-speak loop. |
| `--list-speakers` | off | Print supported speakers + languages, then exit (instant, no model load). |
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Hugging Face model id to load. |
| `--device` | `auto` | Torch device (`cuda:0`, `cpu`, ...). `auto` picks CUDA if available, else CPU. |
| `--dtype` | `bfloat16` | Model dtype (`bfloat16`, `float16`, `float32`). |
| `--flash-attn` | off | Use FlashAttention-2 (if installed). |
| `--max-new-tokens` | `2048` | Max new codec tokens to generate. |

## Speakers

The 1.7B CustomVoice model ships 9 premium timbres (best in their native
language, but each can speak any supported language):

| Speaker | Description | Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

## Troubleshooting

- **`No module named 'qwen_tts'`** — you're not in the venv, or `pip install -r requirements.txt` didn't finish.
- **CUDA out of memory** — try `--dtype float16`, close other GPU apps, or install `flash-attn` and use `--flash-attn`.
- **No sound** — confirm an output device is connected; `sounddevice` bundles PortAudio wheels for Windows, so no extra install is normally needed.
- **Slow first run** — expected; the ~2B-param weights download once and are cached.
- **Gated access** — if the model requires a token, set `HF_TOKEN` (or run `huggingface-cli login`) before running.
- **Startup warnings** — `flash-attn is not installed` and `SoX could not be found` are harmless one-time messages from the `qwen-tts`/`sounddevice` imports. They do not affect playback or synthesis.
