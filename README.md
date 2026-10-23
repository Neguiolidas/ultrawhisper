# Whispero

Audio transcription and deep audio analysis with any OpenAI-compatible
provider. Defaults to Gemini's native audio input (transcribes *and*
reasons in one pass); also works with OpenAI, Groq, Together, or any
Whisper-style `/v1/audio/transcriptions` endpoint. A fully offline mode
(`--provider local`) runs everything on your machine with no API key.

Unlike pure ASR models, the Gemini path processes audio with reasoning: it
transcribes, cleans up, structures output, detects speakers, emotions, and
action items in a single pass.

## Install

```bash
pip install whispero
```

## Setup

```bash
# Gemini (default) - free key at https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-key-here"

# ...or any provider key, picked up automatically
export WHISPERO_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# ...or fully offline, no key at all (see Offline mode below)
```

## Usage

```bash
# Simple transcription, Gemini (Portuguese default)
whispero meeting.mp3

# English transcription
whispero call.mp3 --lang en

# YouTube URL directly, no download needed (Gemini only)
whispero "https://www.youtube.com/watch?v=..." --lang en
whispero-analyze "https://www.youtube.com/watch?v=..."

# OpenAI-compatible provider (Whisper-style API)
whispero meeting.mp3 --provider openai
whispero meeting.mp3 --provider openai \
    --base-url https://api.groq.com/openai/v1 \
    --model whisper-large-v3

# Deep analysis: transcript, speakers, tone, emotions, urgency,
# intent, action items, key moments, irony/sarcasm detection
whispero-analyze meeting.mp3
whispero-analyze meeting.mp3 --provider openai \
    --base-url https://api.groq.com/openai/v1

# Faster upload for large files (compresses to opus 32k first, local files only)
whispero long.mp3 --compress

# Speaker diarization: who said what, with timestamps
whispero meeting.mp3 --diarize
whispero-analyze meeting.mp3 --diarize

# Fully offline: no API key, no upload (pip install "whispero[local]")
whispero meeting.mp3 --provider local --diarize

# Precise diarization engine (needs pyannote.audio + HF_TOKEN)
pip install "whispero[diarize]"
whispero meeting.mp3 --diarize --diarize-engine pyannote

# Configurable fallback models (comma-separated, tried in order)
whispero meeting.mp3 --fallback-models gemini-2.0-flash
# Or via env var (CLI flag takes precedence)
export WHISPERO_FALLBACK_MODELS="gemini-2.0-flash,gemini-1.5-flash"
```

## Speaker diarization

`--diarize` identifies who is speaking in multi-speaker audio:

```bash
whispero debate.mp3 --diarize
# [FALANTE 1] [00:00] Boa noite a todos, vamos começar...
# [FALANTE 2] [00:12] Obrigado pelo convite, é um prazer...
# [FALANTE 1] [00:28] Então, o primeiro ponto da pauta...
```

Two engines:

- **llm** (default): prompt-based diarization using the provider's native
  audio understanding. Works out of the box, no extra dependencies.
  On `--provider openai` it falls back to a text-based heuristic
  (clearly marked), since Whisper doesn't return voice information —
  use `--provider gemini` for voice-based diarization.
- **pyannote**: precise timestamped segments via
  [pyannote.audio](https://github.com/pyannote/pyannote-audio).
  Needs `pip install "whispero[diarize]"` and `HF_TOKEN` (accept the
  model terms at
  https://huggingface.co/pyannote/speaker-diarization-3.1).
  Each speaker chunk is transcribed separately and reassembled with
  exact speaker labels and timestamps.
- **local** (default with `--provider local`): voice embeddings computed
  on your machine, no API and no Hugging Face token. See
  [Offline mode](#offline-mode---provider-local).

`whispero-analyze --diarize` adds a `diarization` array to the JSON output
with `speaker`, `start`, `end` and `text` per segment.

## Offline mode (`--provider local`)

Runs everything on your machine: no API key, no upload, no daily quota.

```bash
pip install "whispero[local]"
whispero meeting.mp3 --provider local                    # transcript
whispero meeting.mp3 --provider local --diarize          # + speakers
whispero meeting.mp3 --provider local --diarize --format srt --output meeting.srt
whispero meeting.mp3 --provider local --diarize --num-speakers 4   # known count
whispero meeting.mp3 --provider local --model large-v3   # most accurate
```

- **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  on CPU (int8, batched). Default model `large-v3-turbo` (about 2x faster);
  `--model large-v3` is a little more accurate. Models download once
  (1.6 GB / 3 GB) to `~/.cache/huggingface`.
- **Diarization:** Silero VAD → 1 s voice windows → WeSpeaker ResNet34-LM
  embeddings (ONNX, the same public model `whispero-enroll` uses) →
  spectral clustering that estimates the number of speakers → one speaker
  per sentence. Overlapping speech gets a single speaker.
- **CPU:** `--threads` defaults to the number of physical cores (max 8).
  More threads than physical cores was 2-10x *slower* in our tests.
  `--batch-size 0` turns batching off.

Measured on an 8-core / 16-thread Xeon with no GPU, on synthetic PT-BR
conversations with exact answer keys (lines with the correct speaker):

| | 3 voices (134 s) | 5 voices, noisy bar + video call (172 s) | time, 3 voices |
|---|---|---|---|
| local, large-v3-turbo | 26/28, WER 3.6% | 39/43 | 24 s |
| local, large-v3 | 27/28, WER 2.3% | 41/43 | 42 s |
| gemini-3.5-flash `--diarize` | 27/28, WER 4.3% | 34/43 (6 speakers) | 70 s |

The remaining misses are short interjections said over someone else
("Sério?", "Aham."). In heavy noise (5 dB SNR) the speaker count can come
out too low. Pass `--num-speakers` if you know it (5-voice test at 5 dB:
28/43 automatic, 40/43 with `--num-speakers 5`).

## Hands-free dictation

Hold a hotkey, speak, release: the text lands in whatever app has focus
(code editor, terminal, browser, ...). SuperWhisper-style, powered by the
Whispero engine instead of a local model.

```bash
pip install "whispero[dictate]"

# Hold F8 (default), speak, release -> text is pasted at the cursor
whispero-dictate

# Toggle mode: press once to start, press again to stop and paste
whispero-dictate --mode toggle --hotkey "<ctrl>+<alt>+space"

# Test the paste pipeline without recording
whispero-dictate --test-inject "hello from whispero"
```

Config lives in `~/.whispero/dictate.json` (hotkey, mode, lang, cleanup,
post-record hook). Esc cancels a recording. See `DICTATE.md` for the full
guide.

### Cleanup LLM providers

After transcription, an LLM pass removes filler words and fixes punctuation.
It defaults to your transcription provider, but you can point it at any of
the three:

```bash
# Claude (Anthropic API) for the cleanup pass
export ANTHROPIC_API_KEY=<redacted>
whispero-dictate --cleanup-provider anthropic

# Any Anthropic-compatible endpoint (proxy, gateway, LiteLLM...)
whispero-dictate --cleanup-provider anthropic \
    --cleanup-base-url https://my-proxy.example/v1 \
    --cleanup-model my-claude-model

# Or keep it on OpenAI-compatible
whispero-dictate --cleanup-provider openai \
    --cleanup-base-url https://api.groq.com/openai/v1
```

Note: the Anthropic Messages API has no audio input, so it only applies to
this text-only cleanup pass, never to transcription itself.

## Ears for agents

Give any agent rich hearing: speaker identity, transcript, tone, emotion,
urgency. Not just text.

```bash
pip install "whispero[voiceid]"

# Enroll a voice once (~10s of clean speech)
whispero-enroll my-voice.wav --name "Neguiolidas"

# Audio -> rich turn JSON (schema whispero-turn/1)
whispero-ears turn.wav --pretty
# {"schema": "whispero-turn/1", "speaker": "Neguiolidas",
#  "speaker_score": 0.93, "transcript": "...", "tone": "animado",
#  "emotion": "curiosidade", "urgency": "baixa", ...}
```

Speaker verification runs locally (ONNX embedding, no API call, no upload).
See `EARS.md` for the turn contract.

## Features

- Three transcription providers: `gemini` (default, native audio + YouTube URLs),
  `openai` (any OpenAI-compatible `/v1/audio/transcriptions` endpoint) and
  `local` (fully offline, faster-whisper on CPU, no API key)
- Third provider for the text layer: `anthropic` (Messages API) powers the
  LLM cleanup pass in `whispero-dictate` (`--cleanup-provider anthropic`)
- Speaker diarization ([FALANTE 1], [FALANTE 2] with voice characteristics)
- Retry with exponential backoff on rate limits and server errors
- `--fallback-models`: configurable comma-separated fallback list
  (or `WHISPERO_FALLBACK_MODELS` env); logs which model responded
- `--output FILE --format {txt,json,srt}`: save to file; srt needs --diarize
- `--compress` flag: converts to opus 32k mono 16kHz via ffmpeg before upload
- JSON validation on analysis output
- Base has zero pip dependencies (stdlib + ffmpeg); each extra installs
  only what its feature needs

## Benchmark

50 seconds of real audio, Gemini (Portuguese):

| Script | Time | Output |
|---|---|---|
| transcribe.py | ~4s | faithful transcript |
| transcribe.py --compress | ~4s | same, half the upload size |
| audio_analyze.py | ~14s | full JSON analysis |

GTA 6 trailer via YouTube URL (no download):

| Script | Time | Output |
|---|---|---|
| transcribe.py | ~12s | dialogue + song lyrics + on-screen text |
| audio_analyze.py | ~23s | 4 speakers, tone, 9 key moments, intent |

## Hear: measured signal analysis (v0.7.0)

```bash
pip install "whispero[hear]"
whispero hear voice.wav --baseline original.wav
whispero-ears voice.wav --hear --hear-baseline original.wav
```

Hear measures the waveform instead of guessing from words: prosody (F0,
range, rate, pauses), fricative spectra, timbre, music ducking, cut
placement and prosodic state, all reported as deviations from a
reference speaker baseline (warn at 30%, fail at 50%, provisional
until calibrated on real voice). No hard emotion labels, no faked
numbers: optional torch models report as unavailable when absent.

## License

Apache 2.0 — see LICENSE.
