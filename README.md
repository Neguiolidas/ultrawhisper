# Whispero

Audio transcription and deep audio analysis with any OpenAI-compatible
provider. Defaults to Gemini's native audio input (transcribes *and*
reasons in one pass); also works with OpenAI, Groq, Together, or any
Whisper-style `/v1/audio/transcriptions` endpoint.

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

# Precise diarization engine (needs pyannote.audio + HF_TOKEN)
pip install "whispero[diarize]"
whispero meeting.mp3 --diarize --diarize-engine pyannote

# Configurable fallback models (comma-separated, tried in order)
whispero meeting.mp3 --fallback-models gemini-2.0-flash
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

`whispero-analyze --diarize` adds a `diarization` array to the JSON output
with `speaker`, `start`, `end` and `text` per segment.

## Features

- Two providers: `gemini` (default, native audio + YouTube URLs) and `openai`
  (any OpenAI-compatible `/v1/audio/transcriptions` endpoint)
- Speaker diarization ([FALANTE 1], [FALANTE 2] with voice characteristics)
- Retry with exponential backoff on rate limits and server errors
- `--fallback-models`: configurable comma-separated fallback list
- `--compress` flag: converts to opus 32k mono 16kHz via ffmpeg before upload
- JSON validation on analysis output
- Zero pip dependencies, only stdlib + ffmpeg

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

## License

Apache 2.0 — see LICENSE.
