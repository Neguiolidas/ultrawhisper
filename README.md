# UltraWhisper

Audio transcription and deep audio analysis with any OpenAI-compatible
provider. Defaults to Gemini's native audio input (transcribes *and*
reasons in one pass); also works with OpenAI, Groq, Together, or any
Whisper-style `/v1/audio/transcriptions` endpoint.

Unlike pure ASR models, the Gemini path processes audio with reasoning: it
transcribes, cleans up, structures output, detects speakers, emotions, and
action items in a single pass.

## Setup

```bash
# Gemini (default) - free key at https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-key-here"

# ...or any provider key, picked up automatically
export ULTRAWHISPER_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

## Usage

```bash
# Simple transcription, Gemini (Portuguese default)
python3 scripts/transcribe.py meeting.mp3

# English transcription
python3 scripts/transcribe.py call.mp3 --lang en

# YouTube URL directly, no download needed (Gemini only)
python3 scripts/transcribe.py "https://www.youtube.com/watch?v=..." --lang en
python3 scripts/audio_analyze.py "https://www.youtube.com/watch?v=..."

# OpenAI-compatible provider (Whisper-style API)
python3 scripts/transcribe.py meeting.mp3 --provider openai
python3 scripts/transcribe.py meeting.mp3 --provider openai \
    --base-url https://api.groq.com/openai/v1 \
    --model whisper-large-v3

# Deep analysis: transcript, speakers, tone, emotions, urgency,
# intent, action items, key moments, irony/sarcasm detection
python3 scripts/audio_analyze.py meeting.mp3
python3 scripts/audio_analyze.py meeting.mp3 --provider openai \
    --base-url https://api.groq.com/openai/v1

# Faster upload for large files (compresses to opus 32k first, local files only)
python3 scripts/transcribe.py long.mp3 --compress

# Configurable fallback models (comma-separated, tried in order)
python3 scripts/transcribe.py meeting.mp3 --fallback-models gemini-2.0-flash
```

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
