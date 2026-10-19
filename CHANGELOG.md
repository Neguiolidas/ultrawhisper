# Changelog

## v0.6.0 (2026-10-19)

- voiceid: adaptive threshold by audio duration + minimum top-2 margin for
  short audio (5-voice benchmark: 79% → 83%, zero misidentifications)
- voiceid: `--delete NAME` and `--rename OLD:NEW` for voiceprint management
- transcribe: `--output FILE` and `--format {txt,json,srt}` (srt from diarization)
- transcribe: `WHISPERO_FALLBACK_MODELS` env var; logs which model responded
- ears: `--history FILE` (JSONL) — appends each turn, last 5 become context
- 9 new tests for transcribe.py pure functions (26/26 passing)
- Docs: faster-whisper large-v3-turbo recipe, BENCHMARK-5-VOZES.md

## v0.5.0 (2026-10-19)

- Agent ears: voice enrollment and rich turn output (whispero-turn/1 schema)

## v0.4.0 (2026-10-19)

- Hands-free dictation daemon (whispero-dictate)

## v0.3.0 (2026-10-18)

- Pitch check for diarization (splits merged same-gender voices)

## v0.2.0 (2026-10-18)

- Speaker diarization (--diarize, llm and pyannote engines)

## v0.1.0 (2026-10-18)

- Initial release: transcription via Gemini
