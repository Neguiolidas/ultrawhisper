# Changelog

## v0.7.0 (2026-10-22)

- hear: measured signal analysis (`whispero hear <audio> --baseline ref.wav`,
  also `whispero-hear`; extra `whispero[hear]` with librosa/scipy/numpy)
- hear M1 prosody: F0 via pyin (scipy autocorrelation fallback), range in
  semitones, syllable-rate proxy, pauses, energy, final intonation
- hear M3 accent: spectral analysis of fricative-like segments with
  center of gravity per segment (phoneme-labelled forced alignment is a
  future optional extra, not faked)
- hear M2 timbre: LPC formants, HNR, jitter/shimmer approximations,
  spectral centroid; ECAPA-TDNN embedding only when torch/speechbrain
  are installed, otherwise honestly null
- hear M5 music/mix: ducking measured on music-only frequency bins
  (8192-point FFT, voice-harmonic exclusion from the speech spectrum)
- hear M6 cuts/tempo: beat tracking with reliability gate, cut
  classification (on-beat / in-pause / mid-voiced)
- hear M4 prosodic state: arousal/valence proxies vs baseline, never a
  hard emotion label; no audio emotion classifier is configured
- hear flags: deviation vs baseline, warn at 30% / fail at 50%
  (provisional, pending calibration on real voice)
- ears: `--hear` / `--hear-baseline` attach whispero-hear/1 as
  turn['hearing'] without changing the whispero-turn/1 contract
- transcribe: `--provider local` runs fully offline (faster-whisper on CPU,
  default `large-v3-turbo`, batched, int8) with no API key
- transcribe: `--diarize-engine local` (default with `--provider local`):
  Silero VAD + WeSpeaker ResNet34-LM ONNX embeddings + spectral clustering
  with automatic speaker count; `--num-speakers` fixes the count,
  `--threads` and `--batch-size` tune CPU use
- New optional extra: `pip install "whispero[local]"`
- transcribe: default model is now gemini-3.5-flash (2.5-flash returns 404
  for new keys); daily-quota 429 skips retries and moves to the next fallback
- transcribe: diarization vote uses the most granular run as reference;
  JSON output includes segments with real end times when diarized;
  SRT crash fixed
- ears: fallback models on analyze_turn; errors surface as an `error`
  field instead of a blank neutral turn, are not fed to history,
  and exit with code 5
- 25 new hear tests + 6 new local backend tests, all synthetic fixtures
  with known ground truth (57/57 passing)

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
