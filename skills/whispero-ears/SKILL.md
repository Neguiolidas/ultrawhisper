---
name: whispero-ears
description: Give an agent rich hearing from audio. Use when audio arrives and the task needs more than transcription: who is speaking (enrolled voiceprint or unknown), the transcript, tone, emotion and urgency, as a versioned JSON turn (whispero-turn/1). Commands: whispero-enroll (one-time voice enrollment) and whispero-ears (audio -> rich turn).
---

# Whispero Ears

Turn audio into a structured turn object any agent can reason about:

```bash
pip install "whispero[voiceid]"          # enroll + local speaker ID
whispero-enroll voice-sample.wav --name "Alice"   # once per person (~10s)
whispero-ears turn.wav --pretty
```

Output (schema `whispero-turn/1`):

```json
{
  "schema": "whispero-turn/1",
  "speaker": "Alice",
  "speaker_score": 0.93,
  "transcript": "faithful transcript",
  "tone": "tense",
  "emotion": "frustration",
  "urgency": "baixa",
  "lang": "pt",
  "duration_s": 4.2
}
```

## Rules for consuming a turn

- Check `schema` before parsing. `whispero-turn/1` is the current version.
- Branch on `speaker` first: enrolled name vs `"unknown"`. In a 1:1 context,
  `"unknown"` means someone else is present.
- `speaker_score` is cosine similarity [-1, 1]. < 0.5 = not them.
  0.5-0.65 = uncertain, say so. > 0.8 = confident.
- `tone`/`emotion` calibrate the reply; they are signal, not diagnosis.
- `transcript` is the content source of truth.
- `urgency` is `baixa` | `media` | `alta`.
- Turns are self-contained: feed them to any conversation loop. The loop
  (brain, transport, TTS) is not Whispero's business.

## Enrollment

- ~10s of clean speech, one speaker, no music. Rejects < 3s or silence.
- Voiceprints are local (`~/.whispero/voiceprints.json`), never uploaded.
- Speaker ID runs fully offline (ONNX embedding); only the transcript +
  tone pass uses the Whispero API engine.

## Limits

- Very similar voices can confuse verification; 0.5-0.65 is the gray zone.
- Under 1s of audio cannot identify anyone.
- Overlapping speech produces one mixed embedding: expect low scores.
