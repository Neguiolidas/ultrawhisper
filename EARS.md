# Whispero Ears

Rich audio understanding for agents: who is speaking, what was said, and
how it was said. Plugs into any agent that normally only gets text.

## Install

```bash
pip install whispero                  # ears (transcribe + tone analysis)
pip install "whispero[voiceid]"       # + speaker identification (ONNX, local)
```

## Voice enrollment (once per person)

```bash
# ~10s of clean speech, one speaker
whispero-enroll my-voice.wav --name "Neguiolidas"
# {"name": "Neguiolidas", "duration_s": 10.2, "dim": 256}

whispero-enroll other.wav --name "Partner"
whispero-enroll --list
whispero-enroll turn.wav --verify "Neguiolidas"
# {"name": "Neguiolidas", "match": true, "score": 0.93}
```

Voiceprints live in `~/.whispero/voiceprints.json` (local, never uploaded).
Enrollment rejects samples under 3s or (near) silence.

How it works: a Wespeaker ResNet34 model (ONNX, ~30MB, no torch) extracts a
256-dim embedding from 80-bin log-mel features; verification is cosine
similarity against the enrolled voiceprint, threshold 0.5 (calibrated:
22/22 correct on a 3-speaker test).

## Ears: audio -> rich turn

```bash
whispero-ears turn.wav --pretty
```

```json
{
  "schema": "whispero-turn/1",
  "speaker": "Neguiolidas",
  "speaker_score": 0.93,
  "transcript": "texto fiel do que foi dito",
  "tone": "animado",
  "emotion": "curiosidade",
  "urgency": "baixa",
  "lang": "pt",
  "duration_s": 4.2
}
```

Pipeline per turn:

1. **Speaker ID** (local, ~0.2s): embedding vs enrolled voiceprints.
   Unknown when nobody is enrolled or score < 0.5.
2. **Transcript + tone + emotion + urgency** (one API pass): Gemini native
   audio by default, or `--provider openai` (transcribe, then text analysis).

## The turn contract (whispero-turn/1)

The schema version is in every turn. Agents should:

- Branch on `speaker`: a known name vs `"unknown"` changes the meaning of
  the turn (someone else joined the conversation)
- Treat `speaker_score` in [0.5, 0.65] as uncertain
- Use `tone`/`emotion` as context for the reply, not as a diagnosis
- Treat `transcript` as the content source of truth

Designed for continuous use: each turn is self-contained, so a conversation
loop can feed turns to any agent brain over any transport (the brain is not
part of Whispero).

## Tests

```bash
python3 tests/test_voiceid.py
python3 tests/test_ears.py
# or: python3 -m pytest tests/
```

All fixtures are synthesized in `tests/conftest.py` (harmonic complexes, no
real voices). Real-voice calibration (threshold 0.5, 22/22 on a 3-speaker
sample) is documented here, not committed: personal audio never belongs in
the repo.

## Options

```bash
whispero-ears turn.wav --lang en --provider openai \
    --base-url https://api.groq.com/openai/v1
whispero-ears turn.wav --no-identify   # skip speaker ID
```
