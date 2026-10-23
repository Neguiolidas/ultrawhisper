"""Phase 4 tests: hear attached to ears turns, contract preserved."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ears
from test_hear import synth_harmonic


def test_attach_hearing_adds_key_only(tmp_path):
    audio = synth_harmonic(str(tmp_path / "t.wav"))
    turn = {"schema": "whispero-turn/1", "speaker": "unknown",
            "speaker_score": 0.0, "transcript": "x", "tone": "neutro",
            "emotion": "neutro", "urgency": "baixa", "lang": "pt",
            "duration_s": 4.0}
    out = ears.attach_hearing(dict(turn), audio)
    for k in turn:
        assert out[k] == turn[k], f"existing key changed: {k}"
    assert out["hearing"]["schema"] == "whispero-hear/1"
    assert "prosody" in out["hearing"]["measurements"]


def test_attach_hearing_with_baseline(tmp_path):
    base = synth_harmonic(str(tmp_path / "b.wav"), f0=120.0)
    target = synth_harmonic(str(tmp_path / "t.wav"), f0=150.0)
    out = ears.attach_hearing({"schema": "whispero-turn/1"}, target, base)
    assert out["hearing"]["baseline"] == base
    assert isinstance(out["hearing"]["flags"], list)


def test_parser_accepts_hear_flags():
    args = ears.build_parser().parse_args(
        ["a.wav", "--hear", "--hear-baseline", "b.wav"])
    assert args.hear is True
    assert args.hear_baseline == "b.wav"
    args2 = ears.build_parser().parse_args(["a.wav"])
    assert args2.hear is False
