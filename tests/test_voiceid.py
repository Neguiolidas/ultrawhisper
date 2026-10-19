"""Universal tests for voiceid.py (speaker enrollment / verification).

Run: python3 -m pytest tests/  (or: python3 tests/test_voiceid.py)
Requires: pip install "whispero[voiceid]"
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import synth_voice, synth_silence, tmp_path

import voiceid


def _isolate_store(monkeypatch=None):
    """Point the voiceprint store at a temp file."""
    path = tmp_path("voiceprints.json")
    voiceid.VOICEPRINTS_PATH = path
    return path


def test_fbank_shape():
    wav = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
    fb = voiceid.fbank(wav)
    assert fb.shape[1] == 80, f"expected 80 mel bins, got {fb.shape}"
    assert fb.dtype == np.float32
    assert fb.shape[0] > 50  # ~100 frames per second


def test_embed_normalized_and_dim():
    p = synth_voice(tmp_path("a.wav"), f0=120.0)
    emb = voiceid.embed_file(p)
    assert emb.shape == (256,), f"expected 256-dim, got {emb.shape}"
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_enroll_verify_roundtrip():
    _isolate_store()
    p = synth_voice(tmp_path("enroll.wav"), f0=120.0)
    info = voiceid.enroll(p, "TestVoice")
    assert info["name"] == "TestVoice"
    assert info["dim"] == 256
    match, score = voiceid.verify(p, "TestVoice")
    assert match is True
    assert score > 0.9, f"same-audio score too low: {score}"


def test_verify_separates_different_voices():
    _isolate_store()
    pa = synth_voice(tmp_path("a.wav"), f0=120.0, tilt=0.3)
    pb = synth_voice(tmp_path("b.wav"), f0=220.0, tilt=0.0)
    voiceid.enroll(pa, "VoiceA")
    match_a, score_a = voiceid.verify(pa, "VoiceA")
    match_b, score_b = voiceid.verify(pb, "VoiceA")
    assert match_a and score_a > score_b, (
        f"same voice should score highest: {score_a} vs {score_b}")
    # the machinery separates; absolute threshold is calibrated on real
    # voices (see EARS.md), synthetics only assert relative ordering


def test_identify_unknown_when_empty():
    _isolate_store()
    p = synth_voice(tmp_path("x.wav"), f0=150.0)
    name, score = voiceid.identify(p)
    assert name == "unknown"
    assert score == 0.0


def test_identify_picks_best_match():
    _isolate_store()
    pa = synth_voice(tmp_path("a.wav"), f0=120.0, tilt=0.3)
    pb = synth_voice(tmp_path("b.wav"), f0=220.0, tilt=0.0)
    voiceid.enroll(pa, "VoiceA")
    voiceid.enroll(pb, "VoiceB")
    name, _ = voiceid.identify(pa)
    assert name == "VoiceA", f"expected VoiceA, got {name}"


def test_enroll_rejects_silence():
    _isolate_store()
    p = synth_silence(tmp_path("sil.wav"))
    try:
        voiceid.enroll(p, "Nobody")
    except ValueError as e:
        assert "silence" in str(e)
    else:
        raise AssertionError("silence should be rejected")


def test_enroll_rejects_short_sample():
    _isolate_store()
    p = synth_voice(tmp_path("short.wav"), f0=120.0, seconds=1.0)
    try:
        voiceid.enroll(p, "Shorty")
    except ValueError as e:
        assert "too short" in str(e)
    else:
        raise AssertionError("short sample should be rejected")


def test_verify_unknown_name_raises():
    _isolate_store()
    p = synth_voice(tmp_path("x.wav"), f0=150.0)
    try:
        voiceid.verify(p, "Ghost")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown name should raise KeyError")


def test_store_roundtrip_json():
    path = _isolate_store()
    p = synth_voice(tmp_path("a.wav"), f0=120.0)
    voiceid.enroll(p, "Json")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    assert "Json" in d
    assert len(d["Json"]["embedding"]) == 256


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
