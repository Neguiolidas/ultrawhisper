"""Phase 2 tests: M2 timbre + M5 music/mix."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hear
from test_hear import write_wav, SR


def synth_chord_mix(path, ducked: bool):
    """1 s chord-only + 2 s speech+chord + 1 s chord-only. When ducked,
    the chord drops to 0.25 gain under speech; otherwise it stays."""
    t = np.arange(int(4.0 * SR)) / SR
    chord = (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t)
             + np.sin(2 * np.pi * 330 * t)) / 3.0
    speech = np.zeros_like(t)
    seg = (t >= 1.0) & (t < 3.0)
    phase = 2 * np.pi * np.cumsum(np.full(len(t), 120.0)) / SR
    voice = sum((1.0 / h) * np.sin(h * phase + h) for h in range(1, 8))
    voice *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    speech[seg] = voice[seg]
    gain = np.ones_like(t)
    if ducked:
        gain[seg] = 0.25
    sig = speech + chord * gain
    return write_wav(path, sig)


def test_timbre_centroid_ordering(tmp_path):
    t = np.arange(int(2.0 * SR)) / SR
    dark = sum((1.0 / h ** 2) * np.sin(2 * np.pi * 120 * h * t)
               for h in range(1, 8))
    bright = sum((1.0 / np.sqrt(h)) * np.sin(2 * np.pi * 120 * h * t)
                 for h in range(1, 8))
    d = hear.analyze(write_wav(str(tmp_path / "d.wav"), dark),
                     modules=("timbre",))["measurements"]["timbre"]
    b = hear.analyze(write_wav(str(tmp_path / "b.wav"), bright),
                     modules=("timbre",))["measurements"]["timbre"]
    assert d["spectral_centroid_mean_hz"] < b["spectral_centroid_mean_hz"]


def test_timbre_hnr_voiced_beats_noise(tmp_path):
    t = np.arange(int(2.0 * SR)) / SR
    tone = sum((1.0 / h) * np.sin(2 * np.pi * 150 * h * t) for h in range(1, 8))
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(t))
    h_tone = hear.analyze(write_wav(str(tmp_path / "t.wav"), tone),
                          modules=("timbre",))["measurements"]["timbre"]
    h_noise = hear.analyze(write_wav(str(tmp_path / "n.wav"), noise),
                           modules=("timbre",))["measurements"]["timbre"]
    assert h_tone["hnr_median_db"] is not None
    assert h_noise["hnr_median_db"] is None or \
        h_tone["hnr_median_db"] > h_noise["hnr_median_db"]


def test_timbre_fields_sane(tmp_path):
    t = np.arange(int(2.0 * SR)) / SR
    tone = sum((1.0 / h) * np.sin(2 * np.pi * 150 * h * t) for h in range(1, 8))
    tim = hear.analyze(write_wav(str(tmp_path / "t.wav"), tone),
                       modules=("timbre",))["measurements"]["timbre"]
    for k in ("formant_f1_median_hz", "formant_f2_median_hz"):
        assert tim[k] is None or 100 <= tim[k] <= 5000
    for k in ("jitter_pct", "shimmer_pct"):
        assert tim[k] is None or tim[k] >= 0
    # No torch in this env: the honest null path, not a fake number.
    assert tim["embedding_available"] is False
    assert tim["embedding"] is None


def test_music_ducked_vs_unducked(tmp_path):
    ducked = hear.analyze(
        synth_chord_mix(str(tmp_path / "duck.wav"), ducked=True),
        modules=("music",))["measurements"]["music"]
    loud = hear.analyze(
        synth_chord_mix(str(tmp_path / "loud.wav"), ducked=False),
        modules=("music",))["measurements"]["music"]
    assert ducked["ducking_ratio"] is not None
    assert loud["ducking_ratio"] is not None
    assert ducked["ducking_ratio"] < loud["ducking_ratio"]
    assert ducked["music_competing"] is False
    assert loud["music_competing"] is True


def test_music_speech_only_not_competing(tmp_path):
    t = np.arange(int(2.0 * SR)) / SR
    phase = 2 * np.pi * np.cumsum(np.full(len(t), 120.0)) / SR
    voice = sum((1.0 / h) * np.sin(h * phase + h) for h in range(1, 8))
    voice *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    m = hear.analyze(write_wav(str(tmp_path / "v.wav"), voice),
                     modules=("music",))["measurements"]["music"]
    assert m["music_competing"] is False
