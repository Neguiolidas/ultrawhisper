"""Tests for whispero hear (Audição de verdade, v0.7.0).

All fixtures are synthetic with known ground truth: no real voices in
the repo. Real-voice calibration on the user's voice stays pending
until he provides the original/clone files.
"""

import json
import os
import struct
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hear

SR = 22050


def write_wav(path, y, sr=SR):
    y = np.asarray(y, dtype=float)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak * 0.5
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % len(y),
                                  *(y * 32767).astype(np.int16)))
    return path


def synth_harmonic(path, seconds=4.0, f0=120.0, f0_sweep=None, am_hz=0.0):
    """Harmonic complex. f0_sweep=(f_start, f_end) for a linear sweep,
    am_hz>0 for amplitude modulation (syllable-rate proxy)."""
    t = np.arange(int(seconds * SR)) / SR
    if f0_sweep:
        f0_t = np.linspace(f0_sweep[0], f0_sweep[1], len(t))
    else:
        f0_t = np.full(len(t), f0)
    phase = 2 * np.pi * np.cumsum(f0_t) / SR
    sig = np.zeros_like(t)
    for h in range(1, 8):
        sig += (1.0 / h) * np.sin(h * phase + h)
    if am_hz:
        sig *= 0.6 + 0.4 * np.sin(2 * np.pi * am_hz * t)
    return write_wav(path, sig)


def synth_vibrato(path, seconds=4.0, f0=120.0, vib_st=0.5, vib_hz=5.0,
                  am_hz=0.0):
    """Constant F0 with sinusoidal vibrato of vib_st semitones."""
    t = np.arange(int(seconds * SR)) / SR
    ratio = 2 ** ((vib_st * np.sin(2 * np.pi * vib_hz * t)) / 12.0)
    f0_t = f0 * ratio
    phase = 2 * np.pi * np.cumsum(f0_t) / SR
    sig = np.zeros_like(t)
    for h in range(1, 8):
        sig += (1.0 / h) * np.sin(h * phase + h)
    if am_hz:
        sig *= 0.6 + 0.4 * np.sin(2 * np.pi * am_hz * t)
    return write_wav(path, sig)


def synth_noise_band(path, seconds=2.0, lo=2000, hi=4000, seed=0):
    """Band-limited white noise via FFT mask (fricative simulation)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mask = (freqs >= lo) & (freqs <= hi)
    spec[~mask] = 0
    sig = np.fft.irfft(spec, n)
    return write_wav(path, sig)


def synth_click_track(path, bpm=120.0, seconds=4.0):
    """Click track: short bursts on the beat."""
    sig = np.zeros(int(seconds * SR))
    period = 60.0 / bpm
    t = 0.0
    while t < seconds:
        i = int(t * SR)
        sig[i:i + 300] += np.hanning(300) * 0.9
        t += period
    return write_wav(path, sig)


def test_f0_accuracy_constant_tone(tmp_path):
    p = synth_harmonic(str(tmp_path / "tone.wav"), f0=150.0)
    y, sr = hear.load_audio(p)
    rep = hear.analyze(p, modules=("prosody",))
    f0 = rep["measurements"]["prosody"]["f0_median_hz"]
    assert f0 is not None and abs(f0 - 150.0) / 150.0 < 0.02


def test_pause_detection(tmp_path):
    sig = np.concatenate([
        np.sin(2 * np.pi * 150 * np.arange(SR) / SR),
        np.zeros(SR),
        np.sin(2 * np.pi * 150 * np.arange(SR) / SR)])
    p = write_wav(str(tmp_path / "pauses.wav"), sig)
    rep = hear.analyze(p, modules=("prosody",))
    pauses = rep["measurements"]["prosody"]["pauses"]
    assert len(pauses) == 1
    assert abs(pauses[0]["duration"] - 1.0) < 0.15


def test_syllable_rate_follows_am(tmp_path):
    slow = synth_harmonic(str(tmp_path / "slow.wav"), am_hz=3.0)
    fast = synth_harmonic(str(tmp_path / "fast.wav"), am_hz=4.5)
    r_slow = hear.analyze(slow, modules=("prosody",))["measurements"]["prosody"]
    r_fast = hear.analyze(fast, modules=("prosody",))["measurements"]["prosody"]
    # Onset-envelope peak picking undercounts AM cycles slightly; the
    # ratio between rates is the meaningful assertion, not the absolute.
    assert 120 <= r_slow["syllable_rate_per_min"] <= 240
    assert 200 <= r_fast["syllable_rate_per_min"] <= 400
    assert r_fast["syllable_rate_per_min"] > r_slow["syllable_rate_per_min"] * 1.3


def test_fricative_cog_ordering(tmp_path):
    low = synth_noise_band(str(tmp_path / "sh.wav"), lo=2000, hi=4000)
    high = synth_noise_band(str(tmp_path / "s.wav"), lo=5000, hi=11000)
    a_low = hear.analyze(low, modules=("accent",))["measurements"]["accent"]
    a_high = hear.analyze(high, modules=("accent",))["measurements"]["accent"]
    assert a_low["segment_count"] >= 1 and a_high["segment_count"] >= 1
    assert a_low["mean_cog_hz"] < a_high["mean_cog_hz"]
    assert a_low["mean_cog_hz"] < 4500 < a_high["mean_cog_hz"]


def test_flags_thresholds():
    meas = {"prosody": {"f0_range_st": 1.2, "syllable_rate_per_min": 100,
                        "rms_mean": 0.1},
            "accent": {"mean_cog_hz": 3000, "segments": []},
            "timbre": {"spectral_centroid_mean_hz": 1000},
            "music": {}, "beats": {"cuts": []}}
    base = {"prosody": {"f0_range_st": 1.0, "syllable_rate_per_min": 100,
                        "rms_mean": 0.1},
            "accent": {"mean_cog_hz": 3000},
            "timbre": {"spectral_centroid_mean_hz": 1000}}
    flags = hear.build_flags(meas, base, 4.0)
    by_metric = {f["metric"]: f["level"] for f in flags}
    assert by_metric["f0_range_st"] == "pass"  # 20% deviation
    meas["prosody"]["f0_range_st"] = 1.35
    flags = hear.build_flags(meas, base, 4.0)
    assert {f["metric"]: f["level"] for f in flags}["f0_range_st"] == "warn"
    meas["prosody"]["f0_range_st"] = 1.6
    flags = hear.build_flags(meas, base, 4.0)
    assert {f["metric"]: f["level"] for f in flags}["f0_range_st"] == "fail"
    assert hear.build_flags(meas, {}, 4.0) == []


def test_no_baseline_no_flags(tmp_path):
    p = synth_harmonic(str(tmp_path / "tone.wav"))
    rep = hear.analyze(p)
    assert rep["schema"] == "whispero-hear/1"
    assert rep["flags"] == []
    assert rep["measurements"]["prosody"]["f0_median_hz"] is not None


def test_silence_is_safe(tmp_path):
    p = write_wav(str(tmp_path / "sil.wav"), np.zeros(SR * 2))
    rep = hear.analyze(p)
    pros = rep["measurements"]["prosody"]
    assert pros["voiced_fraction"] == 0.0
    assert pros["f0_median_hz"] is None


def test_acceptance_happy(tmp_path):
    """Fixture (a): clone with exaggerated prosody flags pitch range
    and rate vs baselines. Two controlled pairs: range via F0 sweep
    (vibrato baseline) and rate via pure AM (constant-F0 baseline),
    because the onset envelope also reacts to vibrato."""
    base1 = synth_vibrato(str(tmp_path / "base1.wav"), vib_st=0.5)
    t = np.arange(int(4.0 * SR)) / SR
    f0_t = np.linspace(100, 160, len(t))
    phase = 2 * np.pi * np.cumsum(f0_t) / SR
    sig = sum((1.0 / h) * np.sin(h * phase + h) for h in range(1, 8))
    clone1 = write_wav(str(tmp_path / "clone1.wav"), sig)
    rep1 = hear.analyze(clone1, baseline_path=base1, modules=("prosody",))
    levels1 = {(f["module"], f["metric"]): f["level"] for f in rep1["flags"]}
    assert levels1.get(("prosody", "f0_range_st")) == "fail"
    base2 = synth_harmonic(str(tmp_path / "base2.wav"), am_hz=2.0)
    clone2 = synth_harmonic(str(tmp_path / "clone2.wav"), am_hz=6.0)
    rep2 = hear.analyze(clone2, baseline_path=base2, modules=("prosody",))
    levels2 = {(f["module"], f["metric"]): f["level"] for f in rep2["flags"]}
    assert levels2.get(("prosody", "syllable_rate_per_min")) == "fail"


def test_acceptance_chiado(tmp_path):
    """Fixture (b): clone without the palatalized fricative flags CoG."""
    base = synth_noise_band(str(tmp_path / "sh.wav"), lo=2000, hi=4000)
    clone = synth_noise_band(str(tmp_path / "s.wav"), lo=5000, hi=11000)
    rep = hear.analyze(clone, baseline_path=base, modules=("accent",))
    seg_flags = [f for f in rep["flags"]
                 if f["module"] == "accent" and f["level"] in ("warn", "fail")]
    assert seg_flags, "expected CoG deviation flags for the dry /s/ clone"
    assert all(f["t_start"] is not None for f in seg_flags)


def test_cli_end_to_end(tmp_path, capsys):
    base = synth_vibrato(str(tmp_path / "base.wav"), vib_st=0.5)
    target = synth_harmonic(str(tmp_path / "t.wav"), f0=150.0)
    argv = ["whispero-hear", target, "--baseline", base, "--prosody"]
    old = sys.argv
    sys.argv = argv
    try:
        assert hear.main() == 0
    finally:
        sys.argv = old
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "whispero-hear/1"
    assert out["baseline"] == base
    assert isinstance(out["flags"], list)


def test_missing_file_returns_2(tmp_path, capsys):
    old = sys.argv
    sys.argv = ["whispero-hear", str(tmp_path / "nope.wav")]
    try:
        assert hear.main() == 2
    finally:
        sys.argv = old
