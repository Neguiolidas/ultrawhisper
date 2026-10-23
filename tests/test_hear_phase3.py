"""Phase 3 tests: M6 beats/cuts + M4 prosodic state."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hear
from test_hear import synth_click_track, synth_harmonic, write_wav, SR
from test_hear_phase2 import synth_chord_mix


def test_beat_tempo_click_track(tmp_path):
    p = synth_click_track(str(tmp_path / "click.wav"), bpm=120.0)
    y, sr = hear.load_audio(p)
    b = hear.analyze_beats(y, sr, [], [], None, None)
    assert b["tempo_bpm"] is not None
    assert abs(b["tempo_bpm"] - 120.0) < 10.0
    assert len(b["beat_times"]) >= 5
    gaps = np.diff(b["beat_times"])
    assert abs(float(np.median(gaps)) - 0.5) < 0.1


def test_cut_on_beat_vs_off_beat(tmp_path):
    p = synth_click_track(str(tmp_path / "click.wav"), bpm=120.0)
    y, sr = hear.load_audio(p)
    b = hear.analyze_beats(y, sr, [1.0, 1.25], [], None, None)
    by_time = {c["time"]: c for c in b["cuts"]}
    assert by_time[1.0]["on_beat"] is True
    assert by_time[1.0]["classification"] == "on-beat"
    assert by_time[1.25]["on_beat"] is False


def test_cut_in_pause(tmp_path):
    sig = np.concatenate([
        np.sin(2 * np.pi * 150 * np.arange(SR) / SR),
        np.zeros(SR),
        np.sin(2 * np.pi * 150 * np.arange(SR) / SR)])
    p = write_wav(str(tmp_path / "pauses.wav"), sig)
    y, sr = hear.load_audio(p)
    pauses = hear.detect_pauses(y, sr)
    b = hear.analyze_beats(y, sr, [1.5], pauses, None, None)
    assert b["cuts"][0]["in_pause"] is True
    assert b["cuts"][0]["classification"] == "in-pause"


def test_state_proxies_vs_baseline():
    pros = {"rms_mean": 0.2, "f0_range_st": 4.0,
            "syllable_rate_per_min": 240.0, "rms_std": 0.05,
            "final_direction": "rising"}
    base = {"rms_mean": 0.1, "f0_range_st": 2.0,
            "syllable_rate_per_min": 120.0}
    st = hear.analyze_state(pros, base)
    assert st["arousal_components"]["energy_ratio"] == 2.0
    assert st["arousal_components"]["pitch_range_ratio"] == 2.0
    assert st["arousal_components"]["rate_ratio"] == 2.0
    assert st["arousal_proxy"] == 2.0
    # No hard emotion label, ever: suggestion stays null with reason.
    assert st["emotion_suggestion"] is None
    assert st["emotion_reason"]


def test_state_without_baseline():
    st = hear.analyze_state({"rms_mean": 0.2, "rms_std": 0.05,
                              "final_direction": "flat"}, None)
    assert st["arousal_proxy"] is None
    assert st["emotion_suggestion"] is None


def test_acceptance_music_and_cut(tmp_path):
    """Fixture (c): unducked bed competing with speech + a cut
    mid-voiced off-beat raise both flags with timestamps."""
    base = synth_chord_mix(str(tmp_path / "duck.wav"), ducked=True)
    target = synth_chord_mix(str(tmp_path / "loud.wav"), ducked=False)
    rep = hear.analyze(target, baseline_path=base,
                       modules=("music", "beats", "prosody"),
                       cuts=[2.0])
    music_flags = [f for f in rep["flags"] if f["module"] == "music"]
    assert music_flags, "expected a music_competing flag"
    cut_flags = [f for f in rep["flags"] if f["module"] == "beats"]
    assert cut_flags, "expected a cut classification flag"
    assert cut_flags[0]["level"] == "fail"  # musical + mid-voiced
    assert cut_flags[0]["t_start"] == 2.0
