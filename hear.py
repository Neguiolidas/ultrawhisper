#!/usr/bin/env python3
"""Whispero Hear - measured signal analysis (Audição de verdade).

Measures what transcription cannot: prosody, timbre, accent, musical
mix, cuts and prosodic state, all from the waveform, compared against
a reference speaker baseline. Never emits absolute labels ("manic",
"sad", "carioca"): every flag is a measured deviation from the
baseline, and the human decides what it means.

Usage:
    whispero hear <audio> [--baseline ref.wav]
                          [--prosody --timbre --accent --state --music
                           --beats] [--cuts t1,t2,...] [--output FILE]

Output (JSON, schema whispero-hear/1):
    {
      "schema": "whispero-hear/1",
      "audio": "...", "duration_s": 4.2,
      "modules": ["prosody", "accent", ...],
      "measurements": { "prosody": {...}, "accent": {...}, ... },
      "flags": [ {"module": ..., "metric": ..., "level": "warn",
                  "measured": ..., "baseline": ..., "deviation_pct": ...,
                  "t_start": ..., "t_end": ..., "note": ...} ],
      "notes": [...]
    }

Install: pip install "whispero[hear]" (librosa, scipy, numpy).
Torch models (speaker embedding, separation, emotion) are optional
extras and never required by the core: when unavailable the report
says so explicitly instead of faking the number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HEAR_SCHEMA = "whispero-hear/1"

# Flag thresholds: relative deviation from baseline, in percent.
# Provisional until calibrated on the user's real voice (SPEC 9).
WARN_PCT = 30.0
FAIL_PCT = 50.0

DEFAULT_SR = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def _need_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        print("error: whispero hear needs: pip install \"whispero[hear]\"",
              file=sys.stderr)
        raise SystemExit(4)


def _need_numpy_scipy():
    try:
        import numpy as np
        from scipy.signal import find_peaks
        return np, find_peaks
    except ImportError:
        print("error: whispero hear needs: pip install \"whispero[hear]\"",
              file=sys.stderr)
        raise SystemExit(4)


def load_audio(path: str, sr: int = DEFAULT_SR):
    """Load an audio file. Returns (samples, sample_rate)."""
    librosa = _need_librosa()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def _st(x: float) -> float:
    """Semitones above 1 Hz (log scale helper)."""
    return 12.0 * math.log2(x) if x > 0 else 0.0


def _frame_rms(y, sr: int, frame_length: int = FRAME_LENGTH,
               hop: int = HOP_LENGTH):
    """Frame RMS with librosa's centered framing, so lengths match the
    other librosa frame features (zcr, centroid) exactly."""
    librosa = _need_librosa()
    if len(y) == 0:
        import numpy as np
        return np.array([])
    return librosa.feature.rms(y=y, frame_length=frame_length,
                               hop_length=hop, center=True)[0]


def librosa_frames(y, frame_length: int, hop: int):
    import numpy as np
    n_frames = 1 + max(0, (len(y) - frame_length) // hop)
    shape = (frame_length, n_frames)
    strides = (y.strides[0], y.strides[0] * hop)
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def _merge_intervals(times, flags, min_dur: float):
    """Merge consecutive True flags into (start_idx, end_idx) frame
    index pairs, dropping runs shorter than min_dur seconds."""
    if len(times) < 2:
        return []
    hop_s = float(times[1] - times[0])
    min_frames = max(1, int(round(min_dur / hop_s)))
    runs = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return [(s, e) for (s, e) in runs if (e - s + 1) >= min_frames]


def detect_pauses(y, sr: int) -> list[dict]:
    """Pause segments from frame RMS: frames below 10% of p95 RMS."""
    np, _ = _need_numpy_scipy()
    rms = _frame_rms(y, sr)
    if len(rms) == 0:
        return []
    thresh = 0.10 * float(np.percentile(rms, 95))
    hop_s = HOP_LENGTH / sr
    times = np.arange(len(rms)) * hop_s
    quiet = rms < thresh
    pauses = []
    for s, e in _merge_intervals(times, quiet, 0.10):
        pauses.append({"start": round(float(times[s]), 3),
                       "end": round(float(times[e] + hop_s), 3)})
    for p in pauses:
        p["duration"] = round(p["end"] - p["start"], 3)
    return pauses


def _f0_autocorr(y, sr: int):
    """Fallback F0 tracker: per-frame autocorrelation peak, 50-500 Hz."""
    np, _ = _need_numpy_scipy()
    frame_length, hop = FRAME_LENGTH, HOP_LENGTH
    if len(y) < frame_length:
        return np.array([]), np.array([], dtype=bool)
    frames = librosa_frames(y, frame_length, hop)
    min_lag, max_lag = int(sr / 500), int(sr / 50)
    f0, voiced = [], []
    for frame in frames.T:
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if corr[0] <= 0:
            f0.append(np.nan)
            voiced.append(False)
            continue
        corr = corr / corr[0]
        seg = corr[min_lag:max_lag + 1]
        lag = int(np.argmax(seg)) + min_lag
        peak = float(seg[lag - min_lag])
        if peak > 0.5:
            f0.append(sr / lag)
            voiced.append(True)
        else:
            f0.append(np.nan)
            voiced.append(False)
    return np.array(f0), np.array(voiced, dtype=bool)


def estimate_f0(y, sr: int):
    """F0 contour: librosa pyin first, scipy autocorrelation fallback.

    Returns (f0_hz array with NaN for unvoiced, voiced mask). The
    fallback triggers when pyin finds voiced frames in under 10% of
    frames (noisy or very short audio).
    """
    librosa = _need_librosa()
    np, _ = _need_numpy_scipy()
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        f0 = np.asarray(f0, dtype=float)
        voiced = np.asarray(voiced_flag, dtype=bool) & ~np.isnan(f0)
        if len(f0) and float(np.mean(voiced)) >= 0.10:
            return f0, voiced
    except Exception:
        pass
    return _f0_autocorr(y, sr)


def analyze_prosody(y, sr: int, duration: float, f0=None, voiced=None) -> dict:
    """M1 prosody: F0, energy, pauses, syllable rate, final direction."""
    librosa = _need_librosa()
    np, find_peaks = _need_numpy_scipy()
    if f0 is None or voiced is None:
        f0, voiced = estimate_f0(y, sr)
    v = f0[voiced] if len(f0) else np.array([])
    out: dict = {"duration_s": round(duration, 3)}
    if len(v) >= 3:
        p5, p95 = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        out["f0_median_hz"] = round(float(np.median(v)), 2)
        out["f0_mean_hz"] = round(float(np.mean(v)), 2)
        out["f0_std_hz"] = round(float(np.std(v)), 2)
        out["f0_range_st"] = round(12.0 * math.log2(p95 / p5), 2) if p5 > 0 else None
        out["f0_min_hz"] = round(float(np.min(v)), 2)
        out["f0_max_hz"] = round(float(np.max(v)), 2)
        # Final intonation: median F0 of the last 0.5 s of voiced frames
        # vs the overall median, in semitones.
        hop_s = HOP_LENGTH / sr
        times = np.arange(len(f0)) * hop_s
        tail = v[times[voiced] >= max(0.0, duration - 0.5)] if len(v) else np.array([])
        if len(tail) >= 2:
            delta = 12.0 * math.log2(float(np.median(tail)) / float(np.median(v)))
            out["final_direction"] = ("rising" if delta > 0.5 else
                                      "falling" if delta < -0.5 else "flat")
            out["final_direction_st"] = round(delta, 2)
        else:
            out["final_direction"] = None
            out["final_direction_st"] = None
    else:
        for k in ("f0_median_hz", "f0_mean_hz", "f0_std_hz", "f0_range_st",
                  "f0_min_hz", "f0_max_hz", "final_direction",
                  "final_direction_st"):
            out[k] = None
    out["voiced_fraction"] = round(float(np.mean(voiced)) if len(voiced) else 0.0, 3)
    rms = _frame_rms(y, sr)
    if len(rms):
        out["rms_mean"] = round(float(np.mean(rms)), 4)
        out["rms_std"] = round(float(np.std(rms)), 4)
        out["rms_p95"] = round(float(np.percentile(rms, 95)), 4)
    else:
        out["rms_mean"] = out["rms_std"] = out["rms_p95"] = None
    pauses = detect_pauses(y, sr)
    out["pauses"] = pauses
    out["pause_count"] = len(pauses)
    out["pause_total_s"] = round(sum(p["duration"] for p in pauses), 3)
    out["pause_ratio"] = round(out["pause_total_s"] / duration, 3) if duration else 0.0
    out["pause_mean_s"] = (round(out["pause_total_s"] / len(pauses), 3)
                           if pauses else 0.0)
    # Syllable-rate proxy: peaks of the onset-strength envelope.
    # Without a transcript this counts syllable nuclei, not words, and
    # is labelled as such. With a transcript, words/min is exact.
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        min_dist = max(1, int(round(0.12 * sr / HOP_LENGTH)))
        peaks, _ = find_peaks(env, distance=min_dist,
                              prominence=(0.1 * float(np.max(env))
                                          if len(env) and float(np.max(env)) > 0 else 0))
        out["syllable_rate_per_min"] = round(len(peaks) / duration * 60.0, 1) if duration else 0.0
        out["syllable_count"] = int(len(peaks))
    except Exception:
        out["syllable_rate_per_min"] = None
        out["syllable_count"] = None
    return out


def _spectral_cog(mag: "np.ndarray", freqs: "np.ndarray") -> float:
    import numpy as np
    total = float(np.sum(mag))
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * mag) / total)


def analyze_accent(y, sr: int) -> dict:
    """M3 accent: spectral analysis of fricative-like segments.

    LIMITATION (documented): without a phoneme recognizer
    available offline, segments are detected spectrally (high
    zero-crossing rate + high centroid + audible level), not by forced
    phoneme alignment. What this reliably separates is the fricative
    spectrum itself: a palatalized carioca [S] has a lower spectral
    center of gravity than a dry [s]. Phoneme-labelled alignment is a
    future optional extra, not faked here.
    """
    librosa = _need_librosa()
    np, _ = _need_numpy_scipy()
    out: dict = {"alignment": "spectral-segments",
                 "note": ("Segments are fricative-like by spectrum, not "
                          "phoneme-labelled. True forced alignment needs a "
                          "phoneme recognizer (future optional extra).")}
    if len(y) < FRAME_LENGTH:
        out.update({"segments": [], "segment_count": 0, "mean_cog_hz": None})
        return out
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms = _frame_rms(y, sr)
    if len(rms) == 0:
        out.update({"segments": [], "segment_count": 0, "mean_cog_hz": None})
        return out
    rms_floor = 0.05 * float(np.percentile(rms, 95))
    hop_s = HOP_LENGTH / sr
    is_fric = (zcr > 0.15) & (cent > 2000) & (rms > rms_floor)
    times = np.arange(len(is_fric)) * hop_s
    mag = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    segments = []
    for s, e in _merge_intervals(times, is_fric, 0.08):
        seg_mag = np.mean(mag[:, s:e + 1], axis=1)
        cog = _spectral_cog(seg_mag, freqs)
        peak = float(freqs[int(np.argmax(seg_mag))])
        segments.append({"start": round(float(times[s]), 3),
                         "end": round(float(times[e] + hop_s), 3),
                         "duration": round(float((e - s + 1) * hop_s), 3),
                         "cog_hz": round(cog, 1),
                         "peak_hz": round(peak, 1)})
    out["segments"] = segments
    out["segment_count"] = len(segments)
    out["mean_cog_hz"] = (round(float(np.mean([s["cog_hz"] for s in segments])), 1)
                          if segments else None)
    return out


def _levinson_durbin(r, order: int):
    """Levinson-Durbin recursion. r: autocorrelation array, order: LPC order."""
    import numpy as np
    a = np.zeros(order + 1)
    e = r[0]
    if e <= 0:
        return a
    a[0] = 1.0
    a[1] = -r[1] / e
    e *= 1.0 - (r[1] / r[0]) ** 2 if r[0] != 0 else 1.0
    for i in range(2, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(1, i))
        k = -(r[i] + acc) / e if e != 0 else 0.0
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= 1.0 - k * k
        if e <= 0:
            break
    return a


def _lpc_formants(frame, sr: int, order: int = 14):
    """Formant frequencies (Hz) of one frame via LPC roots."""
    import numpy as np
    frame = frame - np.mean(frame)
    if len(frame) < order + 2 or float(np.max(np.abs(frame))) <= 0:
        return []
    frame = frame * np.hamming(len(frame))
    r = np.array([float(np.sum(frame[:-k] * frame[k:])) if k else
                  float(np.sum(frame ** 2)) for k in range(order + 1)])
    if r[0] <= 0:
        return []
    a = _levinson_durbin(r, order)
    if not np.any(a[1:]):
        return []
    roots = np.roots(a)
    roots = [rt for rt in roots if np.imag(rt) > 0.01]
    freqs, bws = [], []
    for rt in roots:
        ang = math.atan2(np.imag(rt), np.real(rt))
        freq = ang * sr / (2.0 * math.pi)
        bw = -math.log(abs(rt)) * sr / math.pi
        if 100 <= freq <= 5000 and bw < 400:
            freqs.append(freq)
            bws.append(bw)
    order_idx = sorted(range(len(freqs)), key=lambda i: freqs[i])
    return [freqs[i] for i in order_idx]


def analyze_timbre(y, sr: int, f0=None, voiced=None) -> dict:
    """M2 timbre/identity: spectral core, no torch required.

    Formants via LPC, HNR via autocorrelation peak, jitter/shimmer as
    approximations from the F0 contour and frame RMS (labelled as
    such), spectral centroid. The speaker embedding (ECAPA-TDNN) runs
    only when torch + speechbrain are installed; otherwise the report
    says embedding_available=false instead of faking the number.
    """
    librosa = _need_librosa()
    np, _ = _need_numpy_scipy()
    out: dict = {}
    if len(y) >= FRAME_LENGTH:
        cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        out["spectral_centroid_mean_hz"] = round(float(np.mean(cent)), 1)
    else:
        out["spectral_centroid_mean_hz"] = None
    if f0 is None or voiced is None:
        f0, voiced = estimate_f0(y, sr)
    v = f0[voiced] if len(f0) else np.array([])
    # Formants: LPC on a subsample of voiced frames.
    formants = []
    if len(v) >= 3 and len(y) >= FRAME_LENGTH:
        frames = librosa_frames(y, FRAME_LENGTH, HOP_LENGTH)
        voiced_idx = np.where(voiced)[0]
        step = max(1, len(voiced_idx) // 20)
        for i in voiced_idx[::step]:
            if i < frames.shape[1]:
                formants.append(_lpc_formants(frames[:, i], sr))
        f1 = sorted(f[0] for f in formants if len(f) >= 1)
        f2 = sorted(f[1] for f in formants if len(f) >= 2)
        out["formant_f1_median_hz"] = round(float(np.median(f1)), 1) if f1 else None
        out["formant_f2_median_hz"] = round(float(np.median(f2)), 1) if f2 else None
    else:
        out["formant_f1_median_hz"] = None
        out["formant_f2_median_hz"] = None
    # HNR: median autocorrelation peak in the pitch-lag range.
    hnr_vals = []
    if len(v) >= 3 and len(y) >= FRAME_LENGTH:
        frames = librosa_frames(y, FRAME_LENGTH, HOP_LENGTH)
        voiced_idx = np.where(voiced)[0]
        min_lag, max_lag = int(sr / 500), int(sr / 50)
        for i in voiced_idx[::max(1, len(voiced_idx) // 20)]:
            if i >= frames.shape[1]:
                continue
            frame = frames[:, i] - np.mean(frames[:, i])
            corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
            if corr[0] <= 0:
                continue
            corr = corr / corr[0]
            peak = float(np.max(corr[min_lag:max_lag + 1]))
            peak = min(max(peak, 0.01), 0.99)
            hnr_vals.append(10.0 * math.log10(peak / (1.0 - peak)))
    out["hnr_median_db"] = round(float(np.median(hnr_vals)), 2) if hnr_vals else None
    # Jitter/shimmer approximations from F0 periods and frame RMS.
    if len(v) >= 4:
        periods = 1.0 / v
        out["jitter_pct"] = round(float(np.mean(np.abs(np.diff(periods))) /
                                        np.mean(periods)) * 100.0, 3)
    else:
        out["jitter_pct"] = None
    rms = _frame_rms(y, sr)
    if len(rms) >= 4 and float(np.mean(rms)) > 0:
        out["shimmer_pct"] = round(float(np.mean(np.abs(np.diff(rms))) /
                                         np.mean(rms)) * 100.0, 3)
    else:
        out["shimmer_pct"] = None
    out["embedding_available"] = False
    out["embedding_model"] = None
    out["embedding"] = None
    try:
        import torch  # noqa: F401
        from speechbrain.inference.speaker import SpeakerRecognition  # noqa
        out["embedding_available"] = True
        out["embedding_model"] = "spkrec-ecapa-voxceleb"
        out["note"] = ("ECAPA embedding computed only when the caller "
                       "requests it via compare_embeddings(); the core "
                       "report never blocks on torch.")
    except ImportError:
        out["note"] = ("torch/speechbrain not installed: speaker embedding "
                       "not computed. Install the optional torch extra to "
                       "enable it.")
    return out


def speaker_embedding(y, sr: int):
    """Optional ECAPA-TDNN embedding. Returns None when unavailable."""
    try:
        import numpy as np
        from speechbrain.inference.speaker import SpeakerRecognition
        rec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/whispero/spkrec"))
        emb = rec.encode_batch(np.asarray(y, dtype=np.float32)[None, :])
        return np.asarray(emb).flatten().tolist()
    except Exception:
        return None


def analyze_music(y, sr: int) -> dict:
    """M5 music/mix: ducking measured on music-only frequency bins.

    Method, after two rejected approaches (median filter keeps the
    voice's median level; minimum statistics keeps its AM valleys,
    both swamping the ducking signal):
    1. Speech activity from the onset envelope (syllabic pulses), NOT
       from silence pauses: a music bed playing through a "pause" is
       still non-speech.
    2. Music profile = mean energy per frequency bin over non-speech
       frames. Music bins = bins peaking above 5x the median profile,
       excluding bins within 15 Hz of a voice harmonic (multiples of
       the measured F0 median), so voice harmonics never count as
       "music".
    3. Ducking ratio = mean music-bin energy per frame during speech
       / during non-speech. A ducked bed reads ~0.06 (0.25 gain
       squared); an unducked bed reads ~1.0.
    This is a heuristic, not a music classifier and not source
    separation (Demucs stays a future optional extra). When there are
    no non-speech frames or no music bins, the verdict is honestly
    null instead of a guess.
    """
    librosa = _need_librosa()
    np, _ = _need_numpy_scipy()
    out: dict = {"method": "music-bin-ducking"}
    if len(y) < FRAME_LENGTH:
        out.update({"music_bins_hz": [], "music_share_nonspeech": None,
                    "ducking_ratio": None, "music_competing": None})
        return out
    mag = np.abs(librosa.stft(y, n_fft=8192, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    n_frames = mag.shape[1]
    power = mag ** 2
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    n_env = min(len(env), n_frames)
    thresh = 0.25 * float(np.percentile(env[:n_env], 95)) if n_env else 0.0
    speech_mask = np.zeros(n_frames, dtype=bool)
    speech_mask[:n_env] = env[:n_env] > thresh
    nonspeech_mask = ~speech_mask
    out["music_bins_hz"] = []
    out["music_share_nonspeech"] = None
    out["ducking_ratio"] = None
    out["music_competing"] = None
    if not np.any(nonspeech_mask) or not np.any(speech_mask):
        out["note"] = ("No speech/non-speech contrast in this audio: "
                       "ducking cannot be measured.")
        return out
    profile = np.mean(power[:, nonspeech_mask], axis=1)
    med = float(np.median(profile))
    if med <= 0:
        out["note"] = "Silent non-speech frames: no music profile."
        return out
    # Peak bins of the non-speech profile, excluding voice harmonics.
    # 8192-point FFT (2.7 Hz/bin) keeps voice leakage out of chord
    # bins; a 2048-point FFT was tried and rejected (leakage swamped
    # the bed and the ducking ratio read ~6 for both versions).
    # The exclusion F0 comes from the speech-frame spectrum peak
    # (50-300 Hz), NOT from pyin on the mix: pyin locks onto a
    # spurious 55 Hz subharmonic on voice+chord mixes, which would
    # exclude every chord bin as a "harmonic".
    spec_speech = np.mean(power[:, speech_mask], axis=1)
    band = (freqs >= 50) & (freqs <= 300)
    f0_voice = None
    if np.any(band):
        pk = int(np.argmax(np.where(band, spec_speech, 0)))
        if spec_speech[pk] > 5 * float(np.median(spec_speech[band])):
            f0_voice = float(freqs[pk])
    from scipy.signal import find_peaks as _fp
    peaks, _ = _fp(profile, distance=10, height=5 * med)
    cand = []
    for i in peaks:
        if not (50 <= freqs[i] <= 8000):
            continue
        if f0_voice:
            h = round(freqs[i] / f0_voice)
            if h >= 1 and abs(freqs[i] - h * f0_voice) <= 15:
                continue
        cand.append(i)
    cand = sorted(cand, key=lambda i: profile[i], reverse=True)[:8]
    out["music_bins_hz"] = [round(float(freqs[i]), 1) for i in cand]
    if not cand:
        out["note"] = ("No frequency bins peak outside voice harmonics: "
                       "no music bed detected.")
        out["music_competing"] = False
        return out
    total_ns = float(np.sum(power[:, nonspeech_mask]))
    out["music_share_nonspeech"] = round(
        float(np.sum(power[cand][:, nonspeech_mask])) / total_ns, 3) \
        if total_ns > 0 else 0.0
    mean_s = float(np.mean(np.sum(power[cand][:, speech_mask], axis=0)))
    mean_n = float(np.mean(np.sum(power[cand][:, nonspeech_mask], axis=0)))
    out["ducking_ratio"] = round(mean_s / mean_n, 3) if mean_n > 0 else None
    out["music_competing"] = bool(
        out["music_share_nonspeech"] >= 0.05 and
        out["ducking_ratio"] is not None and out["ducking_ratio"] >= 0.8)
    return out


def analyze_beats(y, sr: int, cuts: list[float], pauses: list[dict],
                  f0=None, voiced=None) -> dict:
    """M6 cuts/tempo: beat tracking + cut classification.

    Each cut is classified: on-beat (within 70 ms of a beat),
    in-pause (inside a pause +/- 50 ms), or mid-voiced (inside a voiced
    segment, away from its edges). Without a transcript, 'mid-voiced'
    is the honest stand-in for 'mid-word'.
    """
    librosa = _need_librosa()
    np, _ = _need_numpy_scipy()
    out: dict = {"cuts": []}
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=env, sr=sr, units="time")
        tempo = float(np.atleast_1d(tempo)[0])
        beats = [round(float(b), 3) for b in np.atleast_1d(beats)]
    except Exception:
        tempo, beats = None, []
    # Beat-grid reliability gate: a grid needs repetition in a sane
    # tempo range. A single "beat" from a 30 BPM guess on a beatless
    # mix is noise, not a grid, and must not classify cuts as on-beat.
    out["tempo_bpm"] = round(tempo, 1) if tempo else None
    reliable = bool(tempo and 40 <= tempo <= 240 and len(beats) >= 3)
    out["beat_grid_reliable"] = reliable
    out["beat_times"] = beats if reliable else []
    beats = out["beat_times"]
    if f0 is None or voiced is None:
        f0, voiced = estimate_f0(y, sr)
    hop_s = HOP_LENGTH / sr
    times = np.arange(len(voiced)) * hop_s if len(voiced) else np.array([])
    voiced_segs = []
    for s, e in _merge_intervals(times, voiced, 0.10):
        voiced_segs.append((float(times[s]), float(times[e] + hop_s)))
    # Bridge gaps under 0.25 s: pyin drops voicing in AM dips, so a
    # single utterance arrives fragmented; without bridging, no cut
    # inside it ever counts as mid-voiced.
    bridged = []
    for s, e in voiced_segs:
        if bridged and s - bridged[-1][1] < 0.25:
            bridged[-1] = (bridged[-1][0], e)
        else:
            bridged.append((s, e))
    voiced_segs = bridged
    for cut in cuts:
        nearest = min(beats, key=lambda b: abs(b - cut)) if beats else None
        dist = abs(nearest - cut) if nearest is not None else None
        on_beat = dist is not None and dist <= 0.07
        in_pause = any(p["start"] - 0.05 <= cut <= p["end"] + 0.05
                       for p in pauses)
        mid_voiced = any(s + 0.10 < cut < e - 0.10 for s, e in voiced_segs)
        if on_beat:
            cls = "on-beat"
        elif in_pause:
            cls = "in-pause"
        elif mid_voiced:
            cls = "mid-voiced-off-beat"
        else:
            cls = "off-beat-boundary"
        out["cuts"].append({
            "time": round(float(cut), 3),
            "classification": cls,
            "nearest_beat": nearest,
            "nearest_beat_dist_s": round(dist, 3) if dist is not None else None,
            "on_beat": on_beat,
            "in_pause": in_pause,
            "mid_voiced": mid_voiced,
        })
    return out


def analyze_state(prosody: dict, baseline_prosody: dict | None) -> dict:
    """M4 prosodic state: arousal/valence proxies, never hard labels.

    Arousal proxy = mean of the ratios (energy, F0 range, syllable
    rate) against the baseline; valence proxy = final intonation
    direction + energy variability. An audio emotion classifier is NOT
    configured: emotion_suggestion stays null with the reason stated.
    """
    out: dict = {"method": "prosody-proxies-vs-baseline"}
    comps = {}
    if baseline_prosody:
        for key, name in (("rms_mean", "energy_ratio"),
                          ("f0_range_st", "pitch_range_ratio"),
                          ("syllable_rate_per_min", "rate_ratio")):
            m, b = prosody.get(key), baseline_prosody.get(key)
            if m is not None and b:
                comps[name] = round(m / b, 3)
    out["arousal_components"] = comps
    out["arousal_proxy"] = (round(float(np_mean(list(comps.values()))), 3)
                            if comps else None)
    rms_mean = prosody.get("rms_mean")
    rms_std = prosody.get("rms_std")
    out["energy_variability"] = (round(rms_std / rms_mean, 3)
                                 if rms_mean and rms_std is not None else None)
    out["final_direction"] = prosody.get("final_direction")
    out["emotion_suggestion"] = None
    out["emotion_reason"] = ("No audio emotion classifier is configured; "
                             "proxies only. A classifier would enter here "
                             "as a suggestion with confidence, never as a "
                             "hard label.")
    return out


def np_mean(values):
    import numpy as np
    return float(np.mean(values)) if values else None


def _dev_pct(measured, baseline):
    if measured is None or baseline is None or baseline == 0:
        return None
    return (measured - baseline) / baseline * 100.0


def _level(dev_pct):
    if dev_pct is None:
        return "pass"
    a = abs(dev_pct)
    if a >= FAIL_PCT:
        return "fail"
    if a >= WARN_PCT:
        return "warn"
    return "pass"


def build_flags(meas: dict, base: dict, duration: float) -> list[dict]:
    """Compare measurements against the baseline; one flag per metric."""
    flags = []
    if not base:
        return flags

    def add(module, metric, measured, baseline, t_start, t_end, note):
        dev = _dev_pct(measured, baseline)
        if dev is None:
            return
        flags.append({
            "module": module, "metric": metric,
            "level": _level(dev),
            "measured": measured, "baseline": baseline,
            "deviation_pct": round(dev, 1),
            "t_start": t_start, "t_end": t_end, "note": note,
        })

    p, bp = meas.get("prosody", {}), base.get("prosody", {})
    for metric, note in (
            ("f0_range_st", "Pitch range vs baseline (wide range reads "
                            "as exaggerated prosody)"),
            ("syllable_rate_per_min", "Syllable rate vs baseline (fast "
                                      "rate reads as rushed delivery)"),
            ("rms_mean", "Mean level vs baseline")):
        add("prosody", metric, p.get(metric), bp.get(metric),
            0.0, round(duration, 3), note)
    a, ba = meas.get("accent", {}), base.get("accent", {})
    add("accent", "mean_cog_hz", a.get("mean_cog_hz"), ba.get("mean_cog_hz"),
        0.0, round(duration, 3),
        "Mean fricative spectral center of gravity vs baseline")
    if a.get("segments") and ba.get("mean_cog_hz"):
        for seg in a["segments"]:
            add("accent", "segment_cog_hz", seg["cog_hz"],
                ba["mean_cog_hz"], seg["start"], seg["end"],
                "Fricative-like segment CoG vs baseline mean")
    t, bt = meas.get("timbre", {}), base.get("timbre", {})
    add("timbre", "spectral_centroid_mean_hz",
        t.get("spectral_centroid_mean_hz"), bt.get("spectral_centroid_mean_hz"),
        0.0, round(duration, 3), "Voice brightness vs baseline")
    m, bm = meas.get("music", {}), base.get("music", {})
    if (m.get("music_competing") is True and
            bm.get("music_competing") is not True):
        flags.append({
            "module": "music", "metric": "music_competing",
            "level": "warn", "measured": True, "baseline": False,
            "deviation_pct": None, "t_start": 0.0, "t_end": round(duration, 3),
            "note": ("Steady background stays loud during speech "
                     "(no ducking) while the baseline ducks it"),
        })
    for cut in meas.get("beats", {}).get("cuts", []):
        if cut["classification"] == "mid-voiced-off-beat":
            musical = (meas.get("music", {}).get("music_competing") is True)
            flags.append({
                "module": "beats", "metric": "cut_classification",
                "level": "fail" if musical else "warn",
                "measured": cut["classification"], "baseline": None,
                "deviation_pct": None,
                "t_start": cut["time"], "t_end": cut["time"],
                "note": ("Cut lands mid-voiced and off-beat; fail when "
                         "the material is musical"),
            })
    return flags


ALL_MODULES = ("prosody", "timbre", "accent", "state", "music", "beats")


def analyze(path: str, baseline_path: str | None = None,
            modules: tuple[str, ...] = ALL_MODULES,
            cuts: list[float] | None = None,
            sr: int = DEFAULT_SR) -> dict:
    """Full hear analysis. Returns the whispero-hear/1 report dict."""
    y, sr = load_audio(path, sr=sr)
    duration = len(y) / sr
    base_meas = None
    if baseline_path:
        by, _ = load_audio(baseline_path, sr=sr)
        base_meas = _measure(by, sr, len(by) / sr, modules)
    base_prosody = base_meas.get("prosody") if base_meas else None
    meas = _measure(y, sr, duration, modules, cuts=cuts or [],
                    base_prosody=base_prosody)
    flags = build_flags(meas, base_meas, duration) if base_meas else []
    notes = []
    if not baseline_path:
        notes.append("No baseline: measurements only, no deviation flags.")
    if "accent" in modules:
        notes.append(meas["accent"].get("note", ""))
    if "timbre" in modules and not meas["timbre"].get("embedding_available"):
        notes.append(meas["timbre"].get("note", ""))
    if "state" in modules:
        notes.append(meas["state"].get("emotion_reason", ""))
    return {
        "schema": HEAR_SCHEMA,
        "audio": path,
        "baseline": baseline_path,
        "duration_s": round(duration, 3),
        "modules": list(modules),
        "measurements": meas,
        "flags": flags,
        "notes": [n for n in notes if n],
    }


def _measure(y, sr: int, duration: float, modules, cuts=None,
             base_prosody=None) -> dict:
    meas: dict = {}
    f0 = voiced = None
    if "prosody" in modules or "timbre" in modules or "beats" in modules:
        f0, voiced = estimate_f0(y, sr)
    if "prosody" in modules:
        meas["prosody"] = analyze_prosody(y, sr, duration, f0=f0, voiced=voiced)
    if "timbre" in modules:
        meas["timbre"] = analyze_timbre(y, sr, f0=f0, voiced=voiced)
    if "accent" in modules:
        meas["accent"] = analyze_accent(y, sr)
    if "music" in modules:
        meas["music"] = analyze_music(y, sr)
    if "beats" in modules:
        pauses = meas.get("prosody", {}).get("pauses")
        if pauses is None:
            pauses = detect_pauses(y, sr)
        meas["beats"] = analyze_beats(y, sr, cuts or [], pauses,
                                      f0=f0, voiced=voiced)
    if "state" in modules:
        meas["state"] = analyze_state(
            meas.get("prosody", {}), base_prosody)
    return meas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whispero Hear - measured signal analysis.")
    p.add_argument("audio", help="Audio file to analyze")
    p.add_argument("--baseline", default=None, metavar="FILE",
                   help="Reference speaker audio: enables deviation flags")
    p.add_argument("--prosody", action="store_true")
    p.add_argument("--timbre", action="store_true")
    p.add_argument("--accent", action="store_true")
    p.add_argument("--state", action="store_true")
    p.add_argument("--music", action="store_true")
    p.add_argument("--beats", action="store_true")
    p.add_argument("--cuts", default="", metavar="T1,T2,...",
                   help="Comma-separated cut timestamps in seconds")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Write JSON report to FILE instead of stdout")
    p.add_argument("--sr", type=int, default=DEFAULT_SR,
                   help="Sample rate for analysis (default 22050)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    picked = tuple(m for m in ALL_MODULES if getattr(args, m))
    modules = picked or ALL_MODULES
    try:
        cuts = [float(t) for t in args.cuts.split(",") if t.strip()]
    except ValueError:
        print("error: --cuts must be comma-separated seconds", file=sys.stderr)
        return 2
    try:
        report = analyze(args.audio, baseline_path=args.baseline,
                         modules=modules, cuts=cuts, sr=args.sr)
    except FileNotFoundError as e:
        print(f"error: file not found: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 1)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
