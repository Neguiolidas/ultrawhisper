"""Shared fixtures for whispero tests: synthetic audio, no real voices.

All audio here is synthesized (harmonic complexes). Real-voice calibration
(threshold 0.5, 22/22 on a 3-speaker sample) is documented in EARS.md, not
committed: personal audio never belongs in the repo.
"""
import os
import struct
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SR = 16000


def synth_voice(path: str, f0: float = 120.0, seconds: float = 4.0,
                tilt: float = 0.3) -> str:
    """Write a voice-like harmonic complex to a WAV file."""
    t = np.arange(int(seconds * SR)) / SR
    sig = np.zeros_like(t)
    for h in range(1, 8):
        sig += (1.0 / (h ** (1.0 + tilt))) * np.sin(2 * np.pi * f0 * h * t + h)
    sig *= 0.8 + 0.2 * np.sin(2 * np.pi * 5 * t)  # vibrato
    sig = sig / np.max(np.abs(sig)) * 0.5
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % len(sig),
                                  *(sig * 32767).astype(np.int16)))
    return path


def synth_silence(path: str, seconds: float = 4.0) -> str:
    n = int(seconds * SR)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % n, *([0] * n)))
    return path


def tmp_path(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(), name)
