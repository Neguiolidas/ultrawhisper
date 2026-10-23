"""Tests for local_engine.py that need no models (clustering + assignment).
Run: python3 -m pytest tests/test_local_engine.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("scipy")
import local_engine as le


def _blobs(k, n=40, dim=64, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim))
    E, y = [], []
    for c in range(k):
        x = centers[c] + spread * rng.normal(size=(n, dim)) * np.linalg.norm(centers[c]) / np.sqrt(dim)
        E.append(x)
        y += [c] * n
    E = np.concatenate(E)
    return E / np.linalg.norm(E, axis=1, keepdims=True), np.array(y)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_spectral_finds_speaker_count(k):
    E, y = _blobs(k)
    lab = le.spectral_cluster(E)
    assert len(set(lab.tolist())) == k
    # every true cluster maps to exactly one label
    for c in range(k):
        assert len(set(lab[y == c].tolist())) == 1


def test_spectral_fixed_k():
    E, _ = _blobs(4)
    assert len(set(le.spectral_cluster(E, k=4).tolist())) == 4


def test_assign_speakers_splits_on_change_and_numbers_by_appearance():
    asr = [{"start": 0.0, "end": 3.0, "text": "oi tudo bem sim",
            "words": [{"start": 0.0, "end": 0.5, "word": "oi"},
                      {"start": 0.6, "end": 1.0, "word": "tudo"},
                      {"start": 1.1, "end": 1.5, "word": "bem"},
                      {"start": 2.0, "end": 2.4, "word": "sim"}]}]
    dsegs = [(0.0, 1.6, 7), (1.9, 3.0, 2)]
    turns = le.assign_speakers(asr, dsegs)
    assert [t["speaker"] for t in turns] == ["FALANTE 1", "FALANTE 2"]
    assert turns[0]["text"] == "oi tudo bem" and turns[1]["text"] == "sim"
    assert turns[1]["start"] == 2.0 and turns[1]["end"] == 2.4


def test_fbank_shape():
    wav = np.random.default_rng(0).normal(scale=0.1, size=16000).astype(np.float32)
    f = le._fbank(wav)
    assert f.shape == (98, 80)
    assert abs(float(f.mean())) < 1e-3
