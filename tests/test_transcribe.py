"""Tests for pure functions in transcribe.py (no API calls)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe import (
    is_url,
    fmt_ts,
    _ts_to_s,
    parse_diarize_lines,
    _segments_to_text,
    _speaker_agreement,
    _build_label_map,
)


def test_is_url():
    assert is_url("https://example.com/a.mp3")
    assert is_url("http://example.com/a.mp3")
    assert not is_url("/tmp/a.mp3")
    assert not is_url("a.mp3")
    assert not is_url("")


def test_fmt_ts():
    assert fmt_ts(0) == "00:00"
    assert fmt_ts(65) == "01:05"
    assert fmt_ts(3661) == "61:01"
    assert fmt_ts(6.16) == "00:06"


def test_ts_to_s():
    assert _ts_to_s("00:00") == 0.0
    assert _ts_to_s("01:05") == 65.0


def test_parse_diarize_lines():
    text = "[FALANTE 0] [00:00] Olá mundo\n[FALANTE 1] [00:05] Oi de volta"
    segs = parse_diarize_lines(text)
    assert len(segs) == 2
    assert segs[0]["speaker"] == "FALANTE 0"
    assert segs[0]["text"] == "Olá mundo"
    assert segs[1]["start"] == "00:05"


def test_parse_diarize_lines_empty():
    assert parse_diarize_lines("") == []
    assert parse_diarize_lines("sem formato nenhum") == []


def test_segments_to_text():
    segs = [
        {"speaker": "A", "start": "00:00", "text": "olá"},
        {"speaker": "B", "start": "00:05", "text": "oi"},
    ]
    out = _segments_to_text(segs)
    assert "olá" in out and "oi" in out


def test_speaker_agreement_identical():
    segs = [{"speaker": "A", "start": 0, "end": 5, "text": "x"}]
    assert _speaker_agreement(segs, segs) == 1.0


def test_speaker_agreement_empty():
    assert _speaker_agreement([], []) == 0.0


def test_build_label_map():
    ref = [{"speaker": "A", "start": 0, "end": 5, "text": "x"}]
    run = [{"speaker": "X", "start": 0, "end": 5, "text": "x"}]
    m = _build_label_map(ref, run)
    assert m.get("X") == "A" or "X" in m
