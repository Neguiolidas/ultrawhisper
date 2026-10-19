"""Universal tests for ears.py (rich turn object).

The API calls are mocked: these tests cover the turn contract
(schema whispero-turn/1), JSON parsing, fallbacks and flags.
Run: python3 -m pytest tests/  (or: python3 tests/test_ears.py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import synth_voice, tmp_path

import ears
import transcribe as t

MOCK_TURN_JSON = json.dumps({
    "transcript": "olá, tudo bem?",
    "tone": "animado",
    "emotion": "alegria",
    "urgency": "baixa",
})


def _mock_api(monkeypatch_response=MOCK_TURN_JSON):
    t.get_api_key = lambda a: "test-key"

    def fake_gemini(base, key, model, prompt, media_part):
        assert "JSON" in prompt
        return monkeypatch_response

    t.gemini_transcribe = fake_gemini


def _no_voiceid():
    # force the identify path to be skipped cleanly
    pass


def test_turn_has_all_schema_keys():
    _mock_api()
    p = synth_voice(tmp_path("t.wav"), f0=150.0)
    turn = ears.analyze_turn(p, {"lang": "pt", "provider": "gemini",
                                 "identify": False})
    for k in ("schema", "speaker", "speaker_score", "transcript", "tone",
              "emotion", "urgency", "lang", "duration_s"):
        assert k in turn, f"missing key: {k}"
    assert turn["schema"] == "whispero-turn/1"


def test_turn_parses_model_json():
    _mock_api()
    p = synth_voice(tmp_path("t.wav"), f0=150.0)
    turn = ears.analyze_turn(p, {"lang": "pt", "provider": "gemini",
                                 "identify": False})
    assert turn["transcript"] == "olá, tudo bem?"
    assert turn["tone"] == "animado"
    assert turn["emotion"] == "alegria"
    assert turn["urgency"] == "baixa"
    assert turn["lang"] == "pt"


def test_turn_tolerates_markdown_fences():
    _mock_api("```json\n" + MOCK_TURN_JSON + "\n```")
    p = synth_voice(tmp_path("t.wav"), f0=150.0)
    turn = ears.analyze_turn(p, {"lang": "pt", "provider": "gemini",
                                 "identify": False})
    assert turn["transcript"] == "olá, tudo bem?"


def test_turn_falls_back_on_garbage():
    _mock_api("this is not json at all")
    p = synth_voice(tmp_path("t.wav"), f0=150.0)
    turn = ears.analyze_turn(p, {"lang": "pt", "provider": "gemini",
                                 "identify": False})
    # whole output becomes the transcript, analysis keeps defaults
    assert turn["transcript"] == "this is not json at all"
    assert turn["tone"] == "neutro"
    assert turn["urgency"] == "baixa"


def test_turn_normalizes_bad_urgency():
    bad = json.dumps({"transcript": "x", "tone": "calmo",
                      "emotion": "neutro", "urgency": "extrema"})
    _mock_api(bad)
    p = synth_voice(tmp_path("t.wav"), f0=150.0)
    turn = ears.analyze_turn(p, {"lang": "pt", "provider": "gemini",
                                 "identify": False})
    assert turn["urgency"] == "baixa"


def test_no_identify_leaves_speaker_unknown():
    _mock_api()
    p = synth_voice(tmp_path("t.wav"), f0=150.0)
    turn = ears.analyze_turn(p, {"lang": "pt", "provider": "gemini",
                                 "identify": False})
    assert turn["speaker"] == "unknown"
    assert turn["speaker_score"] == 0.0


def test_missing_file_raises():
    _mock_api()
    try:
        ears.analyze_turn("/nonexistent/turn.wav",
                          {"lang": "pt", "identify": False})
    except FileNotFoundError:
        return
    raise AssertionError("missing file should raise FileNotFoundError")


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
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
