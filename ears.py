#!/usr/bin/env python3
"""Whispero Ears - rich audio understanding for agents.

Turns audio into a structured turn object an agent can reason about:
who is speaking, what was said, and how it was said. More than text:
tone, emotion, urgency, speaker identity via voiceprint.

Usage:
    whispero-ears <audio> [--lang pt] [--provider gemini|openai]
                          [--api-key KEY] [--no-identify]

Output (JSON, schema whispero-turn/1):
    {
      "schema": "whispero-turn/1",
      "speaker": "Neguiolidas",      # enrolled name or "unknown"
      "speaker_score": 0.93,         # cosine similarity vs voiceprint
      "transcript": "...",
      "tone": "animado",
      "emotion": "curiosidade",
      "urgency": "baixa",
      "lang": "pt",
      "duration_s": 4.2
    }

Install: pip install whispero (+ "whispero[voiceid]" for speaker ID)
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys

TURN_SCHEMA = "whispero-turn/1"

TURN_PROMPT = (
    "Analise este áudio e retorne APENAS um JSON válido, sem comentários, "
    "com exatamente estas chaves:\n"
    '{{"transcript": "transcrição fiel em {lang}", '
    '"tone": "tom da voz em 1-2 palavras (ex: calmo, animado, tenso)", '
    '"emotion": "emoção predominante em 1-2 palavras (ex: curiosidade, '
    'frustração, alegria, neutro)", '
    '"urgency": "baixa|media|alta"}}\n'
    "Seja fiel: não invente palavras na transcrição."
)


def _import_transcribe():
    try:
        import transcribe as t
        return t
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import transcribe as t
        return t


def _audio_duration(path: str) -> float:
    import subprocess  # nosec B404
    try:
        out = subprocess.run(  # nosec B603
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, check=True, timeout=15).stdout.decode()
        return round(float(out.strip()), 1)
    except Exception:
        return 0.0


def analyze_turn(audio_path: str, cfg: dict) -> dict:
    """Process one audio turn into a rich turn dict."""
    t = _import_transcribe()
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    lang = cfg.get("lang") or "pt"
    provider = cfg.get("provider") or "gemini"

    # conversation history: recent turns as context for tone/emotion
    history_ctx = ""
    history_path = cfg.get("history")
    if history_path:
        prev = _load_history(history_path)
        if prev:
            lines = [f"[{p.get('speaker', '?')}] {p.get('transcript', '')[:120]}"
                     for p in prev]
            history_ctx = ("\nContexto da conversa (turnos anteriores):\n"
                           + "\n".join(lines) + "\n")

    class A:
        pass
    a = A()
    a.api_key = cfg.get("api_key")
    a.provider = provider
    key = t.get_api_key(a)

    turn: dict = {
        "schema": TURN_SCHEMA,
        "speaker": "unknown",
        "speaker_score": 0.0,
        "transcript": "",
        "tone": "neutro",
        "emotion": "neutro",
        "urgency": "baixa",
        "lang": lang,
        "duration_s": _audio_duration(audio_path),
    }

    # 1. speaker identification (local, fast, no API)
    if cfg.get("identify", True):
        try:
            import voiceid  # noqa: F401
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import voiceid  # noqa: F401
        try:
            name, score = voiceid.identify(audio_path)
            turn["speaker"] = name
            turn["speaker_score"] = score
        except Exception as e:
            print(f"warn: speaker id failed ({e})", file=sys.stderr)

    # 2. transcript + tone + emotion + urgency (one API pass)
    base_prompt = TURN_PROMPT.format(lang=lang)
    if history_ctx:
        base_prompt = history_ctx + "\n" + base_prompt
    if provider == "gemini":
        model = cfg.get("model") or t.DEFAULT_GEMINI_MODEL
        mime, _ = mimetypes.guess_type(audio_path)
        with open(audio_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        media_part = {"inline_data": {"mime_type": mime or "audio/wav",
                                      "data": b64}}
        text = None
        fallbacks = [m for m in (cfg.get("fallback_models") or "").split(",")
                     if m.strip()]
        for m in [model] + [x.strip() for x in fallbacks]:
            text = t.gemini_transcribe(
                t.GEMINI_API_BASE, key, m, base_prompt, media_part)
            if text:
                turn["model"] = m
                break
    else:
        # OpenAI-compatible: transcribe, then text-analyze the transcript
        model = cfg.get("model") or t.DEFAULT_OPENAI_MODEL
        base = cfg.get("base_url") or t.OPENAI_API_BASE
        transcript = t.openai_transcribe(base, key, model, audio_path, lang)
        text = None
        if transcript:
            analysis = t.openai_chat(
                base, key, model,
                "Analise esta transcrição e retorne APENAS um JSON válido "
                "com as chaves tone (tom em 1-2 palavras), emotion (emoção "
                "em 1-2 palavras) e urgency (baixa|media|alta), sem "
                "comentários.\n\nTranscrição:\n" + transcript)
            if analysis:
                text = json.dumps({"transcript": transcript})[:-1] + "," + \
                    analysis[1:]

    if text:
        try:
            # tolerate markdown fences
            t2 = text.strip()
            if t2.startswith("```"):
                t2 = t2.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(t2)
            for k in ("transcript", "tone", "emotion", "urgency"):
                if k in data and isinstance(data[k], str) and data[k].strip():
                    turn[k] = data[k].strip()
            if turn["urgency"] not in ("baixa", "media", "alta"):
                turn["urgency"] = "baixa"
        except (ValueError, AttributeError):
            # not JSON: treat whole output as transcript
            if text.strip():
                turn["transcript"] = text.strip()

    if not text:
        # API failed: say so instead of returning a blank "neutro" turn
        turn["error"] = "no response from the model (see stderr)"
    return turn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whispero Ears - rich audio understanding for agents. "
                    "Outputs a JSON turn: speaker, transcript, tone, emotion.")
    p.add_argument("audio", help="Audio file to process")
    p.add_argument("--lang", default="pt")
    p.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--fallback-models", default=os.environ.get(
        "WHISPERO_FALLBACK_MODELS", ""),
        help="Comma-separated models to try if --model fails")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--no-identify", action="store_true",
                   help="Skip speaker identification")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print the JSON")
    p.add_argument("--history", default=None, metavar="FILE",
                   help="JSONL conversation history: appends this turn and "
                        "uses recent turns as context")
    p.add_argument("--hear", action="store_true",
                   help="Attach measured signal analysis (whispero-hear/1) "
                        "as turn['hearing'] (needs whispero[hear])")
    p.add_argument("--hear-baseline", default=None, metavar="FILE",
                   help="Reference speaker audio for hear deviation flags")
    return p


def _load_history(path: str, limit: int = 5) -> list[dict]:
    """Load recent turns from a JSONL history file."""
    turns = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        turns.append(json.loads(line))
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    return turns[-limit:]


def _append_history(path: str, turn: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(turn, ensure_ascii=False) + "\n")


def _import_hear():
    try:
        import hear as h
        return h
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hear as h
        return h


def attach_hearing(turn: dict, audio: str,
                   baseline: str | None = None) -> dict:
    """Attach measured signal analysis to a turn, additively only.

    Adds turn["hearing"] (schema whispero-hear/1) and never modifies
    any existing turn key, so the whispero-turn/1 contract is
    preserved. Raises SystemExit(4) when the hear extra (librosa) is
    not installed, same convention as the voiceid extra.
    """
    hear = _import_hear()
    turn["hearing"] = hear.analyze(audio, baseline_path=baseline)
    return turn


def main() -> int:
    args = build_parser().parse_args()
    cfg = {
        "lang": args.lang,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "fallback_models": args.fallback_models,
        "identify": not args.no_identify,
        "history": args.history,
    }
    try:
        turn = analyze_turn(args.audio, cfg)
    except FileNotFoundError:
        print(f"error: file not found: {args.audio}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 1)
    if args.hear:
        try:
            attach_hearing(turn, args.audio, args.hear_baseline)
        except SystemExit as e:
            return int(e.code or 1)
        except FileNotFoundError as e:
            print(f"error: file not found: {e}", file=sys.stderr)
            return 2
    if args.history and "error" not in turn:
        _append_history(args.history, turn)  # never feed blank turns back
    print(json.dumps(turn, ensure_ascii=False,
                     indent=2 if args.pretty else None))
    return 5 if "error" in turn else 0


if __name__ == "__main__":
    sys.exit(main())