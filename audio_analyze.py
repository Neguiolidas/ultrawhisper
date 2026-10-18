#!/usr/bin/env python3
"""Whispero - deep audio analysis with any OpenAI-compatible provider.

Usage:
    audio_analyze.py <audio_file_or_url> [--provider gemini]
    [--model gemini-2.5-flash] [--base-url URL] [--api-key KEY]
    [--fallback-models m1,m2] [--compress]

Providers:
    gemini  - Google AI Studio, native audio input, YouTube URLs supported
              (default, key via GEMINI_API_KEY or --api-key)
    openai  - Any OpenAI-compatible API: transcribes via
              /v1/audio/transcriptions then analyzes the text via
              /v1/chat/completions (key via OPENAI_API_KEY or --api-key,
              URL via --base-url)

Prints JSON with: transcript, speakers, tone, emotions, emotional_arc,
urgency, intent, action_items, speaker_notes, key_moments,
irony_detected, sarcasm_detected.

Zero pip dependencies (stdlib + ffmpeg).
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess  # nosec B404
import sys
import tempfile
import time
import urllib.request
import urllib.error

DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "whisper-1"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_API_BASE = "https://api.openai.com/v1"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]

ANALYZE_PROMPT = (
    "Analise este áudio em profundidade e retorne um JSON com os seguintes campos:\n"
    "- transcript: transcrição fiel em português, com marcação de falante se houver mais de um ([FALANTE 1], [FALANTE 2])\n"
    "- speakers: lista de falantes detectados com características de voz de cada um\n"
    "- tone: tom geral (ex: calmo, tenso, animado, irritado, sarcástico)\n"
    "- emotions: lista das emoções detectadas com intensidade (0-1)\n"
    "- emotional_arc: como as emoções mudam ao longo do áudio (início/meio/fim)\n"
    "- urgency: nível de urgência 0-10 e justificativa curta\n"
    "- intent: o que o falante quer (pedido, desabafo, informação, decisão, etc.)\n"
    "- action_items: lista de ações ou compromissos mencionados\n"
    "- speaker_notes: características de voz (ritmo, pausas, ênfases, sotaque)\n"
    "- key_moments: lista de momentos importantes com timestamp aproximado e descrição\n"
    "- irony_detected: true/false com explicação se houver\n"
    "- sarcasm_detected: true/false com explicação se houver\n"
    "Retorne APENAS o JSON, sem markdown nem comentários."
)

ANALYZE_TEXT_PROMPT = (
    "Analise a transcrição abaixo em profundidade e retorne um JSON com os "
    "seguintes campos:\n"
    "- transcript: a transcrição recebida, com marcação de falante se houver "
    "mais de um ([FALANTE 1], [FALANTE 2])\n"
    "- speakers: lista de falantes detectados (se identificável pelo texto)\n"
    "- tone: tom geral (ex: calmo, tenso, animado, irritado, sarcástico)\n"
    "- emotions: lista das emoções detectadas com intensidade (0-1)\n"
    "- emotional_arc: como as emoções mudam ao longo do texto (início/meio/fim)\n"
    "- urgency: nível de urgência 0-10 e justificativa curta\n"
    "- intent: o que o falante quer (pedido, desabafo, informação, decisão, etc.)\n"
    "- action_items: lista de ações ou compromissos mencionados\n"
    "- key_moments: lista de momentos importantes com descrição\n"
    "- irony_detected: true/false com explicação se houver\n"
    "- sarcasm_detected: true/false com explicação se houver\n"
    "Retorne APENAS o JSON, sem markdown nem comentários.\n\nTranscrição:\n"
)

DIARIZE_INSTRUCTION = (
    "\nDIARIZAÇÃO ATIVA: identifique os diferentes falantes pela voz. "
    "Para cada trecho de fala, use exatamente este formato no campo diarization:\n"
    '{"speaker": "FALANTE 1", "start": "00:00", "end": "00:15", "text": "texto do que foi dito"}\n'
    "Regras:\n"
    "- Atribua o mesmo número ao mesmo falante durante todo o áudio\n"
    "- O timestamp é o momento aproximado em que o trecho começa/termina (MM:SS)\n"
    "- Se houver apenas um falante, use FALANTE 1 em todos os trechos\n"
    "- Não invente falantes: só crie um novo número quando a voz mudar claramente\n"
    "- Atenção: falantes do mesmo gênero (ex.: dois homens) têm vozes diferentes "
    "— preste atenção no timbre, ritmo e tom, não só no gênero\n"
    '- Inclua no JSON o campo "diarization": lista de {"speaker": "FALANTE N", '
    '"start": "MM:SS", "end": "MM:SS", "text": "trecho transcrito"} '
    "em ordem temporal, cobrindo todo o áudio"
)


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def get_api_key(args) -> str:
    for src in (args.api_key, os.environ.get("WHISPERO_API_KEY"),
                os.environ.get("GEMINI_API_KEY"),
                os.environ.get("OPENAI_API_KEY")):
        if src and src.strip():
            return src.strip()
    print(
        "error: no API key. Use --api-key or set WHISPERO_API_KEY "
        "(or GEMINI_API_KEY / OPENAI_API_KEY).",
        file=sys.stderr,
    )
    sys.exit(3)


def compress_audio(src: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()
    try:
        subprocess.run(  # nosec B607 B603
            ["ffmpeg", "-y", "-v", "error", "-i", src,
             "-c:a", "libopus", "-b:a", "32k", "-ar", "16000", "-ac", "1",
             tmp.name],
            check=True, timeout=120,
        )
        return tmp.name
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        os.unlink(tmp.name)
        return src


def http_post_json(url: str, headers: dict, payload: dict,
                   timeout: int = 180) -> dict | None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                last_err = e
            else:
                body = e.read().decode(errors="replace")[:300]
                print(f"error: API {e.code}: {body}", file=sys.stderr)
                return None
        except Exception as e:
            last_err = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    print(f"warn: failed after {MAX_RETRIES} tries: {last_err}", file=sys.stderr)
    return None


def extract_json(data: dict) -> dict | None:
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return None
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# ---------------- Gemini provider ----------------

def gemini_analyze(base: str, key: str, model: str, media_part: dict) -> dict | None:
    url = f"{base}/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": ANALYZE_PROMPT}, media_part]}],
        "generationConfig": {"temperature": 0.2},
    }
    data = http_post_json(url, {"Content-Type": "application/json"}, payload)
    return extract_json(data) if data else None


def _parse_diarize_segments(text: str) -> list[dict]:
    """Parse [FALANTE N] [MM:SS-MM:SS] text into segments."""
    import re
    segments = []
    for line in text.strip().split("\n"):
        m = re.match(
            r"\[FALANTE (\d+)\]\s*\[(\d+:\d+)-(\d+:\d+)\]\s*(.*)",
            line.strip(),
        )
        if m:
            segments.append({
                "speaker": f"FALANTE {m.group(1)}",
                "start": m.group(2),
                "end": m.group(3),
                "text": m.group(4).strip(),
            })
    return segments


def _diarize_agreement(segs_a: list[dict], segs_b: list[dict]) -> float:
    """Agreement score between two diarization runs (0-1)."""
    if not segs_a or not segs_b:
        return 0.0
    from collections import Counter
    mapping = Counter()
    for sa in segs_a:
        words_a = set(sa["text"].lower().split())
        if not words_a:
            continue
        best_b, best_overlap = None, 0
        for sb in segs_b:
            words_b = set(sb["text"].lower().split())
            overlap = len(words_a & words_b) / max(len(words_a), 1)
            if overlap > best_overlap:
                best_overlap, best_b = overlap, sb
        if best_b and best_overlap > 0.5:
            mapping[(sa["speaker"], best_b["speaker"])] += 1
    label_map, used_b = {}, set()
    for (la, lb), _ in mapping.most_common():
        if la not in label_map and lb not in used_b:
            label_map[la] = lb
            used_b.add(lb)
    agree, total = 0, 0
    for sa in segs_a:
        words_a = set(sa["text"].lower().split())
        if not words_a:
            continue
        total += 1
        for sb in segs_b:
            words_b = set(sb["text"].lower().split())
            if len(words_a & words_b) / max(len(words_a), 1) > 0.5:
                if label_map.get(sa["speaker"]) == sb["speaker"]:
                    agree += 1
                break
    return agree / max(total, 1)


def gemini_diarize(base: str, key: str, model: str, media_part: dict,
                  lang: str = "pt", rounds: int = 3) -> list | None:
    """Diarization in separate call with consensus voting.

    Runs N times, picks the run with highest consensus agreement.
    Returns [{"speaker", "start", "end", "text"}] or None.
    """
    url = f"{base}/{model}:generateContent?key={key}"
    prompt = (
        f"Transcreva este áudio de forma fiel em {lang}, identificando os "
        "diferentes falantes pela voz.\n"
        "Para cada trecho de fala, use exatamente este formato:\n"
        "[FALANTE 1] [00:00-00:05] texto do que foi dito\n"
        "[FALANTE 2] [00:06-00:12] texto do que foi dito\n"
        "Regras:\n"
        "- Atribua o mesmo número ao mesmo falante durante todo o áudio\n"
        "- Os timestamps são início e fim aproximados de cada trecho (MM:SS)\n"
        "- Se houver apenas um falante, use [FALANTE 1] em todos os trechos\n"
        "- Não invente falantes: só crie um novo número quando a voz mudar claramente\n"
        "- Atenção: falantes do mesmo gênero (ex.: dois homens) têm vozes diferentes "
        "— preste atenção no timbre, ritmo e tom, não só no gênero\n"
        "- Retorne apenas a transcrição com as marcações, sem comentários."
    )
    runs: list[list[dict]] = []
    for i in range(rounds):
        payload = {
            "contents": [{"parts": [{"text": prompt}, media_part]}],
            "generationConfig": {"temperature": 0.2},
        }
        data = http_post_json(url, {"Content-Type": "application/json"}, payload)
        if not data:
            continue
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue
        segments = _parse_diarize_segments(text)
        n_speakers = len(set(s["speaker"] for s in segments))
        print(f"diarize run {i + 1}/{rounds}: {n_speakers} speakers, "
              f"{len(segments)} segments", file=sys.stderr)
        if segments:
            runs.append(segments)

    if not runs:
        return None
    if len(runs) == 1:
        return runs[0]

    best_idx, best_score = 0, -1.0
    for i in range(len(runs)):
        total = sum(_diarize_agreement(runs[i], runs[j])
                    for j in range(len(runs)) if i != j)
        avg = total / (len(runs) - 1)
        n_spk = len(set(s["speaker"] for s in runs[i]))
        print(f"run {i + 1} consensus score: {avg:.2f}", file=sys.stderr)
        if (avg, n_spk) > (best_score, 0):
            best_score, best_idx = avg, i

    print(f"selected run {best_idx + 1} (consensus {best_score:.2f})",
          file=sys.stderr)
    if best_score < 0.7:
        print("warn: low consensus between runs, result may be unstable",
              file=sys.stderr)
    return runs[best_idx]


# ---------------- OpenAI-compatible provider ----------------

def _mmss_to_s(ts: str) -> int:
    mm, ss = ts.split(":")[-2:]
    return int(mm) * 60 + int(float(ss))


def shared_diarize(base: str, key: str, model: str, media_part: dict,
                   lang: str = "pt", rounds: int = 3,
                   audio_path: str | None = None) -> list | None:
    """Same diarization path as `whispero --diarize` (transcribe.py).

    Reuses the segment-level consensus vote with the simpler transcribe
    prompt, which separates same-gender speakers reliably; the old
    analyze-only prompt collapsed two male voices into one in every run.
    Returns [{"speaker", "start", "end", "text"}] (end = next start).
    """
    try:
        from transcribe import gemini_diarize_vote_segments, pitch_split
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from transcribe import gemini_diarize_vote_segments, pitch_split
    segs = gemini_diarize_vote_segments(base, key, model, media_part,
                                        lang=lang, rounds=rounds)
    if not segs:
        return None
    if audio_path:
        segs = pitch_split(segs, audio_path)
    out = []
    for i, s in enumerate(segs):
        end = segs[i + 1]["start"] if i + 1 < len(segs) else None
        if end is not None and _mmss_to_s(end) < _mmss_to_s(s["start"]):
            end = s["start"]
        out.append({"speaker": s["speaker"], "start": s["start"],
                    "end": end or s["start"], "text": s["text"]})
    return out


def openai_transcribe(base: str, key: str, model: str,
                      audio_path: str) -> str | None:
    import uuid
    boundary = uuid.uuid4().hex
    url = f"{base.rstrip('/')}/audio/transcriptions"
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()
    filename = os.path.basename(audio_path)
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "application/octet-stream"
    parts = []
    for name, value in (("model", model), ("response_format", "text")):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f'\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
        + audio_bytes + b'\r\n'
    )
    parts.append(f'--{boundary}--\r\n'.encode())
    req = urllib.request.Request(
        url, data=b"".join(parts),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # nosec B310
                return resp.read().decode().strip() or None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                last_err = e
            else:
                print(f"error: API {e.code}: {e.read().decode()[:300]}",
                      file=sys.stderr)
                return None
        except Exception as e:
            last_err = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    print(f"warn: failed after {MAX_RETRIES} tries: {last_err}", file=sys.stderr)
    return None


def openai_analyze_text(base: str, key: str, model: str,
                        transcript: str) -> dict | None:
    url = f"{base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": ANALYZE_TEXT_PROMPT + transcript}],
        "temperature": 0.2,
    }
    data = http_post_json(
        url, {"Content-Type": "application/json",
              "Authorization": f"Bearer {key}"},
        payload,
    )
    return extract_json(data) if data else None


# ---------------- main ----------------

def main() -> int:
    p = argparse.ArgumentParser(description="Whispero deep audio analysis.")
    p.add_argument("source", help="Audio file path or YouTube URL (gemini only)")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=["gemini", "openai"])
    p.add_argument("--model", default=None,
                   help="Transcription model (default: gemini-2.5-flash / whisper-1)")
    p.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL,
                   help="Chat model for text analysis step (openai provider)")
    p.add_argument("--base-url", default=None,
                   help="Custom API base URL for OpenAI-compatible providers")
    p.add_argument("--api-key", default=None, help="API key (or env var)")
    p.add_argument("--fallback-models", default="",
                   help="Comma-separated fallback models to try in order")
    p.add_argument("--compress", action="store_true",
                   help="Compress to opus 32k before upload (local files only)")
    p.add_argument("--diarize", action="store_true",
                   help="Identify speakers ([FALANTE 1], [FALANTE 2], ...) "
                        "with timestamps; adds a diarization section to "
                        "the JSON output")
    p.add_argument("--no-pitch-check", action="store_true",
                   help="Disable the pitch check that splits two same-gender "
                        "voices merged into one label (needs numpy)")
    args = p.parse_args()

    key = get_api_key(args)
    fallbacks = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    if args.provider == "gemini":
        model = args.model or DEFAULT_GEMINI_MODEL
        models = [model] + [m for m in fallbacks if m != model]
        base = GEMINI_API_BASE
        tmp_path = None
        if is_url(args.source):
            media_part = {"file_data": {"file_uri": args.source}}
        else:
            if not os.path.isfile(args.source):
                print(f"error: file not found: {args.source}", file=sys.stderr)
                return 2
            src = args.source
            if args.compress:
                tmp_path = compress_audio(src)
                if tmp_path != src:
                    src = tmp_path
            try:
                mime, _ = mimetypes.guess_type(src)
                mime = mime or "audio/ogg"
                with open(src, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                media_part = {"inline_data": {"mime_type": mime, "data": b64}}
            finally:
                if tmp_path and tmp_path != args.source:
                    os.unlink(tmp_path)
        for m in models:
            result = gemini_analyze(base, key, m, media_part)
            if result:
                if args.diarize:
                    print("diarizing in separate call...",
                          file=sys.stderr)
                    dz = shared_diarize(
                        base, key, m, media_part,
                        audio_path=None if (is_url(args.source)
                                            or args.no_pitch_check)
                        else args.source)
                    if dz:
                        result["diarization"] = dz
                    else:
                        print("warn: diarization returned nothing",
                              file=sys.stderr)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            print(f"warn: no usable result from {m}, trying next...",
                  file=sys.stderr)
    else:  # openai-compatible: transcribe then analyze text
        if is_url(args.source):
            print("error: URL input only supported with --provider gemini",
                  file=sys.stderr)
            return 2
        if not os.path.isfile(args.source):
            print(f"error: file not found: {args.source}", file=sys.stderr)
            return 2
        model = args.model or DEFAULT_OPENAI_MODEL
        models = [model] + [m for m in fallbacks if m != model]
        base = args.base_url or OPENAI_API_BASE
        src = args.source
        tmp_path = None
        if args.compress:
            tmp_path = compress_audio(src)
            if tmp_path != src:
                src = tmp_path
        try:
            transcript = None
            for m in models:
                transcript = openai_transcribe(base, key, m, src)
                if transcript:
                    break
                print(f"warn: transcription failed on {m}, trying next...",
                      file=sys.stderr)
            if not transcript:
                print("error: transcription failed on all models",
                      file=sys.stderr)
                return 5
            result = openai_analyze_text(base, key, args.chat_model, transcript)
            if result:
                result.setdefault("transcript", transcript)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
        finally:
            if tmp_path and tmp_path != args.source:
                os.unlink(tmp_path)

    print("error: no usable analysis from any model", file=sys.stderr)
    return 5


if __name__ == "__main__":
    sys.exit(main())
