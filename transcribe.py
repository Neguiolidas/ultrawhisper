#!/usr/bin/env python3
"""Whispero - transcribe audio with any OpenAI-compatible provider.

Usage:
    transcribe.py <audio_file_or_url> [--provider gemini] [--lang pt]
    [--model gemini-2.5-flash] [--base-url URL] [--api-key KEY]
    [--fallback-models m1,m2] [--prompt "..."] [--compress]

Providers:
    gemini  - Google AI Studio, native audio input, YouTube URLs supported
              (default, key via GEMINI_API_KEY or --api-key)
    openai  - Any OpenAI-compatible API (OpenAI, Groq, Together, etc.)
              uses /v1/audio/transcriptions (Whisper-style)
              (key via OPENAI_API_KEY or --api-key, URL via --base-url)

Prints the transcript to stdout. Zero pip dependencies (stdlib + ffmpeg).
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
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_API_BASE = "https://api.openai.com/v1"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]

DIARIZE_PROMPT_TEMPLATE = (
    "Transcreva este áudio de forma fiel em {lang}, identificando os "
    "diferentes falantes pela voz.\n"
    "Para cada trecho de fala, use exatamente este formato:\n"
    "[FALANTE 1] [00:00] texto do que foi dito\n"
    "[FALANTE 2] [00:15] texto do que foi dito\n"
    "Regras:\n"
    "- Atribua o mesmo número ao mesmo falante durante todo o áudio\n"
    "- O timestamp é o momento aproximado em que o trecho começa (MM:SS)\n"
    "- Se houver apenas um falante, use [FALANTE 1] em todos os trechos\n"
    "- Não invente falantes: só crie um novo número quando a voz mudar claramente\n"
    "- Retorne apenas a transcrição com as marcações, sem comentários."
)

DIARIZE_TEXT_PROMPT = (
    "Abaixo está a transcrição de um áudio com possivelmente mais de um "
    "falante. Analise o conteúdo e identifique as trocas de falante por "
    "pistas conversacionais (perguntas e respostas, estilos diferentes, "
    "referências como 'eu' vs 'você').\n"
    "Reescreva a transcrição marcando cada trecho com [FALANTE 1], "
    "[FALANTE 2], etc. Mantenha o texto fiel ao original, só adicione as "
    "marcações.\n"
    "Se não houver pista clara de mais de um falante, use [FALANTE 1] em tudo.\n"
    "IMPORTANTE: esta é uma estimativa por texto, marque com [HEURÍSTICA] no "
    "início se a identificação for incerta.\n"
    "Retorne apenas a transcrição marcada, sem comentários.\n\nTranscrição:\n"
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
    """Convert to low-bitrate opus for faster upload. Returns temp path."""
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


def http_post(url: str, headers: dict, data: bytes | None = None,
              timeout: int = 180) -> dict | None:
    """POST with retry on 429/5xx. Returns parsed JSON or None."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
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


# ---------------- Diarization ----------------

def pyannote_diarize(audio_path: str) -> list[dict] | None:
    """Speaker diarization via pyannote.audio (optional heavy dependency).

    Returns list of {"speaker": "FALANTE N", "start": float, "end": float}.
    Requires pyannote.audio installed and HF_TOKEN env var (the model is
    gated on HuggingFace).
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]
    except ImportError:
        print(
            "error: --diarize-engine pyannote needs pyannote.audio installed:\n"
            "  pip install 'whispero[diarize]'  (or: pip install pyannote.audio)\n"
            "and a HuggingFace token with access to the gated model:\n"
            "  export HF_TOKEN=hf_...  (accept terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1)",
            file=sys.stderr,
        )
        return None
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print(
            "error: --diarize-engine pyannote needs HF_TOKEN env var "
            "(accept terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1)",
            file=sys.stderr,
        )
        return None
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"error: pyannote diarization failed: {e}", file=sys.stderr)
        return None
    # Map pyannote speaker labels (SPEAKER_00, ...) to FALANTE N
    label_map: dict[str, str] = {}
    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in label_map:
            label_map[speaker] = f"FALANTE {len(label_map) + 1}"
        segments.append({
            "speaker": label_map[speaker],
            "start": round(turn.start, 1),
            "end": round(turn.end, 1),
        })
    return segments


def fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def extract_audio_chunk(src: str, start: float, end: float) -> str | None:
    """Extract [start, end) seconds of audio to a temp file. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()
    try:
        subprocess.run(  # nosec B607 B603
            ["ffmpeg", "-y", "-v", "error", "-ss", str(start),
             "-to", str(end), "-i", src,
             "-c:a", "libopus", "-b:a", "32k", "-ar", "16000", "-ac", "1",
             tmp.name],
            check=True, timeout=120,
        )
        return tmp.name
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return None


# ---------------- Gemini provider ----------------

def gemini_transcribe(base: str, key: str, model: str, prompt: str,
                      media_part: dict) -> str | None:
    url = f"{base}/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}, media_part]}]}
    data = http_post(url, {"Content-Type": "application/json"},
                     json.dumps(payload).encode())
    if not data:
        return None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def parse_diarize_lines(text: str) -> list[dict]:
    """Parse [FALANTE N] [MM:SS] text lines into segments."""
    import re
    segments = []
    for line in text.strip().split("\n"):
        m = re.match(
            r"\[FALANTE (\d+)\]\s*\[(\d+:\d+)\]\s*(.*)",
            line.strip(),
        )
        if m:
            segments.append({
                "speaker": f"FALANTE {m.group(1)}",
                "start": m.group(2),
                "text": m.group(3).strip(),
            })
    return segments


def _speaker_agreement(segs_a: list[dict], segs_b: list[dict]) -> float:
    """Agreement score between two diarization runs (0-1).

    Accounts for label permutation (FALANTE 1 in run A might be
    FALANTE 2 in run B). Uses text matching to align segments.
    """
    if not segs_a or not segs_b:
        return 0.0
    # Build mapping from run A labels to run B labels via text overlap
    # Simple: for each segment in A, find best text match in B
    from collections import Counter
    mapping = Counter()  # (label_a, label_b) -> count
    for sa in segs_a:
        best_b = None
        best_overlap = 0
        words_a = set(sa["text"].lower().split())
        if not words_a:
            continue
        for sb in segs_b:
            words_b = set(sb["text"].lower().split())
            overlap = len(words_a & words_b) / max(len(words_a), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_b = sb
        if best_b and best_overlap > 0.5:
            mapping[(sa["speaker"], best_b["speaker"])] += 1
    # Find best label mapping (greedy)
    label_map = {}
    used_b = set()
    for (la, lb), cnt in mapping.most_common():
        if la not in label_map and lb not in used_b:
            label_map[la] = lb
            used_b.add(lb)
    # Score: fraction of A segments whose mapped label matches B
    agree = 0
    total = 0
    for sa in segs_a:
        words_a = set(sa["text"].lower().split())
        if not words_a:
            continue
        total += 1
        for sb in segs_b:
            words_b = set(sb["text"].lower().split())
            overlap = len(words_a & words_b) / max(len(words_a), 1)
            if overlap > 0.5:
                if label_map.get(sa["speaker"]) == sb["speaker"]:
                    agree += 1
                break
    return agree / max(total, 1)


def _build_label_map(ref_segs: list[dict], run_segs: list[dict]) -> dict[str, str]:
    """Map run's speaker labels to reference run's labels via text overlap."""
    from collections import Counter
    mapping = Counter()  # (ref_label, run_label) -> count
    for rs in ref_segs:
        words_ref = set(rs["text"].lower().split())
        if not words_ref:
            continue
        for s in run_segs:
            words_s = set(s["text"].lower().split())
            if not words_s:
                continue
            overlap = len(words_ref & words_s) / max(len(words_ref), 1)
            if overlap > 0.5:
                mapping[(rs["speaker"], s["speaker"])] += 1
                break
    label_map: dict[str, str] = {}
    used_ref = set()
    for (ref_label, run_label), _ in mapping.most_common():
        if run_label not in label_map and ref_label not in used_ref:
            label_map[run_label] = ref_label
            used_ref.add(ref_label)
    return label_map


def gemini_diarize_vote(base: str, key: str, model: str, media_part: dict,
                       lang: str = "pt", rounds: int = 3) -> str | None:
    """Diarization with segment-level majority vote across N runs.

    Runs the diarization prompt N times, aligns speaker labels across runs
    via text matching, then takes majority vote per segment. More robust
    than picking a whole run: a single bad run can't corrupt the output.
    Returns the diarized text with consensus labels.
    """
    from collections import Counter
    prompt = DIARIZE_PROMPT_TEMPLATE.format(lang=lang)
    all_runs: list[list[dict]] = []
    for i in range(rounds):
        text = gemini_transcribe(base, key, model, prompt, media_part)
        if not text:
            continue
        segments = parse_diarize_lines(text)
        n_speakers = len(set(s["speaker"] for s in segments))
        print(f"diarize run {i + 1}/{rounds}: {n_speakers} speakers, "
              f"{len(segments)} segments", file=sys.stderr)
        if segments:
            all_runs.append(segments)

    if not all_runs:
        return None
    if len(all_runs) == 1:
        return _segments_to_text(all_runs[0])

    ref = all_runs[0]
    # Build label maps from each run to reference
    label_maps = [_build_label_map(ref, run) for run in all_runs]

    result = []
    for rs in ref:
        votes: Counter[str] = Counter()
        votes[rs["speaker"]] += 1  # reference votes for itself
        words_ref = set(rs["text"].lower().split())
        for run_idx, run in enumerate(all_runs[1:], 1):
            lmap = label_maps[run_idx]
            for s in run:
                words_s = set(s["text"].lower().split())
                if not words_s:
                    continue
                overlap = len(words_ref & words_s) / max(len(words_ref), 1)
                if overlap > 0.5:
                    # Map to reference label space
                    canonical = lmap.get(s["speaker"], s["speaker"])
                    votes[canonical] += 1
                    break
        winner = votes.most_common(1)[0][0]
        result.append({
            "speaker": winner,
            "start": rs["start"],
            "text": rs["text"],
        })

    # Renumber by first appearance
    seen: dict[str, str] = {}
    counter = 0
    for s in result:
        if s["speaker"] not in seen:
            counter += 1
            seen[s["speaker"]] = f"FALANTE {counter}"
        s["speaker"] = seen[s["speaker"]]

    n_final = len(set(s["speaker"] for s in result))
    print(f"consensus: {n_final} speakers from {rounds} runs",
          file=sys.stderr)
    return _segments_to_text(result)


def _segments_to_text(segments: list[dict]) -> str:
    return "\n".join(
        f"[{s['speaker']}] [{s['start']}] {s['text']}" for s in segments
    )


# ---------------- OpenAI-compatible provider ----------------

def openai_transcribe(base: str, key: str, model: str, audio_path: str,
                      lang: str) -> str | None:
    """Whisper-style /v1/audio/transcriptions with multipart upload."""
    import uuid
    boundary = uuid.uuid4().hex
    url = f"{base.rstrip('/')}/audio/transcriptions"

    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()
    filename = os.path.basename(audio_path)
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "application/octet-stream"

    parts = []
    for name, value in (("model", model), ("language", lang),
                        ("response_format", "text")):
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
    body = b"".join(parts)

    req = urllib.request.Request(
        url, data=body,
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


def openai_chat(base: str, key: str, model: str, prompt: str) -> str | None:
    """Simple chat completion for OpenAI-compatible APIs."""
    url = f"{base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    data = http_post(
        url,
        {"Content-Type": "application/json",
         "Authorization": f"Bearer {key}"},
        json.dumps(payload).encode(),
    )
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip() or None
    except (KeyError, IndexError, TypeError):
        return None


# ---------------- main ----------------

def main() -> int:
    p = argparse.ArgumentParser(description="Whispero transcription.")
    p.add_argument("source", help="Audio file path or YouTube URL (gemini only)")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=["gemini", "openai"])
    p.add_argument("--model", default=None,
                   help="Model name (default: gemini-2.5-flash / whisper-1)")
    p.add_argument("--base-url", default=None,
                   help="Custom API base URL for OpenAI-compatible providers")
    p.add_argument("--api-key", default=None, help="API key (or env var)")
    p.add_argument("--fallback-models", default="",
                   help="Comma-separated fallback models to try in order")
    p.add_argument("--prompt", default=None)
    p.add_argument("--lang", default="pt")
    p.add_argument("--compress", action="store_true",
                   help="Compress to opus 32k before upload (local files only)")
    p.add_argument("--diarize", action="store_true",
                   help="Identify speakers ([FALANTE 1], [FALANTE 2], ...) "
                        "with approximate timestamps")
    p.add_argument("--diarize-engine", default="llm",
                   choices=["llm", "pyannote"],
                   help="Diarization engine: llm (prompt-based, works out of "
                        "the box) or pyannote (precise, needs pyannote.audio "
                        "+ HF_TOKEN). Default: llm")
    args = p.parse_args()

    key = get_api_key(args)
    fallbacks = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    # pyannote engine: precise speaker segments first, then transcribe per chunk
    pyannote_segments: list[dict] | None = None
    if args.diarize and args.diarize_engine == "pyannote":
        if args.provider != "gemini" and is_url(args.source):
            print("error: --diarize-engine pyannote needs a local file",
                  file=sys.stderr)
            return 2
        if not os.path.isfile(args.source):
            print(f"error: file not found: {args.source}", file=sys.stderr)
            return 2
        print("diarizing with pyannote...", file=sys.stderr)
        pyannote_segments = pyannote_diarize(args.source)
        if not pyannote_segments:
            return 5

    if args.provider == "gemini":
        model = args.model or DEFAULT_GEMINI_MODEL
        models = [model] + [m for m in fallbacks if m != model]
        base = GEMINI_API_BASE

        # pyannote engine: transcribe each speaker chunk separately
        if pyannote_segments:
            out_lines: list[str] = []
            for seg in pyannote_segments:
                chunk = extract_audio_chunk(args.source, seg["start"],
                                            seg["end"])
                if not chunk:
                    continue
                try:
                    mime, _ = mimetypes.guess_type(chunk)
                    mime = mime or "audio/ogg"
                    with open(chunk, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("ascii")
                    media_part = {"inline_data": {"mime_type": mime,
                                                  "data": b64}}
                    chunk_prompt = (
                        f"Transcreva este trecho de áudio de forma fiel em "
                        f"{args.lang}. Retorne apenas a transcrição, sem "
                        f"comentários."
                    )
                    text: str | None = None
                    for m in models:
                        text = gemini_transcribe(base, key, m, chunk_prompt,
                                                 media_part)
                        if text:
                            break
                    if text:
                        out_lines.append(
                            f"[{seg['speaker']}] [{fmt_ts(seg['start'])}] "
                            f"{text}"
                        )
                finally:
                    os.unlink(chunk)
            if out_lines:
                print("\n".join(out_lines))
                return 0
            print("error: no usable transcript from any chunk",
                  file=sys.stderr)
            return 5

        prompt = args.prompt or (
            f"Transcreva este áudio de forma fiel em {args.lang}. "
            "Retorne apenas a transcrição, sem comentários."
            if args.lang.startswith("pt")
            else f"Transcribe this audio faithfully in {args.lang}. "
                 "Return only the transcript, no commentary."
        )
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
            if args.diarize and not args.prompt:
                # Diarization in separate call with voting (stable speaker count)
                text = gemini_diarize_vote(base, key, m, media_part,
                                           lang=args.lang)
            else:
                text = gemini_transcribe(base, key, m, prompt, media_part)
            if text:
                print(text)
                return 0
            print(f"warn: empty response from {m}, trying next...",
                  file=sys.stderr)
        print("error: no usable transcript from any model", file=sys.stderr)
        return 5
    else:  # openai-compatible
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
            text: str | None = None
            for m in models:
                text = openai_transcribe(base, key, m, src, args.lang)
                if text:
                    break
                print(f"warn: empty response from {m}, trying next...",
                      file=sys.stderr)
        finally:
            if tmp_path and tmp_path != args.source:
                os.unlink(tmp_path)

        if not text:
            print("error: no usable transcript from any model",
                  file=sys.stderr)
            return 5

        # --diarize on openai provider: best-effort speaker segmentation
        # via chat model (heuristic, text-only — no voice info available)
        if args.diarize and not pyannote_segments:
            print("warn: --diarize on openai provider is heuristic "
                  "(text-based, no voice analysis). Use --provider gemini "
                  "for voice-based diarization.", file=sys.stderr)
            chat_model = args.model if args.model != DEFAULT_OPENAI_MODEL \
                else "gpt-4o-mini"
            marked = openai_chat(base, key, chat_model,
                                 DIARIZE_TEXT_PROMPT + text)
            if marked:
                print(marked)
                return 0
            print("warn: diarization pass failed, printing plain transcript",
                  file=sys.stderr)

        print(text)
        return 0


if __name__ == "__main__":
    sys.exit(main())
