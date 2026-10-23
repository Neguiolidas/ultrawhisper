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
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"  # 2.5-* answer 404 for new keys
DEFAULT_OPENAI_MODEL = "whisper-1"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_API_BASE = "https://api.openai.com/v1"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
MAX_RETRIES = 3
ROUND_DELAY = 2  # seconds between diarization rounds
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
    # Precedence: explicit --api-key > provider-specific env >
    # WHISPERO_API_KEY > any other known env var.
    provider = getattr(args, "provider", None)
    specific = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider)
    order = ([specific] if specific else []) + [
        "WHISPERO_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    seen: set[str] = set()
    order = [e for e in order if not (e in seen or seen.add(e))]
    cands = [getattr(args, "api_key", None)]
    cands += [os.environ.get(e) for e in order]
    for src in cands:
        if src and src.strip():
            return src.strip()
    print(
        "error: no API key. Use --api-key or set WHISPERO_API_KEY "
        "(or GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY), "
        "or run offline with --provider local.",
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
                if e.code == 429:
                    body = e.read().decode(errors="replace")
                    if "PerDay" in body or _retry_delay_s(body) > 60:
                        # daily/long quota: retrying burns time, the next
                        # model in --fallback-models is the only way out
                        print(f"warn: quota exhausted (429, retry in "
                              f"{_retry_delay_s(body):.0f}s); skipping retries",
                              file=sys.stderr)
                        return None
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


def _retry_delay_s(body: str) -> float:
    """Gemini RetryInfo.retryDelay ("44024s") from an error body, else 0."""
    import re
    m = re.search(r'"retryDelay":\s*"([0-9.]+)s"', body)
    return float(m.group(1)) if m else 0.0


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
                       lang: str = "pt", rounds: int = 3,
                       audio_path: str | None = None) -> str | None:
    """Text wrapper around gemini_diarize_vote_segments (kept for the CLI)."""
    segs = gemini_diarize_vote_segments(base, key, model, media_part,
                                        lang=lang, rounds=rounds,
                                        audio_path=audio_path)
    return _segments_to_text(segs) if segs else None


def gemini_diarize_vote_segments(base: str, key: str, model: str,
                                 media_part: dict, lang: str = "pt",
                                 rounds: int = 3,
                                 audio_path: str | None = None
                                 ) -> list[dict] | None:
    """Consensus segments; pitch check applied when audio_path is given."""
    segs = _vote_segments(base, key, model, media_part, lang, rounds)
    if segs and audio_path:
        segs = pitch_split(segs, audio_path)
    return segs


def _vote_segments(base: str, key: str, model: str, media_part: dict,
                   lang: str = "pt", rounds: int = 3) -> list[dict] | None:
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
        if i:
            time.sleep(ROUND_DELAY)  # spread calls: avoids 429 bursts
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
        return all_runs[0]

    # Reference = the most granular run: one run that merged turns
    # (seen: 14 segments vs 28) must not set the segmentation.
    all_runs.sort(key=len, reverse=True)
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
    print(f"consensus: {n_final} speakers from {len(all_runs)} runs",
          file=sys.stderr)
    return result


# ---------------- Pitch check (same-gender split) ----------------

def _decode_mono(audio_path: str, sr: int = 16000):
    """Decode to mono float32 via ffmpeg. Returns numpy array or None."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", audio_path, "-ac", "1",
             "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=300,
        ).stdout
    except Exception:
        return None
    return np.frombuffer(raw, np.float32) if raw else None


def _median_f0(x, sr: int = 16000):
    """Median pitch (Hz) of voiced 40 ms frames, autocorrelation method."""
    import numpy as np
    fr = int(0.04 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    vals = []
    for s in range(0, len(x) - fr, fr):
        f = x[s:s + fr]
        if np.sqrt((f ** 2).mean()) < 0.02:
            continue
        f = f - f.mean()
        c = np.correlate(f, f, "full")[fr - 1:]
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            vals.append(sr / k)
    return float(np.median(vals)) if len(vals) >= 3 else None


def _ts_to_s(ts: str) -> float:
    parts = ts.split(":")
    return int(parts[-2]) * 60 + float(parts[-1])


def pitch_split(segments: list[dict], audio_path: str,
                min_semitones: float = 5.0,
                min_gap: float = 1.4,
                max_semitones: float = 10.0) -> list[dict]:
    """Split a speaker label whose segments fall into two pitch groups.

    LLM diarization tends to merge two same-gender voices into one label
    (systematic, so consensus voting can't fix it). For each label, take
    the median F0 of each segment, run 2-means on log-pitch and split the
    label when the two groups sit >= min_semitones apart with an empty band
    of >= min_gap semitones between them (one expressive voice spreads
    continuously; two voices leave a hole).
    Groups more than max_semitones apart (a man and a woman under one label)
    are left alone: that mix-up comes from text alignment, not from voice.
    Needs numpy + a local file; otherwise returns segments unchanged.
    """
    au = _decode_mono(audio_path)
    if au is None or not segments:
        return segments
    import numpy as np
    sr = 16000
    total = len(au) / sr
    f0s = []
    for i, s in enumerate(segments):
        st = _ts_to_s(s["start"])
        en = _ts_to_s(segments[i + 1]["start"]) if i + 1 < len(segments) else total
        en = max(en, st + 1.0)
        f0s.append(_median_f0(au[int(st * sr):int(en * sr)], sr))
    labels = sorted(set(s["speaker"] for s in segments))
    next_n = len(labels) + 1
    out = [dict(s) for s in segments]
    for lab in labels:
        idx = [i for i, s in enumerate(segments)
               if s["speaker"] == lab and f0s[i]]
        if len(idx) < 4:
            continue
        v = np.log2(np.array([f0s[i] for i in idx])) * 12  # semitones
        vs = np.sort(v)
        n = len(vs)
        # optimal 2-group split of the sorted pitches (max between-group var)
        k = max(range(2, n - 1), key=lambda k: k * (n - k) *
                (vs[:k].mean() - vs[k:].mean()) ** 2)
        lo_g, hi_g = vs[:k], vs[k:]
        sep = hi_g.mean() - lo_g.mean()
        gap = hi_g.min() - lo_g.max()
        if sep < min_semitones or sep > max_semitones or gap < min_gap:
            continue
        thr = (lo_g.max() + hi_g.min()) / 2
        # the split-off group may be an existing speaker the model mislabeled:
        # reuse that label when its median pitch is within 3 semitones
        hi_med = float(np.median(hi_g))
        target, best_d = None, 3.0
        for other in labels:
            if other == lab:
                continue
            ov = [np.log2(f0s[i]) * 12 for i, s in enumerate(segments)
                  if s["speaker"] == other and f0s[i]]
            if ov:
                d = abs(float(np.median(ov)) - hi_med)
                if d < best_d:
                    target, best_d = other, d
        new_lab = target or f"FALANTE {next_n}"
        if not target:
            next_n += 1
        for j, i in enumerate(idx):
            if v[j] > thr:
                out[i]["speaker"] = new_lab
        print(f"pitch check: split {lab} -> {lab} + {new_lab} "
              f"({sep:.1f} semitones apart"
              f"{', merged into existing label' if target else ''})",
              file=sys.stderr)
    # renumber by first appearance
    seen: dict[str, str] = {}
    for s in out:
        if s["speaker"] not in seen:
            seen[s["speaker"]] = f"FALANTE {len(seen) + 1}"
        s["speaker"] = seen[s["speaker"]]
    return out


def _segments_to_text(segments: list[dict]) -> str:
    return "\n".join(
        f"[{s['speaker']}] [{s['start']}] {s['text']}" for s in segments
    )


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    """Segments (with 'start' as MM:SS or seconds, 'text') to SRT."""
    blocks = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        start_s = _ts_to_s(start) if isinstance(start, str) else float(start)
        # end = next segment's start, or +5s for the last one
        if seg.get("end") is not None:
            end_s = float(seg["end"])
        elif i + 1 < len(segments):
            nxt = segments[i + 1]["start"]
            end_s = _ts_to_s(nxt) if isinstance(nxt, str) else float(nxt)
        else:
            end_s = start_s + 5.0
        if end_s <= start_s:
            end_s = start_s + 2.0
        speaker = seg.get("speaker", "")
        prefix = f"[{speaker}] " if speaker else ""
        blocks.append(
            f"{i + 1}\n{_srt_ts(start_s)} --> {_srt_ts(end_s)}\n"
            f"{prefix}{seg['text']}"
        )
    return "\n\n".join(blocks) + "\n"


def emit(text: str, args, model: str,
         segments: list[dict] | None = None) -> None:
    """Output transcript to stdout or --output file, in --format."""
    if args.format == "json":
        payload = {"text": text, "model": model, "lang": args.lang,
                   "source": args.source}
        if segments is None and text and getattr(args, "diarize", False):
            segments = parse_diarize_lines(text) or None
        if segments:
            def _s(v):
                return _ts_to_s(v) if isinstance(v, str) else v
            payload["segments"] = [
                {"speaker": sg["speaker"], "start": _s(sg["start"]),
                 "end": (_s(sg["end"]) if sg.get("end") is not None
                         else _s(segments[i + 1]["start"])
                         if i + 1 < len(segments) else None),
                 "text": sg["text"]} for i, sg in enumerate(segments)]
        out = json.dumps(payload, ensure_ascii=False, indent=2)
    elif args.format == "srt" and segments:
        out = to_srt(segments)
    else:
        out = text
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"info: wrote {args.output}", file=sys.stderr)
    else:
        print(out)


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


def anthropic_chat(base: str, key: str, model: str, prompt: str,
                   system: str | None = None,
                   max_tokens: int = 1024) -> str | None:
    """Text chat via the Anthropic Messages API (or compatible endpoint).

    Note: the Messages API has no audio input, so this is text-only.
    Used for the LLM passes (cleanup, text analysis), not transcription.
    """
    url = f"{base.rstrip('/')}/messages"
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if system:
        payload["system"] = system
    data = http_post(
        url,
        {"Content-Type": "application/json",
         "x-api-key": key,
         "anthropic-version": ANTHROPIC_VERSION},
        json.dumps(payload).encode(),
    )
    if not data:
        return None
    try:
        for block in data["content"]:
            if block.get("type") == "text" and block.get("text", "").strip():
                return block["text"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return None


# ---------------- main ----------------

def main() -> int:
    # `whispero hear ...` dispatches to the hear module (lazy import:
    # plain `whispero` keeps working without the hear extra).
    if len(sys.argv) > 1 and sys.argv[1] == "hear":
        try:
            import hear as _hear
        except ImportError:
            print("error: whispero hear needs: pip install \"whispero[hear]\"",
                  file=sys.stderr)
            return 4
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return _hear.main()
    p = argparse.ArgumentParser(description="Whispero transcription.")
    p.add_argument("source", help="Audio file path or YouTube URL (gemini only)")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=["gemini", "openai", "local"],
                   help="local = faster-whisper on this machine, no API key "
                        '(pip install "whispero[local]")')
    p.add_argument("--model", default=None,
                   help="Model name (default: gemini-3.5-flash / whisper-1 / "
                        "large-v3-turbo for --provider local)")
    p.add_argument("--base-url", default=None,
                   help="Custom API base URL for OpenAI-compatible providers")
    p.add_argument("--api-key", default=None, help="API key (or env var)")
    p.add_argument("--fallback-models", default="",
                   help="Comma-separated fallback models to try in order "
                        "(or WHISPERO_FALLBACK_MODELS env)")
    p.add_argument("--prompt", default=None)
    p.add_argument("--lang", default="pt")
    p.add_argument("--compress", action="store_true",
                   help="Compress to opus 32k before upload (local files only)")
    p.add_argument("--diarize", action="store_true",
                   help="Identify speakers ([FALANTE 1], [FALANTE 2], ...) "
                        "with approximate timestamps")
    p.add_argument("--no-pitch-check", action="store_true",
                   help="Disable the pitch check that splits two same-gender "
                        "voices merged into one label (needs numpy)")
    p.add_argument("--diarize-engine", default=None,
                   choices=["llm", "pyannote", "local"],
                   help="Diarization engine: llm (prompt-based, works out of "
                        "the box), pyannote (needs pyannote.audio + HF_TOKEN) "
                        "or local (voice embeddings on this machine, no key; "
                        "needs --provider local). Default: local with "
                        "--provider local, else llm")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Local engine: fixed number of speakers "
                        "(default: estimated automatically)")
    p.add_argument("--threads", type=int, default=None,
                   help="Local provider: CPU threads (default: physical "
                        "cores, max 8)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Local provider: batched inference size, 0 = off "
                        "(default 8)")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Write transcript to FILE instead of stdout")
    p.add_argument("--format", default="txt", choices=["txt", "json", "srt"],
                   help="Output format: txt (default), json (with metadata), "
                        "srt (subtitles, needs --diarize)")
    args = p.parse_args()
    if args.diarize_engine is None:
        args.diarize_engine = "local" if args.provider == "local" else "llm"

    if args.provider == "local" or args.diarize_engine == "local":
        return _main_local(args)

    key = get_api_key(args)
    fb_arg = args.fallback_models or os.environ.get("WHISPERO_FALLBACK_MODELS", "")
    fallbacks = [m.strip() for m in fb_arg.split(",") if m.strip()]

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
                            print(f"info: chunk via {m}", file=sys.stderr)
                            break
                    if text:
                        out_lines.append(
                            f"[{seg['speaker']}] [{fmt_ts(seg['start'])}] "
                            f"{text}"
                        )
                finally:
                    os.unlink(chunk)
            if out_lines:
                emit("\n".join(out_lines), args, model)
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
                if args.format == "srt":
                    segs = gemini_diarize_vote_segments(
                        base, key, m, media_part, lang=args.lang,
                        audio_path=None if (is_url(args.source)
                                            or args.no_pitch_check)
                        else args.source)
                    if segs:
                        print(f"info: transcript via {m}", file=sys.stderr)
                        emit("", args, m, segments=segs)
                        return 0
                    text = None
                else:
                    text = gemini_diarize_vote(
                        base, key, m, media_part, lang=args.lang,
                        audio_path=None if (is_url(args.source)
                                            or args.no_pitch_check)
                        else args.source)
            else:
                text = gemini_transcribe(base, key, m, prompt, media_part)
            if text:
                print(f"info: transcript via {m}", file=sys.stderr)
                emit(text, args, m)
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
                    print(f"info: transcript via {m}", file=sys.stderr)
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
                emit(marked, args, chat_model)
                return 0
            print("warn: diarization pass failed, printing plain transcript",
                  file=sys.stderr)

        emit(text, args, model)
        return 0


def _main_local(args) -> int:
    """--provider local: everything runs on this machine (local_engine.py)."""
    if args.provider != "local":
        print("error: --diarize-engine local needs --provider local",
              file=sys.stderr)
        return 2
    if is_url(args.source) or not os.path.isfile(args.source):
        print(f"error: --provider local needs a local file: {args.source}",
              file=sys.stderr)
        return 2
    import local_engine
    model = args.model or local_engine.DEFAULT_LOCAL_MODEL
    text, turns = local_engine.run(
        args.source, lang=args.lang, model=model,
        diarize_speakers=args.diarize, num_speakers=args.num_speakers,
        threads=args.threads, batch_size=args.batch_size)
    if not text:
        print("error: empty transcript (no speech found?)", file=sys.stderr)
        return 5
    if turns:
        text = "\n".join(f"[{t['speaker']}] [{fmt_ts(t['start'])}] {t['text']}"
                         for t in turns)
    emit(text, args, f"local/{model}", segments=turns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
