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
    args = p.parse_args()

    key = get_api_key(args)
    fallbacks = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    if args.provider == "gemini":
        model = args.model or DEFAULT_GEMINI_MODEL
        models = [model] + [m for m in fallbacks if m != model]
        base = GEMINI_API_BASE
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
            text = gemini_transcribe(base, key, m, prompt, media_part)
            if text:
                print(text)
                return 0
            print(f"warn: empty response from {m}, trying next...",
                  file=sys.stderr)
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
            for m in models:
                text = openai_transcribe(base, key, m, src, args.lang)
                if text:
                    print(text)
                    return 0
                print(f"warn: empty response from {m}, trying next...",
                      file=sys.stderr)
        finally:
            if tmp_path and tmp_path != args.source:
                os.unlink(tmp_path)

    print("error: no usable transcript from any model", file=sys.stderr)
    return 5


if __name__ == "__main__":
    sys.exit(main())
