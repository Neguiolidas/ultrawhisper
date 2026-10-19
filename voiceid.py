#!/usr/bin/env python3
"""Whispero Voice ID - speaker enrollment and verification.

Lightweight (ONNX, no torch): extracts a 256-dim voiceprint with a
Wespeaker ResNet34 model and compares it with cosine similarity.

Usage:
    whispero-enroll <audio> --name "Neguiolidas"   # enroll ~10s of clean speech
    whispero-enroll <audio> --name X --list         # list enrolled voices

As a module:
    from voiceid import enroll, verify, identify
    enroll("/path/to/sample.wav", "Neguiolidas")
    match, score = verify("/path/to/turn.wav", "Neguiolidas")
    name, score = identify("/path/to/turn.wav")  # "Neguiolidas" | "unknown"

Install: pip install "whispero[voiceid]"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
import sys
import tempfile

import numpy as np

SR = 16000
MODEL_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "onnx/model.onnx"
VOICEPRINTS_PATH = os.path.join(os.path.expanduser("~"), ".whispero",
                                "voiceprints.json")
DEFAULT_THRESHOLD = 0.5

_sess = None


def _need_deps() -> None:
    try:
        import onnxruntime  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        print('error: voiceid needs: pip install "whispero[voiceid]"',
              file=sys.stderr)
        sys.exit(4)


def _get_session():
    global _sess
    if _sess is None:
        _need_deps()
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        _sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sess


def load_wav_16k(path: str) -> np.ndarray:
    raw = subprocess.run(  # nosec B603
        ["ffmpeg", "-v", "error", "-i", path, "-ar", str(SR), "-ac", "1",
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _mel_filterbank(n_fft: int = 512, n_mels: int = 80) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bins[m - 1], bins[m], bins[m + 1]
        if f1 > f0:
            fb[m - 1, f0:f1] = (np.arange(f0, f1) - f0) / (f1 - f0)
        if f2 > f1:
            fb[m - 1, f1:f2] = (f2 - np.arange(f1, f2)) / (f2 - f1)
    return fb


_FB = None


def fbank(wav: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Log-mel filterbank [T, 80], 25ms window / 10ms shift, mean norm."""
    global _FB
    if _FB is None:
        _FB = _mel_filterbank(512, n_mels)
    win_len, hop = int(0.025 * SR), int(0.010 * SR)
    window = np.hamming(win_len)
    specs = []
    for start in range(0, len(wav) - win_len + 1, hop):
        seg = wav[start:start + win_len] * window
        specs.append(np.abs(np.fft.rfft(seg, 512)) ** 2)
    if not specs:
        return np.zeros((1, n_mels), dtype=np.float32)
    mel = np.log(np.maximum(np.dot(np.stack(specs), _FB.T), 1e-10))
    mel = mel - mel.mean(axis=0, keepdims=True)
    return mel.astype(np.float32)


def embed(wav: np.ndarray) -> np.ndarray:
    """256-dim L2-normalized speaker embedding."""
    sess = _get_session()
    feat = fbank(wav)[np.newaxis, :, :]
    emb = sess.run(None, {"input_features": feat})[0][0]
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def embed_file(path: str) -> np.ndarray:
    return embed(load_wav_16k(path))


# ---------------------------------------------------------------------------
# Voiceprint store
# ---------------------------------------------------------------------------

def _load_store() -> dict:
    try:
        with open(VOICEPRINTS_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_store(d: dict) -> None:
    os.makedirs(os.path.dirname(VOICEPRINTS_PATH), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", dir=os.path.dirname(VOICEPRINTS_PATH),
        delete=False, encoding="utf-8")
    json.dump(d, tmp)
    tmp.close()
    os.replace(tmp.name, VOICEPRINTS_PATH)


def _quality_check(wav: np.ndarray, min_seconds: float = 3.0) -> str | None:
    """Returns an error string if the sample is unusable, else None."""
    dur = len(wav) / SR
    if dur < min_seconds:
        return (f"sample too short ({dur:.1f}s < {min_seconds:.0f}s): "
                "use at least 10s of clean speech")
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 0.01:
        return "sample is (near) silence: check the microphone / file"
    return None


def enroll(audio_path: str, name: str,
           min_seconds: float = 3.0) -> dict:
    """Enroll a voiceprint. Returns {"name", "duration_s", "dim"}."""
    wav = load_wav_16k(audio_path)
    err = _quality_check(wav, min_seconds)
    if err:
        raise ValueError(err)
    emb = embed(wav)
    store = _load_store()
    store[name] = {
        "embedding": emb.astype(float).tolist(),
        "duration_s": round(len(wav) / SR, 1),
        "threshold": DEFAULT_THRESHOLD,
    }
    _save_store(store)
    return {"name": name, "duration_s": round(len(wav) / SR, 1), "dim": 256}


def _get_voiceprint(name: str) -> tuple[np.ndarray, float]:
    store = _load_store()
    if name not in store:
        raise KeyError(f'no voiceprint enrolled for "{name}" '
                       f"(enrolled: {sorted(store) or 'none'})")
    vp = store[name]
    emb = np.array(vp["embedding"], dtype=np.float32)
    return emb / np.linalg.norm(emb), float(vp.get("threshold",
                                                 DEFAULT_THRESHOLD))


def verify(audio_path: str, name: str) -> tuple[bool, float]:
    """Verify audio against an enrolled voice.

    Returns (match, score) where score is cosine similarity in [-1, 1].
    """
    wav = load_wav_16k(audio_path)
    if len(wav) < int(1.0 * SR):
        return False, 0.0
    vp, threshold = _get_voiceprint(name)
    emb = embed(wav)
    score = float(np.dot(emb, vp))
    return score >= threshold, round(score, 3)


def identify(audio_path: str) -> tuple[str, float]:
    """Identify the speaker: best enrolled match or "unknown".

    Returns (name | "unknown", score).
    """
    store = _load_store()
    if not store:
        return "unknown", 0.0
    wav = load_wav_16k(audio_path)
    if len(wav) < int(1.0 * SR):
        return "unknown", 0.0
    emb = embed(wav)
    best, best_score = "unknown", -1.0
    for name, vp in store.items():
        v = np.array(vp["embedding"], dtype=np.float32)
        v = v / np.linalg.norm(v)
        score = float(np.dot(emb, v))
        if score > best_score:
            best, best_score = name, score
    threshold = float(store[best].get("threshold", DEFAULT_THRESHOLD)) \
        if best != "unknown" else DEFAULT_THRESHOLD
    if best_score < threshold:
        return "unknown", round(best_score, 3)
    return best, round(best_score, 3)


def list_voices() -> list[dict]:
    store = _load_store()
    return [{"name": n, "duration_s": v.get("duration_s"),
             "threshold": v.get("threshold", DEFAULT_THRESHOLD)}
            for n, v in store.items()]


def main() -> int:
    p = argparse.ArgumentParser(description="Whispero Voice ID enrollment.")
    p.add_argument("audio", nargs="?", help="Audio file with clean speech")
    p.add_argument("--name", default=None, help="Name for this voiceprint")
    p.add_argument("--list", action="store_true", help="List enrolled voices")
    p.add_argument("--verify", default=None, metavar="NAME",
                   help="Verify audio against an enrolled voice")
    args = p.parse_args()

    if args.list:
        voices = list_voices()
        if not voices:
            print("no voices enrolled")
        for v in voices:
            print(f"{v['name']}: {v['duration_s']}s "
                  f"(threshold {v['threshold']})")
        return 0

    if not args.audio:
        p.error("audio file required (or --list)")
    if args.verify:
        try:
            match, score = verify(args.audio, args.verify)
        except KeyError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps({"name": args.verify, "match": match,
                          "score": score}))
        return 0

    if not args.name:
        p.error("--name is required to enroll")
    try:
        info = enroll(args.audio, args.name)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
