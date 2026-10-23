"""Local backend: transcription + speaker diarization with no API key.

    ASR          faster-whisper (CTranslate2, CPU int8), word timestamps
    diarization  Silero VAD (bundled with faster-whisper) -> 1.0 s windows
                 (0.5 s hop) -> WeSpeaker ResNet34-LM speaker embeddings (ONNX,
                 the same public model voiceid.py uses, no HF token) ->
                 spectral clustering with automatic speaker count (eigengap),
                 or a fixed count with --num-speakers
    assignment   each sentence gets the speaker that overlaps its words most;
                 turns break where the speaker changes

Install: pip install "whispero[local]"   (faster-whisper, onnxruntime,
huggingface_hub, numpy, scipy). Models download once to ~/.cache/huggingface.
"""
from __future__ import annotations

import os
import sys
import time

SR = 16000
EMB_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"
EMB_FILE = "onnx/model.onnx"
DEFAULT_LOCAL_MODEL = os.environ.get("WHISPERO_LOCAL_MODEL", "large-v3-turbo")


def _need_deps() -> None:
    missing = []
    for mod in ("faster_whisper", "onnxruntime", "huggingface_hub", "numpy",
                "scipy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"error: --provider local needs {', '.join(missing)}:\n"
              '  pip install "whispero[local]"', file=sys.stderr)
        sys.exit(4)


def default_threads() -> int:
    """Physical cores (capped at 8): SMT siblings don't speed up CTranslate2."""
    import glob
    cores = set()
    for f in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/"
                       "thread_siblings_list"):
        try:
            with open(f) as fh:
                cores.add(fh.read().strip())
        except OSError:
            pass
    n = len(cores) or os.cpu_count() or 4
    return max(1, min(n, 8))


# ---------------- ASR ----------------

def transcribe(path: str, lang: str = "pt", model: str | None = None,
               threads: int | None = None, batch_size: int = 8,
               beam_size: int = 5):
    """Returns (wav16k, segments) with per-word timestamps."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio
    model = model or DEFAULT_LOCAL_MODEL
    threads = threads or default_threads()
    t0 = time.time()
    try:  # cached copy first: no network round trip, works offline
        wm = WhisperModel(model, device="cpu", compute_type="int8",
                          cpu_threads=threads, local_files_only=True)
    except Exception:
        print(f"info: downloading {model} (once)...", file=sys.stderr)
        wm = WhisperModel(model, device="cpu", compute_type="int8",
                          cpu_threads=threads)
    wav = decode_audio(path, sampling_rate=SR)
    kw = dict(language=lang.split("-")[0] if lang else None,
              word_timestamps=True, beam_size=beam_size,
              condition_on_previous_text=False, vad_filter=True)
    if batch_size and batch_size > 1:
        segs, _ = BatchedInferencePipeline(model=wm).transcribe(
            wav, batch_size=batch_size, **kw)
    else:
        segs, _ = wm.transcribe(wav, **kw)
    out = [{"start": s.start, "end": s.end, "text": s.text.strip(),
            "words": [{"start": w.start, "end": w.end, "word": w.word.strip()}
                      for w in (s.words or [])]} for s in segs]
    dt = time.time() - t0
    print(f"info: local ASR {model} ({threads} threads): {dt:.1f}s for "
          f"{len(wav) / SR:.1f}s of audio (RTF {dt / max(1e-9, len(wav) / SR):.2f})",
          file=sys.stderr)
    return wav, out


# ---------------- speaker embeddings ----------------

_sess = None
_fb = None


def _fbank(wav):
    """Kaldi-compatible 80-bin log-mel fbank (25/10 ms, povey window,
    pre-emphasis 0.97), mean-normalized: the input WeSpeaker was trained on."""
    import numpy as np
    global _fb
    x = wav.astype(np.float64) * 32768.0
    fl, fs, nfft = 400, 160, 512
    if len(x) < fl:
        x = np.pad(x, (0, fl - len(x)))
    n = 1 + (len(x) - fl) // fs
    fr = x[np.arange(fl)[None, :] + fs * np.arange(n)[:, None]]
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.concatenate([fr[:, :1] * 0.03, fr[:, 1:] - 0.97 * fr[:, :-1]],
                        axis=1)
    fr *= (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fl) / (fl - 1))) ** 0.85
    spec = np.abs(np.fft.rfft(fr, nfft)) ** 2
    if _fb is None:
        def mel(f):
            return 1127.0 * np.log(1 + f / 700.0)
        pts = np.linspace(mel(20.0), mel(SR / 2), 82)
        fmel = mel(np.arange(nfft // 2 + 1) * SR / nfft)
        _fb = np.stack([np.maximum(0, np.minimum(
            (fmel - pts[m]) / (pts[m + 1] - pts[m]),
            (pts[m + 2] - fmel) / (pts[m + 2] - pts[m + 1])))
            for m in range(80)])
    f = np.log(np.maximum(spec @ _fb.T, np.finfo(np.float32).eps))
    return (f - f.mean(axis=0, keepdims=True)).astype(np.float32)


def embed(wav, threads: int | None = None):
    import numpy as np
    global _sess
    if _sess is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or default_threads()
        try:
            path = hf_hub_download(EMB_REPO, EMB_FILE, local_files_only=True)
        except Exception:
            path = hf_hub_download(EMB_REPO, EMB_FILE)
        _sess = ort.InferenceSession(path, so,
                                     providers=["CPUExecutionProvider"])
    e = _sess.run(None, {"input_features": _fbank(wav)[None]})[0][0]
    return e / (np.linalg.norm(e) + 1e-9)


# ---------------- clustering ----------------

def spectral_cluster(E, k: int | None = None, max_k: int = 10,
                     min_size: int = 3):
    """Spectral clustering on cosine affinity (p-neighbour pruning, as in
    NeMo's NME-SC). k=None picks the speaker count by the largest eigengap,
    choosing the pruning level whose gap is clearest. Returns labels 0..k-1."""
    import numpy as np
    from scipy.cluster.vq import kmeans2
    n = len(E)
    if n < 2 * min_size:
        return np.zeros(n, dtype=int)
    S = np.clip(E @ E.T, -1, 1)
    best = None
    for frac in (0.05, 0.08, 0.12, 0.18, 0.25):
        p = max(2, int(n * frac))
        A = S.copy()
        drop = np.argsort(-A, axis=1)[:, p:]
        np.put_along_axis(A, drop, 0.0, axis=1)
        A = np.maximum(np.maximum(A, A.T), 0)
        d = A.sum(1)
        L = np.eye(n) - A / np.sqrt(np.outer(d, d) + 1e-12)
        w, V = np.linalg.eigh(L)
        gaps = np.diff(w[:max_k + 1])
        kk = k or int(np.argmax(gaps[1:]) + 2)
        ratio = (p / n) / (gaps[kk - 1] + 1e-9)
        if best is None or ratio < best[0]:
            best = (ratio, kk, V[:, :kk])
    _, kk, V = best
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    runs = []
    for seed in range(10):
        c, lab = kmeans2(V, kk, minit="++", seed=seed)
        runs.append((((V - c[lab]) ** 2).sum(), lab))
    lab = min(runs, key=lambda r: r[0])[1]
    # clusters with < min_size windows join the nearest big cluster
    sizes = np.bincount(lab, minlength=kk)
    big = [c for c in range(kk) if sizes[c] >= min_size]
    if big and len(big) < kk:
        cent = {c: E[lab == c].mean(0) for c in big}
        for i in np.where(~np.isin(lab, big))[0]:
            lab[i] = max(big, key=lambda c: E[i] @ cent[c]
                         / np.linalg.norm(cent[c]))
    return np.unique(lab, return_inverse=True)[1]


# ---------------- diarization ----------------

def diarize(wav, num_speakers: int | None = None, win: float = 1.0,
            hop: float = 0.5, threads: int | None = None):
    """Returns [(start_s, end_s, speaker_index)] over speech regions."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    vad = get_speech_timestamps(wav, VadOptions(
        threshold=0.5, min_silence_duration_ms=250, speech_pad_ms=60))
    regions = [(v["start"] / SR, v["end"] / SR) for v in vad
               if v["end"] - v["start"] >= 0.3 * SR]
    wins = []
    for a, b in regions:
        t = a
        while t + win < b:
            wins.append((t, t + win))
            t += hop
        wins.append((max(a, b - win), b))
    if not wins:
        return []
    E = np.stack([embed(wav[int(a * SR):int(b * SR)], threads)
                  for a, b in wins])
    lab = spectral_cluster(E, k=num_speakers)
    # 10 ms frame vote: overlapping windows, triangular weight, speech only
    T = int(np.ceil(len(wav) / SR * 100)) + 1
    votes = np.zeros((T, lab.max() + 1))
    for (a, b), c in zip(wins, lab):
        i0, i1 = int(a * 100), int(b * 100)
        mid, half = (a + b) / 2, (b - a) / 2 + 1e-9
        votes[i0:i1, c] += 1 - np.abs(np.arange(i0, i1) / 100 - mid) / half
    speech = np.zeros(T, bool)
    for a, b in regions:
        speech[int(a * 100):int(b * 100)] = True
    frame = np.where(speech & (votes.sum(1) > 0), votes.argmax(1), -1)
    segs, start = [], 0
    for i in range(1, T + 1):
        if i == T or frame[i] != frame[start]:
            if frame[start] >= 0:
                segs.append((start / 100, i / 100, int(frame[start])))
            start = i
    return segs


def _speaker_at(a: float, b: float, dsegs) -> int | None:
    ov: dict[int, float] = {}
    for s, e, c in dsegs:
        o = min(b, e) - max(a, s)
        if o > 0:
            ov[c] = ov.get(c, 0.0) + o
    if ov:
        return max(ov, key=ov.get)
    if not dsegs:
        return None
    mid = (a + b) / 2
    return min(dsegs, key=lambda d: min(abs(mid - d[0]), abs(mid - d[1])))[2]


def _sentences(words, max_pause: float = 0.3):
    """Split words at sentence punctuation or pauses longer than max_pause."""
    out, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > max_pause:
            out.append(cur)
            cur = []
        cur.append(w)
        if w["word"].endswith((".", "?", "!", "…")):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def assign_speakers(asr_segments, dsegs, label: str = "FALANTE",
                    max_gap: float = 2.0) -> list[dict]:
    """Speaker per sentence (duration-weighted vote of its words) -> turns
    {speaker, start, end, text}. Voting per sentence instead of per word
    stops a sentence from flipping speaker on a blurry window boundary.
    Speakers are numbered by first appearance (FALANTE 1, 2, ...)."""
    words = [w for seg in asr_segments
             for w in (seg["words"] or [{"start": seg["start"],
                                         "end": seg["end"],
                                         "word": seg["text"]}])]
    names: dict[int, str] = {}
    turns: list[dict] = []
    for sent in _sentences(words):
        votes: dict = {}
        for w in sent:
            c = _speaker_at(w["start"], w["end"], dsegs)
            votes[c] = votes.get(c, 0.0) + max(w["end"] - w["start"], 0.01)
        spk = names.setdefault(max(votes, key=votes.get),
                               f"{label} {len(names) + 1}")
        text = " ".join(w["word"] for w in sent)
        if (turns and turns[-1]["speaker"] == spk
                and sent[0]["start"] - turns[-1]["end"] <= max_gap):
            turns[-1]["end"] = sent[-1]["end"]
            turns[-1]["text"] += " " + text
        else:
            turns.append({"speaker": spk, "start": sent[0]["start"],
                          "end": sent[-1]["end"], "text": text})
    for t in turns:
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
        t["text"] = t["text"].strip()
    return turns


def run(path: str, lang: str = "pt", model: str | None = None,
        diarize_speakers: bool = False, num_speakers: int | None = None,
        threads: int | None = None, batch_size: int = 8,
        beam_size: int = 5):
    """Transcribe (and optionally diarize) locally.
    Returns (plain_text, turns or None)."""
    _need_deps()
    wav, segs = transcribe(path, lang, model, threads, batch_size, beam_size)
    text = " ".join(s["text"] for s in segs).strip()
    if not diarize_speakers:
        return text, None
    t0 = time.time()
    dsegs = diarize(wav, num_speakers=num_speakers, threads=threads)
    turns = assign_speakers(segs, dsegs)
    print(f"info: local diarization: {len({t['speaker'] for t in turns})} "
          f"speakers in {time.time() - t0:.1f}s", file=sys.stderr)
    return text, turns
