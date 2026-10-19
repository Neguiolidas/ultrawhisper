#!/usr/bin/env python3
"""Whispero Dictate - hands-free voice dictation for any app.

Hold the hotkey (default F8), speak, release: the transcription is typed
into whatever app has focus (Claude Code, terminal, browser, ...).

Usage:
    whispero-dictate [--hotkey f8] [--mode push|toggle] [--lang pt]
                     [--provider gemini|openai] [--model M] [--api-key KEY]
                     [--no-cleanup] [--hook "cmd"] [--no-tray] [--debug]
                     [--test-inject "text"]

How it works (SuperWhisper-style):
    1. Global hotkey (pynput) starts/stops recording.
    2. Audio captured at 16 kHz mono (sounddevice), saved as WAV.
    3. Transcribed with the Whispero engine (Gemini native or any
       OpenAI-compatible API) - no local model, no VRAM needed.
    4. Optional LLM cleanup: removes filler words, fixes punctuation.
    5. Text injected via clipboard + synthesized Ctrl+V, with the
       save/restore clipboard algorithm (only restores if the clipboard
       still holds our text, so slow apps and user copies are safe).

Install the extra deps once:
    pip install "whispero[dictate]"

Config file (~/.whispero/dictate.json) - CLI flags override it:
    {"hotkey": "f8", "mode": "push", "lang": "pt", "cleanup": true,
     "hook": "", "paste_delay": 1.5}
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import queue
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
import wave

# ---------------------------------------------------------------------------
# Lazy third-party imports: `import dictate` must work with zero extra deps.
# ---------------------------------------------------------------------------

_sd = None       # sounddevice
_np = None       # numpy
_pynput_kb = None
_pyperclip = None
_pystray = None
_PIL_Image = None
_PIL_Draw = None


def _need(*names: str) -> None:
    """Import third-party deps on first use, with a helpful error."""
    global _sd, _np, _pynput_kb, _pyperclip, _pystray, _PIL_Image, _PIL_Draw
    missing: list[str] = []
    for name in names:
        try:
            if name == "sounddevice" and _sd is None:
                import sounddevice as m
                _sd = m
            elif name == "numpy" and _np is None:
                import numpy as m
                _np = m
            elif name == "pynput" and _pynput_kb is None:
                from pynput import keyboard as m
                _pynput_kb = m
            elif name == "pyperclip" and _pyperclip is None:
                import pyperclip as m
                _pyperclip = m
            elif name == "pystray" and _pystray is None:
                import pystray as m
                _pystray = m
            elif name == "PIL" and _PIL_Image is None:
                from PIL import Image as I, ImageDraw as D
                _PIL_Image, _PIL_Draw = I, D
        except ImportError:
            missing.append(name)
    if missing:
        print(
            f"error: missing packages for dictation: {', '.join(missing)}. "
            'Install with: pip install "whispero[dictate]"',
            file=sys.stderr,
        )
        sys.exit(4)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".whispero", "dictate.json")

DEFAULTS = {
    "hotkey": "f8",
    "mode": "push",          # push (hold to talk) | toggle (press to start/stop)
    "lang": "pt",
    "provider": "gemini",    # transcription provider: gemini | openai
    "model": None,
    "base_url": None,
    "api_key": None,
    "cleanup": True,
    "cleanup_provider": None,  # LLM for the cleanup pass; default = provider
    "cleanup_model": None,
    "cleanup_base_url": None,
    "hook": "",
    "paste_delay": 1.5,      # seconds before restoring the clipboard
    "min_seconds": 0.4,      # shorter recordings are discarded
    "max_seconds": 120.0,    # safety cap, auto-stops
    "tray": True,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            user = json.load(fh)
        if isinstance(user, dict):
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


# ---------------------------------------------------------------------------
# Transcription + cleanup (reuses the Whispero engine)
# ---------------------------------------------------------------------------

def _import_transcribe():
    try:
        import transcribe as t
        return t
    except ImportError:
        # dev layout: dictate.py next to transcribe.py, not installed
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import transcribe as t
        return t


def transcribe_wav(wav_path: str, cfg: dict) -> str | None:
    """Transcribe a WAV file with the Whispero engine. Returns plain text."""
    t = _import_transcribe()

    class A:  # minimal namespace for get_api_key
        pass
    a = A()
    a.api_key = cfg.get("api_key")
    a.provider = cfg.get("provider") or "gemini"
    key = t.get_api_key(a)

    provider = cfg.get("provider") or "gemini"
    lang = cfg.get("lang") or "pt"
    if provider == "gemini":
        model = cfg.get("model") or t.DEFAULT_GEMINI_MODEL
        with open(wav_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        media_part = {"inline_data": {"mime_type": "audio/wav", "data": b64}}
        prompt = (
            f"Transcreva este áudio de forma fiel em {lang}. "
            "Retorne apenas a transcrição, sem comentários."
            if lang.startswith("pt") else
            f"Transcribe this audio faithfully in {lang}. "
            "Return only the transcription, no commentary."
        )
        return t.gemini_transcribe(t.GEMINI_API_BASE, key, model, prompt,
                                   media_part)
    else:
        model = cfg.get("model") or t.DEFAULT_OPENAI_MODEL
        base = cfg.get("base_url") or t.OPENAI_API_BASE
        return t.openai_transcribe(base, key, model, wav_path, lang)


CLEANUP_PROMPT = (
    "Você é um corretor de transcrição de ditado por voz. "
    "Receba a transcrição bruta abaixo e devolva APENAS o texto corrigido, "
    "sem comentários.\n"
    "Regras:\n"
    "- Remova palavras de preenchimento (ahn, ééé, tipo, né, sabe) quando não "
    "fizerem parte do sentido\n"
    "- Corrija pontuação, capitalização e parágrafos\n"
    "- NÃO mude o sentido nem adicione informação\n"
    "- Mantenha o idioma original\n\n"
    "Transcrição:\n{text}"
)


def cleanup_text(text: str, cfg: dict) -> str:
    """LLM cleanup pass. Returns original text on any failure."""
    if not text or not text.strip():
        return text
    t = _import_transcribe()

    provider = cfg.get("cleanup_provider") or cfg.get("provider") or "gemini"

    class A:
        pass
    a = A()
    a.api_key = cfg.get("api_key")
    a.provider = provider
    try:
        key = t.get_api_key(a)
    except SystemExit:
        return text

    prompt = CLEANUP_PROMPT.format(text=text.strip())
    try:
        if provider == "anthropic":
            model = cfg.get("cleanup_model") or t.DEFAULT_ANTHROPIC_MODEL
            base = cfg.get("cleanup_base_url") or t.ANTHROPIC_API_BASE
            out = t.anthropic_chat(base, key, model, prompt)
            if out:
                return out
        elif provider == "openai":
            model = cfg.get("cleanup_model") or cfg.get("model") \
                or t.DEFAULT_OPENAI_MODEL
            base = cfg.get("cleanup_base_url") or cfg.get("base_url") \
                or t.OPENAI_API_BASE
            out = t.openai_chat(base, key, model, prompt)
            if out:
                return out
        else:
            model = cfg.get("cleanup_model") or cfg.get("model") \
                or t.DEFAULT_GEMINI_MODEL
            base = t.GEMINI_API_BASE
            url = f"{base}/{model}:generateContent?key={key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.2}}
            data = t.http_post(url, {"Content-Type": "application/json"},
                               json.dumps(payload).encode())
            if data:
                out = data["candidates"][0]["content"]["parts"][0]["text"]
                return out.strip() or text
    except Exception as e:  # never break dictation on cleanup failure
        print(f"warn: cleanup failed ({e}), using raw transcript",
              file=sys.stderr)
    return text


# ---------------------------------------------------------------------------
# Clipboard + paste injection (SuperWhisper algorithm)
# ---------------------------------------------------------------------------

class Injector:
    """Inject text into the focused app via clipboard + synthesized paste.

    Algorithm (ported from SuperWhisper's ClipboardUtil):
      1. Save current clipboard content.
      2. Put our text in the clipboard.
      3. Synthesize Ctrl+V (Cmd+V on macOS).
      4. After `delay` seconds, restore the original clipboard ONLY if the
         clipboard still holds our text. This protects slow apps (which
         process the paste late) and user copies made in between.
    """

    def __init__(self, delay: float = 1.5):
        self.delay = delay
        self._platform = sys.platform

    # -- clipboard backends -------------------------------------------------
    def _copy(self, text: str) -> None:
        _need("pyperclip")
        _pyperclip.copy(text)

    def _paste_text(self) -> str:
        _need("pyperclip")
        try:
            return _pyperclip.paste()
        except Exception:
            return ""

    # -- key synthesis ------------------------------------------------------
    def _send_paste_keystroke(self) -> None:
        if self._platform == "win32":
            self._paste_windows()
        elif self._platform == "darwin":
            self._paste_macos()
        else:
            self._paste_linux()

    def _paste_windows(self) -> None:
        import ctypes
        VK_CONTROL, VK_V = 0x11, 0x56
        KEYEVENTF_KEYUP = 0x0002
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_V, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

    def _paste_macos(self) -> None:
        subprocess.run(  # nosec B607
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            check=False, timeout=10,
        )

    def _paste_linux(self) -> None:
        # xdotool on X11; on Wayland there is no reliable global paste.
        try:
            subprocess.run(["xdotool", "key", "ctrl+v"],  # nosec B607
                           check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            print("warn: xdotool not found; text left in clipboard, "
                  "paste manually with Ctrl+V", file=sys.stderr)

    # -- public API ---------------------------------------------------------
    def inject(self, text: str) -> bool:
        if not text:
            return False
        saved = self._paste_text()
        self._copy(text)
        time.sleep(0.12)  # let the clipboard settle
        self._send_paste_keystroke()

        def _restore() -> None:
            time.sleep(self.delay)
            try:
                if self._paste_text() == text:
                    self._copy(saved)
            except Exception:
                pass

        threading.Thread(target=_restore, daemon=True).start()
        return True


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def record_push_to_talk(stop_event: threading.Event,
                        chunks: list) -> None:
    """Record from the default mic until stop_event is set. Appends float32
    mono chunks to `chunks`."""
    _need("sounddevice", "numpy")

    def _cb(indata, frames, time_info, status):
        if status:
            print(f"warn: audio status {status}", file=sys.stderr)
        chunks.append(indata.copy())

    try:
        with _sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             dtype="float32", callback=_cb):
            stop_event.wait()
    except Exception as e:
        print(f"error: audio capture failed: {e}", file=sys.stderr)
        stop_event.set()


def write_wav(chunks: list, path: str) -> float:
    """Write float32 mono chunks to 16-bit WAV. Returns duration in seconds."""
    _need("numpy")
    audio = _np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    pcm = (_np.clip(audio, -1.0, 1.0) * 32767).astype(_np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return duration


# ---------------------------------------------------------------------------
# Hotkey parsing (pynput format: "f8", "<ctrl>+<alt>+space")
# ---------------------------------------------------------------------------

def parse_hotkey(spec: str):
    """Parse a hotkey spec into a frozenset of pynput keys."""
    _need("pynput")
    kb = _pynput_kb
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    keys = set()
    for p in parts:
        name = p.strip("<>")
        if hasattr(kb.Key, name):
            keys.add(getattr(kb.Key, name))
        elif len(name) == 1:
            keys.add(kb.KeyCode.from_char(name))
        else:
            # function keys etc: f1..f24
            attr = getattr(kb.Key, name, None)
            if attr is not None:
                keys.add(attr)
            else:
                raise ValueError(f"unknown key in hotkey spec: {p!r}")
    if not keys:
        raise ValueError(f"empty hotkey spec: {spec!r}")
    return frozenset(keys)


# ---------------------------------------------------------------------------
# Platform checks
# ---------------------------------------------------------------------------

def check_wayland() -> None:
    """Fail fast on Wayland: the protocol blocks global hotkeys entirely,
    so pynput can never capture the dictation hotkey there."""
    if sys.platform != "linux":
        return
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if os.environ.get("WAYLAND_DISPLAY") or session == "wayland":
        print(
            "error: Wayland detected. The Wayland protocol blocks global "
            "hotkeys, so whispero-dictate cannot capture the hotkey in "
            "this session.\n"
            "Options:\n"
            "  - Log into an X11/Xorg session and run it there\n"
            "  - --test-inject \"text\" still works (paste needs no hotkey)",
            file=sys.stderr,
        )
        sys.exit(6)


# ---------------------------------------------------------------------------
# Dictation daemon
# ---------------------------------------------------------------------------

class Dictator:
    IDLE, RECORDING, PROCESSING = "idle", "recording", "processing"

    def __init__(self, cfg: dict, debug: bool = False):
        self.cfg = cfg
        self.debug = debug
        self.state = self.IDLE
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._chunks: list = []
        self._rec_thread: threading.Thread | None = None
        self._rec_start = 0.0
        self._cancelled = False
        self._hotkey = parse_hotkey(cfg["hotkey"])
        self._pressed: set = set()
        self._toggle_armed = True
        self.injector = Injector(delay=float(cfg["paste_delay"]))
        self._tray = None
        self._log = lambda *a: print("[dictate]", *a, file=sys.stderr,
                                     flush=True) if debug else None

    # -- state --------------------------------------------------------------
    def _set_state(self, s: str) -> None:
        with self._lock:
            self.state = s
        self._log("state ->", s)
        self._tray_update()

    # -- recording control ----------------------------------------------------
    def start_recording(self) -> None:
        with self._lock:
            if self.state != self.IDLE:
                return
            self.state = self.RECORDING
        self._log("state -> recording")
        self._tray_update()
        self._stop_event.clear()
        self._chunks = []
        self._cancelled = False
        self._rec_start = time.time()
        self._rec_thread = threading.Thread(target=record_push_to_talk,
                                            args=(self._stop_event,
                                                  self._chunks),
                                            daemon=True)
        self._rec_thread.start()
        # safety cap
        threading.Timer(float(self.cfg["max_seconds"]),
                        self._auto_stop).start()

    def _auto_stop(self) -> None:
        with self._lock:
            if self.state == self.RECORDING:
                self._log("max duration reached, auto-stopping")
                threading.Thread(target=self.stop_and_process,
                                 daemon=True).start()

    def cancel_recording(self) -> None:
        with self._lock:
            if self.state != self.RECORDING:
                return
            self._cancelled = True
        self._log("recording cancelled")
        self._stop_event.set()
        self._set_state(self.IDLE)

    def stop_and_process(self) -> None:
        with self._lock:
            if self.state != self.RECORDING:
                return
            self.state = self.PROCESSING
            cancelled = self._cancelled
        self._log("state -> processing")
        self._tray_update()
        self._stop_event.set()
        if self._rec_thread:
            self._rec_thread.join(timeout=5)
        try:
            self._process(cancelled)
        finally:
            self._set_state(self.IDLE)

    def _process(self, cancelled: bool) -> None:
        if cancelled or not self._chunks:
            self._log("discarded (cancelled or empty)")
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            duration = write_wav(self._chunks, tmp.name)
            self._log(f"recorded {duration:.1f}s")
            if duration < float(self.cfg["min_seconds"]):
                self._log("too short, discarded")
                return
            text = transcribe_wav(tmp.name, self.cfg)
            if not text or not text.strip():
                self._log("empty transcription")
                return
            self._log("transcript:", text[:80])
            if self.cfg.get("cleanup"):
                text = cleanup_text(text, self.cfg)
                self._log("cleaned:", text[:80])
            self.injector.inject(text)
            self._log("injected")
            hook = (self.cfg.get("hook") or "").strip()
            if hook:
                self._run_hook(hook, text)
        except SystemExit:
            raise
        except Exception as e:
            print(f"error: dictation pipeline failed: {e}", file=sys.stderr)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _run_hook(self, hook: str, text: str) -> None:
        """Post-record hook: runs `hook` with the text on stdin."""
        try:
            subprocess.run(hook, input=text.encode("utf-8"), shell=True,  # nosec B602
                           timeout=60)
        except Exception as e:
            print(f"warn: hook failed: {e}", file=sys.stderr)

    # -- hotkey handling ------------------------------------------------------
    def on_press(self, key) -> None:
        kb = _pynput_kb
        if key == kb.Key.esc:
            self.cancel_recording()
            return
        try:
            self._pressed.add(key)
        except TypeError:
            return
        if self._hotkey <= self._pressed:
            mode = self.cfg.get("mode", "push")
            if mode == "push":
                self.start_recording()
            else:  # toggle
                if self._toggle_armed:
                    self._toggle_armed = False
                    with self._lock:
                        st = self.state
                    if st == self.IDLE:
                        self.start_recording()
                    elif st == self.RECORDING:
                        threading.Thread(target=self.stop_and_process,
                                         daemon=True).start()

    def on_release(self, key) -> None:
        try:
            self._pressed.discard(key)
        except TypeError:
            pass
        if key in self._hotkey:
            self._toggle_armed = True
            if self.cfg.get("mode", "push") == "push":
                with self._lock:
                    st = self.state
                if st == self.RECORDING:
                    # small delay so the Ctrl+V we synthesize later does not
                    # collide with the released hotkey modifiers
                    threading.Thread(target=self._delayed_stop,
                                     daemon=True).start()

    def _delayed_stop(self) -> None:
        time.sleep(0.15)
        self.stop_and_process()

    # -- tray -----------------------------------------------------------------
    def _tray_update(self) -> None:
        if self._tray:
            try:
                self._tray.update_icon()
            except Exception:
                pass

    def _make_icon(self, color: str):
        _need("PIL")
        img = _PIL_Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = _PIL_Draw.Draw(img)
        d.ellipse([8, 8, 56, 56], fill=color)
        return img

    def run_tray(self) -> None:
        _need("pystray", "PIL")
        colors = {self.IDLE: "#888888", self.RECORDING: "#e5484d",
                  self.PROCESSING: "#f5a524"}
        icon = _pystray.Icon(
            "whispero-dictate",
            self._make_icon(colors[self.IDLE]),
            "Whispero Dictate",
            menu=_pystray.Menu(
                _pystray.MenuItem("Quit", lambda i, _: self.quit(i))),
        )
        self._tray = icon

        def update_icon() -> None:
            try:
                icon.icon = self._make_icon(colors.get(self.state, "#888"))
            except Exception:
                pass
        icon.update_icon = update_icon  # type: ignore[attr-defined]
        icon.run()

    def quit(self, icon=None) -> None:
        self._log("quit")
        if icon:
            try:
                icon.stop()
            except Exception:
                pass
        os._exit(0)  # noqa: SLF001 - daemon teardown

    # -- main loop --------------------------------------------------------------
    def run(self) -> int:
        check_wayland()
        _need("pynput")
        kb = _pynput_kb
        mode = self.cfg.get("mode", "push")
        print(f"whispero-dictate on ({self.cfg['hotkey']}, mode={mode}). "
              "Esc cancels while recording.", file=sys.stderr)
        listener = kb.Listener(on_press=self.on_press,
                               on_release=self.on_release)
        listener.start()
        if self.cfg.get("tray"):
            try:
                self.run_tray()
            except Exception as e:
                print(f"warn: tray failed ({e}), running headless",
                      file=sys.stderr)
                self._hang()
        else:
            self._hang()
        return 0

    def _hang(self) -> None:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Whispero Dictate - hands-free voice dictation. "
                    "Hold the hotkey, speak, release: text lands in the "
                    "focused app.")
    p.add_argument("--hotkey", default=None,
                   help='Global hotkey, pynput format ("f8", '
                        '"<ctrl>+<alt>+space"). Default from config.')
    p.add_argument("--mode", default=None, choices=["push", "toggle"],
                   help="push = hold to talk (default), toggle = press "
                        "to start/stop.")
    p.add_argument("--lang", default=None, help="Transcription language.")
    p.add_argument("--provider", default=None,
                   choices=["gemini", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--no-cleanup", action="store_true",
                   help="Skip the LLM cleanup pass (raw transcript).")
    p.add_argument("--cleanup-provider", default=None,
                   choices=["gemini", "openai", "anthropic"],
                   help="LLM for the cleanup pass (default: same as "
                        "--provider). Anthropic works here because cleanup "
                        "is text-only; it cannot transcribe audio.")
    p.add_argument("--cleanup-model", default=None,
                   help="Model for the cleanup pass.")
    p.add_argument("--cleanup-base-url", default=None,
                   help="Custom base URL for the cleanup provider "
                        "(Anthropic-compatible endpoint).")
    p.add_argument("--hook", default=None,
                   help='Post-record hook command, text on stdin. '
                        'Ex.: --hook "xclip -selection clipboard"')
    p.add_argument("--no-tray", action="store_true",
                   help="No system tray icon (console mode).")
    p.add_argument("--paste-delay", type=float, default=None,
                   help="Seconds before restoring the clipboard (default 1.5).")
    p.add_argument("--test-inject", default=None, metavar="TEXT",
                   help="Inject TEXT right now (tests the paste pipeline "
                        "without recording).")
    p.add_argument("--debug", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config()
    for k in ("hotkey", "mode", "lang", "provider", "model", "base_url",
              "api_key", "hook", "paste_delay", "cleanup_provider",
              "cleanup_model", "cleanup_base_url"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.no_cleanup:
        cfg["cleanup"] = False
    if args.no_tray:
        cfg["tray"] = False

    if args.test_inject is not None:
        ok = Injector(delay=float(cfg["paste_delay"])).inject(args.test_inject)
        print("injected" if ok else "nothing to inject")
        return 0 if ok else 1

    if args.debug:
        safe = {k: v for k, v in cfg.items() if k != "api_key"}
        print("config:", json.dumps(safe, indent=1), file=sys.stderr)

    d = Dictator(cfg, debug=args.debug)
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
