# Whispero Dictate

Hands-free voice dictation for any app: hold a hotkey, speak, release, and
the transcribed text is pasted wherever your cursor is. Works with Claude
Code, terminals, browsers, editors, chat apps — anything that accepts paste.

## Per-OS notes

### Windows (primary target)

Everything works out of the box. F8 is free on most keyboards.

### macOS

- F8 is play/pause by default: enable "Use F1, F2 as standard function keys"
  (System Settings → Keyboard) or pick another hotkey, e.g.
  `--hotkey "<cmd>+<shift>+d"`.
- The paste step needs **Accessibility permission**:
  System Settings → Privacy & Security → Accessibility → allow your terminal
  (or the packaged app). macOS will prompt on first use.
- Paste uses Cmd+V, clipboard save/restore works the same.

### Linux (X11)

Works, but needs system packages first:

```bash
sudo apt install portaudio19-dev xclip xdotool   # Debian/Ubuntu
pip install "whispero[dictate]"
```

### Linux (Wayland) — not supported

The Wayland protocol blocks global hotkeys entirely: no app can listen for
a hotkey while another window has focus. `whispero-dictate` detects Wayland
and exits with a clear error. Options:

- Log into an **X11/Xorg** session (available in the login screen menu on
  Ubuntu/Fedora) and run it there
- `--test-inject "text"` still works anywhere (paste needs no hotkey)

## Install

```powershell
pip install "whispero[dictate]"
```

This pulls: `sounddevice` (mic capture), `numpy`, `pynput` (global hotkey),
`pyperclip` (clipboard), `pystray` + `Pillow` (tray icon).

Set your API key once:

```powershell
setx WHISPERO_API_KEY "sua-chave"
# or GEMINI_API_KEY (free at https://aistudio.google.com/apikey)
```

Restart the terminal after `setx`.

## Run

```powershell
whispero-dictate
```

A tray icon appears (gray = idle). Now:

1. Click inside Claude Code (or any text field)
2. **Hold F8**, speak, **release**
3. The text is pasted at the cursor after ~1-3s

Press **Esc** while holding to cancel a recording.

## Options

```powershell
# Toggle mode instead of hold-to-talk
whispero-dictate --mode toggle --hotkey "<ctrl>+<alt>+space"

# Raw transcript, no LLM cleanup
whispero-dictate --no-cleanup

# English
whispero-dictate --lang en

# Use Groq / OpenAI / any OpenAI-compatible API instead of Gemini
whispero-dictate --provider openai --base-url https://api.groq.com/openai/v1 --model whisper-large-v3

# Run a command with the dictated text on stdin after each take
whispero-dictate --hook "python meu_agente.py"

# No tray icon (console mode)
whispero-dictate --no-tray

# Test the paste pipeline without recording anything
whispero-dictate --test-inject "texto de teste"

# Cleanup pass on Claude instead of Gemini (text-only, fast)
whispero-dictate --cleanup-provider anthropic
export ANTHROPIC_API_KEY=<redacted>

# Anthropic-compatible endpoint for cleanup
whispero-dictate --cleanup-provider anthropic \
    --cleanup-base-url https://meu-proxy/v1 \
    --cleanup-model minha-claude
```

Hotkey format follows pynput: `f8`, `f9`, `<ctrl>+<alt>+space`, `<cmd>+d`
(macOS), etc.

## Config file

`%USERPROFILE%\.whispero\dictate.json` (CLI flags override it):

```json
{
  "hotkey": "f8",
  "mode": "push",
  "lang": "pt",
  "cleanup": true,
  "hook": "",
  "paste_delay": 1.5
}
```

## Providers

- **Transcription** (`--provider`): `gemini` (default, native audio) or
  `openai` (any Whisper-style `/v1/audio/transcriptions` endpoint).
- **Cleanup** (`--cleanup-provider`): `gemini`, `openai`, or `anthropic`.
  Anthropic works here because the cleanup pass is text-only.

Why no `--provider anthropic` for transcription: the Anthropic Messages API
has no audio input (verified against current docs) — Claude cannot receive
audio, so transcription through it is impossible. The text layer is fully
supported via `/v1/messages`, including Anthropic-compatible endpoints
through `--cleanup-base-url`.

## How it works

1. **Hotkey** (pynput, global): `push` = hold to record, release to send;
   `toggle` = press to start, press again to stop.
2. **Capture**: 16 kHz mono via sounddevice, written to a temp WAV.
   Recordings under 0.4s are discarded; auto-stops at 120s.
3. **Transcribe**: the Whispero engine (Gemini native audio by default —
   no local model, no VRAM, works on any machine).
4. **Cleanup** (optional): one LLM pass removes filler words ("ahn", "tipo",
   "né"), fixes punctuation and capitalization. Never changes meaning.
   Disable with `--no-cleanup`.
5. **Inject**: clipboard save → Ctrl+V synthesized → clipboard restored after
   1.5s, but only if it still holds our text. This is the SuperWhisper
   algorithm: slow apps (Electron, browsers) get their paste, and if you
   copied something else in between, your copy is never clobbered.

## Autostart on Windows

1. Press `Win+R`, type `shell:startup`, Enter
2. Create a shortcut with target:
   `pythonw -m dictate --no-tray`
   (adjust to where pip installed whispero, e.g.
   `%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe`)

Or run `whispero-dictate` manually when you need it.

## Troubleshooting

- **"missing packages"**: run `pip install "whispero[dictate]"`
- **No tray icon**: Windows may hide it; click the `^` in the taskbar.
  Or use `--no-tray`.
- **Hotkey does nothing**: another app may have grabbed F8. Try
  `--hotkey f9`.
- **Text pastes in the wrong app**: click the target field first, then hold
  the hotkey. The paste goes to whichever window has focus on release.
- **Mic not found**: check Windows Settings → Sound → Input. Run with
  `--debug` to see capture errors.
- **Slow first run**: the first transcription warms up the API path;
  subsequent takes are faster.

## Test checklist

- [ ] `--test-inject "ola"` pastes into Notepad
- [ ] Hold F8 in Claude Code, speak a sentence, release → text appears
- [ ] Esc during recording cancels (nothing pasted)
- [ ] Very short tap (< 0.4s) pastes nothing
- [ ] Toggle mode works with a custom hotkey
- [ ] `--no-cleanup` returns the raw transcript
