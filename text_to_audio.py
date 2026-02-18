import platform
import subprocess
import tempfile
from pathlib import Path


def _speak_macos_say(text: str, rate: int = 185):
    subprocess.run(["say", "-r", str(rate), text], check=False)


def _have_cmd(cmd: str) -> bool:
    try:
        r = subprocess.run(["bash", "-lc", f"command -v {cmd}"], capture_output=True, text=True)
        return r.returncode == 0 and (r.stdout or "").strip() != ""
    except Exception:
        return False


def _speak_linux_pico(text: str, rate: int = 170):
    """
    pico2wave does not support a true 'rate' flag like espeak.
    We'll keep the signature consistent, and rely on chunking + punctuation
    for perceived pacing. (Still much clearer than espeak.)
    Install: sudo apt install -y libttspico-utils
    """
    # pico2wave needs a wav output file
    text = (text or "").replace('"', "").strip()
    if not text:
        return

    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "speech.wav")
        subprocess.run(["pico2wave", "-w", wav_path, text], check=False)
        # aplay is the most reliable on Pi
        subprocess.run(["aplay", "-q", wav_path], check=False)


def _speak_linux_espeak_ng(text: str, rate: int = 135):
    # Install: sudo apt install -y espeak-ng
    # Voice tweaks: en-us+f3 is often easier to understand
    subprocess.run(["espeak-ng", "-v", "en-us+f3", "-s", str(rate), "-p", "45", text], check=False)


def _speak_linux_espeak(text: str, rate: int = 135):
    # Install: sudo apt install -y espeak
    subprocess.run(["espeak", "-v", "en-us", "-s", str(rate), text], check=False)


def convert_text_to_audio(text: str, rate: int = 185):
    """
    Cross-platform OFFLINE TTS:
    - macOS: say
    - Linux/Pi: pico2wave (best fast offline), fallback to espeak-ng/espeak
    """
    if not text or not text.strip():
        return

    # Light pacing improvement for ALL engines
    # (this helps a lot with understanding)
    t = text.strip()
    t = t.replace("\n", ". ")
    t = t.replace(";", "; ")
    t = t.replace(".", ". ")
    t = " ".join(t.split())

    system = platform.system().lower()

    if "darwin" in system:
        _speak_macos_say(t, rate=rate)
        return

    # Linux / Raspberry Pi path
    if _have_cmd("pico2wave") and _have_cmd("aplay"):
        _speak_linux_pico(t, rate=rate)
        return

    # Fallbacks
    if _have_cmd("espeak-ng"):
        # Map your 'rate' scale to something reasonable for espeak-ng
        # Your code passes ~175-190; espeak-ng sounds better around 120-150.
        mapped = int(max(115, min(155, rate - 45)))
        _speak_linux_espeak_ng(t, rate=mapped)
        return

    if _have_cmd("espeak"):
        mapped = int(max(115, min(155, rate - 45)))
        _speak_linux_espeak(t, rate=mapped)
        return

    # If nothing exists, do nothing (avoid crashing your main loop)
    return