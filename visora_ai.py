# visora_ai.py (Pi demo-safe, QuickTime HDMI friendly, V4L2 stable)
#
# Controls (type in terminal):
#   scan  -> capture burst + OCR + speak
#   quit  -> exit
#   help  -> show commands
#
# Display:
#   When SSH'd in, OpenCV windows won't appear unless DISPLAY is set to the Pi desktop (:0).
#   This script auto-sets DISPLAY=:0 if it looks missing, unless VISORA_HEADLESS=1.
#
# Camera:
#   Uses /dev/video0 explicitly and forces MJPG @ 1920x1080 30fps.

import cv2
import time
import sys
import numpy as np
import threading
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

# --- Make imports work on Pi + Mac ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from box_data import detect_text_and_draw_box
from text_to_audio import convert_text_to_audio

# Try old OCR module
try:
    from pic_to_text import perform_ocr  # perform_ocr(image)->(text,conf?) or text
    HAVE_PERFORM_OCR = True
except Exception:
    perform_ocr = None
    HAVE_PERFORM_OCR = False

# Optional accuracy helper
try:
    import accuracy as accuracy_mod
    HAVE_ACCURACY = True
except Exception:
    accuracy_mod = None
    HAVE_ACCURACY = False


# =========================
# Terminal command controls
# =========================
class ConsoleControls:
    """
    Type commands in the terminal running the program:
      scan  -> trigger scan
      quit  -> quit program
      help  -> show commands
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._scan_requested = False
        self._quit_requested = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        print("\n--- VisoraAI Controls ---")
        print("Type: scan  |  quit  |  help\n")
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                return

            if cmd in ("scan", "s", "read"):
                with self._lock:
                    self._scan_requested = True
            elif cmd in ("quit", "q", "exit"):
                with self._lock:
                    self._quit_requested = True
                return
            elif cmd in ("help", "?"):
                print("Commands: scan | quit | help")
            elif cmd == "":
                continue
            else:
                print("Unknown command. Type: help")

    def consume_scan(self) -> bool:
        with self._lock:
            if self._scan_requested:
                self._scan_requested = False
                return True
        return False

    def should_quit(self) -> bool:
        with self._lock:
            return self._quit_requested


# =========================
# Speech (single stream)
# =========================
class SpeechManager:
    def __init__(self, default_rate: int = 175, min_gap: float = 2.0):
        self.default_rate = default_rate
        self.min_gap = min_gap
        self._lock = threading.Lock()
        self._latest = None
        self._event = threading.Event()
        self._last_spoken = None
        self._last_time = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str, rate: Optional[int] = None, force: bool = False):
        t = (text or "").strip()
        if not t:
            return
        with self._lock:
            self._latest = (t, rate if rate is not None else self.default_rate, force)
            self._event.set()

    def _loop(self):
        while True:
            self._event.wait()
            with self._lock:
                item = self._latest
                self._latest = None
                self._event.clear()

            if item is None:
                continue

            text, rate, force = item
            now = time.time()

            if not force:
                if text == self._last_spoken and (now - self._last_time) < self.min_gap:
                    continue
                if (now - self._last_time) < self.min_gap:
                    continue

            try:
                convert_text_to_audio(text, rate=rate)
            except Exception:
                pass

            self._last_spoken = text
            self._last_time = time.time()


# =========================
# Display helpers (Pi HDMI)
# =========================
def ensure_pi_desktop_display_if_needed(headless: bool) -> bool:
    """
    If we're not headless and DISPLAY isn't set, force DISPLAY=:0 so OpenCV windows
    show up on the Pi desktop (HDMI output -> capture card -> QuickTime).
    """
    if headless:
        return True

    disp = os.environ.get("DISPLAY", "").strip()
    if disp == "":
        os.environ["DISPLAY"] = ":0"
        disp = ":0"

    # If DISPLAY is set but points to SSH forwarding, force :0 for the real desktop
    if disp.startswith("localhost:") or disp.startswith("127.0.0.1:"):
        os.environ["DISPLAY"] = ":0"
        disp = ":0"

    if os.environ.get("VISORA_FORCE_DISPLAY", "1") == "1":
        os.environ["DISPLAY"] = ":0"

    return True


# =========================
# Camera
# =========================
def _try_set_manual_camera(cap: cv2.VideoCapture):
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass

    for exp in [-4, -5, -6, -7, -8, -3, -2]:
        try:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exp))
            break
        except Exception:
            continue

    for prop, val in [
        (cv2.CAP_PROP_GAIN, 0),
        (cv2.CAP_PROP_BRIGHTNESS, 0.45),
        (cv2.CAP_PROP_CONTRAST, 0.60),
    ]:
        try:
            cap.set(prop, val)
        except Exception:
            pass


def open_cam():
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    dev = os.environ.get("VISORA_CAM_DEV", "/dev/video0")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    # Force MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Reduce buffering
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    _try_set_manual_camera(cap)

    # Warm-up and verify
    ok = False
    last = None
    for _ in range(50):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            ok = True
            last = frame
            break
        time.sleep(0.05)

    if not ok:
        cap.release()
        raise RuntimeError(f"Camera did not deliver frames from {dev}")

    print(f"[Camera] V4L2 OK: {last.shape[1]}x{last.shape[0]}")
    return cap


# =========================
# Metrics
# =========================
def blur_score(frame_bgr) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def overexposure_fraction(frame_bgr) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= 245))


# =========================
# Tesseract TSV helpers (fallback)
# =========================
def tesseract_available() -> bool:
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def _run_tesseract_tsv(gray: np.ndarray, psm: int = 6) -> Tuple[str, float, int]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            cv2.imwrite(tmp.name, gray)
            r = subprocess.run(
                [
                    "tesseract", tmp.name, "stdout",
                    "-l", "eng",
                    "--oem", "1",
                    "--psm", str(psm),
                    "-c", "preserve_interword_spaces=1",
                    "-c", "user_defined_dpi=300",
                    "tsv",
                ],
                capture_output=True,
                text=True
            )

        tsv = r.stdout or ""
        if not tsv.strip():
            return "", 0.0, 0

        lines = tsv.strip().splitlines()
        if len(lines) <= 1:
            return "", 0.0, 0

        header = lines[0].split("\t")
        if "conf" not in header or "text" not in header:
            return "", 0.0, 0

        conf_idx = header.index("conf")
        text_idx = header.index("text")
        line_idx = header.index("line_num") if "line_num" in header else None
        par_idx = header.index("par_num") if "par_num" in header else None

        OK = 52.0
        HARD_REJECT = 30.0

        words_out = []
        confs = []

        for row in lines[1:]:
            cols = row.split("\t")
            if len(cols) <= max(conf_idx, text_idx):
                continue

            txt = (cols[text_idx] or "").strip()
            if not txt:
                continue

            try:
                c = float(cols[conf_idx])
            except Exception:
                continue

            if c < 0 or c < HARD_REJECT or c < OK:
                continue

            par = cols[par_idx] if par_idx is not None and par_idx < len(cols) else "0"
            line = cols[line_idx] if line_idx is not None and line_idx < len(cols) else "0"
            words_out.append((txt, c, par, line))
            confs.append(c)

        if not words_out:
            return "", 0.0, 0

        words_out.sort(key=lambda t: (
            int(t[2]) if str(t[2]).isdigit() else 0,
            int(t[3]) if str(t[3]).isdigit() else 0
        ))

        rebuilt = []
        cur_key = None
        cur_line = []

        for txt, c, par, line in words_out:
            key = (par, line)
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                if cur_line:
                    rebuilt.append(" ".join(cur_line))
                cur_key = key
                cur_line = []
            cur_line.append(txt)

        if cur_line:
            rebuilt.append(" ".join(cur_line))

        clean_text = "\n".join(rebuilt).strip()
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return clean_text, avg_conf, len(confs)

    except Exception:
        return "", 0.0, 0


def tesseract_best_text(gray: np.ndarray) -> Tuple[str, float]:
    psms = [6, 4, 3, 11, 12]
    best_text = ""
    best_score = -1e9

    for psm in psms:
        txt, avg_conf, nwords = _run_tesseract_tsv(gray, psm=psm)
        score = (avg_conf * 10.0) + (nwords * 3.0) + min(len(txt), 900) * 0.02
        if len(txt) < 10 or nwords < 4:
            score -= 120.0
        if score > best_score:
            best_score = score
            best_text = txt

    return best_text.strip(), float(best_score)


# =========================
# Page -> TextBlock crop
# =========================
def detect_page_rect(frame_bgr) -> Optional[Tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)

    blur = cv2.GaussianBlur(L, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        if np.mean(L[mask > 0]) < np.mean(L[mask == 0]):
            mask = 255 - mask
    except Exception:
        pass

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.04 * (w * h):
        return None

    x, y, ww, hh = cv2.boundingRect(c)
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    ww = min(w - x, ww + 2 * pad)
    hh = min(h - y, hh + 2 * pad)
    return (x, y, ww, hh)


def detect_text_block_rect(page_bgr) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
    gray = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    bh_blur = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, bw = cv2.threshold(bh_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)), iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = gray.shape[:2]
    best = None
    best_area = 0

    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < best_area:
            continue
        if area < 0.01 * (W * H):
            continue
        if ww > 0.98 * W and hh > 0.98 * H:
            continue
        best_area = area
        best = (x, y, ww, hh)

    if best is None:
        return None

    x, y, ww, hh = best
    pad_x = int(0.03 * W)
    pad_y = int(0.03 * H)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    ww = min(W - x, ww + 2 * pad_x)
    hh = min(H - y, hh + 2 * pad_y)
    return (x, y, ww, hh)


def crop_full_text_block(frame_bgr) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    page_rect = detect_page_rect(frame_bgr)
    if page_rect is None:
        return None, None

    px, py, pw, ph = page_rect
    page = frame_bgr[py:py + ph, px:px + pw]
    if page.size == 0:
        return None, None

    block_rect = detect_text_block_rect(page)
    if block_rect is None:
        return None, None

    bx, by, bw, bh = block_rect
    crop = page[by:by + bh, bx:bx + bw]
    if crop.size == 0:
        return None, None

    x1 = px + bx
    y1 = py + by
    x2 = x1 + bw
    y2 = y1 + bh
    return crop, (x1, y1, x2, y2)


# =========================
# Fallback crop: union of word boxes (tight)
# =========================
def crop_union_of_word_boxes_tight(
    frame_bgr: np.ndarray,
    word_boxes: List[Tuple[int, int, int, int]],
    pad: int = 14,
    trim_quantile: float = 0.10
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    if not word_boxes:
        return None, None

    h, w = frame_bgr.shape[:2]

    x1s = np.array([x for (x, y, ww, hh) in word_boxes], dtype=np.float32)
    y1s = np.array([y for (x, y, ww, hh) in word_boxes], dtype=np.float32)
    x2s = np.array([x + ww for (x, y, ww, hh) in word_boxes], dtype=np.float32)
    y2s = np.array([y + hh for (x, y, ww, hh) in word_boxes], dtype=np.float32)

    q = float(trim_quantile)
    x1 = int(np.quantile(x1s, q))
    y1 = int(np.quantile(y1s, q))
    x2 = int(np.quantile(x2s, 1.0 - q))
    y2 = int(np.quantile(y2s, 1.0 - q))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


# =========================
# Enhancement + Median stack
# =========================
def enhance_text_crop_to_gray(crop_bgr: np.ndarray) -> np.ndarray:
    crop_bgr = cv2.resize(crop_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(10, 10))
    L = clahe.apply(L)

    L = cv2.bilateralFilter(L, d=5, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(L, (0, 0), 1.0)
    L = cv2.addWeighted(L, 1.9, blur, -0.9, 0)
    return L


def median_stack(grays: List[np.ndarray]) -> Optional[np.ndarray]:
    if not grays:
        return None
    hh = min(g.shape[0] for g in grays)
    ww = min(g.shape[1] for g in grays)
    arr = []
    for g in grays:
        if g.shape[0] != hh or g.shape[1] != ww:
            g = cv2.resize(g, (ww, hh))
        arr.append(g)
    stack = np.stack(arr, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


# =========================
# Burst scan (ADD tiny sleeps on failed reads)
# =========================
def capture_best_text_block(cap, seconds: float = 1.25, max_frames: int = 34):
    start = time.time()
    best_crop = None
    best_rect = None
    best_score = -1e18
    gray_pool: List[np.ndarray] = []

    while (time.time() - start) < seconds and max_frames > 0:
        max_frames -= 1
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        crop, rect = crop_full_text_block(frame)
        if crop is None:
            continue

        sharp = blur_score(crop)
        over = overexposure_fraction(crop)
        if over > 0.35 or sharp < 45:
            continue

        gray = enhance_text_crop_to_gray(crop)
        gray_pool.append(gray)
        if len(gray_pool) > 13:
            gray_pool.pop(0)

        h, w = crop.shape[:2]
        score = min(sharp, 3200.0) + (w * h) * 1e-5 - (over * 1800.0)

        if score > best_score:
            best_score = score
            best_crop = crop
            best_rect = rect

    super_gray = median_stack(gray_pool)
    return best_crop, best_rect, super_gray


def capture_best_text_via_box_data(cap, seconds: float = 1.25, max_frames: int = 34):
    start = time.time()
    best_crop = None
    best_rect = None
    best_score = -1e18
    gray_pool: List[np.ndarray] = []

    while (time.time() - start) < seconds and max_frames > 0:
        max_frames -= 1
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        _, text_detected, _, _, _, word_boxes = detect_text_and_draw_box(frame)
        if not text_detected or not word_boxes:
            continue

        crop, rect = crop_union_of_word_boxes_tight(frame, word_boxes, pad=14, trim_quantile=0.10)
        if crop is None:
            continue

        sharp = blur_score(crop)
        over = overexposure_fraction(crop)
        if over > 0.35 or sharp < 45:
            continue

        gray = enhance_text_crop_to_gray(crop)
        gray_pool.append(gray)
        if len(gray_pool) > 13:
            gray_pool.pop(0)

        n = len(word_boxes)
        score = (n * 1200.0) + min(sharp, 2600.0) - (over * 1500.0)

        if score > best_score:
            best_score = score
            best_crop = crop
            best_rect = rect

    super_gray = median_stack(gray_pool)
    return best_crop, best_rect, super_gray


# =========================
# OCR router
# =========================
def run_best_ocr(super_gray: Optional[np.ndarray], have_tess: bool) -> Tuple[str, float]:
    if super_gray is None:
        return "", -1e9

    if HAVE_PERFORM_OCR and perform_ocr is not None:
        try:
            out = perform_ocr(super_gray)
            if isinstance(out, tuple) and len(out) >= 1:
                text = (out[0] or "").strip()
                conf = float(out[1]) if len(out) >= 2 and out[1] is not None else 0.0
                score = conf * 10.0
            else:
                text = (out or "").strip()
                score = 0.0

            if HAVE_ACCURACY and accuracy_mod is not None:
                for name in ("clean_text", "postprocess_text", "normalize_text"):
                    fn = getattr(accuracy_mod, name, None)
                    if callable(fn):
                        text = fn(text)
                        break

            if text:
                return text, score
        except Exception:
            pass

    if have_tess:
        return tesseract_best_text(super_gray)

    return "", -1e9


# =========================
# Speaking helper
# =========================
def speak_text_in_chunks(speaker: SpeechManager, text: str, rate: int = 175, chunk_chars: int = 240):
    text = (text or "").strip()
    if not text:
        return
    chunk: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        chunk.append(line)
        if len(" ".join(chunk)) >= chunk_chars:
            speaker.say(" ".join(chunk), rate=rate, force=True)
            chunk = []
            time.sleep(0.05)
    if chunk:
        speaker.say(" ".join(chunk), rate=rate, force=True)


def distance_guide_from_crop(crop: np.ndarray, frame_shape) -> str:
    fh, fw = frame_shape[:2]
    ch, cw = crop.shape[:2]
    ratio = (cw * ch) / float(fw * fh)
    if ratio < 0.12:
        return "Move closer"
    if ratio > 0.85:
        return "Move farther"
    return "Good distance"


# =========================
# Main
# =========================
def main():
    HEADLESS = os.environ.get("VISORA_HEADLESS", "0") == "1"
    DEBUG_SAVE = os.environ.get("VISORA_DEBUG", "0") == "1"

    AUTO_SCAN = os.environ.get("VISORA_AUTOSCAN", "0") == "1"
    AUTO_SCAN_STABLE_SECS = float(os.environ.get("VISORA_AUTOSCAN_STABLE_SECS", "1.1"))
    AUTO_SCAN_COOLDOWN_SECS = float(os.environ.get("VISORA_AUTOSCAN_COOLDOWN_SECS", "4.0"))

    ensure_pi_desktop_display_if_needed(HEADLESS)

    WINDOW_NAME = "VisoraAI - Live (terminal scan/quit)"
    speaker = SpeechManager(default_rate=175, min_gap=2.0)
    have_tess = tesseract_available()

    speaker.say("Visora AI is starting.", force=True)

    # Open camera FIRST (prevents scan/quit confusion if camera stalls)
    try:
        cap = open_cam()
    except Exception as e:
        print(f"[Camera] Failed to open: {e}")
        speaker.say("Camera failed to open.", force=True)
        return

    # Start console AFTER camera is ready
    console = ConsoleControls()

    speaker.say("Visora AI is ready.", force=True)
    speaker.say("Type scan to read. Type quit to exit.", force=True)

    if HAVE_PERFORM_OCR:
        speaker.say("Using perform OCR pipeline.", force=True)
    elif have_tess:
        speaker.say("Using Tesseract fallback.", force=True)
    else:
        speaker.say("No OCR engine found.", force=True)

    if not HEADLESS:
        try:
            cv2.namedWindow(WINDOW_NAME)
        except Exception:
            HEADLESS = True

    stable_guide = None
    stable_count = 0
    last_spoken_guide = None
    REQUIRED_STABLE = 10

    BLUR_MIN = 45.0

    scanning = False
    scan_requested = False
    scan_start = 0.0
    SCAN_HOLD_SECONDS = 0.25

    guide = "Point at the page"
    last_processed = None

    frame_idx = 0
    DETECT_EVERY_N_FRAMES = 3

    autoscan_t0 = None
    last_scan_time = 0.0

    consecutive_fail = 0
    MAX_FAIL = 40

    while True:
        if console.should_quit():
            break

        if console.consume_scan():
            scan_requested = True

        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_fail += 1
            time.sleep(0.02)
            if consecutive_fail >= MAX_FAIL:
                print("[Camera] Reopening camera after repeated read failures...")
                speaker.say("Camera reconnecting.", rate=175)
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.2)
                try:
                    cap = open_cam()
                    consecutive_fail = 0
                except Exception as e:
                    print(f"[Camera] Reopen failed: {e}")
                    time.sleep(0.5)
            continue

        consecutive_fail = 0

        frame_idx += 1

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            processed = frame.copy()

            crop, rect = crop_full_text_block(frame)

            if crop is None:
                _, text_detected, _, _, _, word_boxes = detect_text_and_draw_box(frame)
                if text_detected and word_boxes:
                    crop, rect = crop_union_of_word_boxes_tight(frame, word_boxes, pad=14, trim_quantile=0.10)

            if rect is not None:
                x1, y1, x2, y2 = rect
                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 3)

            good_for_scan = False
            if crop is None:
                guide = "Point at the page"
            else:
                sharp = blur_score(crop)
                over = overexposure_fraction(crop)
                if over > 0.35:
                    guide = "Tilt page to remove glare"
                elif sharp < BLUR_MIN:
                    guide = "Hold steady"
                else:
                    guide = distance_guide_from_crop(crop, frame.shape)

                if guide == "Good distance" and sharp >= 60 and over <= 0.20:
                    good_for_scan = True

            if guide == stable_guide:
                stable_count += 1
            else:
                stable_guide = guide
                stable_count = 1

            if stable_count >= REQUIRED_STABLE and guide != last_spoken_guide:
                speaker.say(guide, rate=185)
                last_spoken_guide = guide

            now = time.time()
            if AUTO_SCAN and not scanning and not scan_requested:
                if (now - last_scan_time) >= AUTO_SCAN_COOLDOWN_SECS:
                    if good_for_scan:
                        if autoscan_t0 is None:
                            autoscan_t0 = now
                        elif (now - autoscan_t0) >= AUTO_SCAN_STABLE_SECS:
                            scan_requested = True
                            autoscan_t0 = None
                    else:
                        autoscan_t0 = None
                else:
                    autoscan_t0 = None
            else:
                autoscan_t0 = None

            try:
                if crop is not None:
                    dbg_gray = enhance_text_crop_to_gray(crop)
                    dbg_small = cv2.resize(dbg_gray, (320, 180))
                    dbg_small = cv2.cvtColor(dbg_small, cv2.COLOR_GRAY2BGR)
                    processed[10:10 + 180, 10:10 + 320] = dbg_small
            except Exception:
                pass

            last_processed = processed

        display_frame = last_processed if last_processed is not None else frame

        if scan_requested and not scanning:
            scanning = True
            scan_requested = False
            scan_start = time.time()
            speaker.say("Hold steady.", force=True)

        if scanning and (time.time() - scan_start) >= SCAN_HOLD_SECONDS:
            scanning = False
            last_scan_time = time.time()
            speaker.say("Reading.", force=True)

            best_crop, best_rect, super_gray = capture_best_text_block(cap, seconds=1.25, max_frames=34)
            if best_crop is None or super_gray is None:
                best_crop, best_rect, super_gray = capture_best_text_via_box_data(cap, seconds=1.25, max_frames=34)

            if best_crop is None or super_gray is None:
                speaker.say("I can't see text yet. Move closer and aim at the paragraph.", force=True)
                continue

            if overexposure_fraction(best_crop) > 0.35:
                speaker.say("Too much glare. Tilt the page slightly.", force=True)
                continue

            if blur_score(best_crop) < 55:
                speaker.say("Too blurry. Hold steadier or move slightly farther.", force=True)
                continue

            if DEBUG_SAVE:
                cv2.imwrite(str(BASE_DIR / "debug_best_crop.png"), best_crop)
                cv2.imwrite(str(BASE_DIR / "debug_super_gray.png"), super_gray)

            text, score = run_best_ocr(super_gray, have_tess=have_tess)
            print(f"[OCR] score={score:.1f} chars={len(text)}")

            if DEBUG_SAVE:
                try:
                    with open(str(BASE_DIR / "debug_text.txt"), "w", encoding="utf-8") as f:
                        f.write(text)
                except Exception:
                    pass

            if not text or len(text.strip()) < 10:
                speaker.say("I could not read that. Move closer and reduce glare.", force=True)
                continue

            compact = text.replace(" ", "").replace("\n", "")
            if len(compact) < 12:
                speaker.say("Not enough readable text. Move closer.", force=True)
                continue

            if HAVE_PERFORM_OCR and score != 0.0 and score < 350:
                speaker.say("Text is unclear. Move slightly closer and hold steady.", force=True)
                continue

            speak_text_in_chunks(speaker, text, rate=175, chunk_chars=240)

        if not HEADLESS:
            try:
                status = "SCANNING..." if scanning else "READY (type scan / quit)"
                if AUTO_SCAN:
                    status = "AUTO ON (type scan / quit)" if not scanning else "SCANNING..."
                cv2.putText(display_frame, f"Guide: {guide}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display_frame, "CONTROL: type 'scan' in terminal. type 'quit' to exit.",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, display_frame)
            except Exception:
                pass

            # optional keyboard fallback for local testing
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    scan_requested = True
                elif key == ord("q"):
                    break
            except Exception:
                pass

    try:
        cap.release()
    except Exception:
        pass

    if not HEADLESS:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()