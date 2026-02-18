# ocr_train/synth_data.py
import csv
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
WORDS_PATH = ROOT / "words.txt"

# -----------------------------
# Load & clean word list
# -----------------------------
def load_words(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = []
    for line in text.splitlines():
        w = line.strip()
        if not w:
            continue
        # keep only alphabetic words (MIT list is usually clean)
        w = re.sub(r"[^a-zA-Z]", "", w)
        if 2 <= len(w) <= 12:
            words.append(w.lower())
    # de-duplicate but keep variety
    words = list(dict.fromkeys(words))
    return words


# -----------------------------
# Text generator: realistic lines
# -----------------------------
def make_line(words, min_words=2, max_words=7):
    n = random.randint(min_words, max_words)
    ws = random.choices(words, k=n)

    # sometimes capitalize first word like a sentence
    if random.random() < 0.35:
        ws[0] = ws[0].capitalize()

    s = " ".join(ws)

    # occasional punctuation at end (realistic)
    if random.random() < 0.25:
        s += random.choice([".", "!", "?"])

    # occasional comma in middle
    if random.random() < 0.15 and len(ws) >= 4:
        parts = s.split(" ")
        i = random.randint(1, len(parts) - 2)
        parts[i] = parts[i] + ","
        s = " ".join(parts)

    return s


# -----------------------------
# Rendering + augmentations
# -----------------------------
def choose_font(font_paths, size):
    if font_paths:
        fp = random.choice(font_paths)
        return ImageFont.truetype(str(fp), size=size)
    return ImageFont.load_default()


def render_text_line(text: str, w=900, h=64, font_paths=None):
    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)

    # vary font size a bit
    font_size = random.randint(22, 30)
    font = choose_font(font_paths, font_size)

    # compute text bbox
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # placement with margins
    x = random.randint(8, max(8, w - tw - 8))
    y = random.randint(6, max(6, h - th - 6))
    draw.text((x, y), text, fill=0, font=font)

    # mild perspective-ish jitter by shifting rows (cheap)
    if random.random() < 0.25:
        arr = img.load()
        for yy in range(h):
            shift = random.randint(-1, 1)
            if shift != 0:
                row = [arr[xx, yy] for xx in range(w)]
                for xx in range(w):
                    src = xx - shift
                    arr[xx, yy] = row[src] if 0 <= src < w else 255

    # blur/noise (light)
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.0)))

    # contrast variation
    if random.random() < 0.35:
        img = ImageOps.autocontrast(img)

    return img


def find_fonts():
    """
    Optional: put .ttf files in ocr_train/fonts/
    If empty, default font is used.
    """
    font_dir = Path(__file__).resolve().parent / "fonts"
    if not font_dir.exists():
        return []
    return [p for p in font_dir.rglob("*.ttf")]


def make_split(out_dir: Path, n: int, words, seed: int):
    random.seed(seed)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font_paths = find_fonts()

    labels_path = out_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for i in range(n):
            text = make_line(words)
            img = render_text_line(text, font_paths=font_paths)

            fname = f"{i:06d}.png"
            img.save(img_dir / fname)
            writer.writerow([fname, text])


if __name__ == "__main__":
    if not WORDS_PATH.exists():
        raise FileNotFoundError(f"Missing {WORDS_PATH}. Put your MIT words.txt in the VisoraAI root.")

    words = load_words(WORDS_PATH)
    print("Loaded words:", len(words), "example:", words[:10])

    # Smaller for CPU training; you can scale up later
    make_split(Path("ocr_train/data/train"), n=20000, words=words, seed=1)
    make_split(Path("ocr_train/data/val"), n=2000, words=words, seed=2)

    print("Synthetic dataset created.")
