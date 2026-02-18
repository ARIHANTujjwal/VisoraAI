# ocr_train/build_charset.py

from pathlib import Path
import json
import string

ROOT = Path(__file__).resolve().parent.parent
WORDS = ROOT / "words.txt"
OUT = ROOT / "weights" / "charset.json"

# 1) Base characters we ALWAYS want for reading real text
BASE_CHARS = set()

# letters
BASE_CHARS.update(string.ascii_lowercase)
BASE_CHARS.update(string.ascii_uppercase)

# digits
BASE_CHARS.update(string.digits)

# space (very important for reading)
BASE_CHARS.add(" ")

# punctuation (edit this list if you want)
PUNCT = ".,!?;:'\"-()[]{}:/\\@#&%+$*=_<>"
BASE_CHARS.update(list(PUNCT))

# 2) Also include anything that appears in words.txt (just in case)
if WORDS.exists():
    text = WORDS.read_text(encoding="utf-8", errors="ignore")
    for ch in text:
        if ch.isprintable():
            BASE_CHARS.add(ch)

# 3) Make stable, sorted output
chars = sorted(BASE_CHARS)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"chars": chars}, indent=2), encoding="utf-8")

print("Saved charset to:", OUT)
print("unique chars:", len(chars))
print("num_classes (with CTC blank):", len(chars) + 1)
print("sample:", "".join(chars[:120]))
print("has A:", "A" in chars, "has 0:", "0" in chars, "has quote:", '"' in chars)
