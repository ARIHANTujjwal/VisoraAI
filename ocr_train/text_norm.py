# ocr_train/text_norm.py
import re

# Replace common “weird” unicode with normal ASCII
_REPL = {
    "\u2018": "'", "\u2019": "'",  # ‘ ’
    "\u201C": '"', "\u201D": '"',  # “ ”
    "\u2013": "-", "\u2014": "-",  # – —
    "\u00A0": " ",                 # non-breaking space
    "\t": " ", "\n": " ", "\r": " ",
}

_multi_space = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    for a, b in _REPL.items():
        s = s.replace(a, b)
    s = _multi_space.sub(" ", s)
    return s
