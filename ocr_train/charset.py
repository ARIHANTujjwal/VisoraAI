'''File used to define the OCR alphabet since it computes everything in numerical form
    it is like a translator.
'''

import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Charset:
    chars: str  # characters allowed ("abcdefghijklmnopqrstuvwxyz0123456789")

    def __post_init__(self):
        # index 0 reserved for CTC blank
        self.blank_idx = 0
        # maps chapracters to token numbers, ex: a -> 1, b -> 2
        self.char_to_idx = {}
        # maps token number to character
        self.idx_to_char = {}

        # helps store translation data from char to numb to assist OCR in translations
        for i in range(len(self.chars)):
            c = self.chars[i]
            self.char_to_idx[c] = i + 1
            self.idx_to_char[i + 1] = c

        self.num_classes = len(self.chars) + 1  # + blank

    # expects function to return list int
    # function converts string into a list of numbers
    def encode(self, s):
        out = []
        for ch in s:
            if ch in self.char_to_idx:
                out.append(self.char_to_idx[ch])
        return out

    # turns model predictions to readable text
    def decode_ctc_greedy(self, indices: List[int]) -> str:
        # collapse repeats + remove blanks
        res = []
        prev = None
        for idx in indices:
            if idx == self.blank_idx:
                prev = idx
                continue
            if idx == prev:
                continue
            res.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(res)

    def save(self, path: str) -> None:
        # saves the character set to a JSON file so that training and inference use the same charset
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Charset":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Charset(d["chars"])
