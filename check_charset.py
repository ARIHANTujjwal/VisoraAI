from pathlib import Path
from ocr_train.charset import Charset

paths = list(Path(".").rglob("charset.json"))

print("Found charset.json files:")
for p in paths:
    cs = Charset.load(str(p))
    print(f"{p} -> num_classes={cs.num_classes}")
