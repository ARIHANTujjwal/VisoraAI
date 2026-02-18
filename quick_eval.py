"""Used to check if traing was useful"""

import csv
import random
from pathlib import Path

import cv2
from ocr_model import OCRModel  # assumes you created ocr_model.py

VAL_DIR = Path("ocr_train/data/val")
LABELS = VAL_DIR / "labels.csv"
IMG_DIR = VAL_DIR / "images"

model = OCRModel(weights_path="weights/crnn_ctc.pth", charset_path="weights/charset.json")

rows = []
with open(LABELS, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for r in reader:
        if len(r) >= 2:
            rows.append((r[0], r[1]))

sample = random.sample(rows, 10)

for fname, gt in sample:
    img = cv2.imread(str(IMG_DIR / fname))
    pred, conf = model.predict(img)
    print(f"{fname} | conf={conf:.3f}")
    print("GT  :", repr(gt))
    print("PRED:", repr(pred))
    print("-" * 60)
