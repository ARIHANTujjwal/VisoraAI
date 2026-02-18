from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

from ocr_train.charset import Charset
from ocr_train.crnn import CRNN
from ocr_train.decode import ctc_greedy_decode


def _pick_device(force_cpu: bool = True) -> str:
    # You said you want CPU only
    return "cpu"


class OCRModel:
    def __init__(
        self,
        weights_path: str = "weights/crnn_ctc.pth",
        charset_path: str = "weights/charset.json",
        img_h: int = 32,
    ):
        self.device = _pick_device(force_cpu=True)
        self.charset = Charset.load(charset_path)
        self.img_h = int(img_h)

        # CRNN expects num_classes == charset.num_classes
        self.model = CRNN(num_classes=self.charset.num_classes, img_h=self.img_h).to(self.device)

        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[OCRModel] Using device: {self.device}")
        print(f"[OCRModel] Loaded weights: {weights_path}")
        print(f"[OCRModel] Charset classes: {self.charset.num_classes} (blank={self.charset.blank_idx})")

    def _preprocess_crop(self, crop: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Match OCRLineDataset preprocessing:
          - grayscale
          - autocontrast (optional, but usually helps; if you want EXACT match, you can remove it)
          - resize to fixed height (img_h) keeping aspect ratio
          - convert to float in [0,1]
          - shape (1,1,H,W)
        """

        # handle empty
        if crop is None:
            arr = np.zeros((self.img_h, 8), dtype=np.float32)
            return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        # Convert input to PIL grayscale
        if isinstance(crop, Image.Image):
            img = crop
        else:
            crop = np.asarray(crop)
            if crop.size == 0:
                arr = np.zeros((self.img_h, 8), dtype=np.float32)
                return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

            # If OpenCV BGR (H,W,3)
            if crop.ndim == 3 and crop.shape[2] == 3:
                img = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
            # If already grayscale (H,W)
            elif crop.ndim == 2:
                img = Image.fromarray(crop)
            else:
                # fallback: force grayscale from first channel
                img = Image.fromarray(crop[..., 0])

        img = img.convert("L")

        # Resize keeping aspect ratio to fixed height
        w, h = img.size
        h = max(1, h)
        scale = self.img_h / float(h)
        new_w = max(8, int(round(w * scale)))
        img = img.resize((new_w, self.img_h), Image.BILINEAR)

        # EXACT match to ocr_train/dataset.py: normalize to [-1, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # [-1, 1]
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return t


    @torch.no_grad()
    def predict(self, crop: Union[np.ndarray, Image.Image]) -> Tuple[str, float]:
        x = self._preprocess_crop(crop).to(self.device)  # (1,1,H,W)
        log_probs = self.model(x)  # (T,B,C) expected
        texts, confs = ctc_greedy_decode(log_probs, self.charset)
        text = texts[0] if texts else ""
        conf = float(confs[0]) if confs else 0.0
        return text, conf
