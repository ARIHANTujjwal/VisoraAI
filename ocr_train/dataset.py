# ocr_train/dataset.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _find_label_file(root_dir: str) -> str:
    """
    Tries common label/annotation filenames.
    Expected format: one sample per line:  <filename>\t<text>  OR  <filename>,<text>
    """
    candidates = [
        "labels.tsv",
        "labels.txt",
        "labels.csv",          # IMPORTANT
        "gt.tsv",
        "gt.txt",
        "annotations.tsv",
        "annotations.txt",
    ]
    for name in candidates:
        p = os.path.join(root_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No label file found in {root_dir}. Expected one of: {candidates}\n"
        "Label file format must be: filename<TAB>text OR filename,text (one per line)."
    )


def _read_labels_file(label_path: str) -> List[Tuple[str, str]]:
    """
    Reads TSV or CSV style labels.
    Skips empty lines and optional header like: filename,text
    """
    items: List[Tuple[str, str]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # skip header row if it looks like it
            low = line.lower()
            if i == 0 and ("filename" in low or "file" in low) and ("text" in low or "label" in low):
                continue

            if "\t" in line:
                fn, text = line.split("\t", 1)
            elif "," in line:
                fn, text = line.split(",", 1)
            else:
                continue

            fn = fn.strip()
            text = text.strip()
            items.append((fn, text))
    return items


def _to_tensor_grayscale(img: Image.Image) -> torch.Tensor:
    """
    Returns float tensor shaped (1, H, W), normalized to [-1, 1]
    """
    img = img.convert("L")
    arr = __import__("numpy").array(img, dtype="float32") / 255.0
    arr = (arr - 0.5) / 0.5  # <- match OCRModel
    x = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
    return x



def _resize_to_height(img: Image.Image, height: int = 32) -> Image.Image:
    """
    Resize keeping aspect ratio to a fixed height.
    """
    w, h = img.size
    if h == height:
        return img
    new_w = max(1, int(round(w * (height / float(h)))))
    return img.resize((new_w, height), Image.BILINEAR)


@dataclass
class Sample:
    img_path: str
    text: str
    enc: List[int]


class OCRLineDataset(Dataset):
    """
    Folder format expected:

    root_dir/
      images/
        xxx.png
        ...
      labels.csv (or labels.tsv / gt.tsv / etc)

    Label format:
      filename<TAB>text
    OR
      filename,text
    """

    def __init__(
        self,
        root_dir: str,
        charset,
        images_subdir: str = "images",
        label_path: Optional[str] = None,
        height: int = 32,
        min_width: int = 1,
        max_width: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.charset = charset
        self.height = int(height)
        self.min_width = int(min_width)
        self.max_width = int(max_width) if max_width is not None else None

        img_dir = os.path.join(root_dir, images_subdir)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing images/: {img_dir}")

        label_path = label_path or _find_label_file(root_dir)
        pairs = _read_labels_file(label_path)

        samples: List[Sample] = []
        missing = 0
        empty = 0
        unencodable = 0

        for fn, text in pairs:
            if text is None or len(text.strip()) == 0:
                empty += 1
                continue

            # accept either "file.png" or full/relative paths
            if os.path.isabs(fn):
                img_path = fn
            else:
                img_path = os.path.join(img_dir, fn)

            if not os.path.isfile(img_path):
                # sometimes labels include paths like "images/xxx.png"
                alt = os.path.join(root_dir, fn)
                if os.path.isfile(alt):
                    img_path = alt
                else:
                    missing += 1
                    continue

            try:
                enc = self.charset.encode(text)
            except Exception:
                unencodable += 1
                continue

            if not isinstance(enc, (list, tuple)) or len(enc) == 0:
                unencodable += 1
                continue

            samples.append(Sample(img_path=img_path, text=text, enc=list(enc)))

        self.samples = samples

        if verbose:
            print(
                f"[Dataset] loaded={len(self.samples)} "
                f"(missing={missing}, empty={empty}, unencodable={unencodable})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        img = Image.open(s.img_path)
        img = _resize_to_height(img, self.height)

        x = _to_tensor_grayscale(img)  # (1,H,W)

        # optional clamp width
        w = x.shape[-1]
        if w < self.min_width:
            pad = self.min_width - w
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)
        if self.max_width is not None and x.shape[-1] > self.max_width:
            x = x[:, :, : self.max_width]

        # return image tensor + raw text + encoded labels (list[int])
        return x, s.text, s.enc

    def collate_fn(self, batch):
        """
        Pads variable-width images and flattens targets for CTC.
        Returns:
          images: (B,1,32,Wmax)
          targets: (sum(target_lengths),)
          target_lengths: (B,)
          texts: list[str]
        """
        images, texts, encs = zip(*batch)

        B = len(images)
        C, H = images[0].shape[0], images[0].shape[1]
        Wmax = max(im.shape[-1] for im in images)

        padded = torch.zeros((B, C, H, Wmax), dtype=images[0].dtype)
        for i, im in enumerate(images):
            padded[i, :, :, : im.shape[-1]] = im

        target_lengths = torch.tensor([len(e) for e in encs], dtype=torch.long)
        targets = torch.tensor([t for e in encs for t in e], dtype=torch.long)

        return padded, targets, target_lengths, list(texts)
