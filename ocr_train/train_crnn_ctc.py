# ocr_train/train_crnn_ctc.py

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ocr_train.dataset import OCRLineDataset
from ocr_train.crnn import CRNN
from ocr_train.charset import Charset


# ----------------------------
# Utils
# ----------------------------
def _pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance (insert/delete/substitute)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Make a the longer string to reduce memory
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _ctc_greedy_decode(log_probs: torch.Tensor, charset: Charset) -> List[str]:
    """
    log_probs: (T, B, C) log-probabilities
    Returns list[str] length B
    """
    # argmax over classes
    pred = log_probs.argmax(dim=2)  # (T, B)

    results: List[str] = []
    blank = charset.blank_idx

    # charset.idx_to_char might be dict or list; handle both
    idx_to_char = charset.idx_to_char

    for b in range(pred.shape[1]):
        seq = pred[:, b].tolist()
        out_chars: List[str] = []
        prev_idx = None
        for idx in seq:
            if idx == blank:
                prev_idx = idx
                continue
            if prev_idx is not None and idx == prev_idx:
                continue
            # map idx -> char
            if isinstance(idx_to_char, dict):
                ch = idx_to_char.get(idx, "")
            else:
                ch = idx_to_char[idx] if 0 <= idx < len(idx_to_char) else ""
            if ch:
                out_chars.append(ch)
            prev_idx = idx
        results.append("".join(out_chars))
    return results


# ----------------------------
# Dataset adapter
# ----------------------------
def make_dataset(split: str, charset: Charset, data_root: str | None = None) -> OCRLineDataset:
    """
    Expects data layout:
      ocr_train/data/train/images + labels.csv (or labels.tsv etc)
      ocr_train/data/val/images   + labels.csv
      ocr_train/data/debug/images + labels.csv
    """
    base = data_root or "ocr_train/data"

    if split == "train":
        root_dir = os.path.join(base, "train")
    elif split in ("val", "valid", "validation"):
        root_dir = os.path.join(base, "val")
    elif split == "debug":
        root_dir = os.path.join(base, "debug")
    else:
        # allow passing custom path
        root_dir = split

    return OCRLineDataset(root_dir=root_dir, charset=charset)


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Dataset collate returns: padded_images, targets, target_lengths, texts
        images, targets, target_lengths, _texts = batch

        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        log_probs = model(images)  # (T, B, C)
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, charset: Charset, device: str):
    model.eval()

    total_edits = 0
    total_chars = 0
    exact = 0
    count = 0

    debug_gt = None
    debug_pr = None

    for batch in loader:
        images, _targets, _target_lengths, gt_texts = batch
        images = images.to(device)

        log_probs = model(images)
        preds = _ctc_greedy_decode(log_probs, charset)  # list[str]

        # store one debug example
        if debug_gt is None and len(gt_texts) > 0:
            debug_gt = gt_texts[0]
            debug_pr = preds[0] if len(preds) else ""

        for gt, pr in zip(gt_texts, preds):
            if gt == pr:
                exact += 1

            total_edits += _edit_distance(gt, pr)
            total_chars += max(1, len(gt))
            count += 1

    cer = total_edits / max(1, total_chars)
    exact_match = exact / max(1, count)
    dbg = (debug_gt, debug_pr)

    return exact_match, cer, dbg


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--charset_path", type=str, default="weights/charset.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="weights/crnn_ctc.pth")
    parser.add_argument("--overfit_k", type=int, default=None)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    device = _pick_device(force_cpu=args.force_cpu)
    print("Device:", device)

    charset = Charset.load(args.charset_path)
    print("Loaded charset num_classes:", charset.num_classes, "blank_idx:", charset.blank_idx)

    # Datasets
    train_ds = make_dataset("train", charset, data_root=args.data_root)
    val_ds = make_dataset("val", charset, data_root=args.data_root)

    if args.overfit_k:
        print(f"[Overfit mode] training on K={args.overfit_k} samples")
        train_ds.samples = train_ds.samples[: args.overfit_k]
        val_ds = train_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        collate_fn=train_ds.collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=val_ds.collate_fn,
    )

    # Model
    model = CRNN(num_classes=charset.num_classes).to(device)

    criterion = nn.CTCLoss(blank=charset.blank_idx, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_exact, val_cer, (dbg_gt, dbg_pr) = evaluate(model, val_loader, charset, device)

        elapsed = time.time() - start

        if dbg_gt is not None:
            print(f"  [VAL] GT: {dbg_gt}")
            print(f"  [VAL] PR: {dbg_pr}")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss {train_loss:.4f} | "
            f"val_exact {val_exact:.3f} | "
            f"val_CER {val_cer:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_cer < best_cer:
            best_cer = val_cer
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("Saved best ->", args.save_path)

    print("Training done. best val_CER:", best_cer)


if __name__ == "__main__":
    main()
