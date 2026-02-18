import torch
from typing import List, Tuple
from .charset import Charset


@torch.no_grad()
def ctc_greedy_decode(logits: torch.Tensor, charset: Charset) -> Tuple[List[str], List[float]]:
    """
    logits: (T, B, C)
    Returns:
      texts: list of decoded strings length B
      confs: list of simple confidence estimates length B
    """
    probs = torch.softmax(logits, dim=-1)  # (T,B,C)
    best_probs, best_idx = probs.max(dim=-1)  # (T,B)

    texts = []
    confs = []
    T, B = best_idx.shape
    for b in range(B):
        seq = best_idx[:, b].tolist()
        text = charset.decode_ctc_greedy(seq)
        texts.append(text)

        # confidence heuristic: average max prob over non-blank steps after collapse
        # not perfect but usable for gating
        pb = best_probs[:, b].tolist()
        non_blank = [p for i, p in zip(seq, pb) if i != charset.blank_idx]
        conf = float(sum(non_blank) / max(1, len(non_blank))) if non_blank else 0.0
        confs.append(conf)

    return texts, confs
