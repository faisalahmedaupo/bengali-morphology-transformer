"""
dataset.py
----------
Loads the TSV files produced by data/generate_data.py and wraps them as a
PyTorch Dataset for the surface-form -> segmented-morpheme task.

TSV columns: surface_form \t segmented_form \t gloss
Only the first two columns are used for training; gloss is kept for
error analysis / qualitative inspection in evaluate.py.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


def read_tsv(path: str) -> List[Tuple[str, str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            surface, segmented = parts[0], parts[1]
            gloss = parts[2] if len(parts) > 2 else ""
            rows.append((surface, segmented, gloss))
    return rows


class MorphDataset(Dataset):
    """Character-level src/tgt pairs for morphological segmentation."""

    def __init__(self, tsv_path: str, tokenizer: CharTokenizer):
        self.rows = read_tsv(tsv_path)
        self.tok = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        surface, segmented, gloss = self.rows[idx]
        src_ids = self.tok.encode(surface, add_bos_eos=True)
        tgt_ids = self.tok.encode(segmented, add_bos_eos=True)
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "surface": surface,
            "segmented": segmented,
            "gloss": gloss,
        }


def build_tokenizer_from_files(paths: List[str]) -> CharTokenizer:
    texts = []
    for p in paths:
        for surface, segmented, _gloss in read_tsv(p):
            texts.append(surface)
            texts.append(segmented)
    return CharTokenizer.build(texts)


def collate_batch(batch, pad_id: int):
    """Pads src/tgt to the max length in the batch. Returns tensors plus
    the raw strings (useful for evaluate.py's qualitative output)."""
    src_lens = [len(b["src"]) for b in batch]
    tgt_lens = [len(b["tgt"]) for b in batch]
    max_src, max_tgt = max(src_lens), max(tgt_lens)

    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, : len(b["src"])] = b["src"]
        tgt[i, : len(b["tgt"])] = b["tgt"]

    return {
        "src": src,
        "tgt": tgt,
        "src_lens": torch.tensor(src_lens),
        "tgt_lens": torch.tensor(tgt_lens),
        "surface": [b["surface"] for b in batch],
        "segmented": [b["segmented"] for b in batch],
        "gloss": [b["gloss"] for b in batch],
    }
