"""
inference.py
------------
Load a trained checkpoint and segment arbitrary Bengali words from the
command line.

Usage:
    python src/inference.py --ckpt_dir checkpoints --word "করছিলাম"
    python src/inference.py --ckpt_dir checkpoints          # interactive REPL
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from tokenizer import CharTokenizer
from model import BengaliMorphTransformer


def load_model(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    tokenizer = CharTokenizer.from_dict(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BengaliMorphTransformer(
        vocab_size=len(tokenizer),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["enc_layers"],
        num_decoder_layers=cfg["dec_layers"],
        dropout=cfg["dropout"],
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
    model.eval()
    return model, tokenizer, device


def segment(model, tokenizer, device, word: str) -> str:
    src = torch.tensor([tokenizer.encode(word, add_bos_eos=True)], dtype=torch.long, device=device)
    pred_ids = model.greedy_decode(src, tokenizer.bos_id, tokenizer.eos_id, max_len=40)
    return tokenizer.decode(pred_ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--word", default=None, help="Single word to segment; omit for interactive mode")
    args = p.parse_args()

    model, tokenizer, device = load_model(args.ckpt_dir)

    if args.word:
        print(segment(model, tokenizer, device, args.word))
        return

    print("Bengali Morphology Transformer -- interactive mode. Ctrl+C to exit.")
    try:
        while True:
            word = input("word> ").strip()
            if not word:
                continue
            print(" ->", segment(model, tokenizer, device, word))
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
