"""Evaluate a trained Bengali morphology segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer import CharTokenizer
from dataset import MorphDataset, collate_batch
from model import BengaliMorphTransformer
from metrics import segmentation_report


def load_model(ckpt_dir: Path, device: torch.device):
    with open(ckpt_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(ckpt_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = CharTokenizer.from_dict(vocab)
    model = BengaliMorphTransformer(
        vocab_size=len(tokenizer),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["enc_layers"],
        num_decoder_layers=cfg["dec_layers"],
        dropout=cfg["dropout"],
        pad_id=tokenizer.pad_id,
    ).to(device)

    model.load_state_dict(
        torch.load(ckpt_dir / "best_model.pt", map_location=device)
    )
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="test.tsv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--show", type=int, default=12)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(Path(args.ckpt_dir), device)

    ds = MorphDataset(str(Path(args.data_dir) / args.split), tokenizer)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_id),
    )

    pairs = []
    examples = []

    for batch in loader:
        src = batch["src"].to(device)
        pred_ids = model.greedy_decode(
            src, tokenizer.bos_id, tokenizer.eos_id, max_len=64
        )

        for i in range(src.size(0)):
            pred = tokenizer.decode(pred_ids[i].tolist())
            gold = batch["segmented"][i]
            surface = batch["surface"][i]
            pairs.append((gold, pred))
            examples.append((surface, gold, pred))

    report = segmentation_report(pairs)

    print(f"Split: {args.split}")
    print(f"Examples: {len(pairs)}")
    print(f"Exact-match accuracy: {report['exact_match']:.2%}")
    print(f"Character error rate: {report['cer']:.2%}")
    print(f"Boundary precision: {report['boundary_precision']:.2%}")
    print(f"Boundary recall: {report['boundary_recall']:.2%}")
    print(f"Boundary F1: {report['boundary_f1']:.2%}")
    print()
    print(f"{'surface':<16} {'gold':<20} {'predicted':<20}")
    print("-" * 60)
    for surface, gold, pred in examples[: args.show]:
        print(f"{surface:<16} {gold:<20} {pred:<20}")


if __name__ == "__main__":
    main()
