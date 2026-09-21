"""
model.py
--------
A compact character-level Transformer encoder-decoder for Bengali
morphological segmentation (surface form -> root+suffix+suffix... string).

The architecture is a standard pre-norm Transformer (Vaswani et al., 2017)
built directly from torch.nn primitives (no huggingface dependency) so the
whole pipeline stays small, auditable, and easy to explain in an admissions
portfolio / interview: every component here is implemented, not imported
from a pretrained checkpoint.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BengaliMorphTransformer(nn.Module):
    """
    Encoder-decoder Transformer over character ids.

    Args:
        vocab_size: size of the shared src/tgt character vocabulary
        d_model: embedding / hidden dimension
        nhead: number of attention heads
        num_encoder_layers / num_decoder_layers
        dim_feedforward: FFN hidden size
        dropout
        pad_id: padding token id, used to build padding masks
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 64,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        # True where padding -> masked out
        return ids == self.pad_id

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """
        src:    (B, S) source character ids
        tgt_in: (B, T) decoder input ids (shifted right, starts with <bos>)
        returns logits: (B, T, vocab_size)
        """
        src_key_padding_mask = self._padding_mask(src)
        tgt_key_padding_mask = self._padding_mask(tgt_in)

        src_emb = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.embed(tgt_in) * math.sqrt(self.d_model))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1)
        ).to(src.device)

        hidden = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out_proj(hidden)

    @torch.no_grad()
    def greedy_decode(
        self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 40
    ) -> torch.Tensor:
        """Autoregressive greedy decoding. src: (B, S) -> returns (B, <=max_len)."""
        self.eval()
        device = src.device
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(src, ys)
            next_ids = logits[:, -1, :].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            finished = finished | (next_ids == eos_id)
            if finished.all():
                break
        return ys
