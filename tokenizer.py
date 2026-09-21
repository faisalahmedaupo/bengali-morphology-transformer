"""
tokenizer.py
------------
A small, dependency-free character-level tokenizer for Bengali text.

Bengali is written with a complex script (independent vowels, consonants,
dependent vowel signs / matras, the virama "hasant", and conjunct
formations). For a morpheme-segmentation model, tokenizing at the Unicode
codepoint level (rather than naive whitespace or "grapheme cluster"
splitting) gives the model direct access to matras and the virama, which
carry a lot of the morphological signal in Bengali. The special "+" symbol
used in segmented targets is treated as an ordinary vocabulary symbol.

Vocabulary is built from the training data itself (closed vocabulary is fine
here: an in-domain morphological segmenter does not need open-vocabulary
subword coverage the way a general LM does).
"""

from __future__ import annotations
from typing import List, Dict, Iterable

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class CharTokenizer:
    def __init__(self, vocab: Dict[str, int] | None = None):
        self.stoi: Dict[str, int] = vocab or {}
        self.itos: Dict[int, str] = {i: s for s, i in self.stoi.items()}

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars = set()
        for t in texts:
            chars.update(list(t))
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in sorted(chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        chars = []
        for i in ids:
            tok = self.itos.get(i, UNK)
            if strip_special and tok in SPECIAL_TOKENS:
                if tok == EOS:
                    break
                continue
            chars.append(tok)
        return "".join(chars)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.stoi)

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "CharTokenizer":
        return cls(d)
