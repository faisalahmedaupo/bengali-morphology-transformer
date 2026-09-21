import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizer import CharTokenizer, PAD, BOS, EOS, UNK  # noqa: E402


def test_build_and_roundtrip():
    tok = CharTokenizer.build(["করছি", "কর+ছ+ি", "বাড়িগুলো"])
    ids = tok.encode("করছি")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert decoded == "করছি"


def test_plus_symbol_is_vocab_char():
    tok = CharTokenizer.build(["কর+ি"])
    assert "+" in tok.stoi
    ids = tok.encode("কর+ি")
    decoded = tok.decode(ids)
    assert decoded == "কর+ি"


def test_unk_for_unseen_char():
    tok = CharTokenizer.build(["ক", "খ"])
    ids = tok.encode("গ", add_bos_eos=False)  # unseen char
    assert ids == [tok.unk_id]


def test_special_tokens_present():
    tok = CharTokenizer.build(["x"])
    for special in (PAD, BOS, EOS, UNK):
        assert special in tok.stoi


def test_to_dict_from_dict_roundtrip():
    tok = CharTokenizer.build(["করি", "খাই"])
    d = tok.to_dict()
    tok2 = CharTokenizer.from_dict(d)
    assert tok.stoi == tok2.stoi
    assert tok2.decode(tok2.encode("করি")) == "করি"


if __name__ == "__main__":
    test_build_and_roundtrip()
    test_plus_symbol_is_vocab_char()
    test_unk_for_unseen_char()
    test_special_tokens_present()
    test_to_dict_from_dict_roundtrip()
    print("All tokenizer tests passed.")
