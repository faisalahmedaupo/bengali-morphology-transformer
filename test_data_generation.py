import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from generate_data import build_verb_pairs, build_noun_pairs  # noqa: E402


def test_every_pair_has_plus_separator():
    for surface, segmented, gloss in build_verb_pairs() + build_noun_pairs():
        assert "+" in segmented, f"missing separator: {segmented}"
        assert "+" in gloss, f"missing gloss separator: {gloss}"


def test_segmented_concat_equals_surface():
    for surface, segmented, _gloss in build_verb_pairs() + build_noun_pairs():
        assert segmented.replace("+", "") == surface, (surface, segmented)


def test_no_empty_surface_forms():
    for surface, segmented, _gloss in build_verb_pairs() + build_noun_pairs():
        assert len(surface) > 0
        assert len(segmented) > 0


def test_verb_and_noun_pair_counts():
    # 15 verb roots x 14 suffix slots, 15 noun roots x (6 or 7) suffix slots
    assert len(build_verb_pairs()) == 15 * 14
    assert len(build_noun_pairs()) > 0


if __name__ == "__main__":
    test_every_pair_has_plus_separator()
    test_segmented_concat_equals_surface()
    test_no_empty_surface_forms()
    test_verb_and_noun_pair_counts()
    print("All data generation tests passed.")
