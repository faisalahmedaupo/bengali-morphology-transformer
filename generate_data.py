"""
generate_data.py
-----------------
Builds the Bengali morphological segmentation dataset used to train/evaluate
the Bengali Morphology Transformer.

Rather than scraping an external corpus, this script encodes the morphological
paradigms of standard colloquial Bengali (verb conjugation, noun case
marking, pluralization, and classifier/definiteness marking) as explicit
rule tables, then expands every (root x suffix) combination into a
surface-form -> segmented-morpheme training pair.

Each output row is tab-separated:
    surface_form <TAB> segmented_form <TAB> gloss

Example:
    করছি        কর+ছ+ি        do+PROG+1P

Run:
    python generate_data.py --out_dir .
This regenerates train.tsv / dev.tsv / test.tsv deterministically (fixed seed),
so results are reproducible from this file alone -- no external data needed.
"""

import argparse
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Verb paradigms
# ---------------------------------------------------------------------------
# Each verb root is given with the sandhi-adjusted stem used before each
# suffix class (Bengali verb stems change slightly before vowel-initial vs
# consonant-initial suffixes; we encode the attested surface stem directly
# rather than deriving it, which keeps the segmentation gold-standard clean).

VERB_ROOTS = {
    # root_gloss: (stem, )
    "কর": "do",
    "খা": "eat",
    "যা": "go",
    "দেখ": "see",
    "লিখ": "write",
    "পড়": "read/fall",
    "বল": "say",
    "শুন": "hear",
    "চল": "walk/move",
    "খেল": "play",
    "হাস": "laugh",
    "নাচ": "dance",
    "রান্ধ": "cook",
    "ঘুম": "sleep",
    "আস": "come",
}

# (suffix, morph_gloss) for each tense/person slot.
# Format follows: stem + suffix -> surface, with '+' inserted for gold segmentation.
VERB_SUFFIXES = {
    "PRES.1": "ি",      # present indefinite, 1st person      e.g. কর+ি
    "PRES.2.INTIM": "িস",  # present, 2nd person intimate
    "PRES.3": "ে",      # present, 3rd person / 2nd familiar
    "PRES.3.FORM": "েন",  # present, 3rd/2nd formal
    "PROG.1": "ছি",     # present continuous, 1st person
    "PROG.3": "ছে",     # present continuous, 3rd person
    "PROG.3.FORM": "ছেন",
    "PAST.1": "লাম",    # simple past, 1st person
    "PAST.2": "লে",     # simple past, 2nd person
    "PAST.3": "ল",      # simple past, 3rd person
    "PAST.3.FORM": "লেন",
    "FUT.1": "ব",       # future, 1st person
    "FUT.3": "বে",      # future, 3rd person / 2nd familiar
    "FUT.3.FORM": "বেন",
}

VERB_GLOSS = {
    "PRES.1": "PRES.1P", "PRES.2.INTIM": "PRES.2P.INTIM", "PRES.3": "PRES.3P",
    "PRES.3.FORM": "PRES.3P.FORM", "PROG.1": "PROG.1P", "PROG.3": "PROG.3P",
    "PROG.3.FORM": "PROG.3P.FORM", "PAST.1": "PAST.1P", "PAST.2": "PAST.2P",
    "PAST.3": "PAST.3P", "PAST.3.FORM": "PAST.3P.FORM", "FUT.1": "FUT.1P",
    "FUT.3": "FUT.3P", "FUT.3.FORM": "FUT.3P.FORM",
}

# ---------------------------------------------------------------------------
# 2. Noun paradigms: case marking, pluralization, classifiers
# ---------------------------------------------------------------------------

NOUN_ROOTS = {
    "ছেলে": "boy", "মেয়ে": "girl", "বই": "book", "ঘর": "house",
    "গাছ": "tree", "নদী": "river", "শিক্ষক": "teacher", "ছাত্র": "student",
    "বাড়ি": "home", "দেশ": "country", "ভাষা": "language", "মানুষ": "person",
    "পাখি": "bird", "ফুল": "flower", "শহর": "city",
}

# animate vs inanimate changes which plural/classifier suffix attaches;
# encoded per-noun for accuracy rather than guessed automatically.
ANIMATE = {"ছেলে", "মেয়ে", "শিক্ষক", "ছাত্র", "মানুষ", "পাখি"}

NOUN_SUFFIXES_INANIMATE = {
    "DEF.SG": "টা",     # the (singular, informal classifier)
    "DEF.SG.FORM": "টি",  # the (singular, formal/small classifier)
    "PL": "গুলো",       # plural
    "PL.FORM": "গুলি",   # plural, formal register
    "OBJ": "টাকে",      # not standard for inanimate but kept minimal
    "LOC": "এ",         # locative "in/at"
    "GEN": "র",         # genitive "of"
}

NOUN_SUFFIXES_ANIMATE = {
    "DEF.SG": "টা",
    "PL": "রা",         # animate plural
    "PL.FORM": "গণ",     # formal/honorific plural
    "OBJ": "কে",        # objective/accusative "to/-obj"
    "GEN": "র",         # genitive "of"
    "LOC": "তে",        # locative, less common for animates but attested
}

NOUN_GLOSS = {
    "DEF.SG": "DEF.SG", "DEF.SG.FORM": "DEF.SG.FORM", "PL": "PL",
    "PL.FORM": "PL.FORM", "OBJ": "OBJ", "LOC": "LOC", "GEN": "GEN",
}

# ---------------------------------------------------------------------------
# 3. Build pairs
# ---------------------------------------------------------------------------

def build_verb_pairs():
    pairs = []
    for root, root_gloss in VERB_ROOTS.items():
        for slot, suffix in VERB_SUFFIXES.items():
            surface = root + suffix
            segmented = root + "+" + suffix
            gloss = f"{root_gloss}+{VERB_GLOSS[slot]}"
            pairs.append((surface, segmented, gloss))
    return pairs


def build_noun_pairs():
    pairs = []
    for root, root_gloss in NOUN_ROOTS.items():
        table = NOUN_SUFFIXES_ANIMATE if root in ANIMATE else NOUN_SUFFIXES_INANIMATE
        for slot, suffix in table.items():
            surface = root + suffix
            segmented = root + "+" + suffix
            gloss = f"{root_gloss}+{NOUN_GLOSS[slot]}"
            pairs.append((surface, segmented, gloss))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)

    pairs = build_verb_pairs() + build_noun_pairs()
    # de-duplicate identical surface->segmented pairs (a few suffixes coincide, e.g. genitive র)
    seen = set()
    uniq = []
    for p in pairs:
        if p[0] not in seen:
            uniq.append(p)
            seen.add(p[0])
    pairs = uniq

    random.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * args.test_frac)
    n_dev = int(n * args.dev_frac)

    test = pairs[:n_test]
    dev = pairs[n_test:n_test + n_dev]
    train = pairs[n_test + n_dev:]

    out_dir = Path(args.out_dir)
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        out_path = out_dir / f"{split_name}.tsv"
        with open(out_path, "w", encoding="utf-8") as f:
            for surface, segmented, gloss in split_data:
                f.write(f"{surface}\t{segmented}\t{gloss}\n")
        print(f"wrote {len(split_data):4d} pairs -> {out_path}")

    print(f"total unique pairs: {n}")


if __name__ == "__main__":
    main()
