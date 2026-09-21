# Data Card

## Dataset name

Bengali Morphology Transformer — Synthetic Morphology Benchmark

## Data type

Synthetic, rule-generated Bengali surface-form / morpheme-segmentation pairs.

## Fields

| Field | Description |
|---|---|
| `surface_form` | Bengali word form |
| `segmented_form` | Same form with `+` inserted at proposed morpheme boundaries |
| `gloss` | Human-readable morphological label |

## Generation

`data/generate_data.py` contains the explicit paradigm tables and creates deterministic train/dev/test files.

The generator uses a fixed seed by default, making the benchmark reproducible.

## Intended purpose

The dataset exists to demonstrate the full machine-learning pipeline, not to serve as a comprehensive Bengali linguistic resource.

## Known limitations

- small vocabulary
- hand-authored paradigms
- incomplete Bengali morphology
- no natural frequency distribution
- no expert annotation protocol
- limited phonological alternation handling
- limited derivational morphology
