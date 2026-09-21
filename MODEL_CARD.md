# Model Card — Bengali Morphology Transformer

## Model summary

A character-level Transformer encoder-decoder trained to transform Bengali surface forms into morpheme-segmented strings.

## Intended use

- NLP portfolio demonstration
- computational-linguistics experimentation
- teaching sequence-to-sequence modelling
- controlled morphology experiments

## Not intended for

- production Bengali morphological analysis
- clinical, legal, or other high-stakes language applications
- claims about complete Bengali morphology
- benchmarking against real-world systems without a real annotated test set

## Training data

The bundled training data is synthetic and generated from explicit rule tables in `data/generate_data.py`.

Because the data is synthetic, high test accuracy on the bundled benchmark should not be interpreted as evidence of strong performance on unrestricted Bengali text.

## Limitations

The model can memorize the small synthetic vocabulary and may fail on:

- unseen morphology
- unseen roots
- irregular forms
- compounds
- spelling variation
- dialectal variation
- natural corpus noise

## Ethical considerations

Bengali NLP is a low-resource setting relative to English. Releasing a small, reproducible benchmark can support experimentation, but synthetic labels should be clearly distinguished from expert annotations.

## Recommended reporting

When publishing results from this repository, report:

- dataset size
- whether roots are shared across train/test
- random seed
- model configuration
- exact-match accuracy
- boundary F1
- qualitative errors
- limitations
