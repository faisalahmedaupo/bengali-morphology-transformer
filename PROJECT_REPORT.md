# Project Report Template

## 1. Research question

Can a small character-level Transformer learn Bengali morphological segmentation from controlled training examples?

## 2. Dataset

Describe the number of examples, paradigm types, split strategy, and synthetic-data limitation.

## 3. Model

Describe the encoder-decoder Transformer, tokenization, positional encoding, and decoding method.

## 4. Experimental setup

Record:

- seed
- optimizer
- learning rate
- batch size
- epochs
- model dimensions
- hardware
- training time

## 5. Results

Report at minimum:

| Metric | Test |
|---|---:|
| Exact match | XX.XX% |
| CER | XX.XX% |
| Boundary precision | XX.XX% |
| Boundary recall | XX.XX% |
| Boundary F1 | XX.XX% |

## 6. Error analysis

Include at least 5 incorrect predictions and explain the likely cause.

## 7. Limitations

Be explicit about synthetic data, limited vocabulary, and incomplete linguistic coverage.

## 8. Future work

Compare against a rule-based baseline and a pretrained multilingual/Bengali model on a real annotated dataset.
