"""Metrics for Bengali morphological segmentation."""

from __future__ import annotations

from typing import Iterable


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = cur
    return prev[-1]


def boundary_positions(segmented: str) -> set[int]:
    """Return character offsets at which a morpheme boundary occurs."""
    pos = 0
    boundaries = set()
    for ch in segmented:
        if ch == "+":
            boundaries.add(pos)
        else:
            pos += 1
    return boundaries


def boundary_f1(gold: str, predicted: str) -> tuple[float, float, float]:
    """Compute boundary precision, recall and F1."""
    gold_b = boundary_positions(gold)
    pred_b = boundary_positions(predicted)

    if not gold_b and not pred_b:
        return 1.0, 1.0, 1.0

    tp = len(gold_b & pred_b)
    precision = tp / len(pred_b) if pred_b else 0.0
    recall = tp / len(gold_b) if gold_b else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def segmentation_report(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Return exact match, CER and mean boundary P/R/F1."""
    pairs = list(pairs)
    if not pairs:
        return {
            "exact_match": 0.0,
            "cer": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
        }

    exact = 0
    edit_total = 0
    gold_len_total = 0
    ps, rs, fs = [], [], []

    for gold, pred in pairs:
        exact += int(gold == pred)
        edit_total += levenshtein(pred, gold)
        gold_len_total += max(len(gold), 1)
        p, r, f = boundary_f1(gold, pred)
        ps.append(p)
        rs.append(r)
        fs.append(f)

    n = len(pairs)
    return {
        "exact_match": exact / n,
        "cer": edit_total / gold_len_total,
        "boundary_precision": sum(ps) / n,
        "boundary_recall": sum(rs) / n,
        "boundary_f1": sum(fs) / n,
    }
