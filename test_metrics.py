import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import boundary_f1, segmentation_report  # noqa: E402


def test_perfect_boundary_score():
    assert boundary_f1("কর+ছি", "কর+ছি") == (1.0, 1.0, 1.0)


def test_boundary_mismatch_is_detected():
    p, r, f = boundary_f1("কর+ছি", "ক+রছি")
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_report_keys_and_values():
    report = segmentation_report([("কর+ছি", "কর+ছি"), ("খা+ব", "খ+াব")])
    assert report["exact_match"] == 0.5
    assert 0.0 <= report["boundary_f1"] <= 1.0
    assert report["cer"] >= 0.0
