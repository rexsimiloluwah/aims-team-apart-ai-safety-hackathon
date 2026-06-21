import numpy as np
import pytest

from src.metrics.calibration import (
    accuracy,
    brier_score,
    expected_calibration_error,
    overconfidence_gap,
    reliability_bins,
)
from src.metrics.selective import (
    accuracy_at_coverage,
    auroc_error_detection,
    build_deployment_card,
    errors_removed_at_threshold,
    risk_coverage_curve,
    threshold_for_coverage,
)
from src.metrics.uncertainty import bootstrap_ci


def test_ece_perfectly_calibrated():
    # confidence 1.0 and always correct -> ECE 0
    conf = np.ones(20)
    correct = np.ones(20)
    assert expected_calibration_error(conf, correct, n_bins=15) == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident():
    # confidence 1.0 but 50% correct -> gap of 0.5 in that bin
    conf = np.ones(100)
    correct = np.array([1, 0] * 50, dtype=float)
    assert expected_calibration_error(conf, correct, n_bins=15) == pytest.approx(0.5, abs=1e-6)


def test_overconfidence_gap():
    conf = np.full(10, 0.9)
    correct = np.array([1, 0] * 5, dtype=float)  # 0.5 accuracy
    assert overconfidence_gap(conf, correct) == pytest.approx(0.4, abs=1e-9)


def test_brier_and_accuracy():
    conf = np.array([1.0, 0.0])
    correct = np.array([1.0, 0.0])
    assert brier_score(conf, correct) == pytest.approx(0.0)
    assert accuracy(correct) == pytest.approx(0.5)


def test_reliability_bins_counts():
    conf = np.array([0.05, 0.95, 1.0])
    correct = np.array([0.0, 1.0, 1.0])
    b = reliability_bins(conf, correct, n_bins=10)
    assert int(b["bin_count"].sum()) == 3
    assert b["bin_count"][-1] == 2  # 0.95 and 1.0 both land in the last bin


def test_auroc_perfect_separation():
    correct = np.array([1, 1, 0, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    assert auroc_error_detection(conf, correct) == pytest.approx(1.0)


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc_error_detection([0.5, 0.6], [1, 1]))


def test_risk_coverage_full_coverage_equals_error_rate():
    conf = np.linspace(0, 1, 10)
    correct = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)  # 70% acc
    rc = risk_coverage_curve(conf, correct)
    assert rc["coverage"][-1] == pytest.approx(1.0)
    assert rc["risk"][-1] == pytest.approx(0.3, abs=1e-9)
    assert np.isfinite(rc["aurc"])  # guards the np.trapz/trapezoid path


def test_threshold_for_coverage():
    conf = np.linspace(0.0, 1.0, 11)  # 0.0,0.1,...,1.0
    thr = threshold_for_coverage(conf, 0.5)  # answer exactly 5 of 11 (k=floor(0.5*11)=5)
    answered = int((conf >= thr).sum())
    assert answered == 5  # exact: catches off-by-one in the order statistic
    assert threshold_for_coverage(conf, 1.0) == float("-inf")


def test_accuracy_at_coverage_prefers_confident():
    conf = np.array([0.1, 0.2, 0.9, 0.95])
    correct = np.array([0.0, 0.0, 1.0, 1.0])
    res = accuracy_at_coverage(conf, correct, 0.5)
    assert res["accuracy"] == pytest.approx(1.0)  # top-2 most confident are both correct


def test_errors_removed_at_threshold():
    conf = np.array([0.1, 0.2, 0.9, 0.95])
    correct = np.array([0.0, 0.0, 1.0, 1.0])  # both errors are low-confidence
    res = errors_removed_at_threshold(conf, correct, threshold=0.5)
    assert res["frac_errors_removed"] == pytest.approx(1.0)
    assert res["coverage"] == pytest.approx(0.5)


def test_deployment_card_shape():
    per_lang = {
        "eng": (np.linspace(0, 1, 20), np.random.default_rng(0).integers(0, 2, 20).astype(float)),
        "yor": (np.linspace(0, 1, 20), np.random.default_rng(1).integers(0, 2, 20).astype(float)),
    }
    rows = build_deployment_card(per_lang, [0.5, 0.8])
    assert len(rows) == 4
    assert {"language", "target_coverage", "threshold", "frac_errors_removed"} <= set(rows[0])


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 200)
    correct = (rng.uniform(0, 1, 200) < conf).astype(float)  # roughly calibrated
    ci = bootstrap_ci(conf, correct, lambda c, y: float(np.mean(y)), n_boot=200, key="t")
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_bootstrap_ci_deterministic():
    conf = np.linspace(0, 1, 50)
    correct = (conf > 0.5).astype(float)
    f = lambda c, y: float(np.mean(y))  # noqa: E731
    a = bootstrap_ci(conf, correct, f, n_boot=100, key="same")
    b = bootstrap_ci(conf, correct, f, n_boot=100, key="same")
    assert a == b  # sha256-seeded -> reproducible
