import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from medunc.metrics import (
    aurc, balanced_accuracy, binary_auc, bootstrap_indices, bootstrap_metrics, brier, coverage_at_accuracy, ece, holm,
    macro_auc, nll, oracle_aurc, paired_p, reliability, risk_coverage, selective_accuracy,
)


def two_class(conf: float, n: int, n_correct: int) -> tuple[np.ndarray, np.ndarray]:
    """n images predicted as class 0 with confidence ``conf``; the first ``n_correct`` are labelled 0."""
    p = np.tile([conf, 1 - conf], (n, 1))
    y = np.r_[np.zeros(n_correct, int), np.ones(n - n_correct, int)]
    return p, y


def test_ece_zero_when_accuracy_matches_confidence():
    p1, y1 = two_class(0.8, 10, 8)
    p2, y2 = two_class(0.6, 10, 6)
    assert ece(np.r_[p1, p2], np.r_[y1, y2]) == pytest.approx(0.0, abs=1e-12)


def test_ece_of_overconfident_model():
    p, y = two_class(0.99, 100, 50)
    assert ece(p, y) == pytest.approx(0.49)
    r = reliability(p, y)
    assert r["count"].sum() == 100 and r.loc[r["count"] > 0, "accuracy"].item() == pytest.approx(0.5)


def test_nll_brier_and_balanced_accuracy():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    y = np.array([0, 1, 1, 1])
    assert nll(p, y) == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4, 0.3])))
    assert brier(p, y) == pytest.approx(np.mean([0.02, 0.08, 0.72, 0.98]))
    assert balanced_accuracy(p, y) == pytest.approx((1.0 + 1 / 3) / 2)


def test_macro_auc_matches_scikit_learn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    logits = rng.normal(size=(500, 4)) + 1.5 * np.eye(4)[y]
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert macro_auc(p, y) == pytest.approx(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    assert binary_auc(np.ones(10), np.arange(10) < 3) == pytest.approx(0.5)


def test_risk_coverage_and_selective_metrics():
    correct = np.array([1, 1, 0, 1, 0], bool)
    good = np.array([0.1, 0.2, 0.9, 0.3, 0.8])  # errors are the most uncertain
    bad = -good
    cov, risk = risk_coverage(good, correct)
    assert np.allclose(cov, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(risk, [0, 0, 0, 0.25, 0.4])
    assert aurc(good, correct) == pytest.approx(oracle_aurc(correct))
    assert aurc(bad, correct) > aurc(good, correct)
    assert selective_accuracy(good, correct, 0.6) == pytest.approx(1.0)
    assert coverage_at_accuracy(good, correct, 0.95) == pytest.approx(0.6)
    assert coverage_at_accuracy(bad, correct, 0.95) == 0.0


def test_bootstrap_paired_difference_and_holm():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    good = np.eye(3)[y] * 0.7 + 0.1
    noisy = np.full((300, 3), 1 / 3)
    draws = bootstrap_indices(300, 200, seed=0)
    a = bootstrap_metrics(good, y, 1 - good.max(1), draws)
    b = bootstrap_metrics(noisy + rng.normal(0, 0.01, (300, 3)).clip(-0.1, 0.1), y, np.zeros(300), draws)
    assert set(a) == {"accuracy", "balanced_accuracy", "auc", "nll", "ece", "aurc"}
    assert paired_p(a["accuracy"] - b["accuracy"]) < 1e-6
    assert paired_p(rng.normal(0, 1, 1000)) > 0.01
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
