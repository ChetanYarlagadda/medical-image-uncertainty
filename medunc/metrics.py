"""Classification, calibration and selective-prediction metrics, paired bootstrap and Holm correction.

All functions take class probabilities (n, C) and integer labels (n,). The expected calibration error is the
top-label ECE with 15 equal-width confidence bins (Guo et al., 2017); the risk-coverage curve ranks images by
confidence and reports the error rate of the most confident fraction (Geifman and El-Yaniv, 2017).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

EPS = 1e-12


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    """Share of images whose most probable class is the label."""
    return float((p.argmax(1) == y).mean())


def balanced_accuracy(p: np.ndarray, y: np.ndarray) -> float:
    """Mean per-class recall over the classes present in ``y``."""
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def per_class_recall(p: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Recall of every class (NaN for a class without images)."""
    pred = p.argmax(1)
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(n_classes)])


def binary_auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for the boolean target ``pos`` (Mann-Whitney U with tie ranks)."""
    n_pos = int(pos.sum())
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(p: np.ndarray, y: np.ndarray) -> float:
    """Mean one-vs-rest AUC over classes (the MedMNIST evaluator's multi-class AUC)."""
    return float(np.nanmean([binary_auc(p[:, c], y == c) for c in range(p.shape[1])]))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the labels."""
    return float(-np.log(p[np.arange(len(y)), y] + EPS).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean squared distance between probabilities and the one-hot label."""
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def _bins(p: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin count, summed confidence and summed correctness of the top-label prediction."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    return (np.bincount(b, minlength=n_bins), np.bincount(b, weights=conf, minlength=n_bins),
            np.bincount(b, weights=correct, minlength=n_bins))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per-bin count, mean confidence and accuracy of the top-label prediction (equal-width bins on [0, 1])."""
    count, s_conf, s_acc = _bins(p, y, n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"bin": np.arange(n_bins), "lower": np.arange(n_bins) / n_bins,
                             "upper": np.arange(1, n_bins + 1) / n_bins, "count": count,
                             "confidence": s_conf / count, "accuracy": s_acc / count})


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: sum over bins of |accuracy - confidence| weighted by the bin's share of images.

    ECE = sum_b (n_b / n) |acc_b - conf_b|; with per-bin sums this is sum_b |sum correct - sum conf| / n.
    """
    _, s_conf, s_acc = _bins(p, y, n_bins)
    return float(np.abs(s_acc - s_conf).sum() / len(y))


def risk_coverage(uncertainty: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and selective risk (error rate of the k most certain images) for k = 1..n.

    Ties in ``uncertainty`` are broken by a stable sort, so the curve is deterministic.
    """
    order = np.argsort(uncertainty, kind="stable")
    err = 1.0 - correct[order].astype(float)
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over all coverages); lower is better."""
    return float(risk_coverage(uncertainty, correct)[1].mean())


def oracle_aurc(correct: np.ndarray) -> float:
    """AURC of a perfect ranking that defers every error first (the lower bound for this error rate)."""
    return aurc((~correct.astype(bool)).astype(float), correct)


def selective_accuracy(uncertainty: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the ``coverage`` share of images with the lowest uncertainty."""
    k = max(1, int(round(coverage * len(correct))))
    order = np.argsort(uncertainty, kind="stable")
    return float(correct[order[:k]].mean())


def coverage_at_accuracy(uncertainty: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Largest coverage whose selective accuracy is at least ``target`` (0 if even the most certain image fails)."""
    cov, risk = risk_coverage(uncertainty, correct)
    ok = np.flatnonzero(1.0 - risk >= target - 1e-12)
    return float(cov[ok[-1]]) if len(ok) else 0.0


def summarize(p: np.ndarray, y: np.ndarray, uncertainty: np.ndarray, target_accuracy: float) -> dict[str, float]:
    """Every headline metric of one predictive distribution with one uncertainty score."""
    correct = p.argmax(1) == y
    return {
        "accuracy": accuracy(p, y), "balanced_accuracy": balanced_accuracy(p, y), "auc": macro_auc(p, y),
        "nll": nll(p, y), "brier": brier(p, y), "ece": ece(p, y),
        "aurc": aurc(uncertainty, correct), "oracle_aurc": oracle_aurc(correct),
        "error_detection_auroc": binary_auc(uncertainty, ~correct),
        "selective_acc_90": selective_accuracy(uncertainty, correct, 0.9),
        "selective_acc_80": selective_accuracy(uncertainty, correct, 0.8),
        "coverage_at_target": coverage_at_accuracy(uncertainty, correct, target_accuracy),
    }


def bootstrap_indices(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n) indices of images resampled with replacement; reuse the same draws for paired comparisons."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


BOOT_METRICS = ("accuracy", "balanced_accuracy", "auc", "nll", "ece", "aurc")


def bootstrap_metrics(p: np.ndarray, y: np.ndarray, uncertainty: np.ndarray, draws: np.ndarray) -> dict[str, np.ndarray]:
    """Metric values on every bootstrap resample of the test images (rows of ``draws``)."""
    correct = p.argmax(1) == y
    out = {m: np.empty(len(draws)) for m in BOOT_METRICS}
    for b, idx in enumerate(draws):
        pb, yb = p[idx], y[idx]
        out["accuracy"][b] = correct[idx].mean()
        out["balanced_accuracy"][b] = balanced_accuracy(pb, yb)
        out["auc"][b] = macro_auc(pb, yb)
        out["nll"][b] = nll(pb, yb)
        out["ece"][b] = ece(pb, yb)
        out["aurc"][b] = aurc(uncertainty[idx], correct[idx])
    return out


def paired_p(diff: np.ndarray) -> float:
    """Two-sided p-value that a paired difference is zero, from its bootstrap mean and standard error (normal approx.)."""
    se = float(np.std(diff, ddof=1))
    if se == 0:
        return 0.0 if float(np.mean(diff)) != 0 else 1.0
    return float(2 * norm.sf(abs(float(np.mean(diff))) / se))


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr)
    m = len(p_arr)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_arr[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
