"""Predictive distributions of the six methods and their uncertainty scores.

Methods are built from stored logits only, so nothing here needs a GPU:

* single model: softmax of ensemble member 0;
* MC dropout: member 0 with its head dropout left on, averaged over S stochastic passes (Gal and Ghahramani, 2016);
* deep ensemble: average softmax of M independently trained members (Lakshminarayanan et al., 2017);
* each of the three with temperature scaling (Guo et al., 2017): logits divided by one T fitted on validation NLL.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax, xlogy

EPS = 1e-12


def mc_dropout_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, p: float, n_samples: int,
                      seed: int) -> np.ndarray:
    """Logits (S, n, C) of ``n_samples`` dropout passes through the linear head on fixed penultimate features.

    The backbone has no dropout, so only the head is stochastic and the features can be reused for every pass.
    Masks keep each feature with probability 1 - p and scale it by 1 / (1 - p) (inverted dropout, as in training).
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, features.shape[0], weight.shape[0]), dtype=np.float32)
    for s in range(n_samples):
        mask = (rng.random(features.shape, dtype=np.float32) >= p).astype(np.float32) / (1 - p)
        out[s] = (features * mask) @ weight.T + bias
    return out


def mean_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Average softmax over the leading sample axis of (S, n, C) logits (or plain softmax of (n, C))."""
    p = softmax(logits / temperature, axis=-1)
    return p.mean(axis=0) if p.ndim == 3 else p


def nll_of(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Negative log-likelihood of the averaged predictive distribution at a temperature."""
    if logits.ndim == 2:
        lp = log_softmax(logits / temperature, axis=-1)
        return float(-lp[np.arange(len(y)), y].mean())
    p = mean_probs(logits, temperature)
    return float(-np.log(p[np.arange(len(y)), y] + EPS).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Temperature minimising validation NLL (bounded scalar search over log T).

    For (S, n, C) logits one shared T scales every sample or member before averaging.
    """
    res = minimize_scalar(lambda lt: nll_of(logits, y, float(np.exp(lt))), bounds=np.log(bounds), method="bounded",
                          options={"xatol": 1e-4})
    return float(np.exp(res.x))


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row of (n, C) probabilities."""
    return -xlogy(p, p).sum(axis=-1)


def mutual_information(sample_probs: np.ndarray) -> np.ndarray:
    """Epistemic part of the predictive entropy: H[mean p] - mean H[p] over the samples (S, n, C)."""
    return entropy(sample_probs.mean(axis=0)) - entropy(sample_probs).mean(axis=0)


def residual_mass(probs: np.ndarray) -> np.ndarray:
    """1 - max probability, computed as the summed probability of the other classes.

    Subtracting the max from 1 rounds every very confident prediction to exactly 0 and turns the ranking of
    those images into ties; summing the small probabilities keeps them distinct.
    """
    return np.sort(probs, axis=-1)[..., :-1].sum(axis=-1)


def scores(probs: np.ndarray, sample_probs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Uncertainty scores (higher = less certain): 1 - max probability, entropy and, with samples, mutual information."""
    out = {"1 - max prob": residual_mass(probs), "entropy": entropy(probs)}
    if sample_probs is not None and sample_probs.shape[0] > 1:
        out["mutual information"] = mutual_information(sample_probs)
    return out


def method_samples(member_logits: np.ndarray, mc_logits: np.ndarray) -> dict[str, np.ndarray]:
    """Logit stacks (S, n, C) of the three base methods: member 0, MC dropout, deep ensemble."""
    return {"Single model": member_logits[:1], "MC dropout": mc_logits, "Deep ensemble": member_logits}


def logit_adjust(probs: np.ndarray, prior: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Post-hoc logit adjustment (Menon et al., 2021): p_c / prior_c ** tau, renormalised (balanced decisions)."""
    q = probs / prior[None, :] ** tau
    return q / q.sum(axis=1, keepdims=True)
