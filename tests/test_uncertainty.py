import numpy as np
import pytest
from scipy.special import softmax

from medunc.uncertainty import (
    entropy, fit_temperature, logit_adjust, mc_dropout_logits, mean_probs, method_samples, mutual_information,
    residual_mass, scores,
)


def test_temperature_scaling_recovers_the_true_temperature():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=(20_000, 5))
    p_true = softmax(z / 2.0, axis=1)  # labels drawn at temperature 2: the raw logits are overconfident
    y = (rng.random((len(z), 1)) > p_true.cumsum(1)).sum(1).clip(0, 4)  # one categorical draw per row
    assert fit_temperature(z, y) == pytest.approx(2.0, rel=0.05)
    # one shared temperature for a stack of identical members gives the same answer
    assert fit_temperature(np.stack([z, z]), y) == pytest.approx(2.0, rel=0.05)


def test_mc_dropout_without_dropout_is_deterministic_and_unbiased_with_it():
    rng = np.random.default_rng(1)
    f = rng.random((50, 16)).astype(np.float32)
    w = rng.normal(size=(3, 16)).astype(np.float32)
    b = rng.normal(size=3).astype(np.float32)
    det = f @ w.T + b
    out = mc_dropout_logits(f, w, b, p=0.0, n_samples=4, seed=0)
    assert out.shape == (4, 50, 3)
    assert np.allclose(out, det[None], atol=1e-5)
    many = mc_dropout_logits(f, w, b, p=0.5, n_samples=4000, seed=0)
    assert np.allclose(many.mean(0), det, atol=0.2)  # inverted dropout keeps the expected logit
    assert many.std(0).mean() > 0.1


def test_mutual_information_and_scores():
    agree = np.tile(softmax(np.array([[2.0, 0.0, 0.0]]), axis=1), (5, 4, 1))
    assert np.allclose(mutual_information(agree), 0.0, atol=1e-9)
    disagree = np.stack([np.tile([[0.98, 0.01, 0.01]], (4, 1)), np.tile([[0.01, 0.98, 0.01]], (4, 1))])
    assert (mutual_information(disagree) > 0.5).all()
    assert entropy(np.full((1, 4), 0.25))[0] == pytest.approx(np.log(4))
    s = scores(disagree.mean(0), disagree)
    assert set(s) == {"1 - max prob", "entropy", "mutual information"}
    assert "mutual information" not in scores(agree[0], agree[:1])


def test_method_stacks_and_mean_probs():
    logits = np.random.default_rng(2).normal(size=(5, 10, 4))
    mc = logits[:3]
    stacks = method_samples(logits, mc)
    assert stacks["Single model"].shape == (1, 10, 4) and stacks["Deep ensemble"].shape == (5, 10, 4)
    p = mean_probs(logits)
    assert np.allclose(p.sum(1), 1.0) and np.allclose(p, softmax(logits, axis=-1).mean(0))


def test_logit_adjustment_moves_mass_to_rare_classes():
    p = np.array([[0.6, 0.4], [0.3, 0.7]])
    assert np.allclose(logit_adjust(p, np.array([0.5, 0.5])), p)
    adj = logit_adjust(p, np.array([0.9, 0.1]))
    assert np.allclose(adj.sum(1), 1.0) and adj[0, 1] > p[0, 1] and adj.argmax(1).tolist() == [1, 1]


def test_residual_mass_keeps_confident_predictions_apart():
    p = softmax(np.array([[30.0, 0.0, 0.0], [25.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32), axis=1)
    naive = 1 - p.max(1)
    assert naive[0] == naive[1] == 0.0  # float32 rounding: a tie
    r = residual_mass(p)
    assert 0 < r[0] < r[1] < r[2]
    assert r[2] == pytest.approx(naive[2], rel=1e-5)
    assert entropy(p)[0] < entropy(p)[1]
