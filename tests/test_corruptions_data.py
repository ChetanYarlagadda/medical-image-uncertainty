import hashlib

import numpy as np
import pytest
import torch

from medunc import corruptions
from medunc.data import CLASSES, Split, class_table, crosscheck, filename, md5


def batch(seed: int = 0) -> torch.Tensor:
    return torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(seed))


def high_freq(x: torch.Tensor) -> float:
    """Mean absolute difference between horizontally adjacent pixels."""
    return float((x[..., 1:] - x[..., :-1]).abs().mean())


def test_every_corruption_keeps_the_shape_and_worsens_with_severity():
    x = batch()
    for name in corruptions.SEVERITY:
        outs = [corruptions.make(name, s)(x, 0).clamp(0, 1) for s in range(1, 6)]
        assert all(o.shape == x.shape for o in outs)
        dist = [float((o - x).abs().mean()) for o in outs]
        assert all(b >= a - 1e-6 for a, b in zip(dist, dist[1:])), name


def test_corruption_parameters():
    x = batch()
    noise = corruptions.make("gaussian_noise", 3)(x, 0) - x
    assert float(noise.std()) == pytest.approx(corruptions.SEVERITY["gaussian_noise"][2], rel=0.05)
    blurred = [corruptions.make("gaussian_blur", s)(x, 0) for s in (1, 5)]
    assert high_freq(blurred[1]) < high_freq(blurred[0]) < high_freq(x)
    c = corruptions.make("contrast", 2)(x, 0)
    ratio = (c - c.mean((1, 2, 3), keepdim=True)).std() / (x - x.mean((1, 2, 3), keepdim=True)).std()
    assert float(ratio) == pytest.approx(corruptions.SEVERITY["contrast"][1], rel=1e-4)
    assert torch.allclose(corruptions.make("brightness", 1)(x, 0), x + 0.1)


def test_noise_is_reproducible_per_batch_offset():
    x = batch()
    f = corruptions.make("gaussian_noise", 2)
    assert torch.equal(f(x, 0), f(x, 0))
    assert not torch.equal(f(x, 0), f(x, 1024))


def test_crosscheck_and_class_table():
    rng = np.random.default_rng(0)
    n = {"train": 7_007, "val": 1_003, "test": 2_005}
    data = {s: Split(np.zeros((k, 2, 2, 3), np.uint8), rng.integers(0, 7, k)) for s, k in n.items()}
    for s in data:
        data[s].y[:7] = np.arange(7)  # every class present
    chk = crosscheck("dermamnist", data)
    assert chk.match.all() and len(chk) == 4
    data["test"] = data["test"].take(np.arange(100))
    assert not crosscheck("dermamnist", data).match.all()
    t = class_table("dermamnist", data)
    assert len(t) == len(CLASSES["dermamnist"]) and t.train.sum() == 7_007
    assert t.train_share.sum() == pytest.approx(1.0)


def test_md5_and_file_names(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"medmnist")
    assert md5(p) == hashlib.md5(b"medmnist").hexdigest()
    assert filename("bloodmnist", 28) == "bloodmnist.npz" and filename("bloodmnist", 64) == "bloodmnist_64.npz"
