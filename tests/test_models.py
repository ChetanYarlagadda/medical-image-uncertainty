import numpy as np
import pytest
import torch

from medunc import gradcam
from medunc.models import Classifier, TrainConfig, augment, class_weights, normalize, predict, to_float, train


@pytest.fixture(scope="module")
def net() -> Classifier:
    torch.manual_seed(0)
    return Classifier(n_classes=3, dropout=0.5, pretrained=False).eval()


def toy_images(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 64 x 64 images whose class is the brightness level (dark, mid, bright)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    x = (y[:, None, None, None] * 90 + rng.integers(0, 40, (n, 64, 64, 3))).astype(np.uint8)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def test_shapes_and_dropout_only_in_training_mode(net):
    x, _ = toy_images(4)
    logits, feats = predict(net, x)
    assert logits.shape == (4, 3) and feats.shape == (4, 512)
    xb = normalize(to_float(x))
    with torch.no_grad():
        assert torch.allclose(net(xb), net(xb))  # eval: deterministic
        net.train()
        assert not torch.allclose(net(xb), net(xb))  # train: dropout active
        net.eval()


def test_augment_applies_one_of_the_eight_symmetries():
    x = torch.rand(6, 3, 8, 8)
    out = augment(x, torch.Generator().manual_seed(0), pad=0)
    for i in range(6):
        cands = [x[i].flip(*dims) if dims else x[i] for dims in [(), (1,), (2,), (1, 2)]]
        cands += [c.transpose(1, 2) for c in cands]
        assert any(torch.equal(out[i], c) for c in cands)
    shifted = augment(x, torch.Generator().manual_seed(1), pad=2)
    assert shifted.shape == x.shape


def test_class_weights_upweight_rare_classes():
    y = np.array([0] * 90 + [1] * 10)
    assert class_weights(y, 2, 0.0) is None
    w = class_weights(y, 2, 0.5).numpy()
    assert w[1] / w[0] == pytest.approx(3.0)
    assert w[y].mean() == pytest.approx(1.0)


def test_training_runs_and_keeps_the_best_checkpoint():
    torch.manual_seed(0)
    x, y = toy_images(48)
    xv, yv = toy_images(24, seed=1)
    m = Classifier(3, pretrained=False)
    state, hist = train(m, x, y, xv, yv, TrainConfig(max_epochs=2, batch_size=16, lr=1e-3), seed=0, log=lambda s: None)
    assert len(hist) == 2 and {"train_loss", "val_nll", "val_acc"} <= set(hist.columns)
    best = hist.val_nll.min()
    xx = predict(m, xv)[0]
    nll = torch.nn.functional.cross_entropy(torch.from_numpy(xx), yv).item()
    assert nll == pytest.approx(best, rel=1e-3)  # the model holds the lowest-validation-NLL weights
    assert set(state) == set(m.state_dict())


def test_gradcam_maps_and_randomisation(net):
    x, _ = toy_images(3)
    xb = normalize(to_float(x))
    target = torch.tensor([0, 1, 2])
    cam = gradcam.gradcam(net, xb, target, "layer3")
    assert cam.shape == (3, 4, 4) and (cam >= 0).all()
    up = gradcam.upsample(cam, 64)
    assert up.shape == (3, 64, 64) and np.all(up <= 1 + 1e-6)
    r = gradcam.randomized(net, ["fc", "layer4"])
    assert not torch.equal(r.fc.weight, net.fc.weight)
    assert torch.equal(r.backbone.layer3[0].conv1.weight, net.backbone.layer3[0].conv1.weight)
    blob = np.zeros((1, 64, 64))
    blob[0, 24:40, 24:40] = 1
    assert gradcam.centre_share(blob)[0] == pytest.approx(1.0)
    assert gradcam.centre_share(np.ones((1, 64, 64)))[0] == pytest.approx(0.25)
    a = np.arange(16.0).reshape(1, 4, 4)
    assert gradcam.rank_similarity(a, a)[0] == pytest.approx(1.0)
    assert gradcam.rank_similarity(a, -a)[0] == pytest.approx(-1.0)
