"""Grad-CAM (Selvaraju et al., 2017) and the model-randomisation sanity check of Adebayo et al. (2018).

At 64 x 64 input, ResNet-18's layer3 output is a 4 x 4 grid (layer4 is 2 x 2), so the maps are coarse: each cell
covers 16 x 16 pixels. The sanity check re-initialises the top of the network in cascade (fc, then layer4, then
layer3) and measures how much the maps change; a map that survives randomisation does not explain the model.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn

from medunc.models import Classifier


def gradcam(model: Classifier, x: torch.Tensor, target: torch.Tensor, layer: str = "layer3") -> np.ndarray:
    """Low-resolution Grad-CAM maps (n, u, v) of the ``target`` class logits at ``layer`` (float32, dropout off).

    alpha_k = mean over the grid of d logit_c / d A_k; map = ReLU(sum_k alpha_k A_k).
    """
    model.eval()
    acts: dict[str, torch.Tensor] = {}
    module = getattr(model.backbone, layer)
    handle = module.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        x = x.float().contiguous()
        logits = model(x)
        a = acts["a"]
        grads = torch.autograd.grad(logits.gather(1, target.view(-1, 1)).sum(), a)[0]
    finally:
        handle.remove()
    alpha = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alpha * a).sum(dim=1))
    return cam.detach().cpu().numpy()


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsampling of (n, u, v) maps to (n, size, size), each scaled to a maximum of 1."""
    t = F.interpolate(torch.from_numpy(cam)[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    m = t.flatten(1).max(1).values.clamp_min(1e-12).view(-1, 1, 1)
    return (t / m).numpy()


def randomized(model: Classifier, blocks: list[str], seed: int = 0) -> Classifier:
    """Copy of ``model`` with the named blocks ("fc", "layer4", "layer3", ...) re-initialised at random."""
    torch.manual_seed(seed)
    out = copy.deepcopy(model)
    for name in blocks:
        mod = out.fc if name == "fc" else getattr(out.backbone, name)
        for m in mod.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
    return out


def rank_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation between paired maps (n, u, v); NaN when a map is constant."""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        fa, fb = a[i].ravel(), b[i].ravel()
        if fa.std() > 0 and fb.std() > 0:
            out[i] = spearmanr(fa, fb).statistic
    return out


def centre_share(cam: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Share of each upsampled map's mass inside the central square covering ``frac`` of the width and height."""
    n, h, w = cam.shape
    y0, y1 = int(h * (1 - frac) / 2), int(h * (1 + frac) / 2)
    x0, x1 = int(w * (1 - frac) / 2), int(w * (1 + frac) / 2)
    tot = cam.reshape(n, -1).sum(1)
    inner = cam[:, y0:y1, x0:x1].reshape(n, -1).sum(1)
    return np.where(tot > 0, inner / np.maximum(tot, 1e-12), np.nan)
