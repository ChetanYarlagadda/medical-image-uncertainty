"""Synthetic image corruptions at five severities, applied on the GPU to float (n, 3, h, w) batches in [0, 1].

Severities follow ImageNet-C (Hendrycks and Dietterich, 2019) for noise, contrast and brightness; the blur
widths are scaled to 64-pixel images. Noise is drawn from a generator seeded per image index, so every ensemble
member sees exactly the same corrupted images.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

SEVERITY = {
    "gaussian_noise": [0.08, 0.12, 0.18, 0.26, 0.38],  # noise standard deviation
    "gaussian_blur": [0.5, 0.75, 1.0, 1.5, 2.0],  # kernel sigma in pixels (64-pixel images)
    "contrast": [0.4, 0.3, 0.2, 0.1, 0.05],  # factor applied to the deviation from the image mean
    "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],  # value added to every pixel
}
LABEL = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur", "contrast": "Contrast loss",
         "brightness": "Brightness shift"}


def gaussian_noise(x: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add N(0, sigma^2) pixel noise drawn from a generator seeded with ``seed``."""
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    return x + sigma * torch.randn(x.shape, device=x.device, generator=gen)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding (kernel radius 3 sigma)."""
    radius = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Shrink every pixel's deviation from its image's mean intensity by ``factor``."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * factor + mean


def brightness(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Add ``delta`` to every pixel (clipped to [0, 1] by the caller)."""
    return x + delta


def make(name: str, severity: int, seed: int = 0) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Corruption ``name`` at ``severity`` (1-5) as ``f(batch, start_index)``."""
    p = SEVERITY[name][severity - 1]
    if name == "gaussian_noise":
        return lambda x, start: gaussian_noise(x, p, seed * 1_000_003 + severity * 10_007 + start)
    if name == "gaussian_blur":
        return lambda x, start: gaussian_blur(x, p)
    if name == "contrast":
        return lambda x, start: contrast(x, p)
    if name == "brightness":
        return lambda x, start: brightness(x, p)
    raise KeyError(name)
