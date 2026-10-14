"""BSDS500 loading, retina/LGN-style prewhitening, and patch sampling."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "images"

# Level-1 module geometry (paper Fig. 1c / Fig. 2a): three 16x16 patches,
# offset horizontally by 5 px, so level 2 spans 16 x 26 pixels.
PATCH = 16
OFFSET = 5
SPAN_W = PATCH + 2 * OFFSET


def dog_whiten(img, sigma_c=1.0, sigma_s=1.6):
    """Center-surround (difference-of-Gaussians) filter, as in the paper's retina/LGN stage."""
    return gaussian_filter(img, sigma_c) - gaussian_filter(img, sigma_s)


def load_images(split="train", n=None, return_scale=False):
    """Prewhitened grayscale images rescaled to unit std (and the scale, if return_scale)."""
    paths = sorted((DATA_DIR / split).glob("*.jpg"))
    if n is not None:
        paths = paths[:n]
    imgs = []
    for p in paths:
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0
        imgs.append(dog_whiten(g))
    scale = image_scale(imgs)
    out = [i / scale for i in imgs]
    return (out, scale) if return_scale else out


def image_scale(imgs):
    """Std of the prewhitened (unscaled) images; dividing by it gives unit-variance input."""
    return float(np.std(np.concatenate([i.ravel() for i in imgs])))


def sample_regions(images, n, rng):
    """Sample n regions of size PATCH x SPAN_W. Returns array (n, PATCH, SPAN_W)."""
    out = np.empty((n, PATCH, SPAN_W))
    for k in range(n):
        im = images[rng.integers(len(images))]
        y = rng.integers(0, im.shape[0] - PATCH + 1)
        x = rng.integers(0, im.shape[1] - SPAN_W + 1)
        out[k] = im[y:y + PATCH, x:x + SPAN_W]
    return out
