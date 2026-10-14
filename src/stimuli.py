"""Synthetic stimuli for the endstopping / extra-classical RF experiments."""
import numpy as np

from .data import PATCH, SPAN_W, dog_whiten

PAD = 12  # canvas padding so the DoG filter has no border artefacts in the crop


def bar_region(length, width=3, angle_deg=0.0, contrast=0.5, scale=1.0, whiten=True,
               amplitude=2.0):
    """Dark bar on a mid-gray field, centered in the PATCH x SPAN_W region.

    The bar is centered at the region's centre, has the given length (px, along its
    axis) and width, and is rotated by angle_deg (0 = horizontal, i.e. extending
    along the 26 px axis into the flanking level-1 modules). The image is prewhitened
    exactly like the natural images and divided by `scale` (the training-set std).
    """
    H, W = PATCH + 2 * PAD, SPAN_W + 2 * PAD
    yy, xx = np.mgrid[0:H, 0:W]
    y = yy - (H - 1) / 2
    x = xx - (W - 1) / 2
    th = np.deg2rad(angle_deg)
    along = x * np.cos(th) + y * np.sin(th)
    across = -x * np.sin(th) + y * np.cos(th)
    mask = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
    canvas = np.full((H, W), 0.5)
    canvas[mask] = 0.5 - contrast
    if whiten:
        filt = dog_whiten(canvas) / scale
    else:  # raw dark bar on zero background, in units of the (unit-std) input
        filt = (canvas - 0.5) / contrast * amplitude
    return filt[PAD:PAD + PATCH, PAD:PAD + SPAN_W]


def multi_bar_region(bars, width=3, contrast=0.5, scale=1.0, whiten=True, amplitude=2.0):
    """Several dark bars, each (cx, cy, length, angle_deg) with (cx, cy) relative to the region centre."""
    H, W = PATCH + 2 * PAD, SPAN_W + 2 * PAD
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W), bool)
    for cx, cy, length, angle in bars:
        th = np.deg2rad(angle)
        x = xx - (W - 1) / 2 - cx
        y = yy - (H - 1) / 2 - cy
        along = x * np.cos(th) + y * np.sin(th)
        across = -x * np.sin(th) + y * np.cos(th)
        mask |= (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
    canvas = np.full((H, W), 0.5)
    canvas[mask] = 0.5 - contrast
    filt = dog_whiten(canvas) / scale if whiten else (canvas - 0.5) / contrast * amplitude
    return filt[PAD:PAD + PATCH, PAD:PAD + SPAN_W]
