"""Fig. 4 check: do BSDS natural images have long-range correlation along the dominant orientation?

For random 16x16 patches, the dominant orientation is the one with the largest oriented energy
(sum of squared quadrature Gabor outputs). Correlation is then computed between the patch and a
copy translated up to +-50 px along that orientation vs. the orthogonal direction.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
P = 16
MAXSHIFT = 50
ORIENTS = {0: "horizontal (0°)", 45: "diagonal (45°)", 90: "vertical (90°)", 135: "diagonal (135°)"}


def gabor_pair(theta_deg, wavelength=6.0, sigma=3.0, size=15):
    """Quadrature Gabor pair whose stripes run ALONG direction theta (0=horizontal, 90=vertical)."""
    th = np.deg2rad(theta_deg)
    ax = np.arange(size) - size // 2
    x, y = np.meshgrid(ax, ax)
    along = x * np.cos(th) - y * np.sin(th)       # y axis points down; 90° = vertical
    across = x * np.sin(th) + y * np.cos(th)
    env = np.exp(-(along ** 2 + across ** 2) / (2 * sigma ** 2))
    return env * np.cos(2 * np.pi * across / wavelength), env * np.sin(2 * np.pi * across / wavelength)


def direction(theta_deg):
    th = np.deg2rad(theta_deg)
    return np.cos(th), -np.sin(th)   # (dx, dy) in image coordinates, y down


def patch_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > 0 else np.nan


def run(images, n_per_image, rng, scale=1):
    shifts = np.arange(-MAXSHIFT, MAXSHIFT + 1, 2)
    acc = {t: [[], []] for t in ORIENTS}   # along, orthogonal -> list of arrays over shifts
    for img in images:
        if scale > 1:
            img = img[::scale, ::scale]
        energy = {}
        for t in ORIENTS:
            e, o = gabor_pair(t)
            energy[t] = convolve(img, e) ** 2 + convolve(img, o) ** 2
        H, W = img.shape
        m = MAXSHIFT + P
        if H < 2 * m + P or W < 2 * m + P:
            continue
        for _ in range(n_per_image):
            y = rng.integers(m, H - m - P); x = rng.integers(m, W - m - P)
            en = {t: energy[t][y:y + P, x:x + P].sum() for t in ORIENTS}
            dom = max(en, key=en.get)
            base = img[y:y + P, x:x + P]
            for k, ang in enumerate((dom, (dom + 90) % 180)):
                dx, dy = direction(ang)
                c = []
                for s in shifts:
                    yy, xx = int(round(y + s * dy)), int(round(x + s * dx))
                    c.append(patch_corr(base, img[yy:yy + P, xx:xx + P]))
                acc[dom][k].append(c)
    return shifts, acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--per_image", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    paths = sorted((DATA_DIR / "train").glob("*.jpg"))[:args.n_images]
    imgs = [np.asarray(Image.open(p).convert("L"), dtype=float) / 255 for p in paths]
    noise = [rng.normal(size=(321, 481))]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    print("orientation   n   corr@+-20 along/orth   corr@+-40 along/orth")
    for row, (name, data, per) in enumerate((("BSDS natural", imgs, args.per_image),
                                             ("white noise", noise, 2000))):
        shifts, acc = run(data, per, rng)
        for ax, t in zip(axes[row], ORIENTS):
            al, orr = (np.nanmean(np.array(acc[t][k]), axis=0) if acc[t][k] else None for k in (0, 1))
            n = len(acc[t][0])
            if al is None:
                ax.set_title(f"{ORIENTS[t]}: no patches"); continue
            ax.plot(shifts, al, "k-", label="dominant dir")
            ax.plot(shifts, orr, "k--", label="orthogonal")
            ax.set_title(f"{name}: {ORIENTS[t]} (n={n})", fontsize=9)
            i20 = np.abs(shifts) == 20; i40 = np.abs(shifts) == 40
            print(f"{name[:5]:5s} {t:3d}° {n:6d}   {al[i20].mean():.2f}/{orr[i20].mean():.2f}"
                  f"            {al[i40].mean():.2f}/{orr[i40].mean():.2f}")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("distance (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("correlation")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig4_autocorrelation_bsds.png", dpi=150)
