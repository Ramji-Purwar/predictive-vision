"""Diagnose why level 2 doesn't give length-dependent suppression.

1. coupling: how much each level-2 unit's weights span center vs flank modules
2. context-only prediction: predict the center level-1 response from the flank responses alone
   (center input to level 2 masked out) - the paper's 'predict the center from the surround'
3. same, with level 2 trained only on high-orientation-coherence patches
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.fig4_autocorrelation import gabor_pair
from src.data import load_images, sample_regions
from src.model import PredictiveCodingNet, Params
from src.stimuli import bar_region

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = np.arange(1, 27, 2)
CONFIGS = {
    "A paper-params n2=128":      dict(p=dict(alpha2=1.0, lam2=0.02, k2_init=0.05), n2=128, sel=False),
    "B tuned       n2=128":       dict(p=dict(alpha2=0.01, lam2=0.002, k2_init=0.5), n2=128, sel=False),
    "C tuned       n2=32":        dict(p=dict(alpha2=0.01, lam2=0.002, k2_init=0.5), n2=32, sel=False),
    "D tuned n2=128 + structured": dict(p=dict(alpha2=0.01, lam2=0.002, k2_init=0.5), n2=128, sel=True),
}


def coherence(region):
    """Max-over-orientation oriented energy divided by the mean over orientations (>1 = oriented)."""
    en = []
    for t in (0, 45, 90, 135):
        e, o = gabor_pair(t)
        en.append((convolve(region, e) ** 2 + convolve(region, o) ** 2).sum())
    en = np.array(en)
    return en.max() / en.mean(), en.sum()


def structured_regions(images, n, rng, keep=0.25, pool=8):
    cand = sample_regions(images, int(n * pool), rng)
    coh = np.array([coherence(r)[0] for r in cand])
    idx = np.argsort(-coh)[: int(len(cand) * keep)]
    return cand[idx]


def masked_r2(net, r1, mask):
    """Settle r2 using only the unmasked (flank) entries of the standardized level-1 input."""
    p = net.p
    x = r1.reshape(len(r1), -1) / net.r1_scale
    m = np.tile(mask, (1, 1))
    r2 = np.zeros((len(r1), p.n2))
    dt = min(1.0, 0.9 / (p.k1_l2 * (np.linalg.norm(net.U2, 2) ** 2 / p.sigma2_td + p.alpha2)))
    for _ in range(200):
        r2 = r2 + dt * ((p.k1_l2 / p.sigma2_td) * (m * (x - r2 @ net.U2.T)) @ net.U2
                        - p.k1_l2 * p.alpha2 * r2)
    return r2


def center_pred(net, r1, masked):
    n1 = net.p.n1
    mask = np.ones(3 * n1)
    if masked:
        mask[n1:2 * n1] = 0.0
    r2 = masked_r2(net, r1, mask)
    pred = net.r1_scale * (r2 @ net.U2.T)
    return pred.reshape(len(r1), 3, n1)[:, 1, :]


def run(name):
    cfg = CONFIGS[name]
    images = load_images("train", 100)
    base = np.load(ROOT / "results" / "baseline_st6_seed0.npz")
    net = PredictiveCodingNet(Params(n2=cfg["n2"], sigma2_td=10.0, decay_every=10 ** 6, **cfg["p"]), seed=1)
    net.U1 = base["U1"].copy()
    net.scale = float(base["scale"])
    rng = np.random.default_rng(1)
    net.fit_r1_scale(sample_regions(images, 512, rng))
    for _ in range(4000):
        reg = structured_regions(images, 4, rng) if cfg["sel"] else sample_regions(images, 4, rng)
        net.train_step(reg[:4], stage=2)

    out = {"name": name}
    # 1. coupling: share of each unit's weight energy in center / flank blocks
    n1 = net.p.n1
    W = net.U2 ** 2
    blocks = np.stack([W[0:n1].sum(0), W[n1:2 * n1].sum(0), W[2 * n1:].sum(0)])
    share = blocks / (blocks.sum(0) + 1e-12)
    out["center_share_mean"] = float(share[1].mean())
    out["mixed_units"] = float(np.mean((share[1] > 0.15) & (share[1] < 0.6)))
    # 2. context-only prediction on natural images
    test = sample_regions(load_images("val", 100), 400, rng)
    r1 = net.infer(test, feedback=False)["r1"]
    c = r1[:, 1, :]
    for key, masked in (("ev_full", False), ("ev_ctx", True)):
        pred = center_pred(net, r1, masked)
        out[key] = float(1 - np.sum((c - pred) ** 2) / np.sum(c ** 2))
    # 3. bars: unexplained fraction of center response, context-only
    for whiten in (True, False):
        for ang in (0, 90):
            regs = np.stack([bar_region(L, 3, ang, 0.5, net.scale, whiten) for L in LENGTHS])
            r1b = net.infer(regs, feedback=False)["r1"]
            cb = r1b[:, 1, :]
            pred = center_pred(net, r1b, True)
            out[(whiten, ang)] = np.linalg.norm(cb - pred, axis=1) / np.linalg.norm(cb, axis=1)
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    with Pool(4) as p:
        res = p.map(run, list(CONFIGS))
    print("lengths", LENGTHS)
    for o in res:
        print(f"\n{o['name']}")
        print(f"  coupling: center share of U2 energy {o['center_share_mean']:.2f}, "
              f"units mixing center+flank {o['mixed_units']:.2f}")
        print(f"  natural images: EV(center | all) {o['ev_full']:.2f}   EV(center | flanks only) {o['ev_ctx']:.2f}")
        for k in ((True, 0), (True, 90), (False, 0), (False, 90)):
            print(f"  bars whiten={k[0]!s:5} angle={k[1]:2d}  unexplained(ctx) {o[k]}")
