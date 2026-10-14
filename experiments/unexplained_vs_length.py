"""Fraction of the central level-1 response left unexplained by the level-2 prediction vs bar length."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_images, sample_regions
from src.model import PredictiveCodingNet, Params
from src.stimuli import bar_region

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = np.arange(1, 27, 2)


def train_level2(sigma2_td, alpha2, lam2, k2, steps=3000, seed=1):
    images = load_images("train", 100)
    base = np.load(ROOT / "results" / "baseline_st6_seed0.npz")
    net = PredictiveCodingNet(Params(sigma2_td=sigma2_td, alpha2=alpha2, lam2=lam2,
                                     k2_init=k2, decay_every=10 ** 6), seed=seed)
    net.U1 = base["U1"].copy()
    net.scale = float(base["scale"])
    rng = np.random.default_rng(seed)
    net.fit_r1_scale(sample_regions(images, 512, rng))
    for _ in range(steps):
        net.train_step(sample_regions(images, 4, rng), stage=2)
    return net


def curve(net, angle, whiten):
    regs = np.stack([bar_region(L, 3, angle, 0.5, net.scale, whiten) for L in LENGTHS])
    o = net.infer(regs, feedback=True, n_iter=200)
    r = o["r1"][:, 1, :]
    e = o["err1"][:, 1, :]
    return np.linalg.norm(r, axis=1), np.linalg.norm(e, axis=1)


if __name__ == "__main__":
    net = train_level2(10.0, 0.01, 0.002, 0.5)
    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    print("lengths", LENGTHS)
    for whiten in (True, False):
        for angle in (0, 90):
            rn, en = curve(net, angle, whiten)
            print(f"whiten={whiten} angle={angle}\n  |r|     {rn}\n  |err|   {en}\n  err/|r| {en / rn}")
