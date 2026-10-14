"""Sweep level-2 hyperparameters on top of a trained level 1; report endstopping with/without feedback."""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.endstopping import response_curves, endstopping_pct
from src.data import load_images, sample_regions
from src.model import PredictiveCodingNet, Params

ROOT = Path(__file__).resolve().parents[1]
L1_WEIGHTS = ROOT / "results" / "baseline_st6_seed0.npz"


def run(cfg):
    sigma2_td, alpha2, lam2, k2 = cfg
    images = load_images("train", 100)
    base = np.load(L1_WEIGHTS)
    net = PredictiveCodingNet(Params(sigma2_td=sigma2_td, alpha2=alpha2, lam2=lam2,
                                     k2_init=k2, decay_every=10 ** 6), seed=1)
    net.U1 = base["U1"].copy()
    net.scale = float(base["scale"])
    rng = np.random.default_rng(1)
    net.fit_r1_scale(sample_regions(images, 512, rng))
    for _ in range(3000):
        net.train_step(sample_regions(images, 4, rng), stage=2)
    # explained variance of level-1 responses by the level-2 prediction (natural images)
    reg = sample_regions(images, 256, rng)
    o = net.infer(reg)
    r1 = o["r1"].reshape(256, -1)
    ev = 1 - np.var(r1 - o["r1_td"].reshape(256, -1)) / np.var(r1)
    res = {}
    for whiten in (True, False):
        w = response_curves(net, True, whiten=whiten)
        n = response_curves(net, False, whiten=whiten)
        best = w.max(axis=1).argmax(axis=0)
        cw = np.stack([w[best[i], :, i] for i in range(w.shape[2])])
        cn = np.stack([n[best[i], :, i] for i in range(w.shape[2])])
        res[whiten] = (np.mean([endstopping_pct(c) for c in cw]),
                       np.mean([endstopping_pct(c) for c in cn]))
    return cfg, float(np.linalg.norm(net.U2, 2)), ev, res


if __name__ == "__main__":
    grid = list(itertools.product([1.0, 10.0], [1.0, 0.1, 0.01], [0.02, 0.002], [0.05, 0.5]))
    with Pool(12) as pool:
        out = pool.map(run, grid)
    print("sigma2_td alpha2 lam2   k2 | |U2|  EV  | DoG-bars ES fb/nofb | raw-bars ES fb/nofb")
    for cfg, u, ev, res in out:
        print(f"{cfg[0]:6.1f} {cfg[1]:6.2f} {cfg[2]:6.3f} {cfg[3]:5.2f} | {u:6.2f} {ev:5.2f} | "
              f"{res[True][0]:5.1f}/{res[True][1]:5.1f} | {res[False][0]:5.1f}/{res[False][1]:5.1f}")
