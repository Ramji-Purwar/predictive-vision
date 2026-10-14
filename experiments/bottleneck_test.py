"""Does a level-2 bottleneck make the center prediction depend on the surround?"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.unexplained_vs_length import curve, LENGTHS, ROOT
from src.data import load_images, sample_regions
from src.model import PredictiveCodingNet, Params


def go(n2):
    images = load_images("train", 100)
    base = np.load(ROOT / "results" / "baseline_st6_seed0.npz")
    net = PredictiveCodingNet(Params(n2=n2, sigma2_td=10.0, alpha2=0.01, lam2=0.002,
                                     k2_init=0.5, decay_every=10 ** 6), seed=1)
    net.U1 = base["U1"].copy()
    net.scale = float(base["scale"])
    rng = np.random.default_rng(1)
    net.fit_r1_scale(sample_regions(images, 512, rng))
    for _ in range(4000):
        net.train_step(sample_regions(images, 4, rng), stage=2)
    reg = sample_regions(images, 256, rng)
    o = net.infer(reg)
    r1 = o["r1"].reshape(256, -1)
    ev = 1 - np.var(r1 - o["r1_td"].reshape(256, -1)) / np.var(r1)
    out = {}
    for w in (True, False):
        for a in (0, 90):
            rn, en = curve(net, a, w)
            out[(w, a)] = en / rn
    return n2, ev, out


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=220, suppress=True)
    with Pool(4) as p:
        res = p.map(go, [4, 8, 16, 32])
    print("lengths", LENGTHS)
    for n2, ev, out in res:
        print(f"n2={n2}  EV(natural)={ev:.2f}")
        for k, v in out.items():
            print(f"   whiten={k[0]!s:5} angle={k[1]:2d} err/|r|", v)
