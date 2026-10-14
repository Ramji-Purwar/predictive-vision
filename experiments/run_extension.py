"""Extension: quantify how error-neuron responses depend on context, with/without feedback, over seeds.

(a) Bar-length sweep (center + flanks along the bar axis): error norm of the central level-1
    module, suppression index S = 1 - plateau/peak, with feedback ('fb') vs without ('nofb').
(b) Surround-orientation sweep: center bar plus flanking bars at relative orientation delta.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_images, sample_regions
from src.model import PredictiveCodingNet, Params
from src.stimuli import bar_region, multi_bar_region

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = np.arange(1, 27, 2)
ANGLES = (0, 90)
DELTAS = np.array([0, 15, 30, 45, 60, 75, 90])


def train(seed, steps1=5000, steps2=4000):
    images, scale = load_images("train", return_scale=True)
    rng = np.random.default_rng(seed)
    net = PredictiveCodingNet(Params(k2_init=0.05, decay_every=10 ** 6), seed=seed)
    net.scale = scale
    for _ in range(steps1):
        net.train_step(sample_regions(images, 4, rng), stage=1)
    net.fit_r1_scale(sample_regions(images, 512, rng))
    # level-2 settings (paper uses alpha=1, lam=0.02, k2=0.05; these were chosen by the sweep)
    net.p.alpha2, net.p.lam2, net.p.k2_init = 0.01, 0.002, 0.5
    for _ in range(steps2):
        net.train_step(sample_regions(images, 4, rng), stage=2)
    return net


def err_norm(net, regs, feedback):
    o = net.infer(regs, feedback=feedback, n_iter=200)
    return np.linalg.norm(o["err1"][:, 1, :], axis=1)


def suppression(curve, plateau_from=18):
    peak = curve.max()
    return 100 * (peak - curve[LENGTHS > plateau_from].mean()) / peak


def one_seed(seed):
    net = train(seed)
    out = {"seed": seed}
    for whiten in (True, False):
        for ang in ANGLES:
            regs = np.stack([bar_region(L, 3, ang, 0.5, net.scale, whiten) for L in LENGTHS])
            for fb in (True, False):
                out[("len", whiten, ang, fb)] = err_norm(net, regs, fb)
        # surround-orientation: centre bar (horizontal, length 10) + flank bars at angle delta
        regs = np.stack([multi_bar_region([(0, 0, 10, 0), (-9, 0, 6, d), (9, 0, 6, d)],
                                          3, 0.5, net.scale, whiten) for d in DELTAS])
        for fb in (True, False):
            out[("ori", whiten, fb)] = err_norm(net, regs, fb)
    # explained variance on validation images (sanity check on level 2)
    reg = sample_regions(load_images("val", 100), 256, np.random.default_rng(seed))
    o = net.infer(reg)
    r1 = o["r1"].reshape(256, -1)
    out["ev"] = float(1 - np.var(r1 - o["r1_td"].reshape(256, -1)) / np.var(r1))
    np.savez(ROOT / "results" / f"extension_seed{seed}.npz", U1=net.U1, U2=net.U2, scale=net.scale,
             r1_scale=net.r1_scale)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    with Pool(args.seeds) as p:
        res = p.map(one_seed, range(args.seeds))
    import pickle
    with open(ROOT / "results" / "extension_results.pkl", "wb") as f:
        pickle.dump(res, f)
    print("saved results/extension_results.pkl;", len(res), "seeds; EV(val):",
          np.round([r["ev"] for r in res], 2))
