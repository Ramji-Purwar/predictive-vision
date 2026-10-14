"""Train the baseline hierarchical network on BSDS500 and plot learned RFs (paper Fig. 2)."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_images, sample_regions, image_scale, PATCH
from src.model import PredictiveCodingNet, Params

ROOT = Path(__file__).resolve().parents[1]


def plot_rfs(net, path):
    n = net.p.n1
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    for i, ax in enumerate(axes.ravel()):
        rf = net.U1[:, i].reshape(PATCH, PATCH)
        v = np.abs(rf).max()
        ax.imshow(rf, cmap="gray", vmin=-v, vmax=v)
        ax.axis("off")
    fig.suptitle("Level-1 basis vectors (U1 columns)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000, help="stage-1 (level-1) steps")
    ap.add_argument("--steps2", type=int, default=1500, help="stage-2 (level-2) steps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--decay_every", type=int, default=40)
    ap.add_argument("--init_norm", type=float, default=3.0)
    ap.add_argument("--k2", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    images, scale = load_images("train", return_scale=True)
    net = PredictiveCodingNet(Params(decay_every=args.decay_every, init_norm=args.init_norm, k2_init=args.k2, input_gain=args.gain), seed=args.seed)
    for step in range(args.steps + args.steps2):
        stage = 1 if step < args.steps else 2
        if step == args.steps:
            net.n_seen = 0  # restart the learning-rate schedule for level 2
            net.fit_r1_scale(sample_regions(images, 512, rng))
            print(f"r1_scale = {net.r1_scale:.4f}", flush=True)
        e1, e2 = net.train_step(sample_regions(images, args.batch, rng), stage)
        if step % 200 == 0:
            print(f"stage {stage} step {step:5d}  inputs {net.n_seen:7d}  k2 {net.k2():.3f}  "
                  f"L1 err {e1:.4f}  L2 err {e2:.4f}", flush=True)
    out = ROOT / "results"
    net.scale = scale
    out.mkdir(exist_ok=True)
    net.save(out / f"baseline{args.tag}_seed{args.seed}.npz")
    plot_rfs(net, ROOT / "figures" / f"fig2_level1_rfs{args.tag}_seed{args.seed}.png")
