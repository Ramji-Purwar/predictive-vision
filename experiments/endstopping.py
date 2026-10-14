"""Endstopping in the error-detecting neurons, with and without level-2 feedback (paper Figs. 3, 5)."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import PredictiveCodingNet
from src.stimuli import bar_region

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = np.arange(1, 27, 2)
ANGLES = [0, 45, 90, 135]


def response_curves(net, feedback, width=3, contrast=0.5, whiten=True):
    """Error-neuron |r - r_td| of the central level-1 module. Shape (angles, lengths, n1)."""
    regs = np.stack([bar_region(L, width, a, contrast, net.scale, whiten)
                     for a in ANGLES for L in LENGTHS])
    out = net.infer(regs, feedback=feedback, n_iter=200)
    resp = np.abs(out["err1"][:, 1, :])
    return resp.reshape(len(ANGLES), len(LENGTHS), -1)


def endstopping_pct(curve, plateau_from=18):
    """(peak - plateau) / peak * 100, plateau = mean response for lengths > plateau_from."""
    peak = curve.max()
    plateau = curve[LENGTHS > plateau_from].mean()
    return 100 * (peak - plateau) / peak if peak > 0 else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_whiten", action="store_true", help="do not DoG-filter the bar stimuli")
    args = ap.parse_args()
    net = PredictiveCodingNet.load(args.weights)

    with_fb = response_curves(net, True, whiten=not args.no_whiten)
    no_fb = response_curves(net, False, whiten=not args.no_whiten)
    n1 = with_fb.shape[2]
    # each neuron's preferred orientation = the angle giving it the largest response
    best = with_fb.max(axis=1).argmax(axis=0)
    cw = np.stack([with_fb[best[i], :, i] for i in range(n1)])
    cn = np.stack([no_fb[best[i], :, i] for i in range(n1)])
    es_w = np.array([endstopping_pct(c) for c in cw])
    es_n = np.array([endstopping_pct(c) for c in cn])
    print(f"endstopping >50%: with feedback {np.sum(es_w > 50)}/{n1}, without {np.sum(es_n > 50)}/{n1}")
    print(f"mean endstopping: with {es_w.mean():.1f}%, without {es_n.mean():.1f}%")

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True)
    top = np.argsort(-es_w)[:4]
    for ax, i in zip(axes[0], top):
        ax.plot(LENGTHS, cw[i] / cw[i].max(), "k-o", ms=3, label="with feedback")
        ax.plot(LENGTHS, cn[i] / cw[i].max(), "k:", label="no feedback")
        ax.set_title(f"neuron {i}  ({ANGLES[best[i]]}°)  ES={es_w[i]:.0f}%")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("response (norm.)")
    b = np.linspace(0, 100, 11)
    axes[1, 0].hist(es_w, bins=b); axes[1, 0].set_title("endstopping %, with feedback")
    axes[1, 1].hist(es_n, bins=b); axes[1, 1].set_title("endstopping %, no feedback")
    axes[1, 2].plot(LENGTHS, (cw / cw.max(axis=1, keepdims=True)).mean(0), "k-o", label="with")
    axes[1, 2].plot(LENGTHS, (cn / cn.max(axis=1, keepdims=True)).mean(0), "k:", label="without")
    axes[1, 2].set_title("mean normalised curve"); axes[1, 2].legend(fontsize=8)
    axes[1, 3].axis("off")
    for ax in axes[1]:
        ax.set_xlabel("bar length (px)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"fig3_5_endstopping{args.tag}.png", dpi=150)
