"""(1) Held-out stimulus check: same trained nets, new bar widths / offsets (no re-tuning).
(2) Alternative explanation: feedback-free divisive normalization of the centre by flank energy.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_extension import LENGTHS, suppression
from src.model import PredictiveCodingNet, Params
from src.stimuli import multi_bar_region

ROOT = Path(__file__).resolve().parents[1]
N = 5
TRAIN_COND = [(3, 0, w) for w in (True, False)]            # (width, y-offset, whiten) used when tuning
HELD_COND = [(w_, off, wh) for w_ in (2, 5) for off in (0, 3) for wh in (True, False)] + \
            [(3, 3, True), (3, 3, False)]


def load(seed):
    return PredictiveCodingNet.load(ROOT / "results" / f"extension_seed{seed}.npz",
                                    Params(alpha2=0.01, lam2=0.002))


def curves(net, width, off, whiten):
    regs = np.stack([multi_bar_region([(0, off, L, 0)], width, 0.5, net.scale, whiten) for L in LENGTHS])
    o_fb = net.infer(regs, feedback=True, n_iter=200)
    o_nf = net.infer(regs, feedback=False, n_iter=200)
    fb = np.linalg.norm(o_fb["err1"][:, 1, :], axis=1)
    nf = np.linalg.norm(o_nf["r1"][:, 1, :], axis=1)
    r = o_nf["r1"]
    s2 = net.r1_scale ** 2
    flank = (np.sum(r[:, 0, :] ** 2, 1) + np.sum(r[:, 2, :] ** 2, 1)) / s2
    return fb, nf, flank


def dn(nf, flank, beta):
    return nf / np.sqrt(1.0 + beta * flank)


def ci(a):
    a = np.asarray(a)
    return stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


if __name__ == "__main__":
    nets = [load(s) for s in range(N)]
    # fit the divisive-normalization strength on the tuning conditions (match mean S of the fb model)
    cache = {}
    for c in TRAIN_COND + HELD_COND:
        cache[c] = [curves(n, *c) for n in nets]
    betas = np.logspace(-2, 3, 60)
    target = np.mean([suppression(fb) for c in TRAIN_COND for fb, _, _ in cache[c]])
    gaps = [abs(np.mean([suppression(dn(nf, fl, b)) for c in TRAIN_COND for _, nf, fl in cache[c]]) - target)
            for b in betas]
    beta = betas[int(np.argmin(gaps))]
    print(f"divisive-norm beta fitted on tuning conditions: {beta:.3g} (target S_fb = {target:.1f}%)")
    print("\nwidth offset stim | S_fb          S_nofb        S_divnorm     | fb-nofb (paired t)")
    for c in TRAIN_COND + HELD_COND:
        width, off, wh = c
        sfb = np.array([suppression(fb) for fb, _, _ in cache[c]])
        snf = np.array([suppression(nf) for _, nf, _ in cache[c]])
        sdn = np.array([suppression(dn(nf, fl, beta)) for _, nf, fl in cache[c]])
        d = sfb - snf
        p = stats.ttest_1samp(d, 0).pvalue
        tag = "tune" if c in TRAIN_COND else "held"
        print(f"{width:3d}  {off:3d}   {'DoG' if wh else 'raw'} {tag} | {sfb.mean():5.1f} ±{ci(sfb):4.1f}  "
              f"{snf.mean():5.1f} ±{ci(snf):4.1f}  {sdn.mean():5.1f} ±{ci(sdn):4.1f}  | {d.mean():+5.1f} ±{ci(d):4.1f} (p={p:.3f})")
