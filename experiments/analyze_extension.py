"""Summarize results/extension_results.pkl: curves with 95% CIs and feedback-attributable suppression."""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_extension import LENGTHS, ANGLES, DELTAS, suppression

ROOT = Path(__file__).resolve().parents[1]
res = pickle.load(open(ROOT / "results" / "extension_results.pkl", "rb"))
n = len(res)


def ci(a):
    a = np.asarray(a)
    return a.mean(0), stats.t.ppf(0.975, n - 1) * a.std(0, ddof=1) / np.sqrt(n)


def stack(key):
    return np.stack([r[key] for r in res])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    t, p = stats.ttest_1samp(d, 0.0)
    return d.mean(), stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n), p


print(f"seeds: {n}; EV(level-2 prediction, val images): {np.mean([r['ev'] for r in res]):.2f}")
print("\n(a) Bar-length sweep: suppression index S = (peak - plateau)/peak [%], plateau = lengths > 18 px")
print("stimulus            S_fb            S_nofb          S_fb - S_nofb (paired t)")
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for row, whiten in enumerate((True, False)):
    for col, ang in enumerate(ANGLES):
        ax = axes[row, col]
        S = {}
        for fb, ls, lab in ((True, "k-o", "feedback"), (False, "k--s", "no feedback")):
            c = stack(("len", whiten, ang, fb))
            m, h = ci(c / c.max(axis=1, keepdims=True))
            ax.errorbar(LENGTHS, m, h, fmt=ls, ms=3, capsize=2, label=lab)
            S[fb] = np.array([suppression(x) for x in c])
        d, h, p = paired(S[True], S[False])
        name = f"{'DoG' if whiten else 'raw'} bar, {ang:2d} deg"
        print(f"{name:18s} {S[True].mean():5.1f} ±{ci(S[True])[1]:4.1f}   {S[False].mean():5.1f} ±{ci(S[False])[1]:4.1f}"
              f"      {d:+5.1f} ±{h:4.1f}  (p={p:.3f})")
        ax.set_title(name + f"\nΔS = {d:+.1f} ± {h:.1f} pts", fontsize=9)
        ax.set_xlabel("bar length (px)")
        ax.set_ylabel("error norm / peak")
        if row == 0 and col == 0:
            ax.legend(fontsize=8)

print("\n(b) Surround orientation: centre bar + flank bars at relative angle delta."
      "\n    error normalised to delta=90 (orthogonal surround); >1 means surround of same orientation is MORE error")
for col, whiten in enumerate((True, False)):
    ax = axes[col, 2]
    for fb, ls, lab in ((True, "k-o", "feedback"), (False, "k--s", "no feedback")):
        c = stack(("ori", whiten, fb))
        c = c / c[:, DELTAS == 90]
        m, h = ci(c)
        ax.errorbar(DELTAS, m, h, fmt=ls, ms=3, capsize=2, label=lab)
        if fb:
            ratio_fb = c[:, 0]
        else:
            ratio_nofb = c[:, 0]
    d, h, p = paired(ratio_fb, ratio_nofb)
    print(f"{'DoG' if whiten else 'raw'}: error(delta=0)/error(delta=90): fb {ratio_fb.mean():.2f} ±{ci(ratio_fb)[1]:.2f}, "
          f"nofb {ratio_nofb.mean():.2f} ±{ci(ratio_nofb)[1]:.2f}; diff {d:+.2f} ±{h:.2f} (p={p:.3f})")
    ax.set_title(f"surround orientation ({'DoG' if whiten else 'raw'})", fontsize=9)
    ax.set_xlabel("surround - centre orientation (deg)")
    ax.set_ylabel("error norm / (delta=90)")
    ax.axhline(1, color="gray", lw=0.5)
for ax in (axes[0, 3], axes[1, 3]):
    ax.axis("off")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "extension_results.png", dpi=150)
