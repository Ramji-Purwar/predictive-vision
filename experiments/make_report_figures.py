"""Regenerate every figure and statistic used in report/report.md from results/*.npz and results/*.pkl."""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import endstopping as es
from experiments.heldout_and_alternative import (TRAIN_COND, HELD_COND, load, curves, dn)
from experiments.run_extension import LENGTHS, ANGLES, DELTAS, suppression
from src.data import PATCH

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
N = 5
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white"})


def tci(a, axis=0):
    a = np.asarray(a)
    return stats.t.ppf(0.975, a.shape[axis] - 1) * a.std(axis, ddof=1) / np.sqrt(a.shape[axis])


res = pickle.load(open(ROOT / "results" / "extension_results.pkl", "rb"))
nets = [load(s) for s in range(N)]

# ---- Fig 2: learned level-1 basis vectors (seed 0) --------------------------------------
fig, axes = plt.subplots(4, 8, figsize=(8, 4.3))
for i, ax in enumerate(axes.ravel()):
    rf = nets[0].U1[:, i].reshape(PATCH, PATCH)
    v = np.abs(rf).max()
    ax.imshow(rf, cmap="gray", vmin=-v, vmax=v)
    ax.axis("off")
fig.suptitle("Level-1 basis vectors (columns of U1), seed 0", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "report_fig2_level1_rfs.png", dpi=170)
plt.close(fig)

# ---- Fig 3: bar-length sweep ------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
panels = [(True, 0), (False, 0), (True, 90), (False, 90)]
summary = {}
for ax, (wh, ang) in zip(axes, panels):
    S = {}
    for fb, col, lab, mk in ((True, BLUE, "with feedback", "o"), (False, ORANGE, "no feedback", "s")):
        c = np.stack([r[("len", wh, ang, fb)] for r in res])
        c = c / c.max(axis=1, keepdims=True)
        m, h = c.mean(0), tci(c)
        ax.plot(LENGTHS, m, color=col, lw=2, marker=mk, ms=4, label=lab)
        ax.fill_between(LENGTHS, m - h, m + h, color=col, alpha=0.18, lw=0)
        S[fb] = np.array([suppression(np.stack([r[("len", wh, ang, fb)] for r in res])[k]) for k in range(N)])
    d = S[True] - S[False]
    p = stats.ttest_1samp(d, 0).pvalue
    summary[f"{'DoG' if wh else 'raw'}_{ang}"] = dict(S_fb=S[True].mean(), S_fb_ci=tci(S[True]),
        S_nofb=S[False].mean(), S_nofb_ci=tci(S[False]), diff=d.mean(), diff_ci=tci(d), p=p)
    ax.set_title(f"{'DoG-filtered' if wh else 'raw'} bar, {ang}°\nΔS = {d.mean():+.1f} ± {tci(d):.1f} pts",
                 fontsize=9.5, color=INK)
    ax.set_xlabel("bar length (px)")
axes[0].set_ylabel("error norm / peak")
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "report_fig3_length_sweep.png", dpi=170)
plt.close(fig)

# ---- Fig 4: surround orientation --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
ori = {}
for ax, wh in zip(axes, (True, False)):
    for fb, col, lab, mk in ((True, BLUE, "with feedback", "o"), (False, ORANGE, "no feedback", "s")):
        c = np.stack([r[("ori", wh, fb)] for r in res])
        c = c / c[:, DELTAS == 90]
        m, h = c.mean(0), tci(c)
        ax.plot(DELTAS, m, color=col, lw=2, marker=mk, ms=4, label=lab)
        ax.fill_between(DELTAS, m - h, m + h, color=col, alpha=0.18, lw=0)
        ori[(wh, fb)] = c[:, 0]
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_title(f"{'DoG-filtered' if wh else 'raw'} bars", fontsize=9.5, color=INK)
    ax.set_xlabel("flank orientation relative to centre (deg)")
    ax.set_xticks(DELTAS)
axes[0].set_ylabel("error norm / error at 90°")
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "report_fig4_surround_orientation.png", dpi=170)
plt.close(fig)
for wh in (True, False):
    d = ori[(wh, True)] - ori[(wh, False)]
    summary[f"ori_{'DoG' if wh else 'raw'}"] = dict(fb=ori[(wh, True)].mean(), fb_ci=tci(ori[(wh, True)]),
        nofb=ori[(wh, False)].mean(), nofb_ci=tci(ori[(wh, False)]), diff=d.mean(), diff_ci=tci(d),
        p=stats.ttest_1samp(d, 0).pvalue)

# ---- Fig 5: held-out conditions and divisive normalization --------------------------------
cache = {c: [curves(n, *c) for n in nets] for c in TRAIN_COND + HELD_COND}
betas = np.logspace(-2, 3, 60)
target = np.mean([suppression(fb) for c in TRAIN_COND for fb, _, _ in cache[c]])
gaps = [abs(np.mean([suppression(dn(nf, fl, b)) for c in TRAIN_COND for _, nf, fl in cache[c]]) - target)
        for b in betas]
beta = float(betas[int(np.argmin(gaps))])
conds = TRAIN_COND + HELD_COND
rows = []
for c in conds:
    w, off, wh = c
    sfb = np.array([suppression(fb) for fb, _, _ in cache[c]])
    snf = np.array([suppression(nf) for _, nf, _ in cache[c]])
    sdn = np.array([suppression(dn(nf, fl, beta)) for _, nf, fl in cache[c]])
    d = sfb - snf
    rows.append(dict(width=w, offset=off, whiten=bool(wh), tuning=c in TRAIN_COND,
                     S_fb=sfb.mean(), S_fb_ci=tci(sfb), S_nofb=snf.mean(), S_nofb_ci=tci(snf),
                     S_dn=sdn.mean(), S_dn_ci=tci(sdn), diff=d.mean(), diff_ci=tci(d),
                     p=stats.ttest_1samp(d, 0).pvalue))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
for ax, wh in zip(axes, (True, False)):
    rr = [r for r in rows if r["whiten"] == wh]
    x = np.arange(len(rr))
    for k, (key, col, lab) in enumerate((("S_fb", BLUE, "with feedback"),
                                         ("S_nofb", ORANGE, "no feedback"),
                                         ("S_dn", AQUA, "divisive normalization"))):
        ax.bar(x + (k - 1) * 0.27, [r[key] for r in rr], 0.25, yerr=[r[key + "_ci"] for r in rr],
               color=col, capsize=2, error_kw=dict(lw=0.8, ecolor=MUTED), label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"w={r['width']}, y={r['offset']}" + ("*" if r["tuning"] else "") for r in rr],
                       fontsize=8.5)
    ax.set_title(f"{'DoG-filtered' if wh else 'raw'} bars (* = conditions used for tuning)",
                 fontsize=9.5, color=INK)
axes[0].set_ylabel("suppression S (%)")
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "report_fig5_heldout_alternative.png", dpi=170)
plt.close(fig)

# ---- per-neuron endstopping on the 5 extension nets (Fig. 3/5-style statistics) -----------
stat = []
for n in nets:
    out = {}
    for whiten in (True, False):
        w = es.response_curves(n, True, whiten=whiten)
        nf = es.response_curves(n, False, whiten=whiten)
        best = w.max(axis=1).argmax(axis=0)
        cw = np.stack([w[best[i], :, i] for i in range(w.shape[2])])
        cn = np.stack([nf[best[i], :, i] for i in range(w.shape[2])])
        ew = np.array([es.endstopping_pct(c) for c in cw])
        en = np.array([es.endstopping_pct(c) for c in cn])
        out[whiten] = (int((ew > 50).sum()), int((en > 50).sum()), ew.mean(), en.mean())
    stat.append(out)
rep = {}
for whiten in (True, False):
    a = np.array([s[whiten] for s in stat])
    rep["DoG" if whiten else "raw"] = dict(n_gt50_fb=a[:, 0].mean(), n_gt50_nofb=a[:, 1].mean(),
        mean_es_fb=a[:, 2].mean(), mean_es_fb_ci=tci(a[:, 2]), mean_es_nofb=a[:, 3].mean(),
        mean_es_nofb_ci=tci(a[:, 3]))
out = dict(beta=beta, S_target=float(target), length_sweep=summary, held_out=rows, replication=rep,
           ev_val=[r["ev"] for r in res])
json.dump(out, open(ROOT / "results" / "report_stats.json", "w"), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else bool(o))
print(json.dumps(out, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else bool(o))[:3500])
