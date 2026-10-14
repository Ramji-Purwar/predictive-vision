"""Paper parameters, joint training; sweep the (arbitrary) input amplitude."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_images, sample_regions
from src.pc_exact import ExactPC, PCParams
from src.stimuli import bar_region

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = np.arange(1, 27, 2)


def run(gain, n_inputs=40000, seed=0):
    images, scale = load_images("train", return_scale=True)
    net = ExactPC(PCParams(gain=gain), seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(n_inputs // 4):
        net.train_step(sample_regions(images, 4, rng))
    val = sample_regions(load_images("val", 100), 256, rng)
    o = net.infer(val)
    r1 = o["r1"].reshape(256, -1)
    ev = 1 - np.var(r1 - o["r1_td"].reshape(256, -1)) / np.var(r1)
    # endstopping counts
    res = {}
    for whiten in (True, False):
        regs = np.stack([bar_region(L, 3, a, 0.5, scale, whiten) for a in (0, 45, 90, 135) for L in LENGTHS])
        resp = {}
        for fb in (True, False):
            oo = net.infer(regs, feedback=fb)
            resp[fb] = np.abs(oo["err1"][:, 1, :]).reshape(4, len(LENGTHS), -1)
        best = resp[True].max(axis=1).argmax(axis=0)
        es = {}
        for fb in (True, False):
            c = np.stack([resp[fb][best[i], :, i] for i in range(net.p.n1)])
            peak = c.max(axis=1)
            plateau = c[:, LENGTHS > 18].mean(axis=1)
            es[fb] = np.where(peak > 0, 100 * (peak - plateau) / peak, 0)
        res[whiten] = (int((es[True] > 50).sum()), int((es[False] > 50).sum()),
                       es[True].mean(), es[False].mean())
    return gain, np.linalg.norm(net.U1, 2), np.linalg.norm(net.U2, 2), \
        float(np.sqrt(np.mean(o["r1"] ** 2))), ev, res, net


if __name__ == "__main__":
    gains = [1, 10, 30, 100, 300, 1000]
    with Pool(len(gains)) as p:
        out = p.map(run, gains)
    print("gain | |U1| |U2| rms(r1) EV | DoG bars: n>50% fb/nofb, meanES fb/nofb | raw bars")
    for g, u1, u2, rr, ev, res, net in out:
        a, b = res[True], res[False]
        print(f"{g:5d} | {u1:7.2f} {u2:7.2f} {rr:6.2f} {ev:5.2f} | {a[0]:2d}/{a[1]:2d} {a[2]:5.1f}/{a[3]:5.1f} "
              f"| {b[0]:2d}/{b[1]:2d} {b[2]:5.1f}/{b[3]:5.1f}")
        np.savez(ROOT / "results" / f"exact_gain{g}.npz", U1=net.U1, U2=net.U2)
