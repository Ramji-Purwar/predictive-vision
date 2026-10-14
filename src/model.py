"""Two-level hierarchical predictive coding network (Rao & Ballard 1999).

Level 1: three modules (left/center/right) sharing one weight matrix U1, each
         explaining a 16x16 Gaussian-windowed patch with 32 causes.
Level 2: one module with 128 causes explaining the concatenated 96 level-1 responses.

Update equations follow the paper's Methods:
  dr/dt  = k1/s2 U^T (I - f(Ur)) + k1/s2td (r_td - r) - k1 * g'(r)/2     (Eq. 7)
  dU/dt  = k2/s2 (I - f(Ur)) r^T - k2 * lam * U                           (Eq. 9)
with a Gaussian prior g(r) = alpha*|r|^2 (g'(r) = 2*alpha*r) and f(x) = x.
"""
from dataclasses import dataclass

import numpy as np

from .data import PATCH, OFFSET, SPAN_W


def gaussian_window(size=PATCH, sigma=4.0):
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g


@dataclass
class Params:
    n1: int = 32          # causes per level-1 module
    n2: int = 128         # causes at level 2
    k1_l1: float = 0.5
    k1_l2: float = 0.05
    sigma2: float = 1.0
    sigma2_td: float = 10.0
    alpha: float = 1.0
    alpha2: float = 1.0   # level-2 prior strength (paper uses one alpha; exposed for sweeps)
    lam2: float = 0.02    # level-2 weight decay
    lam: float = 0.02
    k2_init: float = 1.0
    k2_decay: float = 1.015   # divide k2 by this every `decay_every` inputs
    decay_every: int = 40
    win_sigma: float = 4.0
    n_iter: int = 40
    target_rms: float = 1.0   # gain adaptation: keep rms of each r_i near this (paper ref. 26)
    gain_eta: float = 0.0   # gain adaptation disabled (see r1_scale)
    input_gain: float = 1.0  # amplitude of prewhitened input (units are arbitrary in the paper)
    init_norm: float = 1.0   # initial column norm of U1
    init_norm2: float = 5.0  # initial column norm of U2 (needs U2'U2 >~ sigma2_td*alpha or r2 collapses)


class PredictiveCodingNet:
    def __init__(self, params=None, seed=0):
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        d = PATCH * PATCH
        self.U1 = self._init(d, self.p.n1, self.p.init_norm)
        self.U2 = self._init(3 * self.p.n1, self.p.n2, self.p.init_norm2)
        self.win = gaussian_window(sigma=self.p.win_sigma).ravel()
        self.n_seen = 0
        self.r1_scale = 1.0  # rms of level-1 responses after stage 1; level 2 sees r1 / r1_scale
        self.scale = 1.0   # std of prewhitened training images (for scaling stimuli)

    def _init(self, rows, cols, norm):
        U = self.rng.normal(size=(rows, cols))
        return U / np.linalg.norm(U, axis=0, keepdims=True) * norm

    # ---- input handling -------------------------------------------------
    def split_modules(self, region):
        """region (B, PATCH, SPAN_W) -> windowed inputs (B, 3, PATCH*PATCH)."""
        mods = []
        for m in range(3):
            patch = region[:, :, m * OFFSET:m * OFFSET + PATCH].reshape(len(region), -1)
            mods.append(patch * self.win * self.p.input_gain)
        return np.stack(mods, axis=1)

    # ---- inference ------------------------------------------------------
    def infer(self, region, feedback=True, n_iter=None):
        """Settle r1, r2 by gradient descent on E. Returns dict of responses.

        feedback=False removes the level-2 prediction (r_td = 0, no top-down term),
        i.e. level 1 explains the input on its own.
        """
        p = self.p
        I = self.split_modules(region)                  # (B, 3, d)
        B = I.shape[0]
        r1 = np.zeros((B, 3, p.n1))
        r2 = np.zeros((B, p.n2))
        # Euler step size: scale by the largest curvature of E so that descent is stable
        # (the paper integrates a continuous-time ODE; this is its discretisation).
        L1 = p.k1_l1 * (np.linalg.norm(self.U1, 2) ** 2 / p.sigma2
                        + 1 / p.sigma2_td + p.alpha)
        L2 = p.k1_l2 * (np.linalg.norm(self.U2, 2) ** 2 / p.sigma2_td + p.alpha2)
        dt1 = min(1.0, 0.9 / L1)
        dt2 = min(1.0, 0.9 / L2)
        for _ in range(n_iter or p.n_iter):
            pred1 = r1 @ self.U1.T                      # (B, 3, d)
            err_in = I - pred1
            s = self.r1_scale
            r1_td = s * (r2 @ self.U2.T).reshape(B, 3, p.n1) if feedback else r1
            dr1 = (p.k1_l1 / p.sigma2) * (err_in @ self.U1) \
                + (p.k1_l1 / (p.sigma2_td * s * s)) * (r1_td - r1) \
                - p.k1_l1 * p.alpha * r1
            if feedback:
                err2 = r1.reshape(B, -1) / s - r2 @ self.U2.T
                dr2 = (p.k1_l2 / p.sigma2_td) * (err2 @ self.U2) - p.k1_l2 * p.alpha2 * r2
                r2 = r2 + dt2 * dr2
            r1 = r1 + dt1 * dr1
        r1_td = self.r1_scale * (r2 @ self.U2.T).reshape(B, 3, p.n1) if feedback else np.zeros_like(r1)
        return dict(r1=r1, r2=r2, r1_td=r1_td, err1=r1 - r1_td,
                    resid_in=I - r1 @ self.U1.T)

    def fit_r1_scale(self, region):
        """Set r1_scale to the rms of bottom-up level-1 responses (call after stage 1)."""
        r1 = self.infer(region, feedback=False)["r1"]
        self.r1_scale = float(np.sqrt(np.mean(r1 ** 2)))

    def settle_r2(self, r1, n_iter=None):
        """Settle level-2 responses for fixed level-1 responses r1 (B, 3, n1)."""
        p = self.p
        x = r1.reshape(len(r1), -1) / self.r1_scale
        r2 = np.zeros((len(r1), p.n2))
        dt = min(1.0, 0.9 / (p.k1_l2 * (np.linalg.norm(self.U2, 2) ** 2 / p.sigma2_td + p.alpha2)))
        for _ in range(n_iter or p.n_iter):
            r2 = r2 + dt * ((p.k1_l2 / p.sigma2_td) * ((x - r2 @ self.U2.T) @ self.U2)
                            - p.k1_l2 * p.alpha2 * r2)
        return r2

    # ---- learning -------------------------------------------------------
    def k2(self):
        return self.p.k2_init / self.p.k2_decay ** (self.n_seen // self.p.decay_every)

    def train_step(self, region, stage=1):
        """One batch update. Stage 1 learns U1 (no feedback); stage 2 learns U2 with U1 frozen.

        The paper learns level-1 basis vectors first, then level 2 (Methods, Simulations).
        """
        p = self.p
        B = len(region)
        k2 = self.k2()
        if stage == 1:
            out = self.infer(region, feedback=False)
            r1 = out["r1"]
            dU1 = (k2 / p.sigma2) * np.einsum("bmd,bmn->dn", out["resid_in"], r1) / (B * 3) \
                - k2 * p.lam * self.U1
            # The paper updates after every input; a batch of B applies B such steps at once.
            self.U1 += B * dU1
            self._adapt_gain(r1.reshape(-1, p.n1))
            self.n_seen += B
            return float(np.mean(out["resid_in"] ** 2)), float("nan")
        # Stage 2: level 2 learns to predict the (bottom-up) level-1 responses.
        out = self.infer(region, feedback=False)
        r1 = out["r1"]
        r2 = self.settle_r2(r1)
        err2 = r1.reshape(B, -1) / self.r1_scale - r2 @ self.U2.T
        dU2 = (k2 / p.sigma2_td) * (err2.T @ r2) / B - k2 * p.lam2 * self.U2
        self.U2 += B * dU2
        self.n_seen += B
        return float(np.mean(out["resid_in"] ** 2)), float(np.mean(err2 ** 2))

    def _adapt_gain(self, r1):
        """Rescale each level-1 basis vector so its neuron's rms response drifts toward
        target_rms (paper ref. 26). Keeps r1 at a scale where level 2 can learn."""
        rms = np.sqrt(np.mean(r1 ** 2, axis=0)) + 1e-12
        factor = (rms / self.p.target_rms) ** self.p.gain_eta
        self.U1 *= np.clip(factor, 0.95, 1.05)[None, :]
        # keep every column alive: a zero column gets r=0 and could never recover
        norms = np.linalg.norm(self.U1, axis=0, keepdims=True)
        self.U1 *= np.maximum(1.0, 0.2 / norms)

    def save(self, path):
        np.savez(path, U1=self.U1, U2=self.U2, n_seen=self.n_seen, scale=self.scale,
                 r1_scale=self.r1_scale)

    @classmethod
    def load(cls, path, params=None):
        net = cls(params)
        d = np.load(path)
        net.U1, net.U2 = d["U1"], d["U2"]
        net.n_seen = int(d["n_seen"]) if "n_seen" in d else 0
        net.scale = float(d["scale"])
        net.r1_scale = float(d["r1_scale"])
        return net
