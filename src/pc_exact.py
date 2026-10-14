"""Paper-faithful two-level predictive coding network with exact (steady-state) inference.

With f(x) = x and Gaussian priors, the stationary point of the paper's energy E is the solution
of a linear system, so inference is solved exactly instead of by Euler integration of Eq. 7:

  E = sum_m |I_m - U1 r_m|^2 / s2 + |r - U2 r2|^2 / s2td + alpha |r|^2 + alpha2 |r2|^2

Weights are learned with Eq. 9 (Hebbian + weight decay), both levels simultaneously.
"""
from dataclasses import dataclass

import numpy as np

from .data import PATCH, OFFSET


def gaussian_window(size=PATCH, sigma=4.0):
    ax = np.arange(size) - (size - 1) / 2
    return np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))


@dataclass
class PCParams:
    n1: int = 32
    n2: int = 128
    sigma2: float = 1.0
    sigma2_td: float = 10.0
    alpha: float = 1.0
    alpha2: float = 1.0
    lam: float = 0.02
    lam2: float = 0.02
    k2_init: float = 1.0
    k2_decay: float = 1.015
    decay_every: int = 40
    win_sigma: float = 4.0
    gain: float = 1.0        # input amplitude (units of the prewhitened image; arbitrary in the paper)
    init1: float = 1.0       # initial column norm of U1
    init2: float = 1.0       # initial column norm of U2


class ExactPC:
    def __init__(self, params=None, seed=0):
        self.p = params or PCParams()
        self.rng = np.random.default_rng(seed)
        d = PATCH * PATCH
        self.U1 = self._init(d, self.p.n1, self.p.init1)
        self.U2 = self._init(3 * self.p.n1, self.p.n2, self.p.init2)
        self.win = gaussian_window(sigma=self.p.win_sigma).ravel()
        self.n_seen = 0

    def _init(self, rows, cols, norm):
        U = self.rng.normal(size=(rows, cols))
        return U / np.linalg.norm(U, axis=0, keepdims=True) * norm

    # -- input --------------------------------------------------------------
    def split(self, region):
        mods = [region[:, :, m * OFFSET:m * OFFSET + PATCH].reshape(len(region), -1) * self.win
                for m in range(3)]
        return np.stack(mods, axis=1) * self.p.gain

    # -- inference -----------------------------------------------------------
    def infer(self, region, feedback=True):
        p = self.p
        I = self.split(region)                                   # (B, 3, d)
        B = len(I)
        n1 = p.n1
        G = self.U1.T @ self.U1 / p.sigma2
        b = I @ self.U1 / p.sigma2                               # (B, 3, n1)
        c = 1.0 / p.sigma2_td
        if not feedback:
            Q = np.linalg.inv(G + p.alpha * np.eye(n1))
            r1 = b @ Q.T
            r2 = np.zeros((B, p.n2))
            r1_td = np.zeros_like(r1)
        else:
            Q = np.linalg.inv(G + (c + p.alpha) * np.eye(n1))
            Qb = np.kron(np.eye(3), Q)                           # (96, 96)
            bf = b.reshape(B, -1)
            QU = Qb @ self.U2                                    # (96, n2)
            A = c * self.U2.T @ self.U2 + p.alpha2 * np.eye(p.n2) - c * c * self.U2.T @ QU
            rhs = c * (bf @ Qb.T) @ self.U2                      # (B, n2)
            r2 = np.linalg.solve(A, rhs.T).T
            r1f = (bf + c * r2 @ self.U2.T) @ Qb.T
            r1 = r1f.reshape(B, 3, n1)
            r1_td = (r2 @ self.U2.T).reshape(B, 3, n1)
        return dict(r1=r1, r2=r2, r1_td=r1_td, err1=r1 - r1_td, resid_in=I - r1 @ self.U1.T)

    # -- learning ------------------------------------------------------------
    def k2(self):
        return self.p.k2_init / self.p.k2_decay ** (self.n_seen // self.p.decay_every)

    def train_step(self, region):
        p = self.p
        out = self.infer(region, feedback=True)
        B = len(region)
        r1, r2 = out["r1"], out["r2"]
        k2 = self.k2()
        dU1 = (k2 / p.sigma2) * np.einsum("bmd,bmn->dn", out["resid_in"], r1) / (B * 3) \
            - k2 * p.lam * self.U1
        err2 = r1.reshape(B, -1) - r2 @ self.U2.T
        dU2 = (k2 / p.sigma2_td) * (err2.T @ r2) / B - k2 * p.lam2 * self.U2
        self.U1 += B * dU1
        self.U2 += B * dU2
        self.n_seen += B
        return float(np.mean(out["resid_in"] ** 2)), float(np.mean(err2 ** 2))
