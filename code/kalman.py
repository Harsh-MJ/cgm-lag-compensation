"""Piece 4: linear Kalman filter (causal), RTS smoother (non-causal) and regularised inversion (non-causal cross-check).

Model (continuous time):  dG_I/dt = (G_B - G_I)/tau ,  dG_B/dt = R ,  dR/dt = sqrt(q) * white noise ,  y_k = G_I(t_k) + e_k.
Only the ratio qe = q / sigma^2 (1/min^3) matters for the state estimate, so sigma is fixed to 1 and qe is tuned (diffuse initial covariance).
Batched over subjects so a whole cross-validation fold runs in a few hundred ms.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def model(tau, dt):
    """Exact discretisation: F (S,3,3) and unit-q process noise Q (S,3,3), by Gauss-Legendre quadrature."""
    tau = np.atleast_1d(np.asarray(tau, float)); S = len(tau); e = np.exp(-dt / tau)
    F = np.zeros((S, 3, 3)); F[:, 0, 0] = e; F[:, 0, 1] = 1 - e; F[:, 0, 2] = dt - tau * (1 - e)
    F[:, 1, 1] = 1; F[:, 1, 2] = dt; F[:, 2, 2] = 1
    xg, wg = np.polynomial.legendre.leggauss(12); u = (xg + 1) * dt / 2; w = wg * dt / 2
    v = np.stack([u[None, :] - tau[:, None] * (1 - np.exp(-u[None, :] / tau[:, None])), np.broadcast_to(u, (S, len(u))), np.ones((S, len(u)))], axis=2)
    return F, np.einsum('m,smi,smj->sij', w, v, v)

def kf_batch(Y, tau, dt, qe, smooth=False):
    """Y: (S, n) with NaN for missing. tau: (S,). Returns filtered G_B (S, n) and, if smooth, RTS-smoothed G_B."""
    Y = np.asarray(Y, float); S, n = Y.shape; F, Q0 = model(tau, dt); Q = qe * Q0; Ft = F.transpose(0, 2, 1)
    first = np.array([np.where(~np.isnan(Y[s]))[0][0] for s in range(S)]); y0 = Y[np.arange(S), first]
    x = np.stack([y0, y0, np.zeros(S)], 1); P = np.tile(np.diag([1e4, 1e4, 1e2]), (S, 1, 1))
    xp = np.empty((n, S, 3)); Pp = np.empty((n, S, 3, 3)); xf = np.empty((n, S, 3)); Pf = np.empty((n, S, 3, 3))
    for k in range(n):
        if k > 0: x = np.einsum('sij,sj->si', F, x); P = F @ P @ Ft + Q
        xp[k] = x; Pp[k] = P
        y = Y[:, k]; ob = ~np.isnan(y)
        if ob.any():
            K = P[:, :, 0] / (P[:, 0, 0] + 1.0)[:, None] * ob[:, None]
            x = x + K * np.where(ob, y - x[:, 0], 0.0)[:, None]; P = P - K[:, :, None] * P[:, 0, :][:, None, :]; P = 0.5 * (P + P.transpose(0, 2, 1))
        xf[k] = x; Pf[k] = P
    filt = xf[:, :, 1].T.copy()
    if not smooth: return filt, None
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = np.linalg.solve(Pp[k + 1], F @ Pf[k]).transpose(0, 2, 1)
        xs[k] = xf[k] + np.einsum('sij,sj->si', C, xs[k + 1] - xp[k + 1])
    return filt, xs[:, :, 1].T.copy()

def reginv(y, tau, dt, lam):
    """Regularised inversion: min sum_obs (y - G_I)^2 + lam * sum (second difference of G_B)^2, subject to the ISF dynamics with
    G_B piecewise linear between samples. Solved as a sparse KKT system. Returns G_B."""
    y = np.asarray(y, float); n = len(y); e = np.exp(-dt / tau); b = (dt - tau * (1 - e)) / dt
    ob = ~np.isnan(y); w = ob.astype(float); yy = np.where(ob, y, 0.0)
    rows = np.arange(n - 2); D = sparse.csr_matrix((np.r_[np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                                                    (np.r_[rows, rows, rows], np.r_[rows, rows + 1, rows + 2])), shape=(n - 2, n))
    A = sparse.block_diag([2 * lam * (D.T @ D) + 1e-8 * sparse.eye(n), 2 * sparse.diags(w)]).tocsc()
    r = np.r_[np.zeros(n), 2 * w * yy]
    k = np.arange(n - 1)                                            # constraint k: G_I[k+1] - e G_I[k] - (1-e-b) G_B[k] - b G_B[k+1] = 0
    C = sparse.csr_matrix((np.r_[-(1 - e - b) * np.ones(n - 1), -b * np.ones(n - 1), -e * np.ones(n - 1), np.ones(n - 1)],
                           (np.r_[k, k, k, k], np.r_[k, k + 1, n + k, n + k + 1])), shape=(n - 1, 2 * n))
    K = sparse.bmat([[A, C.T], [C, None]], format='csc'); z = spsolve(K, np.r_[r, np.zeros(n - 1)])
    return z[:n]

# ------------------------------------------------------------------ harness registry
def _kf_estimator(oracle, smooth):
    def batch(ys, dt, hp, taus):
        tau = np.array(taus, float) if oracle else np.full(len(ys), 10.0)
        f, s = kf_batch(np.array(ys), tau, dt, hp['qe'], smooth=smooth); return list(s if smooth else f)
    def fn(y, dt, hp, ctx): return batch([y], dt, hp, [ctx['tau_true']])[0]
    return dict(fn=fn, batch=batch, grid=[dict(qe=10 ** (-7 + 0.5 * k)) for k in range(15)], causal=not smooth, oracle=oracle)

def _m2_estimator():
    def batch(ys, dt, hp, taus): return [reginv(y, 10.0, dt, hp['lam']) for y in ys]
    def fn(y, dt, hp, ctx): return reginv(y, 10.0, dt, hp['lam'])
    return dict(fn=fn, batch=batch, grid=[dict(lam=10 ** (-1 + 0.5 * k)) for k in range(17)], causal=False, oracle=False)

ESTIMATORS = {
    'M1f_fixed':  _kf_estimator(False, False),   # causal Kalman filter, tau = 10 (population value)
    'M1s_fixed':  _kf_estimator(False, True),    # RTS smoother, tau = 10 (retrospective)
    'M2_reginv':  _m2_estimator(),               # regularised inversion, tau = 10 (retrospective cross-check)
    'M1f_oracle': _kf_estimator(True, False),    # told the true tau (ceiling)
    'M1s_oracle': _kf_estimator(True, True),
}

def _kf_taupop(smooth):
    """Population lag tuned on training subjects (tau and qe jointly). Needs reference glucose to tune, so simulation only."""
    def batch(ys, dt, hp, taus):
        f, s = kf_batch(np.array(ys), np.full(len(ys), hp['tau']), dt, hp['qe'], smooth=smooth); return list(s if smooth else f)
    def fn(y, dt, hp, ctx): return batch([y], dt, hp, [ctx['tau_true']])[0]
    return dict(fn=fn, batch=batch, grid=[dict(tau=t, qe=10 ** (-5 + 0.5 * k)) for t in (6, 8, 10, 12, 14, 16, 18, 20) for k in range(9)], causal=not smooth, oracle=False)

ESTIMATORS['M1f_taupop'] = _kf_taupop(False)
ESTIMATORS['M1s_taupop'] = _kf_taupop(True)
