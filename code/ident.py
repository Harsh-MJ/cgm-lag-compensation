"""Piece 5: identifiability of the lag tau.

State x = [G_F, G_I, G_B, R]:  dG_F/dt = (G_I - G_F)/tau_f  (on-device filter; tau_f ~ 0 means none),
dG_I/dt = (G_B - G_I)/tau,  dG_B/dt = R,  dR/dt = sqrt(q) white noise.
CGM: y = G_F + N(0, sigma^2).  Fingerstick (optional): fs = G_B + N(0, r_fs).
Gaussian innovation log-likelihood, batched; profiled over (q, sigma) on a grid. Plus empirical-Bayes population prior over log tau,
and a simple fingerstick regression estimator.
"""
import numpy as np
from scipy.linalg import expm

TAU_GRID = 4.0 * (28.0 / 4.0) ** (np.arange(13) / 12.0)       # log-spaced 4 ... 28 min
QE_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)                        # q / sigma^2 (1/min^3)
SIG_GRID = (1.5, 3.0, 5.0, 8.0, 12.0)                           # CGM noise sd (mg/dL)
NO_FILTER = 0.0                                                 # tau_f <= 0 means no on-device filter

def lti(tau, tau_f, dt):
    """Exact F = expm(A dt) and unit-q Q = int_0^dt expm(A u) e e^T expm(A u)^T du (Gauss-Legendre; stable for stiff tau_f).
    tau_f <= 0 means no filter stage: G_F is tied to G_I."""
    tf = tau_f if tau_f > 0 else 1.0
    A = np.zeros((4, 4)); A[0, 0] = -1 / tf; A[0, 1] = 1 / tf; A[1, 1] = -1 / tau; A[1, 2] = 1 / tau; A[2, 3] = 1
    xg, wg = np.polynomial.legendre.leggauss(16); u = (xg + 1) * dt / 2; w = wg * dt / 2
    Q = np.zeros((4, 4))
    for um, wm in zip(u, w):
        c = expm(A * um)[:, 3]; Q += wm * np.outer(c, c)
    F = expm(A * dt)
    if tau_f <= 0:                                   # G_F := G_I
        F[0, :] = F[1, :]; Q[0, :] = Q[1, :]; Q[:, 0] = Q[:, 1]; Q[0, 0] = Q[1, 1]
    return F, 0.5 * (Q + Q.T)

def kf_loglik(Y, F, Q, R, FS=None, RFS=None, burn=72):
    """Batched log-likelihood. Y (B,n) CGM with NaN; F,Q (B,4,4); R (B,); FS, RFS (B,n) NaN where no fingerstick."""
    B, n = Y.shape; Ft = F.transpose(0, 2, 1)
    first = np.array([np.where(~np.isnan(Y[b]))[0][0] for b in range(B)]); y0 = Y[np.arange(B), first]
    x = np.stack([y0, y0, y0, np.zeros(B)], 1); P = np.tile(np.diag([1e4, 1e4, 1e4, 1.0]), (B, 1, 1)); ll = np.zeros(B)
    for k in range(n):
        if k > 0: x = np.einsum('bij,bj->bi', F, x); P = F @ P @ Ft + Q
        for idx, obs, var in ((0, Y[:, k], R), (2, None if FS is None else FS[:, k], None if RFS is None else RFS[:, k])):
            if obs is None: continue
            ob = ~np.isnan(obs)
            if not ob.any(): continue
            v = np.where(ob, obs - x[:, idx], 0.0); S = P[:, idx, idx] + np.where(ob, var, 1.0)
            if k >= burn: ll += np.where(ob, -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S), 0.0)
            K = P[:, :, idx] / S[:, None] * ob[:, None]
            x = x + K * v[:, None]; P = P - K[:, :, None] * P[:, idx, :][:, None, :]; P = 0.5 * (P + P.transpose(0, 2, 1))
    return ll

def profile(Ys, dt, taus=TAU_GRID, tau_fs=(NO_FILTER,), FSs=None, RFSs=None, qes=QE_GRID, sigs=SIG_GRID, burn=72):
    """Profile log-likelihood (S, len(taus), len(tau_fs)), maximised over the (qe, sigma) grid."""
    Ys = np.asarray(Ys, float); S, n = Ys.shape; combos = [(qe, s) for qe in qes for s in sigs]; C = len(combos)
    Yb = np.repeat(Ys, C, 0); FSb = None if FSs is None else np.repeat(np.asarray(FSs, float), C, 0); RFb = None if RFSs is None else np.repeat(np.asarray(RFSs, float), C, 0)
    R = np.tile([s ** 2 for qe, s in combos], S); qabs = np.tile([qe * s ** 2 for qe, s in combos], S)
    out = np.empty((S, len(taus), len(tau_fs)))
    for i, t in enumerate(taus):
        for j, tf in enumerate(tau_fs):
            F, Q = lti(t, tf, dt); Fb = np.broadcast_to(F, (S * C, 4, 4)); Qb = Q[None] * qabs[:, None, None]
            ll = kf_loglik(Yb, Fb, Qb, R, FSb, RFb, burn); out[:, i, j] = ll.reshape(S, C).max(1)
    return out

def lr_interval(prof, taus=TAU_GRID, crit=3.84):
    """Likelihood-ratio 95% interval on the grid (lo, hi) and ML estimate, for a 1-D profile."""
    ok = 2 * (prof.max() - prof) <= crit; i = np.where(ok)[0]; return taus[i[0]], taus[i[-1]], taus[int(np.argmax(prof))]

FINE = np.linspace(np.log(4.0), np.log(28.0), 121)

def _fine(prof, taus):                                   # profile interpolated onto a fine log-tau grid
    return np.interp(FINE, np.log(taus), prof)

def eb_fit(profs, taus=TAU_GRID):
    """Empirical-Bayes population prior on log tau (normal, mean mu, sd s), prior normalised on the grid:
    maximise sum_i log sum_k L_i(tau_k) prior_k."""
    P = np.array([_fine(p, taus) for p in profs]); L = np.exp(P - P.max(1, keepdims=True)); best = None
    for mu in np.linspace(FINE[0], FINE[-1], 81):
        for s in np.linspace(0.03, 1.0, 50):
            pr = np.exp(-0.5 * ((FINE - mu) / s) ** 2); pr /= pr.sum(); m = np.log((L * pr[None]).sum(1) + 1e-300).sum()
            if best is None or m > best[0]: best = (m, mu, s)
    return best[1], best[2]

def eb_post(prof, mu, s, taus=TAU_GRID):
    """Posterior mean of log tau for one subject under the population prior; returns tau."""
    p = _fine(prof, taus); w = np.exp(p - p.max()) * np.exp(-0.5 * ((FINE - mu) / s) ** 2); w /= w.sum(); return float(np.exp((w * FINE).sum()))

def fs_regression(y, fs, dt, W=3):
    """tau_hat = sum (fs - level) * slope / sum slope^2, with level and slope from a centred local linear fit of CGM (+-W samples)."""
    n = len(y); num = den = 0.0; t = np.arange(-W, W + 1) * dt
    for k in np.where(~np.isnan(fs))[0]:
        if k - W < 0 or k + W >= n: continue
        w = y[k - W:k + W + 1]; m = ~np.isnan(w)
        if m.sum() < W + 2: continue
        p = np.polyfit(t[m], w[m], 1); num += (fs[k] - p[1]) * p[0]; den += p[0] ** 2
    return num / den if den > 0 else np.nan

def make_fingersticks(G, t, rng, cv=0.04, times_h=(7.0, 11.5, 17.5, 22.0), jitter_min=15.0, dt=5):
    """4 per day at about 07:00, 11:30, 17:30, 22:00 (sd 15 min), multiplicative 4% CV error. Returns FS and assumed variance (5% CV + 2 mg/dL)."""
    fs = np.full(len(G), np.nan); days = int(t[-1] // 1440) + 1
    for d in range(days):
        for h in times_h:
            k = int(round((d * 1440 + h * 60 + rng.normal(0, jitter_min) - t[0]) / dt))
            if 0 <= k < len(G): fs[k] = G[k] * (1 + cv * rng.standard_normal())
    return fs, (0.05 * np.nan_to_num(fs)) ** 2 + 4.0
