"""Piece 6: meal- and insulin-driven prior for blood glucose.

The Piece 4/5 models let blood glucose drift as a random walk in its rate (dG_B/dt = R). Meals and insulin make glucose move in
ways a random-walk rate cannot describe, which is the misspecification found in Piece 5. Here the known inputs drive G_B:

    dG_I/dt = (G_B - G_I)/tau
    dG_B/dt = R + a*u1(t) - b*u2(t)          u1 = meal appearance (g/min),  u2 = insulin action (U/min)
    dR/dt   = sqrt(q) * white noise          (what the inputs do not explain)
    da/dt = db/dt = small random walk        (patient gains a, b in mg/dL per g and per U, learned online)
    y_k     = G_I(t_k) + e_k

u1 = logged carbohydrate convolved with a two-compartment gut kernel (time constant TAU_MEAL);
u2 = (insulin - basal) convolved with a three-compartment subcutaneous kernel (time constant TAU_INS).
Kernel shapes are population constants (physics); only the gains a, b are patient-specific states.
The state is 5-D: [G_I, G_B, R, a, b]. The inputs enter through an exact piecewise-constant discretisation.
"""
import numpy as np

TAU_MEAL, TAU_INS = 20.0, 40.0                  # min
A0, B0 = 5.0, 40.0                              # prior means of the gains (mg/dL per g, mg/dL per U)
A_VAR, B_VAR = 100.0, 2500.0                    # prior variances (weakly informative)
QA, QB = 0.01, 0.5                              # random-walk variance of a, b per sample step

def _gamma_kernel(tau, order, n=900):
    t = np.arange(n, dtype=float); h = t ** (order - 1) * np.exp(-t / tau); return h / h.sum()

def drive(cho, ins, tau_meal=TAU_MEAL, tau_ins=TAU_INS):
    """1-min input drives. cho in g/min, ins in U/min. Returns u1 (g/min appearing), u2 (U/min of insulin action above basal)."""
    cho = np.asarray(cho, float); ins = np.asarray(ins, float); n = len(cho)
    u1 = np.convolve(cho, _gamma_kernel(tau_meal, 2))[:n]
    u2 = np.convolve(ins - np.median(ins), _gamma_kernel(tau_ins, 3))[:n]
    return u1, u2

def sample_drive(u1, u2, t, dt):
    """Mean drive over each sample interval [t_k, t_k + dt): (n, 2)."""
    c1 = np.r_[0.0, np.cumsum(u1)]; c2 = np.r_[0.0, np.cumsum(u2)]; i = t.astype(int); j = np.minimum(i + int(dt), len(u1))
    return np.stack([(c1[j] - c1[i]) / np.maximum(j - i, 1), (c2[j] - c2[i]) / np.maximum(j - i, 1)], 1)

def _fq(tau, dt):
    """3-state exact F (S,3,3) and unit-q Q (S,3,3): same as kalman.model."""
    tau = np.atleast_1d(np.asarray(tau, float)); S = len(tau); e = np.exp(-dt / tau)
    F = np.zeros((S, 3, 3)); F[:, 0, 0] = e; F[:, 0, 1] = 1 - e; F[:, 0, 2] = dt - tau * (1 - e)
    F[:, 1, 1] = 1; F[:, 1, 2] = dt; F[:, 2, 2] = 1
    xg, wg = np.polynomial.legendre.leggauss(12); u = (xg + 1) * dt / 2; w = wg * dt / 2
    v = np.stack([u[None, :] - tau[:, None] * (1 - np.exp(-u[None, :] / tau[:, None])), np.broadcast_to(u, (S, len(u))), np.ones((S, len(u)))], axis=2)
    return F, np.einsum('m,smi,smj->sij', w, v, v)

def kfx(Y, U, tau, dt, qe, sig, FS=None, RFS=None, smooth=False, burn=72, ll=False, tune_gain=True, ar=None):
    """Batched Kalman filter (and optional RTS smoother) with input-driven blood glucose.
    Y (S,n) CGM with NaN; U (S,n,2) drives; tau (S,) or scalar; qe, sig scalars or (S,): q = qe*sig^2 is the rate process noise, sig the CGM noise sd.
    FS, RFS (S,n): optional fingersticks on G_B and their variances. tune_gain=False freezes a, b at their prior means.
    ar=(corr_min, share): Piece 8 ablation. A share of the noise variance (sig^2 * share) is modelled as an AR(1) state v (correlation time corr_min) added to
    the CGM, the rest as white noise; ar=None is the original white-noise model.
    Returns dict(filt, smooth: G_B estimates (S,n) (smooth None unless asked); filt_sd, smooth_sd: posterior sd of G_B; ll (S,); a, b final gains)."""
    Y = np.asarray(Y, float); S, n = Y.shape; d = 5 + (ar is not None); tau = np.broadcast_to(np.asarray(tau, float), (S,)).copy(); F3, Q0 = _fq(tau, dt)
    qe = np.broadcast_to(np.asarray(qe, float), (S,)); sg = np.broadcast_to(np.asarray(sig, float), (S,)); share = ar[1] if ar else 0.0
    R = sg ** 2 * (1 - share)
    Q = np.zeros((S, d, d)); Q[:, :3, :3] = (qe * sg ** 2)[:, None, None] * Q0
    Q[:, 3, 3], Q[:, 4, 4] = (QA, QB) if tune_gain else (1e-10, 1e-10)       # frozen gains keep a tiny variance so the smoother stays invertible
    first = np.array([np.where(~np.isnan(Y[s]))[0][0] for s in range(S)]); y0 = Y[np.arange(S), first]
    x = np.zeros((S, d)); x[:, 0] = y0; x[:, 1] = y0; x[:, 3] = A0; x[:, 4] = B0
    P = np.zeros((S, d, d)); P[:, 0, 0] = P[:, 1, 1] = 1e4; P[:, 2, 2] = 1e2
    P[:, 3, 3], P[:, 4, 4] = (A_VAR, B_VAR) if tune_gain else (1e-8, 1e-8)
    F = np.zeros((S, d, d)); F[:, :3, :3] = F3; F[:, 3, 3] = F[:, 4, 4] = 1.0
    H0 = np.zeros(d); H0[0] = 1.0; H1 = np.zeros(d); H1[1] = 1.0
    if ar:
        rho = np.exp(-dt / ar[0]); F[:, 5, 5] = rho; Q[:, 5, 5] = share * sg ** 2 * (1 - rho ** 2); P[:, 5, 5] = share * sg ** 2; H0[5] = 1.0
    xp = np.empty((n, S, d)); Pp = np.empty((n, S, d, d)); xf = np.empty((n, S, d)); Pf = np.empty((n, S, d, d)); Fs = np.empty((n, S, d, d)) if smooth else None
    LL = np.zeros(S)
    for k in range(n):
        if k > 0:
            F[:, 0, 3] = F3[:, 0, 2] * U[:, k - 1, 0]; F[:, 1, 3] = F3[:, 1, 2] * U[:, k - 1, 0]
            F[:, 0, 4] = -F3[:, 0, 2] * U[:, k - 1, 1]; F[:, 1, 4] = -F3[:, 1, 2] * U[:, k - 1, 1]
            x = np.einsum('sij,sj->si', F, x); P = F @ P @ F.transpose(0, 2, 1) + Q
            if smooth: Fs[k] = F
        xp[k] = x; Pp[k] = P
        for H, obs, var in ((H0, Y[:, k], R), (H1, None if FS is None else FS[:, k], None if RFS is None else RFS[:, k])):
            if obs is None: continue
            ob = ~np.isnan(obs)
            if not ob.any(): continue
            PH = P @ H; v = np.where(ob, obs - x @ H, 0.0); Sv = PH @ H + np.where(ob, var, 1.0)
            if ll and k >= burn: LL += np.where(ob, -0.5 * (np.log(2 * np.pi * Sv) + v ** 2 / Sv), 0.0)
            K = PH / Sv[:, None] * ob[:, None]
            x = x + K * v[:, None]; P = P - K[:, :, None] * PH[:, None, :]; P = 0.5 * (P + P.transpose(0, 2, 1))
        xf[k] = x; Pf[k] = P
    out = dict(filt=xf[:, :, 1].T.copy(), filt_sd=np.sqrt(np.maximum(Pf[:, :, 1, 1], 0)).T.copy(), smooth=None, smooth_sd=None, ll=LL, a=xf[-1, :, 3].copy(), b=xf[-1, :, 4].copy())
    if smooth:
        xs = xf.copy(); Ps = Pf.copy()
        for k in range(n - 2, -1, -1):
            C = np.linalg.solve(Pp[k + 1], Fs[k + 1] @ Pf[k]).transpose(0, 2, 1)
            xs[k] = xf[k] + np.einsum('sij,sj->si', C, xs[k + 1] - xp[k + 1]); Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.transpose(0, 2, 1)
        out['smooth'] = xs[:, :, 1].T.copy(); out['smooth_sd'] = np.sqrt(np.maximum(Ps[:, :, 1, 1], 0)).T.copy()
    return out

TAU_GRID = 4.0 * (28.0 / 4.0) ** (np.arange(13) / 12.0)       # same grid as Piece 5
QE_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SIG_GRID = (1.5, 3.0, 5.0, 8.0, 12.0)

def profile(Ys, Us, dt, taus=TAU_GRID, qes=QE_GRID, sigs=SIG_GRID, FSs=None, RFSs=None, burn=72, tune_gain=True):
    """Profile log-likelihood (S, len(taus)) with input-driven prior, maximised over the (qe, sigma) grid."""
    Ys = np.asarray(Ys, float); Us = np.asarray(Us, float); S = len(Ys); combos = [(qe, s) for qe in qes for s in sigs]; C = len(combos)
    Yb = np.repeat(Ys, C, 0); Ub = np.repeat(Us, C, 0); FSb = None if FSs is None else np.repeat(np.asarray(FSs, float), C, 0)
    RFb = None if RFSs is None else np.repeat(np.asarray(RFSs, float), C, 0)
    qeb = np.tile([qe for qe, s in combos], S); sgb = np.tile([s for qe, s in combos], S); out = np.empty((S, len(taus)))
    for i, t in enumerate(taus):
        r = kfx(Yb, Ub, t, dt, qeb, sgb, FSb, RFb, burn=burn, ll=True, tune_gain=tune_gain); out[:, i] = r['ll'].reshape(S, C).max(1)
    return out

# ------------------------------------------------------------------ harness registry (inputs carried in data[n]['TU'] = (tau, U))
def _x_estimator(smooth, tau_source, tune_gain=True, ar=None, tau_key='TU'):
    """tau_source: 'fixed' (tau = 10) or 'key' (per-subject tau stored in data[n]['TU'][0])."""
    def batch(ys, dt, hp, tu):
        tau = np.array([10.0 if tau_source == 'fixed' else t for t, _ in tu]); U = np.array([u for _, u in tu])
        r = kfx(np.array(ys), U, tau, dt, hp['qe'], hp['sig'], smooth=smooth, tune_gain=tune_gain, ar=ar); return list(r['smooth'] if smooth else r['filt'])
    def fn(y, dt, hp, ctx): raise NotImplementedError
    return dict(fn=fn, batch=batch, grid=[dict(qe=10 ** (-7 + 0.5 * k), sig=s) for k in range(13) for s in (2.0, 5.0, 8.0)], causal=not smooth, oracle=False, tau_key=tau_key)

ESTIMATORS = {'X1f_fixed': _x_estimator(False, 'fixed'), 'X1s_fixed': _x_estimator(True, 'fixed'),
              'X1f_key': _x_estimator(False, 'key'), 'X1s_key': _x_estimator(True, 'key')}
