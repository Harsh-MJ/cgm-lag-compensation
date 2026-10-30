"""Estimators: raw CGM, fixed-tau inversion, EKF (fixed / online tau, optional fingersticks)."""
import numpy as np
DT = 3.0  # CGM sample time (min)

def local_fit(y, W):
    """Causal local linear regression over last W samples -> (level at now, slope per min)."""
    n = len(y); lev = np.full(n, np.nan); sl = np.full(n, np.nan)
    t = (np.arange(W) - (W - 1)) * DT; tm = t.mean(); stt = ((t - tm) ** 2).sum()
    for k in range(n):
        if k < W - 1: lev[k] = y[k]; sl[k] = 0.0; continue
        w = y[k - W + 1:k + 1]
        s = ((t - tm) * (w - w.mean())).sum() / stt
        sl[k] = s; lev[k] = w.mean() - s * tm
    return lev, sl

def inversion(y, tau, W):
    lev, sl = local_fit(y, W)
    return lev + tau * sl, sl                      # G_hat, rate

def _prop(x, phi):
    """3 Euler sub-steps of 1 min for glucose/sensor, AR(1) step for sensor noise. x=[G,r,I,theta,e]."""
    x = x.copy(); F = np.eye(5)
    for _ in range(3):
        G, r, I, th = x[:4]; it = np.exp(-th)
        Fs = np.eye(5); Fs[0, 1] = 1.0
        Fs[2, 0] = it; Fs[2, 2] = 1 - it; Fs[2, 3] = -(G - I) * it
        x = np.array([G + r, r, I + (G - I) * it, th, x[4]]); F = Fs @ F
    x[4] *= phi; F[4, :] *= phi
    return x, F

def ekf(y, fs, qr, sig_e, phi=0.7, R=1.0, tau0=10.0, free_tau=False, q_theta=0.005 ** 2, sd_theta0=0.4):
    """State [G, r, I, theta=ln(tau), e]; y = I + e (e = AR(1) sensor noise, sd sig_e)."""
    n = len(y)
    x = np.array([y[0], 0.0, y[0], np.log(tau0), 0.0])
    P = np.diag([20.0 ** 2, 1.0, 20.0 ** 2, sd_theta0 ** 2 if free_tau else 0.0, sig_e ** 2])
    Qd = np.diag([0.5 * 3, qr * 3, 0.1 * 3, q_theta * 3 if free_tau else 0.0, sig_e ** 2 * (1 - phi ** 2)])
    h = np.array([0, 0, 1.0, 0, 1.0]); e0 = np.array([1.0, 0, 0, 0, 0])
    G = np.zeros(n); r = np.zeros(n); th = np.zeros(n)
    for k in range(n):
        if k > 0:
            x, F = _prop(x, phi); P = F @ P @ F.T + Qd
        Ph = P @ h; S = h @ Ph + R; K = Ph / S
        x = x + K * (y[k] - h @ x); P = P - np.outer(K, Ph); P = .5 * (P + P.T)
        if not np.isnan(fs[k]):
            Rf = (0.05 * fs[k]) ** 2 + 4.0; Pe = P @ e0; S = Pe[0] + Rf; K = Pe / S
            x = x + K * (fs[k] - x[0]); P = P - np.outer(K, Pe); P = .5 * (P + P.T)
        if free_tau: x[3] = np.clip(x[3], np.log(2.0), np.log(40.0))
        G[k], r[k], th[k] = x[0], x[1], x[3]
    return G, r, np.exp(th)
