"""Sensor layer for the CGM lag project (Piece 2).

Takes the true plasma glucose (1-min grid, from simglucose) and builds a CGM-like signal with
known, controllable error sources, so every estimator can be scored against known truth.

Chain (all on a 1-min grid, then sampled every `dt` min):
  plasma G_B --[ISF kinetics: 1st order | 2nd order | time-varying tau]--> G_I
             --[on-device filter tau_f, pure delay d]--> G_F
             --[calibration drift, compression lows]--> + noise (slow + fast AR(1)) --> y
"""
import numpy as np

# name -> options. Nominal lag the estimator is told about is tau only; extra delay is unmodelled.
SCENARIOS = {
    'S0_matched':       dict(order=1, tau_cv=0.0,  tau_f=0.0, delay=0.0, drift=False, compression=False),
    'S1_sensor_chain':  dict(order=1, tau_cv=0.0,  tau_f=3.0, delay=2.0, drift=False, compression=False),
    'S2_second_order':  dict(order=2, tau_cv=0.0,  tau_f=0.0, delay=0.0, drift=False, compression=False),
    'S3_tau_varying':   dict(order=1, tau_cv=0.25, tau_f=0.0, delay=0.0, drift=False, compression=False),
    'S4_calib_artifact':dict(order=1, tau_cv=0.0,  tau_f=0.0, delay=0.0, drift=True,  compression=True),
    'S5_full':          dict(order=1, tau_cv=0.25, tau_f=3.0, delay=2.0, drift=True,  compression=True),
}
NOISE_SLOW_SHARE = 0.6      # share of noise variance in the slow component
TAU_SLOW, TAU_FAST = 30.0, 2.0   # correlation times (min) of the two AR(1) components
DRIFT_GAIN_SD, DRIFT_OFF_SD = 0.04, 4.0   # fractional gain, mg/dL offset (OU, 2-day / 1-day correlation)
COMPRESS_P_NIGHT, COMPRESS_DUR, COMPRESS_DEPTH = 0.4, (20, 45), (0.25, 0.45)


def first_order(x, tau, x0=None):
    """tau dy/dt = x - y on a 1-min grid; trapezoidal input (exact lag tau for a ramp). tau scalar or array."""
    n = len(x); y = np.empty(n); y[0] = x[0] if x0 is None else x0
    tau = np.broadcast_to(np.asarray(tau, float), (n,))
    xm = np.empty(n); xm[0] = x[0]; xm[1:] = 0.5 * (x[1:] + x[:-1])
    for k in range(1, n):
        if tau[k] < 0.05: y[k] = x[k]; continue
        a = np.exp(-1.0 / tau[k]); y[k] = a * y[k - 1] + (1 - a) * xm[k]
    return y


def ou(n, corr_min, sd, rng):
    """Zero-mean Ornstein-Uhlenbeck / AR(1) on a 1-min grid with stationary sd `sd`."""
    rho = np.exp(-1.0 / corr_min); z = np.empty(n); z[0] = rng.standard_normal() * sd
    e = rng.standard_normal(n) * sd * np.sqrt(1 - rho ** 2)
    for k in range(1, n): z[k] = rho * z[k - 1] + e[k]
    return z


def noise(n, sd, rng):
    """Slow + fast AR(1) noise, total sd = sd."""
    if sd <= 0: return np.zeros(n)
    return ou(n, TAU_SLOW, sd * np.sqrt(NOISE_SLOW_SHARE), rng) + ou(n, TAU_FAST, sd * np.sqrt(1 - NOISE_SLOW_SHARE), rng)


def delay(x, d):
    if d <= 0: return x.copy()
    t = np.arange(len(x)); return np.interp(t - d, t, x)


def compression(n, rng):
    """Multiplicative dips (night only). Returns factor (1 = none) and boolean flag."""
    f = np.ones(n); flag = np.zeros(n, bool)
    for day in range(n // 1440 + 1):
        if rng.random() < COMPRESS_P_NIGHT:
            s = int(day * 1440 + rng.uniform(23, 30) * 60)          # 23:00 to 06:00 (next day)
            dur = int(rng.uniform(*COMPRESS_DUR)); depth = rng.uniform(*COMPRESS_DEPTH)
            for k in range(s, min(s + dur, n)):
                ramp = min(1.0, (k - s + 1) / 5.0, (s + dur - k) / 5.0)
                f[k] = 1 - depth * ramp; flag[k] = True
    return f, flag


def dropout_mask(m, frac, rng, mean_len=3):
    """True = missing. Bursts of mean length `mean_len` samples, overall fraction ~ frac."""
    miss = np.zeros(m, bool)
    if frac <= 0: return miss
    p = frac / mean_len; k = 0
    while k < m:
        if rng.random() < p:
            L = rng.geometric(1.0 / mean_len); miss[k:k + L] = True; k += L
        else: k += 1
    return miss


def make_sensor(plasma, tau, scenario='S0_matched', noise_sd=5.0, dt=5, seed=0, miss_frac=0.0, offset=0):
    """plasma: 1-min true plasma glucose (mg/dL). Returns dict of arrays sampled every dt min."""
    o = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed); n = len(plasma)
    # ISF kinetics
    if o['tau_cv'] > 0:
        z = ou(n, 720.0, 1.0, rng); tau_t = np.clip(tau * np.exp(o['tau_cv'] * z), 3.0, 40.0)
    else:
        tau_t = np.full(n, float(tau))
    if o['order'] == 1: GI = first_order(plasma, tau_t)
    else:               GI = first_order(first_order(plasma, tau_t / 2), tau_t / 2)
    # sensor chain
    GF = first_order(GI, o['tau_f']) if o['tau_f'] > 0 else GI.copy()
    GF = delay(GF, o['delay'])
    out = GF.copy()
    if o['drift']:
        out = (1 + ou(n, 2880.0, DRIFT_GAIN_SD, rng)) * out + ou(n, 1440.0, DRIFT_OFF_SD, rng)
    flag = np.zeros(n, bool)
    if o['compression']:
        f, flag = compression(n, rng); out = out * f
    y_full = out + noise(n, noise_sd, rng)
    idx = np.arange(offset, n, dt)
    miss = dropout_mask(len(idx), miss_frac, rng)
    y = y_full[idx].copy(); y[miss] = np.nan
    return dict(t=idx.astype(float), y=y, G=plasma[idx], GI=GI[idx], tau=float(tau), tau_t=tau_t[idx],
                tau_eff_nominal=float(tau + o['tau_f'] + o['delay']), compress=flag[idx], missing=miss,
                scenario=scenario if isinstance(scenario, str) else 'custom', noise_sd=noise_sd, dt=dt, seed=seed)
