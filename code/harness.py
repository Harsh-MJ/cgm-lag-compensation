"""Evaluation harness (Piece 3): subject-wise CV, baselines B0-B3, metrics, hypoglycaemia detection, cluster bootstrap.
Estimator interface: fn(y, dt, hp, ctx) -> Ghat (same length as y). y has NaN at dropouts. ctx has only what an estimator may know
('tau_true' is for oracle rows only and they are labelled)."""
import numpy as np
import sensor as S
from scipy.stats import wilcoxon

WARM_MIN = 360
RAPID = 1.0          # mg/dL/min, |true rate| at or above this = rapid change

# ---------------------------------------------------------------- data and splits
def load_patients(simpath):
    D = np.load(simpath)
    names = sorted({k.split('|')[0] for k in D.files}, key=lambda s: (s.split('#')[0], s))
    pat = {n: dict(plasma=D[n + '|plasma'], tau=float(D[n + '|tau'])) for n in names}
    excl = [n for n, p in pat.items() if p['plasma'].min() < 30 or (p['plasma'] < 54).mean() > 0.15]
    return pat, [n for n in names if n not in excl], excl

def make_folds(use, k=5):
    """Subject-wise folds, cycling inside each age group so every fold has all groups."""
    groups = {}
    for n in use: groups.setdefault(n.split('#')[0], []).append(n)
    folds = [[] for _ in range(k)]; c = 0
    for g in sorted(groups):
        for n in groups[g]: folds[c % k].append(n); c += 1
    return folds

def build(pat, use, scenario, noise_sd, dt=5, miss=0.05, seed0=1000):
    out = {}
    for i, n in enumerate(use):
        g = pat[n]['plasma']; r = S.make_sensor(g, pat[n]['tau'], scenario, noise_sd=noise_sd, dt=dt, seed=seed0 + i, miss_frac=miss)
        sm = np.convolve(g, np.ones(5) / 5, 'same'); rate = np.gradient(sm)[r['t'].astype(int)]
        r.update(name=n, rate=rate, rapid=np.abs(rate) >= RAPID, warm=int(WARM_MIN / dt)); out[n] = r
    return out

# ---------------------------------------------------------------- estimators (B0-B3)
def locf(y):
    y = y.copy(); ok = np.where(~np.isnan(y))[0]
    if len(ok) == 0: return y
    y[:ok[0]] = y[ok[0]]
    for k in range(ok[0] + 1, len(y)):
        if np.isnan(y[k]): y[k] = y[k - 1]
    return y

def b0_raw(y, dt, hp, ctx): return locf(y)

def b1_shift(y, dt, hp, ctx):                      # NON-causal: advance the signal by s minutes
    yl = locf(y); k = int(round(hp['s'] / dt)); idx = np.minimum(np.arange(len(y)) + k, len(y) - 1); return yl[idx]

def b2_smooth(y, dt, hp, ctx):                     # causal exponential smoothing (no lag removal)
    yl = locf(y); a = hp['alpha']; o = yl.copy()
    for k in range(1, len(o)): o[k] = a * yl[k] + (1 - a) * o[k - 1]
    return o

def _local(yl, W, dt):
    n = len(yl); lev = yl.copy(); sl = np.zeros(n); t = (np.arange(W) - (W - 1)) * dt; tm = t.mean(); stt = ((t - tm) ** 2).sum()
    for k in range(W - 1, n):
        w = yl[k - W + 1:k + 1]; s = ((t - tm) * (w - w.mean())).sum() / stt; sl[k] = s; lev[k] = w.mean() - s * tm
    return lev, sl

def b3_inverse(y, dt, hp, ctx):                    # causal naive inversion with population tau
    lev, sl = _local(locf(y), hp['W'], dt); return lev + hp['tau'] * sl

def b3_oracle(y, dt, hp, ctx):                     # same, but told the true tau (ceiling, not a fair competitor)
    lev, sl = _local(locf(y), hp['W'], dt); return lev + ctx['tau_true'] * sl

ESTIMATORS = {
    'B0_raw':        dict(fn=b0_raw,     grid=[{}],                                              causal=True,  oracle=False),
    'B1_shift':      dict(fn=b1_shift,   grid=[dict(s=s) for s in range(0, 31, 1)],              causal=False, oracle=False),
    'B2_smooth':     dict(fn=b2_smooth,  grid=[dict(alpha=a) for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)], causal=True, oracle=False),
    'B3_inverse':    dict(fn=b3_inverse, grid=[dict(W=w, tau=10.0) for w in (3, 4, 5, 6, 8, 10, 13, 17)],   causal=True, oracle=False),
    'B3_oracle_tau': dict(fn=b3_oracle,  grid=[dict(W=w) for w in (3, 4, 5, 6, 8, 10, 13, 17)],            causal=True, oracle=True),
}

# ---------------------------------------------------------------- metrics
def hypo_stats(Gh, G, dt, thr=70.0, warm=0):
    gap = int(30 / dt); win = int(30 / dt); low = G < thr; n = len(G); ev = []; k = warm
    while k < n:
        if low[k]:
            s = k
            while k < n and low[k]: k += 1
            ev.append([s, k - 1])
        else: k += 1
    merged = []
    for s, e in ev:
        if merged and s - merged[-1][1] < gap: merged[-1][1] = e
        else: merged.append([s, e])
    al = Gh < thr; det = 0; delays = []
    for s, e in merged:
        ws = max(warm, s - win); w = np.where(al[ws:e + 1])[0]
        if len(w): det += 1; delays.append((ws + w[0] - s) * dt)
    onsets = []
    for k in range(max(warm, 1), n):
        if al[k] and not al[k - 1] and (not onsets or k - onsets[-1] >= gap): onsets.append(k)
    fa = sum(1 for t in onsets if not low[max(0, t - win):t + win + 1].any())
    return dict(ev=len(merged), det=det, dsum=float(sum(delays)), fa=fa, days=(n - warm) * dt / 1440.0)

def residual_lag(Gh, G, dt, jmin=-6, jmax=12):
    """Shift (min) of the estimate that best matches truth; positive = estimate still lags truth."""
    best = None
    for j in range(jmin, jmax + 1):
        a, b = (Gh[j:], G[:len(G) - j]) if j >= 0 else (Gh[:j], G[-j:])
        e = np.sqrt(np.mean((a - b) ** 2))
        if best is None or e < best[0]: best = (e, j)
    return best[1] * dt

def subject_metrics(Gh, r, dt):
    w = r['warm']; G = r['G'][w:]; h = Gh[w:]; e = h - G; hi = G >= 40; rap = r['rapid'][w:]
    k = max(1, int(round(15 / dt)))
    roc_h = (h[k:] - h[:-k]) / (k * dt); roc_t = (G[k:] - G[:-k]) / (k * dt)
    m = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mad=float(np.mean(np.abs(e))), bias=float(e.mean()),
             mard=float(np.mean(np.abs(e[hi]) / G[hi]) * 100),
             rmse_rapid=float(np.sqrt(np.mean(e[rap] ** 2))) if rap.sum() >= 20 else np.nan, n_rapid=int(rap.sum()),
             rmse_roc=float(np.sqrt(np.mean((roc_h - roc_t) ** 2))), lag=residual_lag(h, G, dt),
             rapid_frac=float(rap.mean()))
    m.update(hypo_stats(h, G, dt)); return m

# ---------------------------------------------------------------- cross-validation
def cv_run(data, use, folds, estimators=None, verbose=False):
    """Outer subject-wise CV. Hyper-parameters chosen on the NON-test subjects only, by pooled RMSE against true glucose
    (default) or, if the estimator has tune_on='fs', against fingersticks (data[n]['fs'], NaN elsewhere) - the realistic option.
    Outputs are cached by (estimator function, grid, tau source, hyper-parameter, subject) so variants that differ only in tuning share work."""
    est = estimators or ESTIMATORS; res = {k: {} for k in est}; chosen = {k: [] for k in est}; edge = {k: 0 for k in est}
    dt = data[use[0]]['dt']; cache = {}
    def out(name, i, n):
        E = est[name]; base = (id(E.get('batch', E['fn'])), id(E['grid']), E.get('tau_key', 'tau'), i)
        if base + (n,) not in cache:
            if 'batch' in E:                    # batched estimators run all subjects for one hyper-parameter together
                outs = E['batch']([data[m]['y'] for m in use], dt, E['grid'][i], [data[m][E.get('tau_key', 'tau')] for m in use])
                for m, o in zip(use, outs): cache[base + (m,)] = o
            else:
                r = data[n]; cache[base + (n,)] = E['fn'](r['y'], dt, E['grid'][i], dict(tau_true=r['tau']))
        return cache[base + (n,)]
    for f, test in enumerate(folds):
        tune = [n for n in use if n not in test]; assert not set(tune) & set(test)
        for name, E in est.items():
            sc = []
            for i in range(len(E['grid'])):
                se = 0.0; cnt = 0
                for n in tune:
                    if E.get('tune_on', 'G') == 'fs':
                        fs = data[n]['fs']; m = ~np.isnan(fs); d = out(name, i, n)[m] - fs[m]
                    else:
                        w = data[n]['warm']; d = out(name, i, n)[w:] - data[n]['G'][w:]
                    se += (d ** 2).sum(); cnt += len(d)
                sc.append(np.sqrt(se / cnt))
            b = int(np.argmin(sc)); chosen[name].append(E['grid'][b])
            if len(sc) > 2 and b in (0, len(sc) - 1) and sc[0] != sc[-1]: edge[name] += 1
            for n in test: res[name][n] = subject_metrics(out(name, b, n), data[n], dt)
    return res, chosen, edge

# ---------------------------------------------------------------- uncertainty
def boot_idx(n, B, rng): return rng.integers(0, n, size=(B, n))

def ci_median(v, B=2000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; rng = np.random.default_rng(seed)
    m = np.median(v[boot_idx(len(v), B, rng)], axis=1); return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def ci_paired(a, b, B=2000, seed=0):
    """median(a - b) over subjects with cluster-bootstrap CI and Wilcoxon signed-rank p (two-sided)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[~np.isnan(d)]; rng = np.random.default_rng(seed)
    m = np.median(d[boot_idx(len(d), B, rng)], axis=1)
    try: p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError: p = 1.0
    return float(np.median(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), p

def ci_ratio(num, den, B=2000, seed=0):
    """Pooled ratio sum(num)/sum(den) with subject-level bootstrap (for hypoglycaemia delay and false alarms per day)."""
    num = np.asarray(num, float); den = np.asarray(den, float); rng = np.random.default_rng(seed); ix = boot_idx(len(num), B, rng)
    with np.errstate(invalid='ignore', divide='ignore'): bs = num[ix].sum(1) / den[ix].sum(1)
    bs = bs[~np.isnan(bs)]; pt = num.sum() / den.sum() if den.sum() > 0 else np.nan
    return float(pt), (float(np.percentile(bs, 2.5)) if len(bs) else np.nan), (float(np.percentile(bs, 97.5)) if len(bs) else np.nan)

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()
