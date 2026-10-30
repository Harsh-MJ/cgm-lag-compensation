import numpy as np, json, pickle, sys, itertools, time, os
NS = float(os.environ.get('NS', '1')); SUF = '' if NS == 1 else '_ns%g' % NS
from multiprocessing import Pool
import est
np.random.seed(0)
D = np.load('../data/sim.npz')
names = sorted({k.split('|')[0] for k in D.files}, key=lambda s: (s.split('#')[0], s))
names = [n for n in names]
STEP = 3; WARM = 120; H = 10          # warm-up samples (6 h), forecast horizon 10 samples = 30 min
def load(n, rng):
    N = len(D[f'{n}|plasma']); idx = np.arange(2, N, STEP)
    y = D[f'{n}|cgm'][idx]; G = D[f'{n}|plasma'][idx]; Gs = D[f'{n}|sub'][idx]; y = Gs + NS * (y - Gs)
    fs = np.full(len(idx), np.nan)
    for d in range(N // 1440):
        for tm in (7 * 60, 11.5 * 60, 17.5 * 60, 22 * 60):
            k = int((d * 1440 + tm + rng.normal(0, 15)) // STEP)
            if 0 <= k < len(idx): fs[k] = G[k] * (1 + 0.04 * rng.standard_normal())
    return dict(name=n, y=y, G=G, Gs=Gs, fs=fs, tau=float(D[f'{n}|tau']))
rng = np.random.default_rng(1)
P_all = [load(n, rng) for n in names]
excl = [p['name'] for p in P_all if p['G'].min() < 30 or (p['G'] < 54).mean() > 0.15]
P = [p for p in P_all if p['name'] not in excl]
print('excluded unphysiological virtual patients:', excl, flush=True)
tune = [p for i, p in enumerate(P) if i % 2 == 0]; test = [p for i, p in enumerate(P) if i % 2 == 1]
E_ = [p['y'][5:] - p['Gs'][5:] for p in tune]
SIG = max(float(np.std(np.concatenate(E_))), 0.5)
PHI = float(np.mean([np.corrcoef(e[:-1], e[1:])[0, 1] for e in E_])) if NS > 0 else 0.98
R = SIG ** 2
print('CGM noise sd %.2f mg/dL, lag-1 autocorr %.2f ; tune n=%d test n=%d' % (SIG, PHI, len(tune), len(test)), flush=True)

def run_variant(p, v, qr=None, W=5):
    y, fs = p['y'], p['fs']; nofs = np.full_like(y, np.nan)
    if v == 'raw':
        lev, sl = est.local_fit(y, W); return y.copy(), sl, y.copy()
    if v == 'inv':
        g, sl = est.inversion(y, 10.0, W); return g, sl, g + sl * 30
    cfg = dict(ekf_fix=(nofs, 10.0, False), ekf_fix_fs=(fs, 10.0, False), ekf_est=(nofs, 10.0, True),
               ekf_est_fs=(fs, 10.0, True), ekf_oracle=(nofs, p['tau'], False))[v]
    g, r, th = est.ekf(y, cfg[0], qr, SIG, PHI, 1.0, cfg[1], cfg[2])
    p.setdefault('th', {})[v] = th
    return g, r, g + r * 30

def rmse(pred, pats, key='G'):
    e = np.concatenate([(pr[WARM:] - p[key][WARM:]) for pr, p in zip(pred, pats)]); return float(np.sqrt((e ** 2).mean()))

def job(a):
    p, v, qr, W = a
    g, r, f = run_variant(dict(p), v, qr, W); return g

def score(v, qr, W, pats):
    with Pool(2) as pool: gs = pool.map(job, [(p, v, qr, W) for p in pats])
    return rmse(gs, pats), gs

best = {}
t0 = time.time()
for v in ['inv', 'ekf_fix', 'ekf_fix_fs', 'ekf_est', 'ekf_est_fs', 'ekf_oracle']:
    grid = [dict(W=w) for w in (3, 5, 7, 9)] if v == 'inv' else [dict(qr=q) for q in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)]
    res = [(score(v, g.get('qr'), g.get('W', 5), tune)[0], g) for g in grid]
    best[v] = min(res, key=lambda t: t[0])[1]
    print(v, 'tuned', best[v], ['%.2f' % r[0] for r in res], '%.0fs' % (time.time() - t0), flush=True)
best['raw'] = dict(W=5)
pickle.dump(dict(best=best, R=R), open('../data/tuned%s.pkl' % SUF, 'wb'))

# ---- evaluation on held-out patients
def hypo_stats(Gh, G, thr=70):
    low = G < thr; n = len(G)
    ev = []; k = WARM
    while k < n:
        if low[k]:
            s = k
            while k < n and low[k]: k += 1
            ev.append([s, k - 1])
        else: k += 1
    merged = []
    for s, e in ev:
        if merged and s - merged[-1][1] < 10: merged[-1][1] = e
        else: merged.append([s, e])
    al = Gh < thr; det = 0; delays = []
    for s, e in merged:
        ws = max(WARM, s - 10); w = np.where(al[ws:e + 1])[0]
        if len(w): det += 1; delays.append((ws + w[0] - s) * STEP)
    onsets = []; k = WARM
    for k in range(WARM, n):
        if al[k] and not al[k - 1] and (not onsets or k - onsets[-1] >= 10): onsets.append(k)
    fa = sum(1 for t in onsets if not low[max(0, t - 10):t + 11].any())
    return len(merged), det, delays, fa, (n - WARM) * STEP / 1440.0

out = {}; store = {}
for v in ['raw', 'inv', 'ekf_fix', 'ekf_fix_fs', 'ekf_est', 'ekf_est_fs', 'ekf_oracle']:
    b = best[v]; ests = []; rates = []; fcs = []
    for p in test:
        q = dict(p); g, r, f = run_variant(q, v, b.get('qr'), b.get('W', 5)); ests.append(g); rates.append(r); fcs.append(f)
        if 'th' in q: p.setdefault('th', {}).update(q['th'])
    store[v] = ests
    err = np.concatenate([(g[WARM:] - p['G'][WARM:]) for g, p in zip(ests, test)])
    Gt = np.concatenate([p['G'][WARM:] for p in test])
    ferr = np.concatenate([(f[WARM:-H] - p['G'][WARM + H:]) for f, p in zip(fcs, test)])
    fG = np.concatenate([p['G'][WARM + H:] for p in test]); hy = Gt < 70
    ne = nd = fa = 0; dl = []; days = 0
    for g, p in zip(ests, test):
        a, b_, c, d, e = hypo_stats(g, p['G']); ne += a; nd += b_; dl += c; fa += d; days += e
    out[v] = dict(rmse=float(np.sqrt((err ** 2).mean())), mard=float(np.mean(np.abs(err[Gt >= 40]) / Gt[Gt >= 40]) * 100),
                  rmse_hypo=float(np.sqrt((err[hy] ** 2).mean())) if hy.any() else None,
                  fc30_rmse=float(np.sqrt((ferr ** 2).mean())), fc30_rmse_hypo=float(np.sqrt((ferr[fG < 70] ** 2).mean())) if (fG < 70).any() else None,
                  hypo_events=ne, detected=nd, mean_delay_min=float(np.mean(dl)) if dl else None,
                  median_delay_min=float(np.median(dl)) if dl else None, false_alarms_per_day=fa / days)
    print(v, json.dumps(out[v]), flush=True)
# tau recovery
tr = {}
for v in ['ekf_est', 'ekf_est_fs']:
    t_hat = np.array([p['th'][v][-480:].mean() for p in test]); t_true = np.array([p['tau'] for p in test])
    tr[v] = dict(median_abs_pct_err=float(np.median(np.abs(t_hat - t_true) / t_true * 100)), corr=float(np.corrcoef(t_hat, t_true)[0, 1]),
                 t_hat=t_hat.tolist(), t_true=t_true.tolist(), prior_only_median_pct_err=float(np.median(np.abs(10 - t_true) / t_true * 100)))
    print(v, 'tau', {k: v2 for k, v2 in tr[v].items() if 'hat' not in k and 'true' not in k})
json.dump(dict(noise_scale=NS, excluded=excl, sig=SIG, phi=PHI, metrics=out, tau=tr, best=best, R_sd=R ** .5, n_test=len(test), test_names=[p['name'] for p in test]), open('../data/results%s.json' % SUF, 'w'), indent=1)
pickle.dump(dict(store=store, test=test), open('../data/preds%s.pkl' % SUF, 'wb'))
