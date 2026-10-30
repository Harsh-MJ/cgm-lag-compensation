"""Harness tests. Run: python3 -I test_harness.py <sim.npz>"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S
ok = []
def check(n, c, i=''): ok.append(bool(c)); print(('PASS ' if c else 'FAIL ') + n + ('  ' + i if i else ''))

# 1 hand-checked metrics: constant +10 error on a flat 100 mg/dL trace
n = 400; G = np.full(n, 100.0); r = dict(G=G, warm=0, rapid=np.zeros(n, bool))
m = H.subject_metrics(G + 10, r, 5)
check('metrics: constant +10 error', abs(m['rmse'] - 10) < 1e-9 and abs(m['mad'] - 10) < 1e-9 and abs(m['bias'] - 10) < 1e-9 and abs(m['mard'] - 10) < 1e-9, str({k: round(m[k], 3) for k in ('rmse', 'mad', 'bias', 'mard')}))

# 2 perfect estimator on a trace with a hypo: zero error, zero delay, no false alarms
t = np.arange(600); G = 110 + 60 * np.sin(t / 40.0) - 30; r = dict(G=G, warm=0, rapid=np.zeros(600, bool))
m = H.subject_metrics(G.copy(), r, 5)
check('perfect estimator', m['rmse'] == 0 and m['fa'] == 0 and m['det'] == m['ev'] > 0 and m['dsum'] == 0 and m['lag'] == 0, 'events %d detected %d' % (m['ev'], m['det']))

# 3 estimator that lags truth by 10 min (2 samples): residual lag 10, hypo delay about 10 min
Gl = np.concatenate([np.full(2, G[0]), G[:-2]]); m = H.subject_metrics(Gl, r, 5)
check('10-min lag recovered', m['lag'] == 10 and abs(m['dsum'] / max(m['det'], 1) - 10) <= 5, 'lag %d, mean delay %.1f' % (m['lag'], m['dsum'] / max(m['det'], 1)))

# 4 hypo false alarm counted when estimate dips and truth does not
Gf = np.full(600, 100.0); Gh = Gf.copy(); Gh[300:306] = 60; m = H.hypo_stats(Gh, Gf, 5)
check('false alarm counted', m['fa'] == 1 and m['ev'] == 0, str(m))

# 5 bootstrap helpers
v = np.arange(1, 28.0); pt, lo, hi = H.ci_median(v); check('bootstrap median CI brackets median', lo <= pt <= hi and hi - lo < 15, '%.1f [%.1f, %.1f]' % (pt, lo, hi))
d, lo, hi, p = H.ci_paired(v + 5, v); check('paired diff = 5, p small', d == 5 and lo == 5 and hi == 5 and p < 1e-4, 'p %.2e' % p)
check('holm monotone', H.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06])

# 6 folds: disjoint, complete, balanced by group
SIM = sys.argv[1]; pat, use, excl = H.load_patients(SIM); folds = H.make_folds(use)
flat = [n for f in folds for n in f]
check('folds disjoint and complete', sorted(flat) == sorted(use) and len(set(flat)) == len(flat), 'sizes %s' % [len(f) for f in folds])
check('each fold has adolescent+adult', all({n.split('#')[0] for n in f} >= {'adolescent', 'adult'} for f in folds))

# 7 causality: change future samples, causal estimators must not change the past
data = H.build(pat, use[:2], 'S5_full', 8.0); r = data[use[0]]; y = r['y']; k0 = 500; rng = np.random.default_rng(0)
y2 = y.copy(); y2[k0 + 1:] = rng.normal(150, 40, len(y) - k0 - 1)
for name, E in H.ESTIMATORS.items():
    hp = E['grid'][len(E['grid']) // 2]; a = E['fn'](y, 5, hp, dict(tau_true=r['tau'])); b = E['fn'](y2, 5, hp, dict(tau_true=r['tau']))
    same = np.allclose(a[:k0 + 1], b[:k0 + 1], equal_nan=True)
    check('causality %s (declared causal=%s)' % (name, E['causal']), same == E['causal'])

# 8 tuning never sees test subjects (cv_run asserts disjointness; check output covers each subject once)
data = H.build(pat, use, 'S0_matched', 2.0); res, ch, ed = H.cv_run(data, use, folds, {k: H.ESTIMATORS[k] for k in ('B0_raw', 'B3_inverse')})
check('every subject evaluated exactly once', all(sorted(res[k]) == sorted(use) for k in res))

# 9 sanity of estimators on noise-free matched data: B3 with true tau beats raw, B3 oracle RMSE small
data = H.build(pat, use, 'S0_matched', 0.0, miss=0.0); rr = {k: np.median([v['rmse'] for v in H.cv_run(data, use, folds, {k: H.ESTIMATORS[k]})[0][k].values()]) for k in ('B0_raw', 'B3_oracle_tau', 'B1_shift')}
check('noise-free: oracle inverse << raw, shift < raw', rr['B3_oracle_tau'] < 0.3 * rr['B0_raw'] and rr['B1_shift'] < rr['B0_raw'], str({k: round(v, 2) for k, v in rr.items()}))
print('\n%d/%d passed' % (sum(ok), len(ok))); sys.exit(0 if all(ok) else 1)
