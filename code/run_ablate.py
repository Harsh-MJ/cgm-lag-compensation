"""Piece 8 experiments. python3 -I run_ablate.py <sim.npz> <outdir>
Ablations of the input-driven Kalman model (inputs, gain learning, kernel shape, AR(1) noise state) and posterior-interval coverage,
days 4-5 only, hyper-parameters tuned on training folds as before."""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, kalman as K, inputs as X
simpath, out = sys.argv[1], sys.argv[2]; pat, use, excl = H.load_patients(simpath); folds = H.make_folds(use); D = np.load(simpath); t0 = time.time(); NWIN = 3 * 288
def log(*a): print('[%4.0fs]' % (time.time() - t0), *a, flush=True)
CONDS = [('B', 'S0_matched', 2.0), ('C', 'S0_matched', 8.0), ('D', 'S1_sensor_chain', 2.0), ('E', 'S1_sensor_chain', 8.0)]
AR = (30.0, 0.6)       # slow AR(1) noise: correlation time 30 min, 60% of the noise variance (the simulator's truth; an oracle structure)
# name suffix -> (U key, tune_gain, ar)
VARIANTS = {'M': ('TU_zero', False, None), 'M_ar': ('TU_zero', False, AR), 'X': ('TU', True, None), 'X_frozen': ('TU', False, None),
            'X_k05': ('TU_k05', True, None), 'X_k2': ('TU_k2', True, None), 'X_ar': ('TU', True, AR)}
SPEC = {}; EST = {k: H.ESTIMATORS[k] for k in ('B0_raw', 'B1_shift')}
for v, (key, tg, ar) in VARIANTS.items():
    for sm, pre in ((False, 'f'), (True, 's')):
        nm = pre + '_' + v; EST[nm] = X._x_estimator(sm, 'fixed', tune_gain=tg, ar=ar, tau_key=key); SPEC[nm] = (sm, key, tg, ar)
RES = {}
for tag, sc, ns in CONDS:
    data = H.build(pat, use, sc, ns)
    for n in use:
        u1, u2 = X.drive(D[n + '|cho'], D[n + '|ins']); k1, k2 = X.drive(D[n + '|cho'], D[n + '|ins'], X.TAU_MEAL * 0.5, X.TAU_INS * 0.5); j1, j2 = X.drive(D[n + '|cho'], D[n + '|ins'], X.TAU_MEAL * 2, X.TAU_INS * 2)
        sd = lambda a, b: X.sample_drive(a, b, data[n]['t'], 5)
        data[n]['TU'] = (10.0, sd(u1, u2)); data[n]['TU_k05'] = (10.0, sd(k1, k2)); data[n]['TU_k2'] = (10.0, sd(j1, j2)); data[n]['TU_zero'] = (10.0, np.zeros_like(sd(u1, u2))); data[n]['warm'] = NWIN
    res, chosen, edge = H.cv_run(data, use, folds, EST); f = lambda k, key: [res[k][n][key] for n in use]; cell = {'rmse': {}, 'rapid': {}, 'cov': {}}
    for k in EST: cell['rmse'][k] = H.ci_median(f(k, 'rmse')); cell['rapid'][k] = H.ci_median(f(k, 'rmse_rapid'))
    # posterior interval coverage with each fold's chosen hyper-parameters, on days 4-5 (warm = NWIN)
    for nm, (sm, key, tg, ar) in SPEC.items():
        cov_all = {}; cov_rap = {}; width = {}
        for fi, test in enumerate(folds):
            hp = chosen[nm][fi]; idx = [use.index(n) for n in test]; Y = np.array([data[n]['y'] for n in test]); U = np.array([data[n][key][1] for n in test])
            r = X.kfx(Y, U, 10.0, 5.0, hp['qe'], hp['sig'], smooth=sm, tune_gain=tg, ar=ar); est = r['smooth'] if sm else r['filt']; sdv = r['smooth_sd'] if sm else r['filt_sd']
            for j, n in enumerate(test):
                G = data[n]['G'][NWIN:]; hit = np.abs(est[j, NWIN:] - G) <= 1.96 * sdv[j, NWIN:]; rap = data[n]['rapid'][NWIN:]
                cov_all[n] = float(hit.mean()); cov_rap[n] = float(hit[rap].mean()) if rap.sum() > 20 else np.nan; width[n] = float(2 * 1.96 * sdv[j, NWIN:].mean())
        cell['cov'][nm] = dict(all=H.ci_median(list(cov_all.values())), rapid=H.ci_median(list(cov_rap.values())), width=H.ci_median(list(width.values())))
    PAIRS = [('s_X', 's_M'), ('f_X', 'f_M'), ('s_X', 's_X_frozen'), ('s_X', 's_X_k05'), ('s_X', 's_X_k2'), ('s_X_ar', 's_X'), ('f_X_ar', 'f_X'), ('s_M_ar', 's_M'), ('f_M_ar', 'f_M')]
    cell['pairs_rapid'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse_rapid'), f(b, 'rmse_rapid')) for a, b in PAIRS}
    RES[tag] = dict(scenario=sc, noise=ns, **cell); log(tag, sc, ns, ' '.join('%s %.1f' % (k, cell['rapid'][k][0]) for k in ('s_M', 's_X', 's_X_ar', 'f_M', 'f_X', 'f_X_ar')),
        '| cov s_X %.2f f_X %.2f' % (cell['cov']['s_X']['all'][0], cell['cov']['f_X']['all'][0]))
json.dump(RES, open(out + '/ablate_results.json', 'w'), indent=1); log('done')
