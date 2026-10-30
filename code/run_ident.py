"""Piece 5 experiments. python3 -I run_ident.py <sim.npz> <outdir>"""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S, kalman as K, ident as I
pat, use, excl = H.load_patients(sys.argv[1]); out = sys.argv[2]; folds = H.make_folds(use); t0 = time.time()
NWIN = 3 * 288          # days 1-3 used to estimate tau (5-min samples); days 4-5 used to evaluate
fold_of = {n: f for f, fl in enumerate(folds) for n in fl}
def log(*a): print('[%4.0fs]' % (time.time() - t0), *a, flush=True)

CONDS = [('A', 'S0_matched', 2.0, True), ('B', 'S0_matched', 2.0, False), ('C', 'S0_matched', 8.0, False), ('D', 'S1_sensor_chain', 2.0, False),
         ('E', 'S1_sensor_chain', 8.0, False), ('F', 'S5_full', 8.0, False), ('G', 'S5_full', 2.0, False)]
RES = {}; PROF = {}; DATA = {}
for tag, sc, ns, white in CONDS:
    S.NOISE_SLOW_SHARE = 0.0 if white else 0.6
    data = H.build(pat, use, sc, ns); S.NOISE_SLOW_SHARE = 0.6
    Y = np.array([data[n]['y'][:NWIN] for n in use]); FS = []; RF = []
    for i, n in enumerate(use):
        fs, rf = I.make_fingersticks(data[n]['G'][:NWIN], data[n]['t'][:NWIN], np.random.default_rng(2000 + i)); FS.append(fs); RF.append(rf)
    pc = I.profile(Y, 5.0)[:, :, 0]; pf = I.profile(Y, 5.0, FSs=np.array(FS), RFSs=np.array(RF))[:, :, 0]
    tphys = np.array([data[n]['tau'] for n in use]); teff = np.array([data[n]['tau_eff_nominal'] for n in use])
    est = {'prior_10': np.full(len(use), 10.0), 'ML_cgm': I.TAU_GRID[pc.argmax(1)], 'ML_fs': I.TAU_GRID[pf.argmax(1)],
           'FSreg': np.array([I.fs_regression(data[n]['y'][:NWIN], FS[i], 5.0) for i, n in enumerate(use)])}
    for key, prof in (('EB_cgm', pc), ('EB_fs', pf)):                       # cross-fitted: prior from the other folds
        e = np.empty(len(use)); pops = []
        for f, test in enumerate(folds):
            tr = [i for i, n in enumerate(use) if n not in test]; mu, s_ = I.eb_fit(prof[tr]); pops.append((float(np.exp(mu)), float(s_)))
            for i, n in enumerate(use):
                if n in test: e[i] = I.eb_post(prof[i], mu, s_)
        est[key] = e; est[key + '_popmean'] = np.array([pops[fold_of[n]][0] for n in use]); RES.setdefault(tag, {})[key + '_pop'] = pops
    cell = RES.setdefault(tag, {}); cell.update(scenario=sc, noise=ns, white=white)
    for truth_name, truth in (('eff', teff), ('phys', tphys)):
        for k, v in est.items():
            ok = ~np.isnan(v); ape = np.abs(v[ok] - truth[ok]) / truth[ok] * 100
            cell['%s|%s' % (k, truth_name)] = dict(median_ape=float(np.median(ape)), ci=H.ci_median(ape)[1:], corr=float(np.corrcoef(v[ok], truth[ok])[0, 1]) if v[ok].std() > 0 else None)
    for k, prof in (('cgm', pc), ('fs', pf)):
        iv = np.array([I.lr_interval(p)[:2] for p in prof]); cell['interval_' + k] = dict(
            coverage_eff=float(np.mean((iv[:, 0] <= teff) & (teff <= iv[:, 1]))), coverage_phys=float(np.mean((iv[:, 0] <= tphys) & (tphys <= iv[:, 1]))),
            median_hi_lo=float(np.median(iv[:, 1] / iv[:, 0])), flat_frac=float(np.mean((iv[:, 0] <= I.TAU_GRID[0]) & (iv[:, 1] >= I.TAU_GRID[-1]))))
    cell['estimates'] = {k: v.tolist() for k, v in est.items()}; cell['tau_phys'] = tphys.tolist(); cell['tau_eff'] = teff.tolist(); cell['n_fs'] = [int((~np.isnan(f)).sum()) for f in FS]
    PROF[tag + '_cgm'] = pc; PROF[tag + '_fs'] = pf; DATA[tag] = (data, est)
    log(tag, sc, ns, 'white' if white else '', ' '.join('%s %.0f%%' % (k, cell['%s|eff' % k]['median_ape']) for k in ('prior_10', 'ML_cgm', 'ML_fs', 'EB_cgm', 'EB_fs', 'FSreg')),
        '| flat cgm %.2f fs %.2f' % (cell['interval_cgm']['flat_frac'], cell['interval_fs']['flat_frac']))
np.savez_compressed(out + '/ident_profiles.npz', taus=I.TAU_GRID, **PROF)

# ---- physiology vs sensor filter: 2-D profile (tau, tau_f) for condition D (S1, noise 2) with fingersticks
data = DATA['D'][0]; Y = np.array([data[n]['y'][:NWIN] for n in use]); FS = []; RF = []
for i, n in enumerate(use):
    fs, rf = I.make_fingersticks(data[n]['G'][:NWIN], data[n]['t'][:NWIN], np.random.default_rng(2000 + i)); FS.append(fs); RF.append(rf)
TF = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0); p2 = I.profile(Y, 5.0, tau_fs=TF, FSs=np.array(FS), RFSs=np.array(RF))
best_t = I.TAU_GRID[p2.argmax(1)]                       # (S, len(TF)): best tau for each tau_f
ridge = p2.max(1)                                       # (S, len(TF)): best log-lik for each tau_f
rng_ll = ridge.max(1) - ridge.min(1); sums = best_t + np.array(TF)[None]
RES['ridge_D'] = dict(tau_f=list(TF), best_tau_median=np.median(best_t, 0).tolist(), sum_sd_median=float(np.median(sums.std(1))), tau_sd_median=float(np.median(best_t.std(1))),
                      frac_flat_ridge=float(np.mean(2 * rng_ll < 3.84)), median_ridge_range_2dll=float(np.median(2 * rng_ll)),
                      best_tau_f=np.array(TF)[ridge.argmax(1)].tolist(), tau_phys=[data[n]['tau'] for n in use])
np.save(out + '/ident_profile2d_D.npy', p2); log('ridge D', json.dumps({k: v for k, v in RES['ridge_D'].items() if k not in ('best_tau_f', 'tau_phys')}))

# ---- downstream: per-subject tau estimates in the Kalman filter / smoother, evaluated on days 4-5 only
def kf_key(key, smooth):
    E = dict(K._kf_estimator(True, smooth)); E['tau_key'] = key; return E
EST = {k: H.ESTIMATORS[k] for k in ('B0_raw', 'B1_shift')}
EST.update({'M1f_fixed': K.ESTIMATORS['M1f_fixed'], 'M1s_fixed': K.ESTIMATORS['M1s_fixed'], 'M1s_taupop': K.ESTIMATORS['M1s_taupop'],
            'M1f_EB': kf_key('tau_eb', False), 'M1s_EB': kf_key('tau_eb', True), 'M1s_ML': kf_key('tau_ml', True),
            'M1f_oracle_eff': kf_key('tau_eff', False), 'M1s_oracle_eff': kf_key('tau_eff', True)})
PAIRS = [('M1s_EB', 'M1s_fixed'), ('M1s_EB', 'M1s_ML'), ('M1s_EB', 'M1s_oracle_eff'), ('M1s_EB', 'M1s_taupop'), ('M1s_EB', 'B1_shift'), ('M1f_EB', 'M1f_fixed'), ('M1f_oracle_eff', 'M1f_fixed')]
DOWN = {}
for tag in ('B', 'C', 'D', 'E', 'G', 'F'):
    data, est = DATA[tag]
    for i, n in enumerate(use):
        data[n]['tau_eb'] = float(est['EB_fs'][i]); data[n]['tau_ml'] = float(est['ML_fs'][i]); data[n]['tau_eff'] = data[n]['tau_eff_nominal']; data[n]['warm'] = NWIN
    res, chosen, edge = H.cv_run(data, use, folds, EST); f = lambda k, key: [res[k][n][key] for n in use]; cell = {}
    for k in EST:
        cell[k] = dict(rmse=H.ci_median(f(k, 'rmse')), rmse_rapid=H.ci_median(f(k, 'rmse_rapid')), fa_per_day=H.ci_ratio(f(k, 'fa'), f(k, 'days')),
                       hypo_det=(int(sum(f(k, 'det'))), int(sum(f(k, 'ev')))), edge=edge[k])
    cell['pairs_rapid'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse_rapid'), f(b, 'rmse_rapid')) for a, b in PAIRS}
    cell['pairs_all'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse'), f(b, 'rmse')) for a, b in PAIRS}
    DOWN[tag] = cell; log('downstream', tag, ' '.join('%s %.1f' % (k, cell[k]['rmse_rapid'][0]) for k in EST))
RES['downstream'] = DOWN
json.dump(RES, open(out + '/ident_results.json', 'w'), indent=1); log('done')
