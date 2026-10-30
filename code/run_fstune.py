"""Piece 5b: choose the lag with fingersticks instead of ground truth. python3 -I run_fstune.py <sim.npz> <outdir>"""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S, kalman as K, ident as I
pat, use, excl = H.load_patients(sys.argv[1]); out = sys.argv[2]; folds = H.make_folds(use); t0 = time.time(); NWIN = 3 * 288
def log(*a): print('[%4.0fs]' % (time.time() - t0), *a, flush=True)
def kf_key(key, smooth):
    E = dict(K._kf_estimator(True, smooth)); E['tau_key'] = key; return E
B1fs = dict(H.ESTIMATORS['B1_shift'], tune_on='fs')
EST = {'B0_raw': H.ESTIMATORS['B0_raw'], 'B1_shift_truth': H.ESTIMATORS['B1_shift'], 'B1_shift_fs': B1fs,
       'M1f_fixed': K.ESTIMATORS['M1f_fixed'], 'M1f_tau_truth': K.ESTIMATORS['M1f_taupop'], 'M1f_tau_fs': dict(K.ESTIMATORS['M1f_taupop'], tune_on='fs'),
       'M1s_fixed': K.ESTIMATORS['M1s_fixed'], 'M1s_tau_truth': K.ESTIMATORS['M1s_taupop'], 'M1s_tau_fs': dict(K.ESTIMATORS['M1s_taupop'], tune_on='fs'),
       'M1s_FSreg': kf_key('tau_fsreg', True), 'M1f_oracle_eff': kf_key('tau_eff', False), 'M1s_oracle_eff': kf_key('tau_eff', True)}
PAIRS = [('M1s_tau_fs', 'M1s_fixed'), ('M1s_tau_fs', 'M1s_tau_truth'), ('M1s_tau_fs', 'M1s_oracle_eff'), ('M1s_tau_fs', 'B1_shift_fs'),
         ('M1s_FSreg', 'M1s_fixed'), ('M1f_tau_fs', 'M1f_fixed'), ('M1s_tau_fs', 'M1s_FSreg')]
R = {}
for tag, sc, ns in [('B', 'S0_matched', 2.0), ('C', 'S0_matched', 8.0), ('D', 'S1_sensor_chain', 2.0), ('E', 'S1_sensor_chain', 8.0), ('G', 'S5_full', 2.0), ('F', 'S5_full', 8.0)]:
    data = H.build(pat, use, sc, ns)
    for i, n in enumerate(use):
        r = data[n]; fs, _ = I.make_fingersticks(r['G'][:NWIN], r['t'][:NWIN], np.random.default_rng(2000 + i))
        r['fs'] = np.r_[fs, np.full(len(r['y']) - NWIN, np.nan)]                     # fingersticks only in days 1-3 (calibration period)
        r['tau_fsreg'] = float(np.clip(I.fs_regression(r['y'][:NWIN], fs, 5.0), 4.0, 28.0)); r['tau_eff'] = r['tau_eff_nominal']; r['warm'] = NWIN
    res, chosen, edge = H.cv_run(data, use, folds, EST); f = lambda k, key: [res[k][n][key] for n in use]; cell = dict(scenario=sc, noise=ns)
    for k in EST:
        cell[k] = dict(rmse=H.ci_median(f(k, 'rmse')), rmse_rapid=H.ci_median(f(k, 'rmse_rapid')), fa_per_day=H.ci_ratio(f(k, 'fa'), f(k, 'days')),
                       hypo_det=(int(sum(f(k, 'det'))), int(sum(f(k, 'ev')))), edge=edge[k], chosen=[list(h.values()) for h in chosen[k]])
    cell['pairs_rapid'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse_rapid'), f(b, 'rmse_rapid')) for a, b in PAIRS}
    cell['pairs_all'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse'), f(b, 'rmse')) for a, b in PAIRS}
    cell['n_fs_tune_median'] = int(np.median([sum((~np.isnan(data[n]['fs'])).sum() for n in use if n not in t) for t in folds]))
    R[tag] = cell; log(tag, sc, ns, ' '.join('%s %.1f' % (k, cell[k]['rmse_rapid'][0]) for k in EST))
json.dump(R, open(out + '/fstune_results.json', 'w'), indent=1); log('done')
