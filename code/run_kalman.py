"""python3 -I run_kalman.py <sim.npz> <outdir>   (B0, B1, B3 + Kalman family, scenarios x noise 2 and 8)"""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S, kalman as K
pat, use, excl = H.load_patients(sys.argv[1]); out = sys.argv[2]; folds = H.make_folds(use)
EST = {k: H.ESTIMATORS[k] for k in ('B0_raw', 'B1_shift', 'B3_inverse')}; EST.update(K.ESTIMATORS)
PAIRS = [('M1f_fixed', 'B3_inverse'), ('M1s_fixed', 'B1_shift'), ('M2_reginv', 'M1s_fixed'), ('M1f_oracle', 'M1f_fixed'),
         ('M1f_taupop', 'M1f_fixed'), ('M1f_taupop', 'B3_inverse'), ('M1s_taupop', 'B1_shift'), ('M1s_taupop', 'M1s_fixed')]
R = {}; t0 = time.time()
for sc in S.SCENARIOS:
    for ns in (2.0, 8.0):
        data = H.build(pat, use, sc, ns); res, chosen, edge = H.cv_run(data, use, folds, EST); cell = {}
        f = lambda k, key: [res[k][n][key] for n in use]
        for k in EST:
            cell[k] = dict(rmse=H.ci_median(f(k, 'rmse')), rmse_rapid=H.ci_median(f(k, 'rmse_rapid')), lag=H.ci_median(f(k, 'lag')), rmse_roc=H.ci_median(f(k, 'rmse_roc')),
                           hypo_delay=H.ci_ratio(f(k, 'dsum'), f(k, 'det')), hypo_det=(int(sum(f(k, 'det'))), int(sum(f(k, 'ev')))), fa_per_day=H.ci_ratio(f(k, 'fa'), f(k, 'days')),
                           chosen=[list(h.values()) for h in chosen[k]], edge_picks=edge[k])
        names = [k for k in EST if k != 'B0_raw']
        pc = {k: H.ci_paired(f(k, 'rmse_rapid'), f('B0_raw', 'rmse_rapid')) for k in names}; adj = H.holm([pc[k][3] for k in names])
        for k, a in zip(names, adj): cell[k]['vs_raw_rapid'] = dict(median_diff=pc[k][0], lo=pc[k][1], hi=pc[k][2], p_holm=a)
        cell['pairs'] = {'%s - %s' % (a, b): dict(zip(('median_diff', 'lo', 'hi', 'p'), H.ci_paired(f(a, 'rmse_rapid'), f(b, 'rmse_rapid')))) for a, b in PAIRS}
        cell['pairs_all'] = {'%s - %s' % (a, b): dict(zip(('median_diff', 'lo', 'hi', 'p'), H.ci_paired(f(a, 'rmse'), f(b, 'rmse')))) for a, b in PAIRS}
        R['%s|%g' % (sc, ns)] = cell
        print('%-17s n%-2g rapid: %s (%.0fs)' % (sc, ns, ' '.join('%s %.1f' % (k[:9], cell[k]['rmse_rapid'][0]) for k in EST), time.time() - t0), flush=True)
json.dump(R, open(out + '/kalman_results.json', 'w'), indent=1)
