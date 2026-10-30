"""Run B0-B3 over scenarios and noise levels. python3 -I run_baselines.py <sim.npz> <outdir>"""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S
pat, use, excl = H.load_patients(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
folds = H.make_folds(use); json.dump(dict(excluded=excl, folds=folds), open(out + '/split_manifest.json', 'w'), indent=1)
NOISE = (2.0, 5.0, 8.0); R = {}; t0 = time.time()
for sc in S.SCENARIOS:
    for ns in NOISE:
        data = H.build(pat, use, sc, ns); res, chosen, edge = H.cv_run(data, use, folds)
        cell = {}
        for k, per in res.items():
            f = lambda key: [per[n][key] for n in use]
            cell[k] = dict(rmse=H.ci_median(f('rmse')), rmse_rapid=H.ci_median(f('rmse_rapid')), mard=H.ci_median(f('mard')), rmse_roc=H.ci_median(f('rmse_roc')),
                           lag=H.ci_median(f('lag')),
                           hypo_delay=H.ci_ratio(f('dsum'), f('det')), hypo_det=(int(sum(f('det'))), int(sum(f('ev')))),
                           fa_per_day=H.ci_ratio(f('fa'), f('days')),
                           vs_raw_rapid=H.ci_paired(f('rmse_rapid'), [per[n]['rmse_rapid'] for n in use]) if k == 'B0_raw' else None,
                           chosen=chosen[k], edge_picks=edge[k], per_subject=dict(rmse=f('rmse'), rmse_rapid=f('rmse_rapid'), rapid_frac=f('rapid_frac')))
        # paired comparisons against raw on rapid-change RMSE, Holm over the four comparisons
        names = [k for k in res if k != 'B0_raw']
        pc = {k: H.ci_paired([res[k][n]['rmse_rapid'] for n in use], [res['B0_raw'][n]['rmse_rapid'] for n in use]) for k in names}
        adj = H.holm([pc[k][3] for k in names])
        for k, a in zip(names, adj): cell[k]['vs_raw_rapid'] = dict(median_diff=pc[k][0], lo=pc[k][1], hi=pc[k][2], p=pc[k][3], p_holm=a)
        R['%s|%g' % (sc, ns)] = cell
        print('%-18s noise %-3g  RMSE rapid: %s   (%.0fs)' % (sc, ns, '  '.join('%s %.1f' % (k[:6], cell[k]['rmse_rapid'][0]) for k in cell), time.time() - t0), flush=True)
json.dump(R, open(out + '/baseline_results.json', 'w'), indent=1)
