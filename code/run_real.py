"""Piece 7: exploratory real-data check. python3 -I run_real.py <dir with OhioT1DM *.xml> <outdir>
Needs the OhioT1DM files (data-use agreement; not in this repository). Fingersticks are the only reference. Leave-one-patient-out tuning."""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real as R, harness as H
files = sorted(glob.glob(sys.argv[1] + '/*.xml')); assert files, 'no .xml files found'
P = [R.load_xml(f) for f in files]; G = [R.to_grid(p) for p in P]; out = {}
for name, ui in (('with_inputs', True), ('random_walk', False)):
    res = R.loo_check(G, use_inputs=ui); out[name] = res
    for k in ('rmse_raw', 'rmse_tau10', 'rmse_tuned', 'mard_raw', 'mard_tuned'): print(name, k, 'median %.1f' % np.median([r[k] for r in res]), 'CI', [round(x, 1) for x in H.ci_median([r[k] for r in res])[1:]])
    print(name, 'tuned tau per left-out patient', [r['tau'] for r in res])
json.dump(out, open(sys.argv[2] + '/real_results.json', 'w'), indent=1)
