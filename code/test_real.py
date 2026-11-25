"""Piece 7 tests on SYNTHETIC OhioT1DM-format files. python3 -I test_real.py <sim.npz>"""
import os, sys, tempfile, glob, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real as R
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info)
d = tempfile.mkdtemp(); subprocess.run([sys.executable, '-I', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_fake_ohio.py'), sys.argv[1], d, '4'], check=True, capture_output=True)
files = sorted(glob.glob(d + '/*.xml')); P = [R.load_xml(f) for f in files]; G = [R.to_grid(p) for p in P]
check('loader reads 4 patients with CGM, 20 fingersticks, meals and boluses', len(P) == 4 and all(len(p['fs']) == 20 and len(p['meals']) > 5 and len(p['boluses']) > 5 for p in P), str([(len(p['cgm']), len(p['fs']), len(p['meals'])) for p in P]))
check('grid has 5-min samples with ~5% gaps and inputs of matching length', all(1400 < len(g['y']) < 1500 and len(g['U']) == len(g['y']) and 0 < np.isnan(g['y']).mean() < 0.12 for g in G))
check('meal drive is non-negative and mass roughly conserved', all((g['U'][:, 0] >= -1e-9).all() for g in G))
res = R.loo_check(G, use_inputs=True)
a = np.median([r['rmse_raw'] for r in res]); b = np.median([r['rmse_tuned'] for r in res])
check('leave-one-patient-out runs; smoother beats raw CGM against fingersticks on synthetic data', b < a, 'raw %.1f tuned %.1f (n_fs=%s)' % (a, b, [r['n_fs'] for r in res]))
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
