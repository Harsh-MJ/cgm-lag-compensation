"""Piece 8: posterior interval calibration with LIKELIHOOD-chosen noise settings (no reference glucose needed).
python3 -I run_coverage.py <sim.npz> <outdir>. Per subject, (qe, sigma) maximise the innovation log-likelihood on days 1-3 (tau = 10 fixed); the smoother's
95% interval for blood glucose is then scored on days 4-5 against true plasma glucose."""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, inputs as X
simpath, out = sys.argv[1], sys.argv[2]; pat, use, excl = H.load_patients(simpath); D = np.load(simpath); NWIN = 3 * 288; t0 = time.time()
CONDS = [('B', 'S0_matched', 2.0), ('C', 'S0_matched', 8.0), ('D', 'S1_sensor_chain', 2.0), ('E', 'S1_sensor_chain', 8.0)]
AR = (30.0, 0.6); VAR = {'M': ('zero', False, None), 'X': ('U', True, None), 'X_ar': ('U', True, AR), 'M_ar': ('zero', False, AR)}
QES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]; SIGS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0]; combos = [(q, s) for q in QES for s in SIGS]; C = len(combos)
RES = {}
for tag, sc, ns in CONDS:
    data = H.build(pat, use, sc, ns); S = len(use)
    for n in use:
        u1, u2 = X.drive(D[n + '|cho'], D[n + '|ins']); data[n]['U'] = X.sample_drive(u1, u2, data[n]['t'], 5)
    Y = np.array([data[n]['y'] for n in use]); U = np.array([data[n]['U'] for n in use]); Z = np.zeros_like(U); cell = {}
    for v, (uk, tg, ar) in VAR.items():
        UU = U if uk == 'U' else Z
        ll = X.kfx(np.repeat(Y[:, :NWIN], C, 0), np.repeat(UU[:, :NWIN], C, 0), 10.0, 5.0, np.tile([q for q, s in combos], S), np.tile([s for q, s in combos], S), ll=True, tune_gain=tg, ar=ar)['ll'].reshape(S, C)
        best = ll.argmax(1); qe = np.array([combos[b][0] for b in best]); sg = np.array([combos[b][1] for b in best])
        r = X.kfx(Y, UU, 10.0, 5.0, qe, sg, smooth=True, tune_gain=tg, ar=ar); rf = X.kfx(Y, UU, 10.0, 5.0, qe, sg, smooth=False, tune_gain=tg, ar=ar); d = {}
        for nm, est, sd in (('smoother', r['smooth'], r['smooth_sd']), ('filter', rf['filt'], rf['filt_sd'])):
            ca, cr, wd, rm = [], [], [], []
            for i, n in enumerate(use):
                G = data[n]['G'][NWIN:]; hit = np.abs(est[i, NWIN:] - G) <= 1.96 * sd[i, NWIN:]; rap = data[n]['rapid'][NWIN:]
                ca.append(hit.mean()); cr.append(hit[rap].mean() if rap.sum() > 20 else np.nan); wd.append(2 * 1.96 * sd[i, NWIN:].mean()); rm.append(np.sqrt(np.mean((est[i, NWIN:] - G) ** 2)))
            d[nm] = dict(cov_all=H.ci_median(ca), cov_rapid=H.ci_median(cr), width=H.ci_median(wd), rmse=H.ci_median(rm))
        d['sig_median'] = float(np.median(sg)); d['qe_median'] = float(np.median(qe)); d['ll_total'] = float(ll.max(1).sum()); cell[v] = d
    RES[tag] = dict(scenario=sc, noise=ns, **cell)
    print('[%4.0fs]' % (time.time() - t0), tag, sc, ns, ' '.join('%s sm cov %.2f (rapid %.2f) w %.0f sig %.1f' % (v, cell[v]['smoother']['cov_all'][0], cell[v]['smoother']['cov_rapid'][0], cell[v]['smoother']['width'][0], cell[v]['sig_median']) for v in VAR), flush=True)
json.dump(RES, open(out + '/coverage_results.json', 'w'), indent=1)
