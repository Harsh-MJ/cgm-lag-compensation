"""Piece 6 experiments. python3 -I run_inputs.py <sim.npz> <outdir>
(1) per-patient tau likelihood with the meal/insulin-driven prior vs the Piece 5 random-walk prior (same data, same seeds), plus a placebo
    with the inputs shifted by 400 samples; (2) downstream real-time filter and RTS smoother with inputs, days 4-5 only."""
import sys, json, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S, kalman as K, ident as I, inputs as X
simpath, out = sys.argv[1], sys.argv[2]; pat, use, excl = H.load_patients(simpath); folds = H.make_folds(use); D = np.load(simpath); t0 = time.time()
NWIN = 3 * 288; P5 = np.load(out + '/ident_profiles.npz')
def log(*a): print('[%4.0fs]' % (time.time() - t0), *a, flush=True)
CONDS = [('B', 'S0_matched', 2.0), ('C', 'S0_matched', 8.0), ('D', 'S1_sensor_chain', 2.0), ('E', 'S1_sensor_chain', 8.0)]

def attach_inputs(data):
    for n in use:
        u1, u2 = X.drive(D[n + '|cho'], D[n + '|ins']); data[n]['U'] = X.sample_drive(u1, u2, data[n]['t'], 5)

def summarize(prof, teff):
    est = I.TAU_GRID[prof.argmax(1)]; iv = np.array([I.lr_interval(p)[:2] for p in prof]); err = est - teff
    return dict(median_ape=float(np.median(np.abs(err) / teff * 100)), median_signed_err=float(np.median(err)), ci_ape=H.ci_median(np.abs(err) / teff * 100)[1:],
                corr=float(np.corrcoef(est, teff)[0, 1]) if est.std() > 0 else None, coverage_eff=float(np.mean((iv[:, 0] <= teff) & (teff <= iv[:, 1]))),
                median_hi_lo=float(np.median(iv[:, 1] / iv[:, 0])), at_floor=float(np.mean(est <= I.TAU_GRID[0] + 1e-9)), est=est.tolist())

RES = {}; PROF = {}; DATA = {}
for tag, sc, ns in CONDS:
    data = H.build(pat, use, sc, ns); attach_inputs(data)
    Y = np.array([data[n]['y'][:NWIN] for n in use]); U = np.array([data[n]['U'][:NWIN] for n in use]); teff = np.array([data[n]['tau_eff_nominal'] for n in use])
    px = X.profile(Y, U, 5.0); pp = X.profile(Y, np.roll(U, 400, axis=0), 5.0); pz = X.profile(Y, np.zeros_like(U), 5.0, tune_gain=False)
    p5 = P5[tag + '_cgm']; PROF.update({tag + '_x': px, tag + '_placebo': pp, tag + '_zero': pz})
    RES[tag] = dict(scenario=sc, noise=ns, rw_prior=summarize(p5, teff), inputs=summarize(px, teff), placebo=summarize(pp, teff), zero_inputs_5state=summarize(pz, teff),
                    tau_eff=teff.tolist(), tau_phys=[data[n]['tau'] for n in use])
    DATA[tag] = data
    log(tag, sc, ns, ' '.join('%s ape %.0f%% bias %+.1f cov %.2f floor %.2f' % (k, RES[tag][k]['median_ape'], RES[tag][k]['median_signed_err'], RES[tag][k]['coverage_eff'], RES[tag][k]['at_floor']) for k in ('rw_prior', 'inputs', 'placebo', 'zero_inputs_5state')))
np.savez_compressed(out + '/inputs_profiles.npz', taus=I.TAU_GRID, **PROF)

# ---- downstream, days 4-5 only. tau variants: fixed 10; per-subject ML from the inputs likelihood; true effective tau (oracle)
def xkey(key, smooth):
    E = dict(X.ESTIMATORS['X1s_key' if smooth else 'X1f_key']); E['tau_key'] = key; return E
EST = {k: H.ESTIMATORS[k] for k in ('B0_raw', 'B1_shift')}
EST.update({'M1f_fixed': K.ESTIMATORS['M1f_fixed'], 'M1s_fixed': K.ESTIMATORS['M1s_fixed'], 'M1s_taupop': K.ESTIMATORS['M1s_taupop'],
            'X1f_fixed': X.ESTIMATORS['X1f_fixed'], 'X1s_fixed': X.ESTIMATORS['X1s_fixed'], 'X1f_ML': xkey('TU_ml', False), 'X1s_ML': xkey('TU_ml', True),
            'X1f_eff': xkey('TU_eff', False), 'X1s_eff': xkey('TU_eff', True)})
PAIRS = [('X1f_fixed', 'M1f_fixed'), ('X1s_fixed', 'M1s_fixed'), ('X1s_ML', 'X1s_fixed'), ('X1s_ML', 'M1s_fixed'), ('X1s_eff', 'X1s_fixed'), ('X1s_fixed', 'B1_shift'), ('X1f_ML', 'X1f_fixed')]
DOWN = {}
for tag, sc, ns in CONDS:
    data = DATA[tag]; ml = np.array(RES[tag]['inputs']['est'])
    for i, n in enumerate(use):
        data[n]['TU'] = (10.0, data[n]['U']); data[n]['TU_ml'] = (float(ml[i]), data[n]['U']); data[n]['TU_eff'] = (data[n]['tau_eff_nominal'], data[n]['U']); data[n]['warm'] = NWIN
    res, chosen, edge = H.cv_run(data, use, folds, EST); f = lambda k, key: [res[k][n][key] for n in use]; cell = {}
    for k in EST:
        cell[k] = dict(rmse=H.ci_median(f(k, 'rmse')), rmse_rapid=H.ci_median(f(k, 'rmse_rapid')), fa_per_day=H.ci_ratio(f(k, 'fa'), f(k, 'days')), hypo_det=(int(sum(f(k, 'det'))), int(sum(f(k, 'ev')))), edge=edge[k])
    cell['pairs_rapid'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse_rapid'), f(b, 'rmse_rapid')) for a, b in PAIRS}
    cell['pairs_all'] = {'%s - %s' % (a, b): H.ci_paired(f(a, 'rmse'), f(b, 'rmse')) for a, b in PAIRS}
    DOWN[tag] = cell; log('downstream', tag, ' '.join('%s %.1f' % (k, cell[k]['rmse_rapid'][0]) for k in EST))
RES['downstream'] = DOWN; json.dump(RES, open(out + '/inputs_results.json', 'w'), indent=1); log('done')
