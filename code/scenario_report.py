"""Scenario report: raw-CGM error and measured lag per scenario. Run: python3 -I scenario_report.py <sim.npz> <outdir>"""
import sys, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensor as S
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = np.load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
names = sorted({k.split('|')[0] for k in D.files}, key=lambda s: (s.split('#')[0], s))
P = {n: (D[n + '|plasma'], float(D[n + '|tau'])) for n in names}
excl = [n for n, (g, _) in P.items() if g.min() < 30 or (g < 54).mean() > 0.15]
use = [n for n in names if n not in excl]; WARM = 360   # 6 h

def lag_min(y, G, smax=45):
    best = None
    for s in range(0, smax + 1):
        e = np.sqrt(np.mean((y[s:] - G[:len(G) - s]) ** 2))
        if best is None or e < best[0]: best = (e, s)
    return best[1]

res = {}
for sc in S.SCENARIOS:
    rows = []
    for i, n in enumerate(use):
        g, tau = P[n]
        r = S.make_sensor(g, tau, sc, noise_sd=5.0, dt=5, seed=1000 + i, miss_frac=0.05)
        m = (r['t'] >= WARM) & ~r['missing']; y, G = r['y'][m], r['G'][m]; e = y - G
        hi = G >= 40
        r1 = S.make_sensor(g, tau, sc, noise_sd=0.0, dt=1, seed=1000 + i)
        rows.append(dict(rmse=float(np.sqrt((e ** 2).mean())), mard=float(np.mean(np.abs(e[hi]) / G[hi]) * 100), bias=float(e.mean()),
                         lag=lag_min(r1['y'][WARM:], r1['G'][WARM:]), nominal=r['tau_eff_nominal'], tau=tau,
                         compress=float(r['compress'][m].mean() * 100)))
    A = lambda k: np.array([x[k] for x in rows])
    res[sc] = dict(n=len(rows), rmse_med=float(np.median(A('rmse'))), mard_med=float(np.median(A('mard'))), bias_med=float(np.median(A('bias'))),
                   lag_med=float(np.median(A('lag'))), tau_mean=float(A('tau').mean()), nominal_med=float(np.median(A('nominal'))),
                   lag_minus_nominal_med=float(np.median(A('lag') - A('nominal'))), compress_pct=float(A('compress').mean()))
    print('%-18s RMSE %5.1f  MARD %5.1f%%  bias %5.1f  measured lag %4.1f min (nominal tau+tau_f+d %4.1f)  compression %.2f%% of samples' %
          (sc, res[sc]['rmse_med'], res[sc]['mard_med'], res[sc]['bias_med'], res[sc]['lag_med'], res[sc]['nominal_med'], res[sc]['compress_pct']))
json.dump(dict(excluded=excl, n=len(use), noise_sd=5.0, dt=5, miss_frac=0.05, results=res), open(out + '/scenario_report.json', 'w'), indent=1)

# figure: one patient, day 3 (06:00 to 30:00 hours), all scenarios
n = 'adult#002'; g, tau = P[n]; fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey=True)
for a, sc in zip(ax.ravel(), S.SCENARIOS):
    r = S.make_sensor(g, tau, sc, noise_sd=5.0, dt=5, seed=1000 + use.index(n), miss_frac=0.05)
    t0, t1 = 2 * 1440, 3 * 1440; m = (r['t'] >= t0) & (r['t'] < t1); th = (r['t'][m] - t0) / 60
    a.plot(th, r['G'][m], color='#E4572E', lw=1.6, label='plasma (truth)'); a.plot(th, r['y'][m], color='#2D6A6F', lw=1.2, label='CGM-like signal')
    if r['compress'][m].any(): a.fill_between(th, 0, 400, where=r['compress'][m], color='#F2A541', alpha=.3, label='compression')
    a.set_title(sc, fontsize=10); a.set_ylim(40, 320); a.grid(alpha=.25)
ax[0, 0].legend(fontsize=8, loc='upper left'); [a.set_xlabel('hour of day 3') for a in ax[-1]]; [a.set_ylabel('mg/dL') for a in ax[:, 0]]
fig.suptitle('Piece 2 scenarios: %s, noise sd 5 mg/dL, 5-min sampling, 5%% dropouts (gaps not drawn)' % n, fontsize=11)
fig.tight_layout(); fig.savefig(out + '/fig_scenarios.png', dpi=130)
