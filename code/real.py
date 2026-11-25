"""Piece 7: OhioT1DM loader and exploratory real-data check with fingersticks as the reference.

OhioT1DM XML (Marling and Bunescu 2020): one file per patient/split, <patient id=..>, with <glucose_level> (CGM, 5 min), <finger_stick>,
<basal> (U/h step changes), <temp_basal>, <bolus> (ts_begin, ts_end, dose), <meal> (ts, carbs). Timestamps 'DD-MM-YYYY HH:MM:SS'.
The data are NOT in this repository (data-use agreement). Fingersticks are the only blood-glucose reference; they are noisy (~5-10%) and
on some sensors were also used for calibration, so every number from this check is exploratory and is not a validation of lag removal.
"""
import numpy as np, xml.etree.ElementTree as ET
from datetime import datetime
import ident as I, inputs as X, kalman as K

FMT = '%d-%m-%Y %H:%M:%S'
def _t(s): return datetime.strptime(s, FMT)

def load_xml(path):
    """Returns dict(id, t0, cgm=[(min, value)], fs=[(min, value)], meals=[(min, g)], boluses=[(min0, min1, U)], basal=[(min, U/h)], temp=[(min0, min1, U/h)])."""
    root = ET.parse(path).getroot(); ev = lambda tag: root.find(tag).findall('event') if root.find(tag) is not None else []
    cg = [(_t(e.get('ts')), float(e.get('value'))) for e in ev('glucose_level')]; t0 = min(t for t, _ in cg).replace(second=0)
    m = lambda s: (_t(s) - t0).total_seconds() / 60.0
    return dict(id=root.get('id'), t0=t0, cgm=[(m(e.get('ts')), float(e.get('value'))) for e in ev('glucose_level')], fs=[(m(e.get('ts')), float(e.get('value'))) for e in ev('finger_stick')],
                meals=[(m(e.get('ts')), float(e.get('carbs') or 0)) for e in ev('meal')],
                boluses=[(m(e.get('ts_begin')), max(m(e.get('ts_end')), m(e.get('ts_begin')) + 1), float(e.get('dose') or 0)) for e in ev('bolus')],
                basal=[(m(e.get('ts')), float(e.get('value'))) for e in ev('basal')], temp=[(m(e.get('ts_begin')), m(e.get('ts_end')), float(e.get('value'))) for e in ev('temp_basal')])

def to_grid(p, dt=5):
    """5-min grid: CGM (NaN if no reading within +-dt/2), fingersticks (NaN elsewhere), inputs U (n,2) from the Piece 6 drive."""
    tmax = max(t for t, _ in p['cgm']); n = int(tmax // dt) + 1; y = np.full(n, np.nan); fs = np.full(n, np.nan); best = np.full(n, np.inf)
    for t, v in p['cgm']:
        k = int(round(t / dt))
        if 0 <= k < n and abs(t - k * dt) < best[k]: y[k] = v; best[k] = abs(t - k * dt)
    for t, v in p['fs']:
        k = int(round(t / dt))
        if 0 <= k < n: fs[k] = v
    N = n * dt; cho = np.zeros(N); ins = np.zeros(N)
    for t, g in p['meals']:
        if 0 <= int(t) < N: cho[int(t)] += g
    bs = sorted(p['basal']); rate = np.zeros(N)
    for i, (t, v) in enumerate(bs): rate[max(int(t), 0):(int(bs[i + 1][0]) if i + 1 < len(bs) else N)] = v
    if bs: rate[:max(int(bs[0][0]), 0)] = bs[0][1]
    for a, b, v in p['temp']: rate[max(int(a), 0):max(int(b), 0)] = v
    ins += rate / 60.0
    for a, b, d in p['boluses']:
        a, b = max(int(a), 0), min(int(b), N)
        if b > a: ins[a:b] += d / (b - a)
    u1, u2 = X.drive(cho, ins); U = X.sample_drive(u1, u2, np.arange(n) * float(dt), dt)
    return dict(id=p['id'], y=y, fs=fs, U=U, dt=dt)

def estimates(g, tau, qe, sig, use_inputs=True, smooth=True):
    """Blood-glucose estimate on the grid: input-driven (X) or random-walk (Piece 4 model, via zero inputs and frozen gains)."""
    U = g['U'][None] if use_inputs else np.zeros_like(g['U'])[None]
    r = X.kfx(g['y'][None], U, tau, g['dt'], qe, sig, smooth=smooth, tune_gain=use_inputs); return (r['smooth'] if smooth else r['filt'])[0]

def fs_errors(est, g, warm=72):
    """Signed errors (estimate - fingerstick) at fingerstick times after warm-up, and the fingerstick values."""
    k = np.where(~np.isnan(g['fs']) & ~np.isnan(est) & ~np.isnan(g['y']))[0]; k = k[k >= warm]; return est[k] - g['fs'][k], g['fs'][k]

TAUS = (4, 6, 8, 10, 12, 15, 18, 22); QES = (1e-5, 1e-4, 1e-3, 1e-2); SIGS = (3.0, 6.0, 10.0)

def loo_check(grids, use_inputs=True):
    """Leave-one-patient-out: choose (tau, qe, sig) pooled on the other patients to minimise fingerstick RMSE; evaluate on the held-out one.
    Returns per-patient RMSE/MARD for raw CGM, the smoother with tau fixed at 10 (default qe, sig tuned the same way) and the tuned smoother."""
    out = []; cache = {}
    def est(i, tau, qe, sig):
        key = (i, tau, qe, sig, use_inputs)
        if key not in cache: cache[key] = estimates(grids[i], tau, qe, sig, use_inputs)
        return cache[key]
    def score(idx, tau, qe, sig):
        e = np.concatenate([fs_errors(est(i, tau, qe, sig), grids[i])[0] for i in idx]); return float(np.sqrt(np.mean(e ** 2)))
    for h in range(len(grids)):
        rest = [i for i in range(len(grids)) if i != h]
        best = min(((score(rest, t, q, s), t, q, s) for t in TAUS for q in QES for s in SIGS)); bt = best[1:]
        fx = min(((score(rest, 10, q, s), 10, q, s) for q in QES for s in SIGS)); ft = fx[1:]
        e_t, f = fs_errors(est(h, *bt), grids[h]); e_10, _ = fs_errors(est(h, *ft), grids[h])
        k = np.where(~np.isnan(grids[h]['fs']) & ~np.isnan(grids[h]['y']))[0]; k = k[k >= 72]; e_raw = grids[h]['y'][k] - grids[h]['fs'][k]
        r = lambda e: float(np.sqrt(np.mean(e ** 2))); mard = lambda e: float(np.mean(np.abs(e) / f) * 100)
        out.append(dict(id=grids[h]['id'], n_fs=len(f), tau=bt[0], rmse_raw=r(e_raw), rmse_tau10=r(e_10), rmse_tuned=r(e_t), mard_raw=mard(e_raw), mard_tau10=mard(e_10), mard_tuned=mard(e_t)))
    return out
