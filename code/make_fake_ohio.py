"""Writes synthetic files in OhioT1DM XML format from the simulator (for testing the loader and pipeline only; NOT real data).
python3 -I make_fake_ohio.py <sim.npz> <outdir> [n_patients]"""
import sys, os, numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, sensor as S
pat, use, _ = H.load_patients(sys.argv[1]); D = np.load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True); npat = int(sys.argv[3]) if len(sys.argv) > 3 else 4
T0 = datetime(2021, 6, 1, 0, 3, 20); f = lambda t: t.strftime('%d-%m-%Y %H:%M:%S')
for i, n in enumerate(use[:npat]):
    r = S.make_sensor(pat[n]['plasma'], pat[n]['tau'], 'S0_matched', noise_sd=5.0, dt=5, seed=77 + i, miss_frac=0.05); rng = np.random.default_rng(5 + i)
    L = ['<patient id="%d" weight="80" insulin_type="Humalog">' % (900 + i), '<glucose_level>'] + ['<event ts="%s" value="%d"/>' % (f(T0 + timedelta(minutes=float(t))), round(max(v, 40))) for t, v in zip(r['t'], r['y']) if not np.isnan(v)] + ['</glucose_level>', '<finger_stick>']
    for d in range(5):
        for h in (7.0, 11.5, 17.5, 22.0):
            k = int(d * 1440 + h * 60 + rng.normal(0, 15)); L.append('<event ts="%s" value="%d"/>' % (f(T0 + timedelta(minutes=k)), round(pat[n]['plasma'][k] * (1 + 0.04 * rng.standard_normal()))))
    L += ['</finger_stick>', '<basal>', '<event ts="%s" value="%.3f"/>' % (f(T0), np.median(D[n + '|ins']) * 60), '</basal>', '<temp_basal>', '</temp_basal>', '<bolus>']
    cho, ins = D[n + '|cho'], D[n + '|ins']; base = np.median(ins)
    for k in np.where(ins - base > 0.5)[0]: L.append('<event ts_begin="%s" ts_end="%s" type="normal" dose="%.2f" bwz_carb_input="0"/>' % (f(T0 + timedelta(minutes=int(k))), f(T0 + timedelta(minutes=int(k) + 1)), ins[k] - base))
    L += ['</bolus>', '<meal>'] + ['<event ts="%s" type="Lunch" carbs="%d"/>' % (f(T0 + timedelta(minutes=int(k))), round(cho[k])) for k in np.where(cho > 0.5)[0]] + ['</meal>', '</patient>']
    open('%s/%d-ws-training.xml' % (out, 900 + i), 'w').write('\n'.join(L))
print('wrote', npat, 'synthetic files to', out)
