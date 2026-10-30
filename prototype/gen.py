"""Generate simulated T1D data: true plasma glucose, lagged+noisy CGM, meals, insulin.
Ground truth: plasma G = x[3]/Vg ; CGM = subcutaneous Gs + Dexcom noise (tau_true = 1/ksc)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from simglucose.patient.t1dpatient import T1DPatient, Action
from simglucose.sensor.cgm import CGMSensor

Q = pd.read_csv(__import__('simglucose').__path__[0] + '/params/Quest.csv').set_index('Name')

def run(name, days=7, seed=0):
    rng = np.random.default_rng(seed)
    p = T1DPatient.withName(name)
    s = CGMSensor.withName('Dexcom', seed=seed)
    P = p._params
    basal = P.u2ss * P.BW / 6000.0           # U/min
    CR = Q.loc[name, 'CR']
    N = days * 1440
    TDI = Q.loc[name, 'TDI']
    meals = {}                                # minute -> grams ; sized from CR and TDI (about 50% of TDI as meal boluses)
    for d in range(days):
        for share, tm, tsd in [(0.13, 7.5, .5), (0.17, 12.5, .7), (0.20, 19.0, .7)]:
            t = int((d * 24 + tm + rng.normal(0, tsd)) * 60)
            meals[t] = CR * TDI * share * rng.uniform(.8, 1.2)
        if rng.random() < .5:                 # snack
            meals[int((d * 24 + rng.uniform(15, 17)) * 60)] = CR * TDI * 0.04 * rng.uniform(.5, 1.5)
    bolus = {t: g / CR * np.clip(rng.normal(0.92, .18), .5, 1.4) for t, g in meals.items()}
    plasma = np.zeros(N); cgm = np.zeros(N); sc = np.zeros(N); cho = np.zeros(N); ins = np.zeros(N)
    for k in range(N):
        g = meals.get(k, 0.0); b = bolus.get(k, 0.0)
        p.step(Action(insulin=basal + b, CHO=g))
        cgm[k] = s.measure(p)
        sc[k] = p.observation.Gsub
        plasma[k] = p.state[3] / P.Vg
        cho[k] = g; ins[k] = basal + b
    return dict(plasma=plasma, cgm=cgm, sub=sc, cho=cho, ins=ins, tau=1.0 / P.ksc, Gb=P.Gb)

if __name__ == '__main__':
    import time; t0 = time.time()
    r = run('adult#001', days=1)
    print('1 day took %.1fs' % (time.time() - t0), 'tau', r['tau'], 'plasma range', r['plasma'].min(), r['plasma'].max())
