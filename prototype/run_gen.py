import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import gen
names = pd.read_csv(__import__('simglucose').__path__[0] + '/params/vpatient_params.csv').Name.tolist()
def job(a):
    i, n = a
    return n, gen.run(n, days=5, seed=100 + i)
if __name__ == '__main__':
    with Pool() as pool:
        res = pool.map(job, list(enumerate(names)))
    out = {}
    for n, r in res:
        for k, v in r.items():
            out[f'{n}|{k}'] = v
    np.savez_compressed('../data/sim.npz', **out)
    print('saved', len(res), 'patients')
