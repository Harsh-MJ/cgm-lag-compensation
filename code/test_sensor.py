"""Unit tests for sensor.py. Run: python3 -I test_sensor.py <path-to-sim.npz>   (also pytest-friendly with SIM env var)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensor as S

SIM = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SIM', '')
ok = []
def check(name, cond, info=''):
    ok.append(bool(cond)); print(('PASS ' if cond else 'FAIL ') + name + ('  ' + info if info else ''))

def ramp(n=1200, s=2.0, g0=100.0): return g0 + s * np.arange(n)

# 1 analytic ramp lag: steady error = slope * (tau [+ tau_f + delay])
for scen, tau, expect in [('S0_matched', 10.0, 20.0), ('S1_sensor_chain', 10.0, 30.0), ('S2_second_order', 10.0, 20.0)]:
    r = S.make_sensor(ramp(), tau, scen, noise_sd=0, dt=1)
    err = (r['G'] - r['y'])[800:].mean()
    check('ramp lag ' + scen, abs(err - expect) < 0.6, 'error %.2f mg/dL, expected %.1f' % (err, expect))

# 2 tiny tau and no chain -> sensor equals plasma, no index shift
x = 100 + 30 * np.sin(np.arange(2000) / 90.0)
r = S.make_sensor(x, 0.01, 'S0_matched', noise_sd=0, dt=5, offset=2)
check('no shift when tau->0', np.allclose(r['y'], r['G']), 'max |y-G| %.2e' % np.abs(r['y'] - r['G']).max())

# 3 noise: sd and autocorrelation
rng = np.random.default_rng(1); e = S.noise(200000, 5.0, rng)
acf = lambda h: np.corrcoef(e[:-h], e[h:])[0, 1]
want = lambda h: 0.6 * np.exp(-h / 30.0) + 0.4 * np.exp(-h / 2.0)
check('noise sd', abs(e.std() - 5.0) < 0.3, 'sd %.2f' % e.std())
check('noise acf', all(abs(acf(h) - want(h)) < 0.04 for h in (1, 5, 15, 30)),
      ' '.join('h=%d %.2f/%.2f' % (h, acf(h), want(h)) for h in (1, 5, 15, 30)))

# 4 time-varying tau
r = S.make_sensor(ramp(14400 * 3), 10.0, 'S3_tau_varying', noise_sd=0, dt=1, seed=3)
lt = np.log(r['tau_t']); check('tau_varying spread', 0.15 < lt.std() < 0.35 and abs(lt.mean() - np.log(10)) < 0.2,
                              'sd(log tau) %.2f mean tau %.1f' % (lt.std(), np.exp(lt).mean()))

# 5 dropouts and compression
m = S.dropout_mask(200000, 0.05, np.random.default_rng(2)); check('dropout fraction', abs(m.mean() - 0.05) < 0.01, '%.3f' % m.mean())
f, fl = S.compression(1440 * 100, np.random.default_rng(4))
check('compression depth/rate', 0.25 <= (1 - f[fl]).max() <= 0.45 and 0.3 < len(np.unique(np.where(fl)[0] // 1440)) / 100 < 0.5,
      'max depth %.2f, nights with event %.2f' % ((1 - f[fl]).max(), len(np.unique(np.where(fl)[0] // 1440)) / 100))

# 6 reproducibility
a = S.make_sensor(x, 10, 'S5_full', seed=7, miss_frac=0.05); b = S.make_sensor(x, 10, 'S5_full', seed=7, miss_frac=0.05)
c = S.make_sensor(x, 10, 'S5_full', seed=8, miss_frac=0.05)
check('seed reproducible', np.array_equal(a['y'], b['y'], equal_nan=True) and not np.array_equal(a['y'], c['y'], equal_nan=True))

# 7 own ISF integrator vs simglucose subcutaneous state (needs sim.npz)
if SIM:
    D = np.load(SIM); worst = 0; rm = []
    for n in ['adult#002', 'adolescent#004', 'child#003']:
        gi = S.first_order(D[n + '|plasma'], float(D[n + '|tau'])); d = gi[360:] - D[n + '|sub'][360:]
        rm.append(np.sqrt((d ** 2).mean())); worst = max(worst, np.abs(d).max())
    check('own ISF == simglucose subcutaneous', max(rm) < 0.5 and worst < 3.0, 'rmse %s max %.2f mg/dL' % (np.round(rm, 3), worst))
print('\n%d/%d passed' % (sum(ok), len(ok))); sys.exit(0 if all(ok) else 1)
