"""Piece 6 tests. python3 -I test_inputs.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inputs as X, kalman as K
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info)

# 1. with zero inputs the 5-state filter equals the Piece 4 3-state filter (same init, sigma = 1)
rng = np.random.default_rng(0); S, n, dt = 3, 300, 5.0
Y = 120 + np.cumsum(rng.normal(0, 3, (S, n)), 1); Y[:, 40] = np.nan
f4, s4 = K.kf_batch(Y, np.array([8., 10., 14.]), dt, 1e-3, smooth=True)
r = X.kfx(Y, np.zeros((S, n, 2)), np.array([8., 10., 14.]), dt, 1e-3, 1.0, smooth=True, tune_gain=False)
check('zero inputs reproduce Piece 4 filter', np.abs(r['filt'] - f4).max() < 1e-6, 'max diff %.1e' % np.abs(r['filt'] - f4).max())
check('zero inputs reproduce Piece 4 smoother', np.abs(r['smooth'] - s4).max() < 1e-6, 'max diff %.1e' % np.abs(r['smooth'] - s4).max())

# 2. kernels are normalised and have the right peak times
h2 = X._gamma_kernel(20, 2); h3 = X._gamma_kernel(40, 3)
check('kernels sum to 1, peaks at tau and 2 tau', abs(h2.sum() - 1) < 1e-12 and abs(h3.sum() - 1) < 1e-12 and h2.argmax() == 20 and h3.argmax() == 80, '%d %d' % (h2.argmax(), h3.argmax()))
u1, u2 = X.drive(np.r_[np.zeros(10), 50.0, np.zeros(989)], np.full(1000, 0.02))
check('basal insulin gives no drive; meal mass conserved', np.abs(u2).max() < 1e-12 and abs(u1.sum() - 50) < 1e-6)

# 3. simulate from the model itself and recover tau and gains
def simulate(tau, a, b, seed, n=864, dt=5.0, sig=2.0, q=1e-4):
    rng = np.random.default_rng(seed); U = np.zeros((n, 2)); U[rng.integers(20, n - 20, 12), 0] = rng.uniform(0.5, 1.5, 12) * 3.0; U[rng.integers(20, n - 20, 12), 1] = 0.05
    from scipy.ndimage import gaussian_filter1d
    U = gaussian_filter1d(U, 3, axis=0, mode='constant'); F3, Q0 = X._fq([tau], dt); F3 = F3[0]; Q = q * Q0[0]; L = np.linalg.cholesky(Q + 1e-12 * np.eye(3))
    x = np.array([120., 120., 0.]); y = np.empty(n); G = np.empty(n)
    for k in range(n):
        y[k] = x[0] + sig * rng.standard_normal(); G[k] = x[1]
        x = F3 @ x + L @ rng.standard_normal(3) + np.array([F3[0, 2], F3[1, 2], 0.0]) * (a * U[k, 0] - b * U[k, 1])
    return y, U, G
Ys, Us = [], []
for i in range(6):
    y, U, G = simulate(10.0, 6.0, 40.0, i); Ys.append(y); Us.append(U)
P = X.profile(np.array(Ys), np.array(Us), 5.0, qes=(1e-5, 1e-4, 1e-3), sigs=(1.5, 3.0))
est = X.TAU_GRID[P.argmax(1)]
check('ML tau recovers truth 10 on model-generated data (median within 20%)', abs(np.median(est) - 10) < 2.0, str(np.round(est, 1)))
r = X.kfx(np.array(Ys), np.array(Us), 10.0, 5.0, 1e-4, 2.0)
check('gain a, b estimates near truth (6, 40)', abs(np.median(r['a']) - 6) < 2 and abs(np.median(r['b']) - 40) < 15, 'a %.1f b %.1f' % (np.median(r['a']), np.median(r['b'])))

# 4. inputs help: on model-generated data the input-driven smoother has lower RMSE than the zero-input one
G = np.array([simulate(10.0, 6.0, 40.0, i)[2] for i in range(6)])
r1 = X.kfx(np.array(Ys), np.array(Us), 10.0, 5.0, 1e-4, 2.0, smooth=True); r0 = X.kfx(np.array(Ys), np.zeros_like(np.array(Us)), 10.0, 5.0, 1e-4, 2.0, smooth=True, tune_gain=False)
e1 = np.sqrt(np.mean((r1['smooth'] - G) ** 2)); e0 = np.sqrt(np.mean((r0['smooth'] - G) ** 2))
check('inputs improve smoother RMSE on model-generated data', e1 < e0, 'with %.2f without %.2f' % (e1, e0))
# 5. posterior sd: nominal 95% intervals cover the truth on model-generated data (smoother)
cov = np.mean(np.abs(r1['smooth'] - G) <= 1.96 * r1['smooth_sd'])
check('smoother 95% interval coverage on model-generated data in [0.90, 0.99]', 0.90 <= cov <= 0.99, 'coverage %.3f' % cov)
# 6. AR(1) noise ablation: on data with a slow AR(1) noise component the AR model has higher likelihood than the white-noise model
rng = np.random.default_rng(3); Yn = []; Un = []
for i in range(6):
    y, U, _ = simulate(10.0, 6.0, 40.0, 50 + i, sig=1.0); v = np.zeros(len(y)); rho = np.exp(-5 / 30.0)
    for k in range(1, len(y)): v[k] = rho * v[k - 1] + 4.0 * np.sqrt(1 - rho ** 2) * rng.standard_normal()
    Yn.append(y + v); Un.append(U)
l_w = X.kfx(np.array(Yn), np.array(Un), 10.0, 5.0, 1e-4, 4.0, ll=True)['ll'].sum(); l_a = X.kfx(np.array(Yn), np.array(Un), 10.0, 5.0, 1e-4, 4.0, ll=True, ar=(30.0, 0.9))['ll'].sum()
check('AR(1) noise state raises the log-likelihood on AR(1)-noise data', l_a > l_w, 'white %.0f AR %.0f' % (l_w, l_a))
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
