"""Kalman tests. Run: python3 -I test_kalman.py <sim.npz>"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalman as K, harness as H, sensor as S
from scipy.linalg import expm
ok = []
def check(n, c, i=''): ok.append(bool(c)); print(('PASS ' if c else 'FAIL ') + n + ('  ' + i if i else ''))

# 1 discretisation: F and Q against Van Loan's matrix-exponential method
tau, dt = 10.0, 5.0; a = 1 / tau
A = np.array([[-a, a, 0], [0, 0, 1], [0, 0, 0]]); B = np.array([[0], [0], [1.0]]); M = np.zeros((6, 6)); M[:3, :3] = -A; M[:3, 3:] = B @ B.T; M[3:, 3:] = A.T
E = expm(M * dt); Fv = E[3:, 3:].T; Qv = Fv @ E[:3, 3:]
F, Q = K.model([tau], dt)
check('F matches expm', np.allclose(F[0], expm(A * dt), atol=1e-10), 'max diff %.1e' % np.abs(F[0] - expm(A * dt)).max())
check('Q matches Van Loan', np.allclose(Q[0], Qv, rtol=1e-8, atol=1e-8), 'max rel diff %.1e' % (np.abs(Q[0] - Qv).max() / np.abs(Qv).max()))

# 2 noise-free ramp, tau known: filter and smoother recover plasma exactly
n = 400; G = 100 + 2.0 * 5 * np.arange(n); Gi = S.first_order(np.repeat(G, 5)[:n * 5], 10.0)[2::5][:n]   # quick lagged ramp on 5-min grid
t1 = np.arange(n * 5); Gp = 100 + 2.0 * t1; GI = S.first_order(Gp, 10.0)[::5]; Gt = Gp[::5]
f, s = K.kf_batch(GI[None, :], np.array([10.0]), 5.0, 1e-2, smooth=True)
check('ramp, tau known: filter error', np.abs(f[0, 100:] - Gt[100:]).max() < 0.5, 'max %.3f mg/dL' % np.abs(f[0, 100:] - Gt[100:]).max())
check('ramp, tau known: smoother error', np.abs(s[0, 100:-20] - Gt[100:-20]).max() < 0.5, 'max %.3f mg/dL' % np.abs(s[0, 100:-20] - Gt[100:-20]).max())

# 3 tau mismatch on the ramp: steady error = (tau_assumed - tau_true) * slope = (8 - 10) * 2 = -4
f, _ = K.kf_batch(GI[None, :], np.array([8.0]), 5.0, 1e-2); e = (f[0] - Gt)[200:].mean()
check('tau mismatch gives (8-10)*2 = -4 mg/dL', abs(e + 4.0) < 0.5, 'mean error %.2f' % e)

# 4 causality: filter ignores the future, smoother does not
pat, use, excl = H.load_patients(sys.argv[1]); data = H.build(pat, use[:3], 'S5_full', 8.0); Y = np.array([data[m]['y'] for m in use[:3]]); T = np.array([10.0] * 3)
Y2 = Y.copy(); Y2[:, 600:] = np.random.default_rng(0).normal(150, 40, Y.shape[1] - 600)
fa, sa = K.kf_batch(Y, T, 5.0, 1e-3, True); fb, sb = K.kf_batch(Y2, T, 5.0, 1e-3, True)
check('filter causal', np.allclose(fa[:, :600], fb[:, :600])); check('smoother non-causal', not np.allclose(sa[:, :600], sb[:, :600]))

# 5 batch equals single, no NaN outputs with 5% dropouts
f1, s1 = K.kf_batch(Y[:1], T[:1], 5.0, 1e-3, True)
check('batch == single row', np.allclose(f1[0], fa[0]) and np.allclose(s1[0], sa[0])); check('no NaN with dropouts', not (np.isnan(fa).any() or np.isnan(sa).any()))

# 6 independent cross-check: RTS smoother vs regularised inversion with matched prior, lam = 1/(qe dt^3)
qe = 1e-3; d = []; ds = []
for i in range(3):
    m2 = K.reginv(Y[i], 10.0, 5.0, 1 / (qe * 5.0 ** 3)); w = slice(72, -10)
    d.append(np.sqrt(np.mean((m2[w] - sa[i][w]) ** 2))); ds.append(np.sqrt(np.mean((sa[i][w] - data[use[i]]['G'][w]) ** 2)))
check('smoother and regularised inversion agree', max(d) < 0.25 * min(ds), 'RMS difference %s vs smoother error %s mg/dL' % (np.round(d, 2), np.round(ds, 2)))
print('\n%d/%d passed' % (sum(ok), len(ok))); sys.exit(0 if all(ok) else 1)
