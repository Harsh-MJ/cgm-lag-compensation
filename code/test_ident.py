"""Piece 5 tests. python3 -I test_ident.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ident as I, kalman as K
ok = []
def check(n, c, i=''): ok.append(bool(c)); print(('PASS ' if c else 'FAIL ') + n + ('  ' + i if i else ''))

# 1 with no filter stage the 4-state model reduces to the Piece 4 3-state model
F4, Q4 = I.lti(10.0, 0.0, 5.0); F3, Q3 = K.model([10.0], 5.0)
check('4-state reduces to 3-state', np.allclose(F4[1:, 1:], F3[0], atol=1e-6) and np.allclose(Q4[1:, 1:], Q3[0], rtol=1e-4, atol=1e-6),
      'max dF %.1e dQ %.1e' % (np.abs(F4[1:, 1:] - F3[0]).max(), np.abs(Q4[1:, 1:] - Q3[0]).max()))

# 2 well-specified simulation from the state-space model: LR interval covers true tau; fingersticks shrink it
def sim(tau, q, sig, n, rng, dt=5.0):
    F, Q = I.lti(tau, I.NO_FILTER, dt); L = np.linalg.cholesky(Q * q + 1e-12 * np.eye(4)); x = np.array([120., 120., 120., 0.]); Y = np.empty(n); G = np.empty(n)
    for k in range(n):
        x = F @ x + L @ rng.standard_normal(4); Y[k] = x[0] + sig * rng.standard_normal(); G[k] = x[2]
    return Y, G
rng = np.random.default_rng(5); cov = covf = 0; wid = []; widf = []; est = []
for r in range(8):
    Y, G = sim(10.0, 2e-3 * 4, 2.0, 1440, rng)
    t = np.arange(1440) * 5.0; fs, rfs = I.make_fingersticks(G, t, rng)
    p = I.profile(Y[None], 5.0, qes=(2e-3,), sigs=(2.0,))[0, :, 0]; pf = I.profile(Y[None], 5.0, FSs=fs[None], RFSs=rfs[None], qes=(2e-3,), sigs=(2.0,))[0, :, 0]
    lo, hi, ml = I.lr_interval(p); lof, hif, mlf = I.lr_interval(pf); cov += lo <= 10 <= hi; covf += lof <= 10 <= hif; wid.append(hi / lo); widf.append(hif / lof); est.append(ml)
check('well-specified: CGM-only LR interval covers true tau', cov >= 6, '%d/8 covered, median hi/lo ratio %.2f, ML %s' % (cov, np.median(wid), np.round(est, 1)))
check('fingersticks: still covers, interval not wider', covf >= 6 and np.median(widf) <= np.median(wid) + 1e-9, '%d/8 covered, median hi/lo ratio %.2f' % (covf, np.median(widf)))

# 3 fingerstick regression exact on noise-free lagged ramp
t1 = np.arange(3000.); Gp = 100 + 1.5 * t1; import sensor as S; GI = S.first_order(Gp, 12.0)
y = GI[::5]; G = Gp[::5]; fs = np.full(len(y), np.nan); fs[100::50] = G[100::50]
check('fingerstick regression on ramp', abs(I.fs_regression(y, fs, 5.0) - 12.0) < 0.3, 'tau_hat %.2f (true 12)' % I.fs_regression(y, fs, 5.0))

# 4 empirical Bayes recovers population and shrinks toward it
rng = np.random.default_rng(3); lt = np.log(I.TAU_GRID); true = rng.normal(np.log(11), 0.3, 40); obs = true + rng.normal(0, 0.4, 40)
profs = np.array([-0.5 * ((lt - o) / 0.4) ** 2 for o in obs]); mu, s = I.eb_fit(profs)
post = np.array([np.log(I.eb_post(p, mu, s)) for p in profs]); ml = lt[np.argmax(profs, 1)]
check('EB recovers population', abs(mu - np.log(11)) < 0.2 and 0.1 < s < 0.55, 'mu %.2f (true %.2f) s %.2f (true 0.30)' % (mu, np.log(11), s))
check('EB posterior beats ML', np.mean((post - true) ** 2) < np.mean((ml - true) ** 2), 'rmse log-tau %.3f vs %.3f' % (np.sqrt(np.mean((post - true) ** 2)), np.sqrt(np.mean((ml - true) ** 2))))
print('\n%d/%d passed' % (sum(ok), len(ok))); sys.exit(0 if all(ok) else 1)
