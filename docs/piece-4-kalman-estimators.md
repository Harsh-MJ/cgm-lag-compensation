# Piece 4: Kalman filter, RTS smoother and regularised-inversion cross-check (30 Oct 2026)

Code: `code/kalman.py`, `code/test_kalman.py`, `code/run_kalman.py`, `code/plot_kalman.py`; `code/harness.py` was updated (batched estimators). Written by Claude (AI assistant); the team must read and sign off before claiming it. Setup as in Piece 3 (27 subjects, 5-fold subject-wise CV, 5-min sampling, 5% dropouts); noise sd 2 and 8 mg/dL, scenarios S0 to S5.

## Method
- Model: dG_I/dt = (G_B - G_I)/tau, dG_B/dt = R, dR/dt = sqrt(q) white noise, y = G_I + noise. Exact discretisation (F and Q checked against Van Loan). Only qe = q / sigma^2 matters; tuned on a half-decade grid 1e-7 to 1.
- M1f: causal Kalman filter (real time). M1s: RTS smoother (uses future). M2: regularised inversion solved as a sparse KKT system (independent implementation, second-difference penalty). Variants: tau fixed at 10 min ("fixed"), told the true tau ("oracle"), population tau tuned jointly with qe on training subjects ("taupop"; needs reference glucose, so simulation only).

## Tests: 10 of 10 pass
F and Q match matrix-exponential results (1e-16); noise-free ramp with correct tau recovered to 0.017 mg/dL (filter and smoother); wrong tau (8 vs 10) gives -4.02 mg/dL, theory -4; filter is causal, smoother is not; batch equals single; no NaNs with dropouts; smoother and regularised inversion agree to 0.05 to 0.09 mg/dL against about 10 mg/dL error.

## Results: RMSE on rapid-change samples (mg/dL, median over 27 subjects)
Noise sd 2:

| Scenario | Raw | B3 inversion | M1f fixed | M1f tuned tau | B1 shift | M1s fixed | M1s tuned tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 matched | 19.7 | 10.2 | 7.5 | 7.7 | 8.0 | 4.4 | 3.8 |
| S1 sensor chain | 27.5 | 17.0 | 14.8 | 13.4 | 8.3 | 12.3 | 5.2 |
| S5 full | 30.2 | 19.7 | 19.9 | 18.6 | 10.3 | 15.1 | 9.1 |

Noise sd 8:

| Scenario | Raw | B3 inversion | M1f fixed | M1f tuned tau | B1 shift | M1s fixed | M1s tuned tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 matched | 21.5 | 17.9 | 18.4 | 18.5 | 11.3 | 8.4 | 8.9 |
| S1 sensor chain | 27.7 | 23.6 | 24.9 | 25.8 | 11.2 | 13.9 | 9.4 |
| S5 full | 31.3 | 26.3 | 25.6 | 26.3 | 12.4 | 15.8 | 12.0 |

All scenarios and confidence intervals: `results/kalman_results.json`.

## Findings (paired differences are median over subjects, 95% bootstrap CI)
1. Implementation is validated: the independent regularised inversion matches the smoother (M2 minus M1s: 0.0 [-0.0, 0.1] at S0, noise 8).
2. Real-time filtering is noise-limited. At noise sd 2 (S0) the Kalman filter beats naive inversion by 2.1 [0.9, 2.6] mg/dL and reaches the non-causal shift (7.5 vs 8.0), with zero residual lag. At noise sd 8 it does not beat naive inversion (0.0 [-0.5, 3.0]) and recovers only about 3 mg/dL of the 10 mg/dL lag error versus raw.
3. Retrospective smoothing beats the non-causal shift when the model is right: S0 noise 8, M1s - B1 = -1.9 [-3.3, -1.3] mg/dL (p < 0.001); and cuts false alarms to 0.23 per day versus 0.93 raw with the same detection (57 of 58 events).
4. Knowing tau does not help the real-time filter: oracle minus fixed = -0.1 [-0.1, 0.1] at S0 noise 8 and -0.2 [-0.9, 0.0] at noise 2. Noise, not tau error, is the bottleneck.
5. Unmodelled sensor lag matters for the smoother: in S5 noise 8 fixed-tau smoother is worse than the shift by 3.3 [1.6, 4.1] mg/dL, and tuning a population tau recovers it (M1s tuned minus fixed -3.8 [-5.3, -1.8]; tuned tau 16 to 18 min).
6. A tuned tau is not the physical lag: at S0 noise 8 the real-time filter's tuned tau is 6 min (below the true mean 11.9), acting as noise regularisation.
7. Real-time filtering raises false alarms: S0 noise 8 hypoglycaemia false alarms 1.24 per day versus 0.93 raw, with earlier alarms (-11.9 min versus -4.9); do not read earlier alerts as better without the false-alarm cost.
8. Grid edges: no Kalman or regularised-inversion picks at a grid edge after widening the regularisation grid; only B3 at noise 2 picks the smallest window (3 samples).

## Caveats
- Tuned-tau and shift results use ground-truth glucose to tune; real data only has sparse fingersticks (addressed in Piece 5).
- Filter treats noise as white; simulated noise has a slow component. An AR(1) noise state is a candidate ablation.
- Posterior coverage of the filter's intervals is not yet checked.
