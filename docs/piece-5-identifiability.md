# Piece 5: identifiability of the lag tau (31 Oct 2026)

Code: `code/ident.py`, `code/test_ident.py`, `code/run_ident.py`, `code/run_fstune.py`, `code/plot_ident.py`; `code/harness.py` updated (per-subject tau sources via `tau_key`, fingerstick tuning via `tune_on='fs'`, shared cache). Written by Claude (AI assistant); the team must read and sign off before claiming it. Setup: 27 subjects, 5-fold subject-wise CV, 5-min sampling, 5% dropouts. Days 1 to 3 are used to estimate the lag (with about 12 simulated fingersticks per patient: 4 per day, 4% CV); days 4 and 5 are used only for evaluation.

## Methods
- Likelihood: 4-state model [sensor-filtered glucose, ISF glucose, blood glucose, rate]; Gaussian innovation log-likelihood, profiled over process noise and sensor noise; fingerstick updates on blood glucose optional. Per-patient maximum likelihood, 95% likelihood-ratio interval, and an empirical-Bayes population prior on log tau (fitted on training folds only).
- Fingerstick regression (model-light): tau = sum (fs - CGM level) x CGM slope / sum slope^2 at fingerstick times.
- Fingerstick tuning (realistic replacement for truth tuning): population tau and process noise chosen by RMSE against the about 264 fingersticks of the training patients.

## Tests: 6 of 6 pass
4-state model reduces exactly to the Piece 4 model; on data simulated from the model itself, the likelihood interval covers the true tau in 7 of 8 runs (8 of 8 with fingersticks) and ML estimates cluster at 9 to 10.6 min for true 10; fingerstick regression exact on a ramp (12.01 vs 12); empirical Bayes recovers the population (mu log 10.1 vs 11; s 0.37 vs 0.30) and beats ML.

## Results
1. Per-patient likelihood estimates of tau are confidently wrong on simulator data. Intervals are narrow (hi/lo about 1.2 to 1.4) but cover the true effective lag in only 4 to 19% of patients; estimates are biased low (median 4 to 10.6 min vs truth 11.2 to 16.2; at noise sd 8 they pile up at the 4-min grid floor). This holds with near-white noise too, while the same code is accurate on model-generated data, so the cause is model misspecification (the smooth random-walk prior on blood glucose does not describe meal-driven glucose), not lack of information. Fingersticks do not fix it: 12 fingersticks against 864 CGM samples barely move the likelihood. Empirical Bayes inherits the bias (population mean at the grid floor).
2. Fingerstick regression is roughly unbiased at noise sd 2 (median signed error -1.4 to +0.3 min; correlation with truth about 0.5) but attenuates at noise sd 8 (median 6.0 vs 11.2 min in S0), the expected errors-in-variables effect of a noisy slope.
3. Physiology versus sensor filter (H3), S1 at noise 2 with fingersticks, 2-D profile over (tau, tau_f): the best tau falls as the assumed filter constant rises (10.6, 10.6, 9.0, 7.7, 6.5, 4.7 min for tau_f 0 to 10), so their sum is much more stable than either (median per-patient sd 1.2 vs 2.7 min). Best tau_f per patient spreads over the whole 0 to 10 min range (truth 3 min filter + 2 min delay). Physiological and sensor lag are not separable from these data; only an effective lag is.
4. Downstream (RTS smoother, rapid-change RMSE on days 4 to 5, median over 27 patients, mg/dL):

| Scenario, noise | tau = 10 | per-patient likelihood tau | population tau from fingersticks | population tau from truth (sim only) | true effective tau (oracle) |
| --- | --- | --- | --- | --- | --- |
| S0, 2 | 4.9 | 7.5 | 5.7 | 4.5 | 2.5 |
| S0, 8 | 8.3 | 11.9 | 10.0 | 9.5 | 7.7 |
| S1, 2 | 13.5 | 10.9 | 5.8 | 5.9 | 4.5 |
| S1, 8 | 13.7 | 16.2 | 11.5 | 10.1 | 8.8 |
| S5, 2 | 15.0 | 16.1 | 9.0 | 9.0 | 10.5 |
| S5, 8 | 16.0 | 19.9 | 13.7 | 12.6 | 12.8 |

- Fingerstick-tuned population tau versus tau = 10 (paired, median, 95% CI): S1 noise 2 -7.1 [-8.3, -5.3] mg/dL; S5 noise 2 -5.2 [-7.8, -2.3]; S1 noise 8 -3.0 [-3.2, -2.4]; S5 noise 8 -1.5 [-2.6, -0.6]. It chose tau = 16 min in S1 (truth-tuned 16 to 18; true effective median 16.2). In matched S0 it is slightly worse than tau = 10 (noise 8: +0.5 [0.0, 2.3]), the cost of choosing two parameters from noisy fingersticks.
- Per-patient likelihood tau makes things worse than tau = 10 in 5 of 6 conditions.
- Real-time Kalman filter: tuning tau (from truth or fingersticks) does not help; fingerstick tuning hurts at noise 8 (S0 +1.6 [1.2, 2.4] mg/dL). Consistent with Piece 4: noise, not tau, limits real-time compensation.
- Time shift tuned on fingersticks equals the truth-tuned shift in 5 of 6 conditions (8 min in S0, 13 min in S1 and S5).

## What this means for the project question
- From CGM alone, tau is not reliably identifiable per patient under a generic glucose model; the likelihood is sharp but biased. This is a misspecification problem.
- What can be identified, and is useful, is a population effective lag (physiology plus sensor) chosen with fingersticks. It recovers most of the oracle gain for retrospective smoothing when the sensor adds lag the model does not know about.
- Splitting that effective lag into physiological and sensor parts is not possible with these data (H3 confirmed in simulation).

## Caveats
- Simulation only; fingerstick timing (pre-meal, bedtime) and error model are assumptions. Real fingersticks may have been used for sensor calibration.
- Population tuning is coarse (8 tau values x 9 noise values); results at the S0 optimum differ by one grid step.
- A glucose model driven by meal and insulin inputs (Piece 6, optional) is the natural fix for the misspecification in finding 1.
