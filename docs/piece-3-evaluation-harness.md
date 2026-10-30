# Piece 3: evaluation harness and baselines (30 Oct 2026)

Code: `code/harness.py`, `code/test_harness.py`, `code/run_baselines.py`, `code/plot_baselines.py` (needs `sensor.py` from Piece 2). Written by Claude (AI assistant); the team must read and sign off before claiming it. Assumptions: headline noise sd 8 mg/dL (2 and 5 as sensitivity), prototype meal sizes unchanged, 5-min sampling, 5% dropouts.

## Protocol implemented
- 27 usable virtual patients (same exclusions as the prototype). Subject-wise 5-fold CV (folds of 6, 6, 5, 5, 5; each has adolescents and adults). Manifest saved in `results/split_manifest.json`.
- Hyper-parameters are chosen by pooled RMSE on the four non-test folds only; test subjects are evaluated once. The baselines have no fitted parameters, so there is no separate train/validation split yet.
- Estimators: B0 raw (last value carried over dropouts), B1 fixed time shift (non-causal), B2 causal exponential smoothing, B3 causal naive inversion with tau = 10, B3 oracle (told the true tau; a ceiling, not a fair competitor).
- Metrics per subject: RMSE, MAD, bias, MARD (plasma >= 40), RMSE on rapid-change samples (|true rate| >= 1 mg/dL/min, at least 20 samples), rate-of-change RMSE (15-min window), residual lag (best shift), hypoglycaemia detection (threshold 70 mg/dL, events merged if gaps < 30 min, detection window 30 min before onset to end, false alarms per day). Summaries: median over subjects with cluster bootstrap 95% CI (2000 resamples); paired differences versus raw with Wilcoxon test and Holm correction; pooled-ratio bootstrap for hypoglycaemia delay and false alarms.

## Tests: 16 of 16 pass
Hand-checked metrics (constant +10 error gives RMSE, MAD, bias, MARD all 10); perfect estimator; a 10-min lagged estimate gives residual lag 10 and hypo delay 10; false alarm counted; bootstrap and Holm helpers; folds disjoint and complete; causality test (changing future samples leaves past output of B0, B2, B3 unchanged, and B1 correctly fails it); every subject evaluated once; noise-free sanity (oracle inverse RMSE 1.68 vs raw 7.29 mg/dL).

## Results at noise sd 8 mg/dL (median over 27 subjects, 95% CI)
RMSE on rapid-change samples, mg/dL:

| Scenario | B0 raw | B1 shift (non-causal) | B3 inverse (causal, tau 10) | B3 true tau (oracle) |
| --- | --- | --- | --- | --- |
| S0 matched | 21.5 [19.2, 24.7] | 11.3 [10.0, 12.6] | 17.9 [15.1, 20.2] | 17.7 [15.1, 20.0] |
| S1 sensor chain | 27.7 [23.8, 33.2] | 11.2 [9.9, 13.1] | 23.6 [19.9, 26.9] | 22.9 [18.7, 26.6] |
| S5 full | 31.3 [23.0, 32.9] | 12.4 [11.4, 14.0] | 26.3 [21.7, 27.3] | 25.5 [21.6, 28.5] |

Paired change versus raw (rapid RMSE, Holm-adjusted p): S0 shift -10.3 [-12.5, -7.3] (p < 0.001), B3 -3.0 [-4.6, -1.3] (p = 0.001). S1: shift -16.5, B3 -4.6. S5: shift -18.4, B3 -4.2. All scenarios and noise levels are in `results/baseline_results.json`.

## Findings
1. The non-causal shift (advance by 8 to 13 min, chosen in CV) cuts rapid-change error by about half in every scenario. It uses future samples, so it is a retrospective ceiling, not a real-time method. Causal inversion recovers only 3 to 5 mg/dL of the 10 to 18 mg/dL available at noise sd 8.
2. Overall RMSE (all samples): causal inversion does not beat raw at noise sd 8 (S0: 11.3 vs 11.1 mg/dL), consistent with the prototype. Its gain is confined to rapid-change periods and costs more false alarms (S0 1.30 vs 0.93 per day; S5 1.86 vs 1.19).
3. B2 causal smoothing picked alpha = 1 (no smoothing) in every fold, so it equals raw: smoothing adds lag and never lowers error versus truth. Kept as a sanity row.
4. B3 with the true tau is not better than B3 with tau = 10 at noise sd 8: the local-slope estimate over a 25 to 30 min window leaves about 5 min residual lag and noise limits the gain.
5. Rapid-change share varies from 0.0% to 27.7% of samples per patient (median 8.6%); results are driven by the patients who move.
6. No problematic grid-edge picks: B2 hits alpha = 1, a hard physical limit; B1 and B3 optima are interior.
7. Hypoglycaemia: 57 to 58 events pooled across test subjects. Negative delays mean the estimate crossed 70 mg/dL before truth, mostly noise-driven; interpret delay together with false alarms per day.
