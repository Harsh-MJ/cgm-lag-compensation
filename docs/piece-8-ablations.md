# Piece 8: ablations, posterior coverage and final report (26 Nov 2026)

Code: `code/run_ablate.py`, `code/run_coverage.py`, `code/inputs.py` (AR(1) noise state and posterior sd added; tests 9 of 9 pass); results `results/ablate_results.json`, `results/coverage_results.json`, figure `results/fig_pieces678.png`; report `docs/tex/final-report.tex`. Written by Claude (AI assistant); the team must read and sign off before claiming it.

## Ablations (rapid-change RMSE in mg/dL, median over 27 patients, days 4 to 5, tau = 10; S0/S1 and noise 2/8; difference = full model minus variant, 95% CI)
| Variant | filter (S0-2, S0-8, S1-2, S1-8) | smoother (S0-2, S0-8, S1-2, S1-8) |
| --- | --- | --- |
| no inputs (random walk) | 7.8, 20.3, 16.0, 25.8 | 4.9, 8.3, 13.5, 13.7 |
| full input model | 8.4, 14.9, 18.9, 21.2 | 4.6, 8.4, 11.9, 12.1 |
| gains frozen at prior means | 11.1, 19.7, 18.1, 25.7 | 4.2, 8.0, 11.3, 11.9 |
| kernel times x 0.5 | 8.2, 15.8, 14.7, 20.2 | 4.5, 8.3, 12.4, 12.3 |
| kernel times x 2 | 7.4, 13.3, 16.3, 19.4 | 4.6, 8.0, 13.3, 14.2 |
| + AR(1) noise state | 8.9, 14.0, 17.7, 21.3 | 4.9, 8.0, 11.9, 11.9 |

- Gain learning matters for the real-time filter (frozen gains cost +2.7 to +4.8 mg/dL, as differences of medians, in three of four conditions) but not for the smoother (differences +0.13 [+0.10, +0.35], +0.20 [-0.43, +0.62], -0.02 [-0.33, +0.53], +1.09 [-0.43, +1.47]).
- Kernel shape is forgiving: halving the time constants changes the smoother by -0.1 to +0.7 mg/dL (variant minus full model); doubling them costs 0.1 to 1.1 mg/dL in all four conditions (full model minus variant -0.08, -1.11, -0.86, -0.95).
- The AR(1) noise state does not improve point accuracy: smoother difference to the full model -0.00 [-0.03, +0.22], -0.22 [-0.37, +0.24], +0.26 [+0.08, +0.40], -0.19 [-0.45, 0.00]; real-time filter -0.18 to -0.71 (CI excludes 0 in two conditions).

## Posterior coverage (95% interval for blood glucose, smoother, fraction of days 4 to 5 inside; median over patients; noise settings chosen per patient by likelihood on days 1 to 3)
| Condition | white noise, no inputs | white noise + inputs | AR(1) noise + inputs | AR(1) noise, no inputs |
| --- | --- | --- | --- | --- |
| S0, noise 2 | 0.70 | 0.67 | 0.89 | 0.91 |
| S0, noise 8 | 0.66 | 0.59 | 0.89 | 0.91 |
| S1, noise 2 | 0.52 | 0.50 | 0.70 | 0.74 |
| S1, noise 8 | 0.59 | 0.58 | 0.88 | 0.87 |

At rapid-change times the AR(1) model covers 0.57, 0.82, 0.08 and 0.59 (S0-2, S0-8, S1-2, S1-8) against 0.34, 0.49, 0.04 and 0.28 for white noise with inputs. When noise settings are chosen for RMSE instead of likelihood, smoother coverage is 0.86, 0.17, 0.42, 0.15: RMSE-tuned settings give overconfident intervals. The AR structure used is the simulator's own (30 min, 60% of the variance), so this is an oracle-structure ablation; real sensor noise structure is unknown.

## Conclusions
1. The AR(1) noise state is for uncertainty, not accuracy: it brings 95% intervals to about 0.9 coverage in the matched sensor case but does not lower RMSE.
2. When the sensor adds lag the model does not know (S1 at noise 2), intervals fail at rapid change (0.04 to 0.08 coverage): model uncertainty from the lag cannot be fixed by noise modelling.
3. See `docs/tex/final-report.tex` for the integrated account.
