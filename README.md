# Physics-informed lag compensation for continuous glucose monitoring

DH 803 (Wearable Health Technology) group project, group **Lag Compensators**: Harsh Jadhav (Chemical Engineering) and Shreyas Kamble (Mechanical Engineering).

CGM sensors measure glucose in interstitial fluid, which lags blood glucose by roughly 5 to 15 minutes. This project models that lag as first-order transport and asks two questions in simulation:

1. How much of the lag error can a physics-based state estimator (Kalman filter, RTS smoother) remove, in real time and retrospectively, compared with simple baselines?
2. Can the lag time constant tau be identified from data, per patient or for a population, and can physiological lag be separated from sensor lag?

Status: simulation study complete through Piece 8 (26 Nov 2026), final report in `docs/tex/final-report.tex`. No real-patient data yet: the OhioT1DM check is implemented and tested on synthetic files but not run, because the dataset needs a data use agreement (see `docs/piece-7-real-data.md`).



## Main results (simulation, 27 virtual patients, 5-fold subject-wise cross-validation)

Error during rapid glucose change (|rate| >= 1 mg/dL/min), scenario S0 (model matches), RMSE in mg/dL, median over patients:

| Method | Uses future data? | Noise sd 2 | Noise sd 8 |
| --- | --- | --- | --- |
| Raw CGM | no | 19.7 | 21.5 |
| Naive inversion, tau = 10 | no | 10.2 | 17.9 |
| Kalman filter, tau = 10 | no | 7.5 | 18.4 |
| Time shift (tuned) | yes | 8.0 | 11.3 |
| RTS smoother, tau = 10 | yes | 4.4 | 8.4 |

Key findings:
- **Real-time compensation is noise-limited.** The Kalman filter matches the future-peeking shift at low noise but does not beat naive inversion at noise sd 8, and earlier hypoglycaemia alerts come with more false alarms.
- **Retrospective smoothing works best.** The smoother beats the tuned time shift (S0, noise 8: -1.9 mg/dL, 95% CI -3.3 to -1.3) and cuts false alarms from 0.93 to 0.23 per day.
- **Knowing the true tau barely helps the real-time filter** (difference about 0.1 mg/dL); noise, not tau uncertainty, is the bottleneck.
- **Per-patient tau from the model likelihood is confidently wrong.** 95% intervals cover the truth in only 4 to 19% of patients, because the generic glucose model is misspecified for meal-driven glucose.
- **A population lag chosen from fingersticks works.** Using about 264 training fingersticks (no ground truth), it picked tau = 16 min when the sensor added 5 min of unmodelled lag, cutting smoother error from 13.5 to 5.8 mg/dL (true-lag oracle 4.5).
- **Physiological and sensor lag cannot be separated** from these data; only their sum (an effective lag) is identifiable.
- **A meal/insulin-driven prior (Piece 6) helps the state estimate but does not fix lag identifiability.** It improves the smoother in all four tested conditions and the real-time filter by about 5 mg/dL at noise sd 8, but per-patient lag intervals still cover the truth in only 19 to 37% of patients.
- **An AR(1) sensor-noise state (Piece 8) fixes the uncertainty, not the accuracy.** Smoother 95% intervals cover about 0.9 of the time with it and 0.5 to 0.7 without, while RMSE does not improve.

Figures: [scenarios](results/fig_scenarios.png), [baselines](results/fig_baselines.png), [Kalman estimators](results/fig_kalman.png), [identifiability](results/fig_ident.png), [Pieces 6 to 8](results/fig_pieces678.png).

## Repository layout

```
code/        sensor layer, evaluation harness, Kalman/smoother, identifiability (modules, tests, run and plot scripts)
data/        sim.npz: 30 simglucose virtual patients x 5 days, 1-min plasma and subcutaneous glucose
results/     result JSON files, figures, split manifest, likelihood profiles
prototype/   first prototype (EKF with online tau) from chat 1, with its results and figures
docs/        project brief, prototype handoff, one note per piece (methods, tests, numbers, caveats), LaTeX versions in docs/tex/, OhioT1DM request draft
run_all.sh   runs every test and experiment in order
```

| Piece | What it adds | Code | Note |
| --- | --- | --- | --- |
| 1 | Verification of the prototype | `prototype/` | [piece-1](docs/piece-1-prototype-verification.md) |
| 2 | Realistic sensor layer, scenarios S0 to S5 | `sensor.py` | [piece-2](docs/piece-2-sensor-layer.md) |
| 3 | Evaluation harness, baselines B0 to B3 | `harness.py` | [piece-3](docs/piece-3-evaluation-harness.md) |
| 4 | Kalman filter, RTS smoother, regularised inversion | `kalman.py` | [piece-4](docs/piece-4-kalman-estimators.md) |
| 5 | Identifiability of tau, fingerstick tuning | `ident.py` | [piece-5](docs/piece-5-identifiability.md) |
| 6 | Meal- and insulin-driven prior | `inputs.py` | [piece-6](docs/piece-6-input-prior.md) |
| 7 | OhioT1DM loader and fingerstick check (not yet run on real data) | `real.py` | [piece-7](docs/piece-7-real-data.md) |
| 8 | Ablations, posterior coverage, final report | `run_ablate.py`, `run_coverage.py` | [piece-8](docs/piece-8-ablations.md), [final report](docs/tex/final-report.tex) |

Scenarios: S0 matched first-order lag; S1 extra 3-min sensor filter and 2-min delay; S2 two-stage lag; S3 tau drifting about 25%; S4 calibration drift and compression lows; S5 all of these together.

## How to run

Python 3.10 or newer.

```bash
pip install -r requirements.txt
bash run_all.sh            # tests (about 1 min) then all experiments (about 15 min on a laptop)
```

Or step by step from `code/`:

```bash
cd code
python3 test_sensor.py ../data/sim.npz
python3 test_harness.py ../data/sim.npz
python3 test_kalman.py ../data/sim.npz
python3 test_ident.py
python3 run_kalman.py ../data/sim.npz ../results && python3 plot_kalman.py ../results/kalman_results.json ../results/fig_kalman.png
```

`data/sim.npz` can be regenerated with `cd prototype && python3 run_gen.py` (needs `simglucose`, about 7 min). That writes to `data/sim.npz` and overwrites the shipped copy.

## Caveats

- Simulation only. In simglucose the lag is exactly first-order, so matched-scenario results are best-case; scenarios S1 to S5 add deliberate mismatch.
- Tuned time-shift and "population tau from truth" results use ground-truth glucose and are simulation-only references.
- Sensor noise levels are anchored loosely to one published accuracy study; noise sd 8 mg/dL is the headline stress level.
- Hypoglycaemia is over-represented in the simulated patients; three virtual children were excluded as unphysiological.

## Next steps

- Get the OhioT1DM files (draft request in `docs/ohiot1dm-request.md`) and run `code/run_real.py`.
- Make the meal/insulin logs noisy and incomplete in simulation, to see how much of the Piece 6 gain survives.
- Estimate the population effective lag from fingersticks on real data and compare with the simulated value.

## Key references

1. A. Basu et al., "Time lag of glucose from intravascular to interstitial compartment in humans," *Diabetes*, vol. 62, no. 12, pp. 4083-4087, 2013. https://doi.org/10.2337/db13-1132
2. G. Schmelzeisen-Redeker et al., "Time delay of CGM sensors: relevance, causes, and countermeasures," *J. Diabetes Sci. Technol.*, vol. 9, no. 5, pp. 1006-1015, 2015. https://doi.org/10.1177/1932296815590154
3. K. Rebrin, G. M. Steil, W. P. van Antwerp, J. J. Mastrototaro, "Subcutaneous glucose predicts plasma glucose independent of insulin," *Am. J. Physiol. Endocrinol. Metab.*, vol. 277, no. 3, pp. E561-E571, 1999. https://doi.org/10.1152/ajpendo.1999.277.3.e561
4. B. W. Bequette, "Continuous glucose monitoring: real-time algorithms for calibration, filtering, and alarms," *J. Diabetes Sci. Technol.*, vol. 4, no. 2, pp. 404-418, 2010. https://doi.org/10.1177/193229681000400222
5. A. Facchinetti et al., "Modeling the glucose sensor error," *IEEE Trans. Biomed. Eng.*, vol. 61, no. 3, pp. 620-629, 2014. https://doi.org/10.1109/TBME.2013.2284023
6. C. Dalla Man et al., "The UVA/PADOVA type 1 diabetes simulator: new features," *J. Diabetes Sci. Technol.*, vol. 8, no. 1, pp. 26-34, 2014. https://doi.org/10.1177/1932296813514502
7. J. Xie, "simglucose v0.2.1," 2018. https://github.com/jxx123/simglucose
8. C. Marling and R. Bunescu, "The OhioT1DM dataset for blood glucose level prediction: update 2020," *CEUR Workshop Proc.*, vol. 2675, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/

This is a research and education project. Nothing here is a medical device or a basis for treatment decisions.
