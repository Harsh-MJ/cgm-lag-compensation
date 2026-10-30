# Physics-informed lag compensation for CGM: project brief and prototype findings

## Idea
Interstitial (sensor) glucose lags plasma glucose by a diffusion delay. Model it as first-order transport and estimate plasma glucose with a state estimator instead of a black-box forecaster:

$$\tau \frac{dC_{isf}}{dt} = C_{bg} - C_{isf} \;\Rightarrow\; C_{bg} \approx C_{isf} + \tau \frac{dC_{isf}}{dt}$$

Worked check: CGM reads 90 mg/dL, falling 1.5 mg/dL/min, $\tau = 10$ min -> $C_{bg} \approx 90 + 10 \times (-1.5) = 75$ mg/dL.

## Prototype (simulation only)
- Data: `simglucose` UVA/Padova virtual patients (30; 3 excluded as unphysiological -> 27; 5 days each), meals sized from each patient's CR and TDI with +/-bolus error. Ground truth = plasma glucose; CGM = subcutaneous glucose (true $\tau = 1/k_{sc}$, 6.6 to 19.9 min) + simulator Dexcom noise.
- Split by patient: 14 tuning / 13 test. Estimators: raw CGM, fixed-tau inversion, EKF (fixed tau = 10 min), EKF with online tau (+/- 4 fingersticks/day), EKF with true tau (oracle). EKF state = [G, dG/dt, I, ln tau, AR(1) sensor noise].

## Results (13 held-out virtual patients, RMSE vs plasma, mg/dL)
| Sensor noise scale | Raw CGM | Fixed-tau inversion | EKF fixed tau=10 | EKF online tau + FS | EKF true tau |
|---|---|---|---|---|---|
| 0 (lag only) | 8.71 | 2.66 | 2.36 | 5.05 | 0.93 |
| 0.3 | 9.46 | 5.30 | 5.26 | 7.48 | 5.14 |
| 1 (sd ~12 mg/dL) | 15.08 | 15.50 | 15.56 | 14.48 | 15.56 |

Hypoglycemia (<70) detection delay, noise scale 0: raw 12.4 min late; fixed-tau EKF 1.2 min; true-tau EKF -0.3 min.
At noise scale 1 the earlier alerts cost more false alarms (raw 0.36/day; fixed-tau EKF 0.87/day).

## Findings
1. With a roughly correct tau, lag compensation removes most lag error (RMSE -73% at noise 0; delay 12 -> ~1 min).
2. At simulator-level noise (slow, lag-1 autocorrelation 0.98, sd ~12 mg/dL) no estimator beats raw CGM on pooled RMSE; the noise dominates.
3. Online tau estimation fails: median error 57-82% vs 22% for just using the 10-min prior; 4 fingersticks/day did not fix it. tau is weakly identifiable when glucose has no dynamic model.

## Caveats
Simulation only (lag model is exactly first-order in the simulator, so results are optimistic); process-noise grid edge hit for some EKF variants; some virtual children had severe simulated hypoglycemia (excluded); no real-data (OhioT1DM) validation yet.

## Plan
1. Weeks 1-2: harden simulation (model mismatch: unmodelled extra delay, calibration drift, compression lows; realistic noise); tune process noise properly.
2. Weeks 3-4: make tau identifiable (add meal/insulin inputs with a minimal glucose-insulin model as the glucose prior; joint estimation with fingersticks; hierarchical population prior over tau).
3. Weeks 5-6: validate on OhioT1DM (needs data use agreement; fingersticks as reference) and compare with LSTM/linear forecasters.
4. Week 7: write-up: error vs noise, detection lead time vs false alarms, tau identifiability analysis.
