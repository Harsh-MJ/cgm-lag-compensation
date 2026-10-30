# Piece 1: verification of the chat-1 prototype (30 Oct 2026)

Scope: check the inherited simglucose + EKF prototype (`prototype/`) before building on it. The prototype was written by Claude (AI assistant) in an earlier chat; nobody on the team has yet run or reviewed it, so it must not be presented as team work until Piece 3-4 re-implement or sign off each part.

## Checked and confirmed
- Raw-CGM RMSE vs true plasma recomputed directly from `sim.npz` on the 13 held-out patients: 8.71 (NS=0), 9.46 (NS=0.3), 15.08 (NS=1). Matches the handoff.
- Every row of the handoff results table, hypoglycaemia delays and tau-recovery figures matches the saved results JSON files (tau-recovery correlation is 0.0 to 0.36, not "about 0.1 to 0.2").
- Patient exclusions (child#002, #004, #008), 14 tune / 13 test split and tau range (6.6 to 19.9 min, mean 11.9, sd 3.2) reproduce.
- simglucose 0.2.11 source: subcutaneous state obeys dx12/dt = -ksc*x12 + ksc*x3, so the simulator lag is exactly first-order with tau = 1/ksc.
- CGM samples update at minute indices = 2 (mod 3), which matches the `arange(2, N, 3)` sampling in `evaluate.py`; no staleness offset.

## Resolved open item
- Handoff said the 0.98 lag-1 autocorrelation of sensor noise was "not verified". Verified: simglucose `noise_gen.py` draws Johnson-transformed noise every 15 min (PACF 0.7) and cubic-interpolates it to the sample time. Measured noise autocorrelation: 0.98 at 3 min, 0.72 at 15 min, 0.47 at 30 min, 0.24 at 60 min. So the simulator noise is slow and drift-like with no fast component; "noise dominates at NS=1" is partly a property of this noise model, not of real CGM.

## Issues to fix (carried into later pieces)
1. Noise model has no fast component and is unrealistic for real sensors (Piece 2).
2. Several tuned process-noise values sit at the grid edge (qr = 0.3 at NS=0 for every EKF variant); widen the grid (Piece 3).
3. Noise sd and phi are estimated from (cgm - true subcutaneous glucose), which needs ground truth; fine in simulation, not available on real data.
4. At NS=0 online-tau EKF (RMSE 5.05) is worse than fixed tau=10 (2.36), and tau error (57-63%) is worse than using the 10-min prior (22%). Treat as an open problem and a tuning question, not a settled result.
5. Lag is exactly first-order in the generator, so all results are best-case (no mismatch yet).
6. `fc30_*` forecast metrics are not meaningful; do not report.
7. Time below 70 mg/dL on the test set is 4.6% (handoff quotes about 6.6% overall); minor.

## Differences from the project plan doc
- Prototype samples at 3 min (Dexcom model in simglucose), plan assumes 5 min; run 1, 5 and 15 min sensitivity in Piece 3.
- Prototype uses a nonlinear EKF with ln(tau) in the state; plan's core method is a linear Kalman filter and RTS smoother with a regularised-inversion cross-check. Keep the EKF as a comparison.
- Plan's hypotheses H1 to H4 are not yet tested on mismatched generators (G2, G3).

## Proposed pieces
1. Verify prototype (this note).
2. Realistic sensor layer and mismatch generators (fast noise, extra delay, drift, compression lows).
3. Evaluation harness: baselines, widened tuning, subject-wise splits, bootstrap intervals, analytic ramp test.
4. Linear Kalman, RTS smoother and regularised inversion; cross-check.
5. Identifiability study of tau (profile likelihood, population prior, fingerstick-timed regression).
6. Meal and insulin-aware prior (optional).
7. OhioT1DM access and exploratory real-data check.
8. Ablations, final results, report.
