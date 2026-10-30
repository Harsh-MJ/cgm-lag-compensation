# HANDOFF: DH 803 project, physics-informed lag compensation for CGM

Written 30 Oct 2026 at the end of chat 1. Everything a new chat needs to continue. Read this fully before doing anything.

## 0. How to use this
- New chat should be inside the Claude Project "DH 803" (the code and this brief are stored there under `claude/`). If it is not, upload `cgm_lag_prototype.zip` and this file.
- Project docs: `claude/cgm-lag-compensation-plan.md` (short brief), `claude/HANDOFF-cgm-lag-project.md` (this file), `claude/cgm-lag-code/{gen.py, run_gen.py, est.py, evaluate.py, plots.py}`.
- Large binaries (`sim.npz`, figures, results JSON) are only in the zip. `sim.npz` is reproducible (see section 6).

## 1. Project context (as stated by Harsh)
- Course: DH 803, wearable health technology. Constraint: project must need no hardware.
- Group project, 2 members: Harsh Jadhav (Chemical Engineering, IIT Bombay) and Shreyas Kamble (Mechanical).
- Chosen topic: **Physics-informed lag compensation for CGM**, a transport-based state estimator instead of black-box glucose forecasters. Tools: simglucose (UVA/Padova simulator), OhioT1DM (needs a data use agreement).
- Status stated for Assignment 2: literature-review stage. Final deliverables: a document and a PPT.
- Attribution rule: keep each member's contribution accurately attributed; never present a teammate's work as Harsh's own, and do not claim work neither of them did. The prototype in section 3 to 5 was produced by Claude in chat 1 at Harsh's request and must be described as such or re-done/verified by the team before being claimed.

## 2. Idea and model
Interstitial (sensor) glucose $C_{isf}$ lags plasma glucose $C_{bg}$ by diffusion. First-order transport model:

$$\tau \frac{dC_{isf}}{dt} = C_{bg} - C_{isf} \;\Rightarrow\; C_{bg} \approx C_{isf} + \tau \frac{dC_{isf}}{dt}$$

Worked check: CGM reads 90 mg/dL, slope $-1.5$ mg/dL/min, $\tau = 10$ min gives $C_{bg} \approx 90 + 10\times(-1.5) = 75$ mg/dL.

Naive inversion amplifies noise (differentiates it), so use a state estimator (EKF). Core research problem: **$\tau$ is patient-specific and, from CGM alone, weakly identifiable.**

## 3. Simulator facts (simglucose 0.2.11, `pip install --break-system-packages simglucose`)
- 30 virtual patients: adolescent#001-010, adult#001-010, child#001-010.
- In the patient ODE, `x[3]` = plasma glucose mass (mg/kg), plasma glucose = `x[3]/Vg`; `x[12]` = subcutaneous glucose; `dx12/dt = -ksc*x12 + ksc*x3`. So the lag is exactly first-order with true $\tau = 1/k_{sc}$. Across patients $\tau$ = 6.6 to 19.9 min (mean 11.9, sd 3.2).
- `patient.observation.Gsub` is the SUBCUTANEOUS glucose, not plasma. simglucose's own `BG` output is therefore already lagged. Ground truth plasma must be taken from `patient.state[3]/Vg`.
- Dexcom sensor: sample time 3 min, noise from `CGMNoise` (Johnson transform, PACF 0.7). Measured noise sd about 12 mg/dL; **lag-1 autocorrelation at 3-min sampling is 0.98** (slow, drift-like; probably because the generator works on a coarser grid and interpolates; not verified). This noise dominates the lag error.
- Quest.csv gives CR, CF, TDI per patient. Basal = `u2ss*BW/6000` U/min.

## 4. Pipeline as built (code in `claude/cgm-lag-code/`)
1. `gen.py::run(name, days, seed)`: 1-min steps. Meals sized from CR and TDI: breakfast/lunch/dinner shares 0.13/0.17/0.20 of TDI as insulin-equivalent (grams = CR*TDI*share*U(0.8,1.2)) at about 07:30/12:30/19:00 (time jitter), snack on 50% of days (0.04*CR*TDI*U(0.5,1.5)), bolus = grams/CR * clip(N(0.92, 0.18), 0.5, 1.4). Returns plasma, cgm, sub, cho, ins, tau.
2. `run_gen.py`: 30 patients x 5 days, seed `100+i`, multiprocessing, writes `../data/sim.npz` (keys `name|field`). About 6.5 min on 2 cores.
3. `est.py`: `local_fit` (causal local linear regression), `inversion`, `ekf`.
   - EKF state $[G, \dot G, I, \ln\tau, e]$; $G$ random-walk-rate model ($\dot G$ random walk with variance `qr` per min), $I$ follows $\dot I = (G-I)/\tau$ (3 Euler sub-steps of 1 min per 3-min sample), measurement $y = I + e$, $e$ is AR(1) sensor noise (sd from tuning data, $\phi$ from data), small extra R=1. Optional fingerstick update on $G$: simulated 4/day (about 07:00, 11:30, 17:30, 22:00, jitter sd 15 min) with 4% CV noise; the filter assumes 5% CV plus 2 mg/dL.
   - `free_tau=True`: $\ln\tau$ random walk (q_theta = 0.005^2 per min, prior sd 0.4, prior mean ln 10, clipped to 2..40 min).
4. `evaluate.py` (env `NS` = sensor-noise scale: `y = Gs + NS*(cgm - Gs)`): see section 5 for protocol. Outputs `../data/results[_nsX].json`, `preds[_nsX].pkl`, `tuned[_nsX].pkl`. About 3 min per noise level.
5. `plots.py`: fig1 (RMSE vs noise), fig2 (example trace), fig3 (tau recovery).
Run order: `python3 run_gen.py` then `python3 evaluate.py` (NS=1), `NS=0 python3 evaluate.py`, `NS=0.3 python3 evaluate.py`, `python3 plots.py`. Scripts must run from the scripts directory (relative `../data`, `../figs`). Create `../data` and `../figs` first.

## 5. Evaluation protocol and results
- Patients excluded as unphysiological (min plasma <30 mg/dL or >15% of time <54): child#002, child#004, child#008 -> 27 used. Alternating split in sorted order: 14 tune / 13 test. Test = adolescent#002,004,006,008,010; adult#002,004,006,008,010; child#003,006,009.
- 6 h warm-up (120 samples) dropped. Tuning criterion: RMSE vs true plasma on tune patients (uses ground truth; in real data tune against fingersticks).
- Hypoglycemia event: plasma <70 mg/dL, events merged if gaps <30 min. Detected if estimate <70 within [event start - 30 min, event end]; delay = first alarm minus true start (negative = early). False alarm: alarm onset with no true <70 within +/-30 min.
- Tuned hyperparameters: NS=1: inversion W=3; ekf_fix qr=0.01; ekf_fix_fs 0.003; ekf_est 0.3; ekf_est_fs 0.3; oracle 0.001. NS=0: inversion W=3; all EKF variants qr=0.3. **Several sit at the edge of the grid (1e-3..3e-1); widen the grid.**

RMSE vs true plasma (mg/dL), 13 held-out patients:

| Noise scale | Raw CGM | Fixed-tau inversion | EKF tau=10 | EKF online tau + fingersticks | EKF true tau (oracle) |
|---|---|---|---|---|---|
| 0 | 8.71 | 2.66 | 2.36 | 5.05 | 0.93 |
| 0.3 | 9.46 | 5.30 | 5.26 | 7.48 | 5.14 |
| 1 | 15.08 | 15.50 | 15.56 | 14.48 | 15.56 |

Hypoglycemia detection (26 events in test set), mean delay in min / false alarms per day:
- NS=0: raw 12.4 / 0.00; inversion 1.0 / 0.00; EKF tau=10 1.2 / 0.00; EKF online tau + fingersticks 7.3 / 0.00; oracle -0.3 / 0.00.
- NS=1: raw 4.7 / 0.36; inversion -7.5 / 1.33; EKF tau=10 -6.3 / 0.87; EKF online tau 1.8 / 0.44; EKF online + fingersticks -0.7 / 0.63; oracle -5.1 / 0.66.

Tau recovery (median absolute % error of mean $\hat\tau$ over last day): NS=0: 63% (no fingersticks), 57% (with); NS=1: 82% both. Simply using the 10-min prior: 22%. Correlation $\hat\tau$ vs true $\tau$ about 0.1 to 0.2.

Independent sanity checks done: recomputed raw RMSE (8.71) and oracle RMSE (0.93) at NS=0 directly from saved arrays.

## 6. Findings
1. With roughly correct $\tau$, lag compensation removes most lag error (RMSE -73%, hypo detection delay about 12 min to about 1 min at NS=0).
2. At full simulator noise nothing beats raw CGM clearly on RMSE; earlier alerts cost more false alarms.
3. Online $\tau$ estimation fails (no better than the prior). This is the open research problem.

## 7. Caveats and known issues
- Simulation only; lag is exactly first-order in the model, so results are optimistic. No real-data validation.
- Version history: first data run overdosed children (meals not scaled) and had a MARD bug caused by plasma near 0; fixed by CR/TDI-scaled meals, patient exclusion and a MARD mask (plasma >= 40).
- Hypoglycemia is over-represented (about 6.6% of simulated time <70 mg/dL).
- Naive 30-min forecast (`rate*30`) lost to persistence; `fc30_*` metrics in the JSONs are not meaningful. Do not report them.
- Process-noise grid edge hit; online-$\tau$ failure may be partly tuning.
- No literature has been checked yet; every claim about published sensor lag or published Kalman-filter CGM work still needs a source.

## 8. Next steps (proposed)
- Phase 1: harden the simulation with model mismatch (extra unmodelled sensor delay, calibration drift, compression lows, realistic noise with a fast component); widen and re-tune qr properly.
- Phase 2 (the contribution): make $\tau$ identifiable. Options: meal/insulin inputs through a minimal glucose-insulin model as the glucose prior; hierarchical/population prior over $\tau$ (e.g., log-normal) with per-patient shrinkage; fingerstick-timed regression of $(G - I)$ on $\dot I$; compare against an LSTM baseline.
- Phase 3: OhioT1DM validation (request data use agreement now), fingersticks as reference.
- Assignment 2 (literature review): topics to search and verify: blood-to-interstitial glucose kinetics and time-lag; Kalman-filter based CGM calibration/denoising and lag correction (e.g., Knobbe and Buckingham; Facchinetti et al.); UVA/Padova model (Dalla Man, Kovatchev); OhioT1DM dataset papers (Marling and Bunescu); CGM accuracy metrics and Clarke/Parkes error grids. These names are from memory and unverified: confirm each before citing.

## 9. Working preferences for the next chat
Concise, structured: bold key terms, bullets, tables; inline math with single `$...$`, display equations with `$$...$$` (valid LaTeX only); stepwise calculations for worked examples; cite a source for every number or fact in any market/literature material; no interactive widgets unless asked.

## 10. Starter prompt to paste in the new chat
"Continue the DH 803 project 'Physics-informed lag compensation for CGM'. Read `claude/HANDOFF-cgm-lag-project.md` and the code in `claude/cgm-lag-code/` in this Project first. Then [state the task: e.g., build phase 2 (identifiable tau with meal/insulin-aware estimator) / write the Assignment 2 literature review / prepare the document and PPT]."
