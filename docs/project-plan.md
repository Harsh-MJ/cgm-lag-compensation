# Physics-Informed Lag Compensation for CGM — Course Project Plan

Oct 30, 2026 · @Harsh Jadhav

## Assumptions and how to read this plan

This plan assumes a three-person master's-level team working in Python on public data and simulation, with no clinical access. Change any row below and the scope scales as noted in Section 4.

| Item | Assumption |
| --- | --- |
| Course level | Master's (graduate-numbered course); advanced undergraduates can follow the minimum viable version |
| Team size | 3 students (2 students: drop the neural-network extension; 4: deepen the real-data route) |
| Duration | 14-week semester |
| Software | Python 3.10+, NumPy, SciPy, pandas, scikit-learn, Matplotlib, simglucose; PyTorch only for the optional extension; Git |
| Compute | Laptop CPU is enough; no GPU required |
| Data | No clinical site and no reference glucose analyser; public datasets and simulation only |
| Units and sampling | Glucose in mg/dL, time in minutes, CGM sampled every 5 min unless stated |

**Evidence labels.** Statements with a bracketed number such as \[1\] are sourced facts; the numbers point to the References section, which also states how each citation was verified. Anything marked *(proposed)* is a recommendation or design choice, not an established result. Reference entries marked ⚠ could only be partly verified.

## 1. Project overview

The project asks when a physics-based model of interstitial-fluid (ISF) glucose kinetics can undo CGM lag, and when the lag cannot be separated from sensor and filtering effects.

**Proposed title:** Physics-Informed Lag Compensation for Continuous Glucose Monitoring: Identifiability and Benchmarking of Interstitial-Kinetics Estimators on Simulated and Public Data

**Abstract *(proposed)*.** CGM sensors measure glucose in ISF, so readings trail blood glucose when it changes quickly. Controlled studies report delays of roughly 4–10 minutes \[1\]\[2\]\[6\]\[7\], and part of that delay comes from sensor chemistry and on-device filtering rather than physiology \[3\]. This project builds a state-space estimator of blood glucose from CGM data using a first-order ISF kinetic model. It benchmarks the estimator against no correction, a fixed time shift and causal smoothing, using a UVa/Padova-based simulator with known ground truth and, where licences allow, public CGM datasets. The central output is a map of conditions under which error falls and the lag parameter is identifiable. Results are research findings only; nothing here is a medical device.

**Problem statement.** CGM output is a delayed, noisy, calibrated view of blood glucose. Because blood glucose is not observed in daily-use data, the delay cannot be measured directly and must be inferred through a model whose assumptions may not hold.

**Motivation.** Lag distorts accuracy assessment as well as readings: the choice of compensation method changes the apparent accuracy of the same CGM data \[14\]. Lag also matters for closed-loop systems, where delays limit how fast a controller can respond \[1\]\[4\]. Finally, published lag estimates differ widely, and some of the variation is attributed to the sensing systems used to measure it \[4\]\[10\].

### What "lag compensation" means here

In this project, lag compensation is the estimation of a lag-reduced blood-glucose trajectory from CGM samples using a dynamic model of the delays. Its output is judged against a known (simulated) or reference blood-glucose signal.

The total delay between a blood-glucose value and its CGM display is the sum of the plasma-to-ISF lag, the electrochemical sensor delay and front-end signal-processing delay \[3\]. The four components differ in origin and in how far they can be corrected:

| Component | Origin | Reported magnitude | Correctable from CGM data alone? |
| --- | --- | --- | --- |
| Physiological plasma-to-ISF lag | Capillary transport, ISF diffusion and local uptake, modelled as a first-order process \[4\]\[9\] | 5.3–6.2 min in fasted healthy adults \[1\]; 6–8 min around insulin-induced hypoglycaemia \[6\] | Partly, with model-based inverse filtering; limited by noise \[4\]\[16\] |
| Sensor (electrochemical) delay | Membrane diffusion and enzyme reaction \[17\] | In vitro 95% response generally under 1.5 min \[10\]; in vivo "a few minutes" assumed, little published \[2\] | No; confounded with physiology unless a reference is available |
| Signal-processing delay | On-device smoothing and filters \[3\] | Device-specific, rarely published \[2\] | Yes if the filter is known; otherwise confounded |
| Alignment offset | Clock mismatch or resampling between CGM and reference | Study-specific *(proposed category)* | Yes, by protocol and timestamp checks |

A first-order lag is a time constant, which differs from a pure transport delay; transport delay is harder to correct in real time but easier to identify retrospectively \[4\].

### Why it matters, and what it does not mean

Lag matters mostly when glucose changes quickly, such as after meals or insulin. The example below shows the size of the effect in a simple case.

**Worked example (illustrative values, *assumed*).** Blood glucose rises on a ramp at a = 2 mg/dL/min, and the ISF lag is a first-order process with time constant τ = 10 min, of the order of the 9.5 min mean raw-signal delay in \[2\].

1. Model: τ · dG\_I/dt = −G\_I + G\_B, where G\_B is blood glucose and G\_I is ISF glucose.
2. For a ramp G\_B = G\_0 + a·t, the steady-state solution is G\_I = G\_0 + a·(t − τ).
3. Error: e = G\_B − G\_I = a·τ = 2 × 10 = 20 mg/dL.
4. At G\_B = 140 mg/dL the relative error is 20 / 140 = 14.3%.

A 14% error from lag alone is not small next to the MARD of about 9–10% reported in one recent clinical accuracy study \[32\], so lag matters when comparing methods \[14\].

**Limits of this project.** It is a course-level methods study on simulated and de-identified public data. It does not show that any algorithm is safe, accurate enough for treatment, or suitable for display to a person with diabetes.

## 2. Background and literature review

CGM lag is the sum of slow physiological transport and device-specific delays, and published estimates disagree partly because those parts are hard to separate \[3\]\[4\]\[10\].

### Physiology and the measurement chain

Glucose moves from plasma across the capillary wall into ISF and is cleared by surrounding cells; this is commonly described by a two-compartment model with rate constants for exchange and for ISF uptake \[6\]\[9\]. A subcutaneous enzymatic sensor produces a current related to ISF glucose, and a calibration step converts that current to mg/dL \[3\]\[16\].

Facchinetti and colleagues decompose CGM error into three parts: distortion from blood-to-interstitial diffusion, a time-varying calibration and drift term, and autoregressive measurement noise \[11\]. Using several simultaneous sensors, the same group separated physiology-related from technology-related error components \[11\].

### What studies report about delay

Reported numbers are not interchangeable, because the studies define "delay" differently (tracer appearance, equilibration time, or total sensor-versus-reference shift) \[4\]\[30\].

| Study | Setting | Quantity reported | Value |
| --- | --- | --- | --- |
| Basu 2013 \[1\] | 8 healthy adults, overnight fast, microdialysis with glucose tracers | Tracer appearance time in ISF | 5.3–6.2 min; earlier work cited as 4–50 min |
| Steil 2005 \[6\] | 10 non-diabetic adults, insulin-induced hypoglycaemia | ISF equilibration delay | 8.3 ± 0.67 min (fall) vs 6.3 ± 1.1 min (recovery) |
| Boyne 2003 \[7\] | 14 adults with T1D, two sensors each, 8 h with meals | Timing difference, ISF vs venous glucose | 4–10 min; ISF lagged in 81% of cases; two sensors in one person differed by 6.7 ± 5.1 min |
| Rebrin 1999 \[5\] | 4 dogs, hyperglycaemic clamps | ISF equilibration delay | 3.3–12.3 min across conditions; no significant insulin effect |
| Schmelzeisen-Redeker 2015 \[2\] | 37 people, prototype sensor, 108 data sets | Raw-signal delay vs blood glucose | 9.5 min mean (SD 3.7), median 9; patient-dependent |
| Plasma-to-ISF kinetics study 2015 \[30\] ⚠ | Healthy and T1D adults, fasting steady state | Two-compartment "equilibration time" | Median 9.1 min (healthy), 11.0 min (T1D) |
| Aussedat 2000 \[8\] | Anaesthetised rats | ISF vs plasma glucose behaviour | Discrepancies attributed to insulin-driven uptake; differs from \[6\] |

### Modelling and compensation approaches

| Approach | How it treats lag | Evidence in the literature | Main limit |
| --- | --- | --- | --- |
| Two-compartment ODE | Plasma and ISF exchange with rate constants; lag is a time constant \[5\]\[6\]\[9\] | Fits sensor current in dogs and humans \[5\]\[6\] | Parameters vary by person and condition \[2\]\[7\] |
| Time-shift alignment | Shifts CGM earlier by a fixed or variable delay | Compensation method changes apparent CGM accuracy \[14\] | Ignores noise and shape change |
| Inverse (Wiener) filtering | Inverts the lag filter while limiting noise | On 20 sensor traces with added 10 ± 2 min delays, cut delay-related error about 50% with noise and about 75% without \[15\] | Delays were injected into signals, not measured in vivo |
| Kalman filtering | State-space estimate of blood glucose, ISF glucose and rate of change \[16\] | Compensates lag even with substantial noise; Q/R tuning trades lag against noise \[16\] | Needs a good lag and noise model |
| Prediction algorithms | Predict ahead to offset delay | Reduced delay by 4 min on average \[2\] | Patient-specific; prediction adds noise |
| Multi-sensor error models | Separate physiological and technical error \[11\]\[13\] | Dexcom G6 model with 79 traces \[13\] | Needs parallel sensors |
| Physics-informed learning | Embed glucose–insulin dynamics in a network \[27\]\[28\] | 12.33% gain over no-physics model in a small-sample test \[27\] | Aimed at forecasting glucose, not ISF lag |

### Limitations and open questions

- Lag estimates span 4–50 min across studies \[1\], and part of the spread is attributed to the sensing systems and filters used \[4\]\[10\].
- Lag differs between people \[2\] and between two sensors in the same person \[7\], so a single population value is a weak assumption.
- Whether insulin or hypoglycaemia changes ISF dynamics is contested: an animal study suggests it does \[8\], a human study found no protracted delay \[6\].
- Small errors in recalibration or in the assumed blood-to-ISF dynamics can severely distort sensor-error estimates \[12\], which is an identifiability warning.
- Nonlinear on-device filters confound linear deconvolution methods \[4\].
- In my searches, recent physics-informed glucose models target forecasting from glucose–insulin dynamics \[27\]\[28\]; I did not find one aimed at ISF-lag compensation. This is a search-limited observation, not proof of absence.

### Literature-search strategy *(proposed)*

| Element | Plan |
| --- | --- |
| Databases | PubMed/MEDLINE, IEEE Xplore, Scopus or Web of Science, Google Scholar for citation chasing, arXiv and medRxiv for preprints |
| Core strings | ("continuous glucose monitor\*" OR CGM) AND (lag OR "time delay") AND (interstitial OR "blood glucose") |
|  | (interstitial glucose) AND (Kalman OR Wiener OR deconvolution OR "inverse filter" OR "compartment model") |
|  | ("physics-informed" OR "physiology-informed" OR hybrid) AND glucose AND (CGM OR "continuous glucose") |
|  | ("continuous glucose") AND (dataset OR "open data" OR "data descriptor") |
| Date range | 1999–2015 for kinetics, delay and filtering foundations; 2018–2026 for factory-calibrated sensors, physics-informed methods and datasets; re-run just before submission |
| Include | Peer-reviewed papers or clearly labelled preprints; human data or a validated simulator; reports lag, ISF kinetics, a compensation method or a documented dataset; English |
| Exclude | Non-minimally-invasive glucose sensing unless it informs kinetics; forecasting papers with no discussion of lag; abstracts without methods; patents (use only to describe industry practice, labelled as such) |
| Process | Two reviewers screen titles and abstracts, resolve conflicts by discussion, and log counts at each step; build an annotated matrix of device, population, lag definition, method, metrics and data access; target 15–20 cited papers |

**Where to start.** Foundational: \[1\]\[3\]\[4\]\[5\]\[6\]\[7\]\[9\]\[11\]. Method papers: \[14\]\[15\]\[16\]. Recent and data: \[13\]\[21\]–\[28\]. Every entry in the References section states how it was verified.

## 3. Research gap, questions and hypotheses

The gap is not another lag filter but a controlled comparison of lag-compensation methods that shows when the lag parameter can be recovered and when apparent gains come only from a favourable simulation.

**Research gap *(proposed framing)*.** Existing compensation methods such as Wiener, Kalman and time-shift approaches each assume a lag value or estimate it from reference data \[14\]\[15\]\[16\]. Reported lag values differ widely and mix physiological, sensor and filter effects \[4\]\[10\]. Evidence that a method works often comes from a single sensor, injected delays \[15\], or a model that matches the estimator. A reproducible benchmark with known ground truth, deliberate model mismatch and an explicit identifiability analysis is the missing piece this project targets.

**Primary research question.** Does a physics-based ISF-kinetics estimator reduce blood-glucose estimation error and improve trend and timing metrics relative to no compensation, a fixed time shift and causal smoothing, and under which noise, sampling and model-mismatch conditions can its lag parameter be identified?

**Supporting questions.**

1. How quickly does the benefit degrade when the assumed lag is wrong, or when the true ISF dynamics differ from the estimator's model?
2. Can the physiological lag be separated from sensor and filtering delay using CGM data alone, or only the combined effective lag?
3. What is the trade-off between lag reduction and noise amplification for a real-time (causal) estimator, compared with a retrospective (non-causal) one?

### Hypotheses and how each is tested

Thresholds below are pre-registered targets *(proposed)*. The 30% target is deliberately below the roughly 50% reduction in delay-related error that a Wiener filter achieved with noise in \[15\], to allow for model mismatch.

| ID | Hypothesis | Test | Success criterion |
| --- | --- | --- | --- |
| H1 | With the correct model class, the Kalman estimator lowers error versus raw CGM during rapid-change periods | Simulation; paired per-subject RMSE on segments with rate of change at least 1 mg/dL/min | Median RMSE reduction of at least 30%; paired Wilcoxon test, Holm-corrected, p < 0.05 |
| H2 | An estimated, subject-specific lag beats a fixed population lag when true lags differ across subjects | Simulation; draw true lag from a 5–15 min range; compare estimated vs fixed value | Lower median RMSE with a non-overlapping bootstrap interval on the paired difference |
| H3 | The combined effective lag is recoverable, but its physiological part is not, from CGM alone | Simulation; profile likelihood over lag, filter constant and delay | Effective-lag error of 2 min or less at the best signal-to-noise setting; flat likelihood along the physiology-versus-filter direction |
| H4 | Under deliberate model mismatch, gains shrink but remain positive for lags up to ±50% wrong | Simulation with a different ISF generator than the estimator | Positive median RMSE reduction at ±25% lag error; report where it crosses zero |
| H5 *(exploratory)* | On real data the compensated trace agrees better with sparse fingerstick values during rapid change | Public data with fingersticks, leave-one-subject-out | Direction and effect size only; no confirmatory claim, because fingersticks are sparse and may have been used for device calibration ⚠ |

## 4. Project scope and objectives

The aim is a reproducible simulation-first comparison of lag-compensation methods, with a smaller exploratory check on public real data. All items below are proposed.

**Aims.** (A1) Quantify how much a physics-based estimator reduces lag-related error versus simple baselines. (A2) Establish when the lag parameter is identifiable. (A3) Document failure modes honestly.

### Measurable objectives

| ID | Objective | Measure of completion | Week |
| --- | --- | --- | --- |
| O1 | Simulation generator with known blood glucose, ISF glucose and CGM output | 30 virtual subjects, at least 3 meal scenarios each, fixed seeds, unit tests pass | 4 |
| O2 | Baselines B0–B3 and evaluation harness | All metrics in Section 6 computed on a toy case with hand-checked values | 4 |
| O3 | Kalman filter and smoother | Reproduces the analytic ramp lag (a · τ) of the Section 1 example within 1% | 6 |
| O4 | Lag and noise parameter estimation with identifiability analysis | Profile-likelihood plots for lag, filter constant and delay | 8 |
| O5 | Mismatch and sensitivity study | Full grid in Section 6 run with bootstrap intervals | 10 |
| O6 | Exploratory real-data check | At least one dataset with documented access, subject-wise results | 11 |
| O7 | Reproducible repository | One command regenerates every figure and table on a laptop | 13 |

### Boundaries, assumptions and exclusions

- **In scope:** adult-like type 1 diabetes simulation; a lumped model of ISF lag, sensor filtering and pure delay; 5-minute sampling, with 1- and 15-minute sensitivity runs.
- **Assumptions:** the simulator's blood glucose is ground truth; sensor noise is autoregressive, following the structure in \[11\]; calibration is treated as a slowly varying gain and offset.
- **Exclusions:** insulin-dose advice, glucose forecasting as a product goal, hardware or clinical data collection, regulatory or usability work, paediatric or type 2 conclusions.

### What the project will and will not claim

| Will claim | Will not claim |
| --- | --- |
| Relative error reduction versus named baselines under stated simulation settings | That any method is accurate or safe enough for treatment decisions |
| Conditions under which the effective lag is or is not identifiable | That the estimated lag is the true physiological lag in real people |
| Sensitivity to wrong lag values, noise, sampling interval and model mismatch | That simulation gains transfer to real sensors |
| Exploratory direction of effect on public real data | Confirmatory real-world performance from sparse fingersticks |

### Minimum viable version and extensions

| Tier | Content | Choose when |
| --- | --- | --- |
| Minimum viable | O1–O3, hypotheses H1, H2 and H4, baselines B0–B3, simulation only | 2 students, or any plan that slips by 3 or more weeks |
| Standard | MVP plus O4 (H3) and O6 (H5 with one dataset) | 3 students, full 14 weeks |
| Extended | Standard plus a physics-informed neural network (Section 5), insulin-dependent ISF clearance in the simulator, or a second real dataset | 4 students or a strong team with spare capacity |

## 5. Methodology

The recommended method is a state-space (Kalman) estimator built on a first-order ISF lag model, tested first on simulation with known truth and then on public data; a physics-informed neural network is an optional extension.

### 5.1 Signal model

The plasma-to-ISF exchange is described by the two-compartment model used in the sensor-kinetics literature \[6\]\[9\]. Here G\_B is blood (plasma) glucose, G\_I is ISF glucose, V\_1 and V\_2 are compartment volumes, K\_21 and K\_12 are exchange rate constants and K\_02 is ISF clearance.

```latex
\frac{d\,(V_2 G_I)}{dt} = K_{21} V_1 G_B - (K_{12}+K_{02})\, V_2 G_I
```

Dividing by V\_2 gives a first-order lag with time constant τ and gain g:

```latex
\tau\,\frac{dG_I}{dt} = -G_I(t) + g\,G_B(t), \qquad \tau = \frac{1}{K_{12}+K_{02}}, \qquad g = \frac{K_{21}V_1}{V_2\,(K_{12}+K_{02})}
```

The sensor adds on-device filtering, a pure delay, calibration and noise *(proposed model structure, following the error components in \[11\])*:

```latex
\tau_f\,\frac{dG_F}{dt} = -G_F(t) + G_I(t), \qquad y_k = \alpha\,G_F(t_k - d) + \beta + \varepsilon_k, \qquad \varepsilon_k = \phi\,\varepsilon_{k-1} + w_k, \quad w_k \sim \mathcal{N}(0,\sigma_w^2)
```

For slowly varying glucose, the group delays of cascaded stages add, so CGM data alone constrain only the combined effective lag:

```latex
\tau_{\mathrm{eff}} \approx \tau + \tau_f + d
```

This sum is the root of the identifiability question in H3. The product α·g is also not separable when G\_B is unobserved, so the MVP assumes a locally calibrated sensor (α·g ≈ 1, β ≈ 0) and tests calibration drift as an ablation.

| Symbol | Meaning | Unit | How obtained |
| --- | --- | --- | --- |
| G\_B(t) | Blood glucose (ground truth in simulation) | mg/dL | Simulator output |
| G\_I(t) | ISF glucose | mg/dL | Generated by the ISF layer |
| G\_F(t) | ISF glucose after on-device filter | mg/dL | Generated by the sensor layer |
| τ | Plasma-to-ISF time constant | min | Drawn per subject in simulation (5–15 min) |
| g | Steady-state ISF-to-plasma gain | none | Set to 1 in the MVP |
| τ\_f | On-device filter time constant | min | Drawn in simulation; unknown in real data |
| d | Pure processing or transport delay | min | Drawn in simulation (0–5 min) *(proposed)* |
| τ\_eff | Combined effective lag | min | Estimated; main target of H3 |
| α, β | Sensor gain and offset after calibration | none, mg/dL | Fixed to 1 and 0, then perturbed in ablations |
| ε\_k, φ, σ\_w | Autoregressive noise, its coefficient and innovation s.d. | mg/dL, none, mg/dL | Drawn in simulation; estimated from data |
| Δt | Sampling interval | min | 1, 5 or 15 |

### 5.2 Estimators

**M1: Kalman filter and RTS smoother (core method).** Following the state structure in \[16\], the state holds ISF glucose, blood glucose and blood-glucose rate of change R, which is modelled as a random walk. The lumped lag τ\_eff stands in for τ in the MVP.

```latex
\frac{d}{dt}\begin{bmatrix} G_I \\ G_B \\ R \end{bmatrix} = \begin{bmatrix} -1/\tau & 1/\tau & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}\begin{bmatrix} G_I \\ G_B \\ R \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} q\,\eta(t), \qquad y_k = G_I(t_k) + \varepsilon_k
```

Discretise with a matrix exponential at the sampling interval; skip the update step when a sample is missing. The filter is causal and suitable for real-time use. The RTS smoother reuses the same model with future data and is a retrospective reference. The ratio of process noise q to measurement noise sets the trade-off between lag and noise sensitivity \[16\].

**M2: Regularised inversion (cross-check).** Estimate the blood-glucose trajectory by penalising roughness, subject to the ISF dynamics. This is a linear least-squares problem and should agree with the M1 smoother under matched priors, which makes it a good implementation test.

```latex
\hat G_B = \arg\min_{G_B}\ \sum_k \big(y_k - \hat G_I[k]\big)^2 + \lambda \sum_k \big(G_B[k+1] - 2G_B[k] + G_B[k-1]\big)^2 \quad \text{subject to the ISF dynamics}
```

**M3: Physics-informed neural network (extension).** A small network outputs blood and ISF glucose as functions of time, and the lag equation is enforced as a residual at collocation points. Reference \[27\] uses a related idea for glucose–insulin dynamics, not ISF lag.

```latex
\mathcal{L} = \frac{1}{N}\sum_k \big(y_k - \hat G_I(t_k)\big)^2 + \lambda_p \frac{1}{M}\sum_j \Big(\tau\,\frac{d\hat G_I}{dt} + \hat G_I - \hat G_B\Big)^2_{t_j} + \lambda_s \frac{1}{M}\sum_j \Big(\frac{d^2 \hat G_B}{dt^2}\Big)^2_{t_j}
```

In this linear setting M3 solves the same problem as M2, so expect similar accuracy; its value is as a teaching and flexibility comparison, for example when the physics term is made nonlinear.

### 5.3 Why not simply invert the lag?

The naive inverse adds a scaled derivative to the signal and amplifies noise. This worked example uses assumed values: τ = 10 min, Δt = 5 min, white noise s.d. σ = 5 mg/dL.

1. Naive inverse: G\_B\[k\] ≈ y\[k\] + τ·(y\[k\] − y\[k−1\])/Δt.
2. Let r = τ/Δt = 10/5 = 2. Then G\_B\[k\] ≈ (1 + r)·y\[k\] − r·y\[k−1\] = 3·y\[k\] − 2·y\[k−1\].
3. For independent noise, variance = \[(1 + r)² + r²\]·σ² = (9 + 4)·σ² = 13·σ².
4. Standard deviation = √13 × 5 = 3.61 × 5 ≈ 18 mg/dL.
5. The ramp lag error removed in Section 1 was 20 mg/dL, while noise grew from 5 to 18 mg/dL, so the net gain can be small. Finite-difference estimates are known to be very noise-sensitive, which is why filtering is used \[16\].

### 5.4 Parameter estimation and selection

| Parameter | Simulation route | Real-data route |
| --- | --- | --- |
| τ\_eff | (a) Supervised: minimise RMSE against G\_B on training subjects. (b) Unsupervised: maximise the Kalman innovation likelihood | Only (b) is possible without a reference; also run a fixed-τ sensitivity band of 5–12 min *(proposed, spanning \[1\]\[2\]\[6\]\[30\])* |
| q and measurement variance | Tune on validation subjects or by innovation likelihood | Innovation likelihood on the training segment of each subject |
| λ (M2) | Validation grid | Validation grid |
| Time shift s\* (B1) | Choose the shift maximising correlation with G\_B on training data; evaluate on a 1-min grid | Not identifiable without a reference; report as a sensitivity sweep |

Causal evaluation rule: any per-subject parameter is estimated only from that subject's earlier data, never from data after the evaluation time.

### 5.5 Sampling, missing data and noise

- Resample to a regular grid at the device interval and carry a missing-data mask; do not interpolate across gaps longer than 15 min *(proposed)*.
- A 15-minute interval is larger than the expected lag, so the ratio r = τ/Δt drops below 1 and compensation is weakly identifiable; expect and report this \[1\]\[2\].
- Exclude the sensor warm-up period and flagged compression or dropout segments, using device-specific rules documented in each dataset *(proposed)*.
- Test white versus autoregressive noise, since the structure in \[11\] is autoregressive.

### 5.6 Preventing data leakage

- Split by subject, never by time window within a subject; use different meal-scenario and noise seeds in train, validation and test.
- Tune only on validation; run the test set once after configurations are frozen and tagged in Git.
- Never pass true G\_B or true τ to an estimator at test time.
- Causality unit test: change all samples after time t and confirm the causal output at t is unchanged.
- Fit any scaling or priors on training subjects only.
- Real data: use fingersticks only as evaluation targets, never as inputs; check whether the device used them for calibration ⚠.

### 5.7 Baselines

| ID | Method | Causal? |
| --- | --- | --- |
| B0 | No compensation (raw CGM) | Yes |
| B1 | Fixed time shift s\* | No: advancing the signal needs future samples; the causal version is extrapolation |
| B2 | Causal smoothing (exponential or Savitzky–Golay, window tuned) | Yes |
| B3 | Naive inverse with smoothed derivative | Yes |
| B4 *(optional)* | Wiener inverse filter, as in \[15\] | Yes |
| M1 | Kalman filter | Yes |
| M1s | RTS smoother | No |
| M2 | Regularised inversion | No |
| M3 *(extension)* | Physics-informed neural network | No |

### 5.8 Data routes

**Route A: simulation (primary, always available).**

1. Generate blood glucose with simglucose \[20\], a Python implementation of the 2008 UVa/Padova simulator with 30 virtual patients (10 adolescents, 10 adults, 10 children) under an MIT licence. The newer S2013 version adds features not assumed here \[19\].
2. Apply your own ISF and sensor layer rather than relying on the simulator's built-in sensor model. I did not verify the built-in model's delay and noise settings in this session, so inspect the code first ⚠.
3. Use three generators: G1 matched first-order lag with τ drawn from 5–15 min; G2 mismatched dynamics, such as insulin-dependent clearance K\_02 \[9\] or a second-order response; G3 time-varying τ across days, motivated by patient-dependent delay \[2\].
4. Add autoregressive noise at three levels (s.d. 2, 5 and 8 mg/dL), sample at 1, 5 and 15 min, and drop 0%, 5% or 20% of samples *(all proposed)*.

Without G2 and G3, results suffer from the "inverse crime": the estimator is tested on data from its own model, which overstates performance.

**Route B: public real data (secondary).** Access conditions below are as stated in the cited sources; confirm current terms before downloading.

| Dataset | Content | Access | Fit for this project |
| --- | --- | --- | --- |
| OhioT1DM \[21\] | 12 people with T1D, 8 weeks each, Medtronic Enlite CGM every 5 min, fingersticks, insulin, meals | De-identified; data use agreement required | Best fit: 5-min CGM plus sparse fingersticks; check calibration use ⚠ |
| T1D-UOM \[24\] | 17 people with T1D, 3 months, glucose with insulin, nutrition, activity, sleep | Zenodo; repository licence CC BY 4.0 | Open; check CGM device and sampling interval in the data descriptor ⚠ |
| ShanghaiT1DM / T2DM \[23\] | 12 T1D and 100 T2D patients, 3–14 days | Open (figshare); CC BY 4.0 reported in \[25\] | Weak fit: CGM at 15-min intervals reported in \[31\] |
| DiaTrend \[22\] | 54 people with T1D, 27,561 days of CGM from Dexcom, Abbott and Medtronic devices | Controlled access via Synapse | Large, but approval time is unknown |
| T1DiabetesGranada \[26\] | 736 people, FreeStyle Libre, four years | Data usage agreement and request on Zenodo | Check sampling interval before use ⚠ |
| MetaboNet \[25\] | Consolidated T1D datasets with CGM and insulin, 5-min grid | Public subset plus DUA-governed subset | Convenient if preprocessing time is short |

None of the sources I opened describes a public dataset that pairs CGM with frequent venous or laboratory reference glucose, so lag cannot be measured directly in these data. I did not search exhaustively, so treat this as a limitation of my search ⚠.

**Route C: reference-paired data (only if provided).** If the course or a collaborator supplies CGM paired with frequent reference glucose, add it as a held-out validation of H1 and H3. Do not plan around it.

### 5.9 Reproducible workflow

1. Fix seeds and parameters in one configuration file per experiment.
2. Generate simulation sets or fetch real data; keep data covered by agreements outside the repository.
3. Preprocess to a regular grid with masks and save to a versioned file.
4. Create and save a subject-wise split manifest.
5. Fit and tune on training subjects; select on validation; log every run.
6. Freeze configurations and tag the Git commit.
7. Run the test set once; compute metrics with a subject-level bootstrap.
8. Build all figures and tables only from saved results, with no refitting in plotting code.
9. Re-run end to end in a clean environment (lockfile or container) and compare outputs.

## 6. Evaluation plan

Every method is scored on point accuracy, trend tracking and timing, on subject-wise held-out data, with bootstrap intervals and an explicit list of conditions where it fails.

### Metrics

```latex
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{k=1}^{N}\big(\hat G_B[k]-G_B[k]\big)^2}, \qquad \mathrm{MARD} = \frac{100\%}{N}\sum_{k=1}^{N}\frac{\big|\hat G_B[k]-G_B[k]\big|}{G_B[k]}
```

| Family | Metric | What it reveals | Caveat |
| --- | --- | --- | --- |
| Point accuracy | Mean absolute difference (mg/dL), RMSE (mg/dL), signed bias (mg/dL) | Size and direction of error; RMSE weights large errors | Pooled values hide rapid-change behaviour; also report the rapid-change subset |
| Point accuracy | MARD (%) | Relative error, common in CGM accuracy studies \[14\]\[32\] | Inflates at low glucose; stratify by range (below 70, 70–180, above 180 mg/dL) *(proposed bands)* |
| Trend | Rate-of-change RMSE (mg/dL/min) | Whether slope is recovered, where lag does most damage | Differentiation amplifies noise; compute the same way for truth and estimate |
| Trend | Trend-class agreement and Cohen's kappa (falling, steady, rising at ±1 mg/dL/min) *(proposed thresholds)* | Whether the direction is right | Depends on thresholds; report sensitivity to them |
| Timing | Lag-estimation error: τ\_eff estimate minus truth (min) | Identifiability of the lag parameter | Defined only in simulation |
| Timing | Cross-correlation lag between estimate and G\_B (min) | Residual delay; same idea as time-shift alignment in \[14\] | Insensitive in flat segments |
| Timing | Detection delay at crossings of 70 and 180 mg/dL, with event sensitivity and false events per day | Practical timing benefit against noise cost | Thresholds here are conventional placeholders; confirm against your course references |

**Worked example: detection delay.** Blood glucose falls at 1 mg/dL/min and crosses 70 mg/dL at t = 100 min. Assume raw CGM has an effective lag of 10 min and a compensated estimate has 3 min residual lag *(assumed values)*.

1. Raw CGM at time t shows the blood glucose of 10 min earlier, so it reads 70 mg/dL when t − 10 = 100 min. It therefore crosses at t = 100 + 10 = 110 min.
2. Detection delay for raw CGM = 110 − 100 = 10 min.
3. Compensated estimate crosses at 100 + 3 = 103 min, so delay = 3 min.
4. Improvement = 10 − 3 = 7 min, bought at the price of extra noise, which is why false events per day are reported alongside.

### Train, validation and test procedure

- **Simulation:** 5-fold subject-wise cross-validation over the 30 virtual patients (6 per fold, balanced across adolescent, adult and child groups). Within each training set, hold out one fold for validation and tune there only. Use disjoint scenario and noise seeds for training and testing, and repeat with at least 10 noise seeds.
- **Real data:** leave-one-subject-out. For causal estimators, also test time-forward within a subject, using earlier data only for fitting.
- **Freezing rule:** configurations are fixed and tagged before the first test evaluation; any later change creates a new, labelled experiment.

### Ablation and sensitivity analyses

Run one factor at a time, plus two pairings (noise × sampling interval; lag error × generator).

| Factor | Levels | Question answered |
| --- | --- | --- |
| Assumed lag error | −50%, −25%, 0, +25%, +50% | How wrong can τ be before gains vanish? (H4) |
| Noise s.d. | 2, 5, 8 mg/dL | Where does noise amplification overtake lag removal? |
| Sampling interval | 1, 5, 15 min | Is compensation meaningful at 15 min? |
| Missing data | 0%, 5%, 20% | Does the filter degrade gracefully? |
| Generator | G1 matched, G2 mismatched, G3 time-varying | How much is the "inverse crime" worth? |
| Noise model | White vs autoregressive | Does the structure in \[11\] matter? |
| Calibration drift | None vs slow offset | Is drift misread as lag? |
| Causality | M1 vs M1s | Cost of real-time operation |
| Tuning ratio q to noise | Sweep | Lag versus noise trade-off \[16\] |
| State design | With vs without rate-of-change state | Is the third state needed? |

### Uncertainty estimation

- Subject-level cluster bootstrap with 2,000 resamples for all summary statistics and paired differences; report median and interquartile range, not only means.
- Kalman posterior covariance gives 95% intervals for the glucose estimate; check coverage against truth in simulation.
- Holm correction across the pre-registered hypotheses.
- Profile likelihood over τ\_eff, τ\_f and d to show parameter uncertainty and the flat direction behind H3.

### Likely failure cases and how to read results

| Failure case | Why it happens | How to treat it |
| --- | --- | --- |
| Flat glucose for hours | Lag is invisible, so τ is unidentifiable | Wide intervals; do not report a point estimate |
| High process noise | Noise amplification, false events | Report false events per day with every timing gain |
| Compression lows and dropouts | Sensor artefact looks like physiology | Flag and exclude, and state the rule |
| Calibration jumps or drift | Misattributed to lag | Include the drift ablation |
| Hypoglycaemia or high insulin | ISF dynamics may differ \[6\]\[8\] | Report errors by range, not only pooled |
| τ varies by day or sensor | Population value is wrong for that person \[2\]\[7\] | Compare fixed against estimated τ (H2) |
| Clock offset in real data | Alignment error mimics lag | Check timestamps; report as limitation |
| Matched simulation | Estimator tested on its own model | Treat G1 results as an upper bound |

**Interpretation rules.** State every gain with its scope, for example "in simulation with generator G2 at 5-min sampling". Report losses against baselines too, such as flat segments. A null or negative result on H1 to H5 is a valid finding and is graded on the quality of its analysis.

## 7. Ethics, safety and limitations

This is a research and education project; its outputs are not a medical device, must not be shown to patients, and must not inform any treatment decision.

### Data privacy and data-use terms

- Use only de-identified public or controlled-access data. OhioT1DM was de-identified by the Safe Harbor method and requires a signed data use agreement before release \[21\]. DiaTrend uses a controlled-access mechanism through Synapse \[22\]. T1DiabetesGranada requires accepting a usage agreement and submitting a justification \[26\].
- Never attempt re-identification, linkage with other data, or redistribution of agreement-covered files; keep them out of the public repository and out of shared cloud folders.
- Check whether your institution requires an ethics-office determination even for de-identified secondary data, and file the data-use applications in week 1 because approval time is unknown *(proposed)*.
- Cite each dataset as its terms require, and record licence and access date in the data card.

### Bias and generalisability

- Most public CGM sets are small and skew to adults with type 1 diabetes using specific devices; for example OhioT1DM has 12 people on one sensor family \[21\]. Findings may not transfer to other populations, devices, skin sites or type 2 diabetes.
- The UVa/Padova-type simulator is a model of average physiology with inter-subject variability \[18\]\[19\], not a population sample. Report results by virtual cohort and avoid claims about real subgroups.
- Where demographic fields exist, report performance by group descriptively and state that group sizes are too small for inference.

### Risks of glucose-related predictions

- An estimator that over-corrects can show false reassurance or false alarms; a lag-compensated value that looks authoritative may be mistaken for a measurement.
- Errors are largest when glucose changes fastest and near low glucose, which is where the stakes are highest.
- Safeguards: label every figure "simulation" or "research data"; do not build a patient-facing display, alerting or dosing logic; keep any code and demos clearly marked as non-clinical.

### Technical limitations to state in the report

- Physiological delay is hard to separate from sensor, filtering and alignment effects; only the combined effective lag is constrained by CGM data \[3\]\[4\]\[10\]. See H3.
- Small errors in assumed dynamics or recalibration can severely distort inferred sensor error \[12\].
- Real public data lack frequent reference glucose, so real-data findings are exploratory.
- Nonlinear on-device filters violate the linear model and can bias deconvolution \[4\].
- Simulation matched to the estimator overstates performance; hence the mismatched generators.

## 8. Execution plan

The 14-week plan has four gates; the test data are touched once, at the week-11 freeze, and report writing runs from week 11 to submission.

&#91;embedded content: 14-week schedule by workstream · 4 gates\]

Gate 3 in week 11 freezes all configurations before the single test run; the optional neural-network work is dropped if Gate 2 results are not in.

### Week-by-week milestones and deliverables

| Week | Work | Deliverable |
| --- | --- | --- |
| 1 | Agree scope; write search protocol; set up Git and environment; send data-access requests | Proposal v1; requests filed |
| 2 | Screen literature; build annotated matrix; install simglucose and simulate one virtual patient | Matrix of 20 or more papers; smoke-test notebook |
| 3 | Build ISF and sensor layer with known τ, d and noise (generator G1) | Generator v0 with unit tests |
| 4 | Implement baselines B0–B3 and the metric harness; hand-check metrics | **Gate 1:** harness reproduces toy values |
| 5 | Implement M1 Kalman filter; verify the ramp example | Filter passing the analytic test |
| 6 | Add RTS smoother and M2; tune q, noise variance and λ on validation | **Gate 2 (midterm report):** M1 and M2 agree; first H1 plots |
| 7 | Add generators G2 and G3; start identifiability analysis | Mismatch results; profile-likelihood draft |
| 8 | Finish H3 analysis; begin real-data preprocessing | Identifiability figures; real-data card |
| 9 | Real-data exploratory run (H5); start the neural-network extension only if on track | Exploratory real-data results |
| 10 | Full ablation and sensitivity grid with bootstrap intervals | Result tables with intervals |
| 11 | Freeze configurations; run test set once | **Gate 3:** frozen tag and test results |
| 12 | Draft report; clean code; reproduce everything in a clean environment | Report draft; reproduction log |
| 13 | Slides; peer review; revisions | Slides; revised report |
| 14 | Buffer and final checks | **Gate 4:** final submission |

### Skills, software and compute

- **Skills:** Python scientific computing; linear systems and ODE discretisation; basic Kalman filtering; bootstrap and paired statistics; Git and reproducible workflows; reading clinical-engineering papers critically.
- **Software:** NumPy, SciPy, pandas, scikit-learn, Matplotlib, simglucose \[20\]; PyTorch for the extension only; a lockfile or container for the final run.
- **Compute:** a laptop CPU is enough, since the models are low-dimensional; a bootstrap of 2,000 resamples over 30 subjects is cheap.

### Suggested roles for three students

| Role | Owns | Also |
| --- | --- | --- |
| Data and simulation lead | Generators G1–G3, dataset access, data cards, ethics paperwork | Literature on kinetics and datasets |
| Estimation lead | M1, M2, optional M3, parameter estimation, identifiability | Literature on filtering methods |
| Evaluation and reporting lead | Metrics, splits, bootstrap, figures, reproducibility, report assembly | Literature on accuracy metrics |

Every module gets a second reviewer from another role. With two students, merge the first two roles; with four, add a dedicated real-data owner.

### Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Data-use approval arrives late or is refused | High | Medium | File requests in week 1; design simulation-first; use open sets such as T1D-UOM \[24\] as fallback |
| Simulator's built-in sensor model limits control | Medium | Medium | Use your own ISF layer; inspect the code in week 2 |
| Kalman tuning consumes weeks | Medium | Medium | Use M2 as an independent cross-check; cap the tuning grid; follow the lag-noise trade-off in \[16\] |
| Lag is unidentifiable on real data | High | Low | Expected; report it as a finding under H3; keep real data exploratory |
| Gains look too good because the estimator matches the generator | Medium | High | Generators G2 and G3; report G1 as an upper bound |
| Scope creep into the neural network | Medium | Medium | Start M3 only after Gate 2 and only if H1 and H2 results exist |
| Data leakage bug | Low | High | Causality unit test; saved split manifests; code review before the freeze |
| A team member becomes unavailable | Medium | Medium | Pair on each module; fall back to the minimum viable tier; weekly status check |
| Results read as clinical validation | Low | High | Labelling rules in Section 7; supervisor reviews all claim wording |

## 9. Final deliverables

Deliver a report, a talk, a repository free of restricted data, and a reproducibility package, and judge completeness against the checklist below.

| Deliverable | Contents |
| --- | --- |
| Final report | 12–15 pages *(proposed)* following the outline below |
| Presentation | 15 minutes plus questions: problem and lag definitions, method, two headline results with intervals, failure cases, limits and non-clinical statement |
| Code repository | src, configs, tests, notebooks, README, licence, lockfile or container file; split manifests and result files; no restricted data |
| Reproducibility materials | Data cards for each dataset, experiment log, seeds and configuration for every figure, clean-environment reproduction log, hypotheses file with a timestamp from before the test run, data-use confirmation |

### Proposed report outline

1. Abstract (200 words or fewer)
2. Introduction: motivation and definition of lag components
3. Background and related work, with verified citations
4. Research gap, questions and hypotheses
5. Methods: signal model, estimators, baselines, parameter estimation
6. Data: simulation generators, public datasets, access terms, preprocessing
7. Evaluation protocol: metrics, splits, uncertainty
8. Results: H1–H5 with tables, figures and intervals; ablations
9. Failure analysis and sensitivity
10. Discussion: what is and is not claimed; identifiability
11. Ethics, safety and limitations statement
12. Conclusion and future work
13. References and appendices (parameter tables, extra figures, reproduction instructions)

### Evaluation rubric

| Criterion | Weight | Strong work looks like |
| --- | --- | --- |
| Problem framing and literature | 15 | Lag components defined precisely; citations verified; gap stated narrowly |
| Model and method correctness | 20 | Equations and variables correct; analytic ramp test passes; M1 and M2 agree |
| Experimental design and leakage control | 20 | Subject-wise splits; single frozen test run; mismatched generators; fair baselines |
| Results and uncertainty | 15 | Bootstrap intervals; each hypothesis reported as supported, unsupported or inconclusive; ablations |
| Honest interpretation and limitations | 15 | Claims scoped to conditions; failure cases shown; non-clinical statement; ethics handled |
| Reproducibility and communication | 15 | One-command rerun; clear figures; well-paced talk |

### Completion checklist

- [ ] Lag components defined and sourced
- [ ] Every citation verified, or labelled as unverified
- [ ] Hypotheses file committed before the test run
- [ ] Baselines B0–B3 and M1 implemented and unit-tested
- [ ] Analytic ramp test passes within 1%
- [ ] Subject-wise splits and causality test in place
- [ ] Test set run once after the freeze tag
- [ ] Bootstrap intervals on every headline number
- [ ] Mismatched generators G2 and G3 included
- [ ] Identifiability analysis (H3) with figure
- [ ] Failure cases reported
- [ ] Data-use terms followed; no restricted data in the repository
- [ ] Non-clinical statement in report, slides and README
- [ ] Clean-environment reproduction completed
- [ ] Talk rehearsed within time

### One-page proposal draft (for course approval)

**Title:** Physics-Informed Lag Compensation for Continuous Glucose Monitoring: Identifiability and Benchmarking of Interstitial-Kinetics Estimators on Simulated and Public Data

**Course, term, team, supervisor:** \[course code\], \[term\], \[three student names\], \[supervisor name\]

**Background.** CGM sensors measure interstitial glucose, so readings lag blood glucose by roughly 4–10 minutes in controlled studies \[1\]\[2\]\[6\]\[7\]. The delay combines physiological transport, sensor response and on-device filtering \[3\], and published estimates vary partly because these are hard to separate \[4\]\[10\]. Model-based corrections exist (Wiener, Kalman) \[15\]\[16\], but their benefit under model mismatch and noise is rarely mapped.

**Research question.** Does a physics-based interstitial-kinetics estimator reduce blood-glucose estimation error and improve trend and timing metrics versus no compensation, a fixed time shift and causal smoothing, and when can its lag parameter be identified?

**Approach.** Generate ground truth with a UVa/Padova-based simulator \[20\] and apply our own interstitial and sensor layer with known lag, filter, delay and autoregressive noise. Compare a first-order Kalman filter and smoother, a regularised inversion and four baselines under matched and mismatched dynamics. Add an exploratory check on public data with agreement-governed access \[21\]\[24\].

**Evaluation.** MAD, RMSE and MARD; rate-of-change error and trend agreement; lag-estimation error, residual lag and threshold-crossing detection delay with false events. Subject-wise cross-validation, a frozen single test run, cluster-bootstrap intervals, and sensitivity to lag error, noise, sampling and missing data.

**Deliverables and timeline.** Gate 1 (week 4) harness check; Gate 2 (week 6) midterm; Gate 3 (week 11) freeze and test; Gate 4 (week 14) report, talk and reproducible repository.

**Resources, risks, ethics.** Laptops and Python only. Main risks: data approvals (mitigated by simulation-first design), model mismatch inflating results (mitigated by mismatched generators), and scope creep (neural-network extension is conditional). Only de-identified data under their terms will be used.

**Statement.** This is a research and education project. It is not a medical device and its outputs must not be used for treatment decisions.

## References

"Verified" means the record, abstract or visible page text was opened in this session on 2026-10-30; most full texts were not read, so numerical claims rest on abstracts. ⚠ means the paper exists but some bibliographic fields (title, authors, volume or DOI) could not be captured. Re-check ⚠ entries before final submission.

| # | Citation | Link | Status |
| --- | --- | --- | --- |
| 1 | A. Basu et al., "Time lag of glucose from intravascular to interstitial compartment in humans," Diabetes, vol. 62, no. 12, pp. 4083–4087, 2013 | https://doi.org/10.2337/db13-1132 | Verified |
| 2 | G. Schmelzeisen-Redeker et al., "Time delay of CGM sensors: relevance, causes, and countermeasures," J. Diabetes Sci. Technol., vol. 9, no. 5, pp. 1006–1015, 2015 | https://doi.org/10.1177/1932296815590154 | Verified |
| 3 | D. B. Keenan, J. J. Mastrototaro, G. Voskanyan, G. M. Steil, "Delays in minimally invasive continuous glucose monitoring devices: a review of current technology," J. Diabetes Sci. Technol., vol. 3, no. 5, pp. 1207–1214, 2009 | https://doi.org/10.1177/193229680900300528 | Verified |
| 4 | K. Rebrin, N. F. Sheppard, G. M. Steil, "Use of subcutaneous interstitial fluid glucose to estimate blood glucose: revisiting delay and sensor offset," J. Diabetes Sci. Technol., vol. 4, no. 5, pp. 1087–1098, 2010 | https://doi.org/10.1177/193229681000400507 | Verified (abstract and search text; full-text page was blocked) |
| 5 | K. Rebrin, G. M. Steil, W. P. van Antwerp, J. J. Mastrototaro, "Subcutaneous glucose predicts plasma glucose independent of insulin," Am. J. Physiol. Endocrinol. Metab., vol. 277, no. 3, pp. E561–E571, 1999 | https://doi.org/10.1152/ajpendo.1999.277.3.e561 | Verified |
| 6 | G. M. Steil et al., "Interstitial fluid glucose dynamics during insulin-induced hypoglycaemia," Diabetologia, vol. 48, no. 9, pp. 1833–1840, 2005 | https://doi.org/10.1007/s00125-005-1852-x | Verified |
| 7 | M. S. Boyne, D. M. Silver, J. Kaplan, C. D. Saudek, "Timing of changes in interstitial and venous blood glucose measured with a continuous subcutaneous glucose sensor," Diabetes, vol. 52, no. 11, pp. 2790–2794, 2003 | https://doi.org/10.2337/diabetes.52.11.2790 | Verified |
| 8 | B. Aussedat et al., "Interstitial glucose concentration and glycemia: implications for continuous subcutaneous glucose monitoring," Am. J. Physiol. Endocrinol. Metab., vol. 278, no. 4, pp. E716–E728, 2000 | https://doi.org/10.1152/ajpendo.2000.278.4.e716 | Verified |
| 9 | E. Cengiz, W. V. Tamborlane, "A tale of two compartments: interstitial versus blood glucose monitoring," Diabetes Technol. Ther., vol. 11, suppl. 1, pp. S11–S16, 2009 | https://doi.org/10.1089/dia.2009.0002 | Verified |
| 10 | G. Voskanyan, D. B. Keenan, J. J. Mastrototaro, G. M. Steil, "Putative delays in interstitial fluid glucose kinetics can be attributed to the glucose sensing systems used to measure them rather than the delay in interstitial fluid glucose itself," J. Diabetes Sci. Technol. | https://pmc.ncbi.nlm.nih.gov/articles/PMC2769670 | ⚠ Volume, pages, year and DOI not captured |
| 11 | A. Facchinetti et al., "Modeling the glucose sensor error," IEEE Trans. Biomed. Eng., vol. 61, no. 3, pp. 620–629, 2014 | https://doi.org/10.1109/TBME.2013.2284023 | Verified (abstract) |
| 12 | A. Facchinetti, G. Sparacino, C. Cobelli, "Modeling the error of continuous glucose monitoring sensor data: critical aspects discussed through simulation studies," J. Diabetes Sci. Technol., vol. 4, no. 1, pp. 4–14, 2010 | https://www.research.unipd.it/handle/11577/2443081 | ⚠ DOI not captured |
| 13 | M. Vettoretti, C. Battocchio, G. Sparacino, A. Facchinetti, "Development of an error model for a factory-calibrated continuous glucose monitoring sensor with 10-day lifetime," Sensors, vol. 19, no. 23, art. 5320, 2019 | https://doi.org/10.3390/s19235320 | Verified |
| 14 | C. Scuffi, F. Lucarelli, F. Valgimigli, "Minimizing the impact of time lag variability on accuracy evaluation of continuous glucose monitoring systems," J. Diabetes Sci. Technol., vol. 6, no. 6, pp. 1383–1391, 2012 | https://doi.org/10.1177/193229681200600618 | Verified |
| 15 | D. B. Keenan, J. J. Mastrototaro, G. M. Steil, S. A. Weinzimer, "Interstitial fluid glucose time-lag correction for real-time continuous glucose monitoring," Biomed. Signal Process. Control, vol. 8, pp. 81–89, 2013 | https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi...........2147b0177761e62c2efb92df6cef3057 | ⚠ Abstract verified via library record; DOI not captured |
| 16 | B. W. Bequette, "Continuous glucose monitoring: real-time algorithms for calibration, filtering, and alarms," J. Diabetes Sci. Technol., vol. 4, no. 2, pp. 404–418, 2010 | https://doi.org/10.1177/193229681000400222 | Verified |
| 17 | S. Pleus et al., J. Diabetes Sci. Technol., vol. 9, no. 4, pp. 801–807, 2015 | https://doi.org/10.1177/1932296815578716 | ⚠ Title not captured; page text blocked on fetch |
| 18 | C. Dalla Man, R. A. Rizza, C. Cobelli, "Meal simulation model of the glucose-insulin system," IEEE Trans. Biomed. Eng., vol. 54, no. 10, pp. 1740–1749, 2007 | https://doi.org/10.1109/TBME.2007.893506 | Verified |
| 19 | C. Dalla Man et al., "The UVA/PADOVA type 1 diabetes simulator: new features," J. Diabetes Sci. Technol., vol. 8, no. 1, pp. 26–34, 2014 | https://doi.org/10.1177/1932296813514502 | Verified |
| 20 | J. Xie, "Simglucose v0.2.1," 2018 (software, MIT licence) | https://github.com/jxx123/simglucose | Verified (README, PyPI) |
| 21 | C. Marling, R. Bunescu, "The OhioT1DM dataset for blood glucose level prediction: update 2020," CEUR Workshop Proc., vol. 2675, 2020 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904 | Verified |
| 22 | T. Prioleau, A. Bartolome, R. Comi, C. Stanger, "DiaTrend: a dataset from advanced diabetes technology to enable development of novel analytic solutions," Sci. Data, vol. 10, art. 556, 2023 | https://doi.org/10.1038/s41597-023-02469-5 | Verified |
| 23 | Q. Zhao et al., "Chinese diabetes datasets for data-driven machine learning," Sci. Data, vol. 10, art. 35, 2023 | https://doi.org/10.1038/s41597-023-01940-7 | Verified |
| 24 | A. Alsuhaymi et al., "T1D-UOM: a longitudinal multimodal dataset of type 1 diabetes," Sci. Data, 2025 | https://doi.org/10.1038/s41597-025-05695-1 | Verified (record); licence taken from the repository metadata |
| 25 | MetaboNet: consolidated public dataset for type 1 diabetes management, arXiv:2601.11505, 2026 | https://doi.org/10.48550/arXiv.2601.11505 | Verified (preprint; authors not captured) |
| 26 | C. Rodriguez-Leon et al., "T1DiabetesGranada: a longitudinal multi-modal dataset of type 1 diabetes mellitus," Sci. Data, vol. 10, art. 916, 2023 | https://doi.org/10.1038/s41597-023-02737-4 | Verified |
| 27 | W. Wang et al., "A physics-informed glucose-insulin neural network model for glucose prediction," Tsinghua Sci. Technol., 2025 | https://doi.org/10.26599/TST.2025.9010140 | Verified (abstract) |
| 28 | "Personalized blood glucose prediction using physiology-informed machine learning," Mach. Learn. Knowl. Extr., vol. 8, no. 4, art. 96, 2026 | https://www.mdpi.com/2504-4990/8/4/96 | ⚠ Authors and DOI not captured |
| 29 | Glucose-ML: a collection of longitudinal diabetes datasets, arXiv:2507.14077, 2025 | https://arxiv.org/abs/2507.14077 | Verified (preprint); additional reading, not cited in text |
| 30 | Plasma-to-interstitium glucose kinetics model, Diabetes Technol. Ther., vol. 17, no. 11, pp. 825–831, 2015 | https://doi.org/10.1089/dia.2015.0119 | ⚠ Title and authors not captured |
| 31 | Bayesian and expert-knowledge forecasting of CGM values in type 2 diabetes using the Shanghai datasets, arXiv:2409.07315, 2024 | https://arxiv.org/abs/2409.07315 | ⚠ Authors not captured |
| 32 | Clinical evaluation of the AiDEX CGM system (MARD 9.08% vs venous, 10.1% vs capillary reference) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9846405 | ⚠ Authors, journal and DOI not captured |
