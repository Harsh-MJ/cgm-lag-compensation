# Piece 2: realistic sensor layer and mismatch generators (30 Oct 2026)

Code: `code/sensor.py`, `code/test_sensor.py`, `code/scenario_report.py`. Input: the plasma traces in `data/sim.npz` (1-min, 27 usable virtual patients). Written by Claude (AI assistant); the team must read and sign off before claiming it.

## What it does
Builds a CGM-like signal from true plasma glucose with known, switchable error sources: ISF kinetics (1st order, 2nd order, time-varying tau), on-device filter tau_f and pure delay d, calibration drift, night compression lows, slow plus fast AR(1) noise, sampling at 1/5/15 min and burst dropouts. Seeds make every trace reproducible.

| Scenario | ISF kinetics | Extra sensor chain | Artifacts |
| --- | --- | --- | --- |
| S0 matched | 1st order, tau known | none | none |
| S1 sensor chain | 1st order | tau_f 3 min + delay 2 min (unmodelled) | none |
| S2 second order | two stages, same total lag | none | none |
| S3 tau varying | 1st order, tau x exp(0.25 z), 12 h correlation | none | none |
| S4 calibration/artifact | 1st order | none | gain drift sd 4%, offset drift sd 4 mg/dL, compression lows |
| S5 full | S3 | S1 | S4 |

## Tests (11 of 11 pass)
- Ramp at 2 mg/dL/min, tau 10: steady error 20.02 (S0), 30.07 (S1), 20.07 (S2) mg/dL; theory 20, 30, 20.
- Own ISF integrator reproduces simglucose's subcutaneous state: RMSE 0.002 to 0.008 mg/dL, max 0.04 mg/dL (3 patients).
- Noise sd 4.96 for target 5; autocorrelation matches 0.6 exp(-h/30) + 0.4 exp(-h/2) within 0.01 at 1, 5, 15, 30 min.
- tau varying: sd(log tau) 0.22 (target 0.25, clipped to 3 to 40 min); dropout 4.8% (target 5%); compression depth within 0.25 to 0.45; seeds reproducible; no index shift when tau tends to 0.

## Scenario report (raw signal, noise sd 5 mg/dL, 5-min sampling, 5% dropouts, median over 27 patients)
| Scenario | RMSE (mg/dL) | MARD (%) | Best-shift lag (min) | Nominal tau + tau_f + d (min) |
| --- | --- | --- | --- | --- |
| S0 | 8.7 | 4.9 | 10 | 11.2 |
| S1 | 11.2 | 5.5 | 15 | 16.2 |
| S2 | 8.9 | 5.0 | 11 | 11.2 |
| S3 | 8.8 | 4.7 | 10 | 11.2 |
| S4 | 11.2 | 7.0 | 10 | 11.2 |
| S5 | 14.1 | 7.2 | 15 | 16.2 |

S0 RMSE 8.7 equals the prototype's raw RMSE at noise scale 0, as expected. "Best-shift lag" is the integer shift that minimises RMSE on noise-free data; it sits about 1 min below the nominal lag because a first-order lag is not a pure delay.

## Findings that change the plan
1. Realism: S5 raw MARD is 6.2, 7.3, 8.7 and 10.7% at noise sd 2, 5, 8, 12 mg/dL. One clinical accuracy study reports 9.08% against venous reference (AiDEX, PMC9846405, details only partly verified), so use noise sd 8 mg/dL as the stress level; 2 and 5 are optimistic. This is a loose anchor only.
2. Rapid change is rare and uneven: per patient, time with |rate| >= 1 mg/dL/min has median 8.9% (range 0.1 to 27.8%); >= 2 mg/dL/min median 3.1%. Some virtual patients barely move, so lag is nearly invisible for them. Report results stratified by this fraction and use the rapid-change subset for H1.
3. S1 and S5 add 5 min of unmodelled lag. An estimator that knows only tau will be systematically behind by about 5 min there; this is the physiology-versus-sensor identifiability test (H3).

## Open decisions
- Meal sizes: the prototype's meals give median peak 243 mg/dL (range 138 to 336); confirm this is acceptable or rescale.
- Noise sd default for headline results (chosen: 8 mg/dL, with 2 and 5 as sensitivity).
