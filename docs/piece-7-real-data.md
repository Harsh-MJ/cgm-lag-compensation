# Piece 7: OhioT1DM and the real-data check (26 Nov 2026)

Code: `code/real.py` (loader and leave-one-patient-out fingerstick check), `code/make_fake_ohio.py`, `code/test_real.py` (4 of 4 pass), `code/run_real.py`; request draft `docs/ohiot1dm-request.md`. Written by Claude (AI assistant); the team must read and sign off before claiming it.

## Status: not run on real data
OhioT1DM needs a signed data-use agreement. No copy was downloaded: the dataset hosts (Kaggle, PhysioNet, Ohio and Charlotte pages) are blocked from this workspace, and copies posted on third-party sites would breach the agreement. What exists is the whole pipeline, tested on synthetic files in OhioT1DM XML format, so it runs as soon as the files are available:

`python3 -I code/run_real.py <folder with the OhioT1DM .xml files> results`

## Access (checked 26 Nov 2026)
Dataset pages say to email Razvan Bunescu (razvan.bunescu@charlotte.edu), subject `OhioT1DM Request`, with name, job title and institutional email; the link and password arrive by email within about a week. One page instead says to complete the DUA form and email it to the Ohio University technology coordinator. The draft in `docs/ohiot1dm-request.md` handles both and has blanks for the instructor and the IITB email. A DUA is normally signed by faculty, so ask the DH 803 instructor to sign.

## What the check does
Parser: CGM (5 min), fingersticks, meals, boluses, basal and temp basal to a 5-min grid and the Piece 6 input drives. For each left-out patient, tau, process noise and sensor noise are chosen to minimise fingerstick error on the other patients; the held-out patient is scored on its own fingersticks: raw CGM, smoother with tau = 10, tuned smoother, with and without the input prior. Tests (synthetic): loader reads CGM, fingersticks, meals and boluses; the grid has 5-min samples and about 5% gaps; the drives are non-negative; the leave-one-patient-out run completes and the smoother beats raw CGM against the synthetic fingersticks (8.1 vs 7.9 mg/dL, a plumbing check only).

## Cautions for the real run
Fingersticks are noisy (about 5 to 10%), sparse, usually taken before meals, and on some sensors were used to calibrate the CGM, so agreement with them is not the same as accuracy against blood glucose. Report this as exploratory. Per-patient lag will be as unidentifiable on real data as in simulation; the useful output is the population lag chosen on other patients.
