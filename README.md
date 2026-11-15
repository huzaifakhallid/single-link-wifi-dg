# Class-discriminative single-link WiFi domain generalization for CSI gesture recognition

Phase 3 (implementation and experimental evaluation) of a Computer Networks research project.

**Research question.** Can source-trained class-consistency learning, combined with same-link phase/Doppler
fusion and an augmentation curriculum, reduce confusion between similar gestures (clap vs slide) and improve
single-link cross-orientation accuracy without any target-domain calibration?

**Setting.** Strict domain generalization on Widar3.0, Room 1 (session 20181130, 9 users, gestures 1-6).
One transmitter and one receiver (`r1`); two `r1` antennas form a CSI ratio. No other receivers, no BVP, no
dataset DFS files. Models, normalisation statistics and prototype banks are frozen before a target trial is touched.

| Task | Train orientations | Source validation | Target |
|---|---|---|---|
| O1 | 3, 4, 5 | 2 | 1 |
| O3 | 1, 2, 5 | 4 | 3 |
| O5 | 2, 3, 4 | 1 | 5 |

20% of the trials of every (gesture, train orientation) cell form a locked in-domain test set.

## Repository layout

| File | Purpose |
|---|---|
| `preprocess.py` | Signal pipeline: decoding, timestamp repair, Hampel, CSI ratio, 200 Hz resampling, phase and Doppler maps. The locked variant is `preprocess.LOCKED`. |
| `prepare_data.py` | Extraction from the Widar zips (CRC-checked), provenance (SHA-256), audit, feature cache, manifest, splits. |
| `diagnose.py` | Preprocessing gate: plain ERM per preprocessing variant on orientations 2 and 4 only (never targets). |
| `train.py` | Models, episodic training (C0-C5), DG baselines (B1-B4), unit tests, resumable job queue. |
| `analyze.py` | All tables (CSV + LaTeX) and figures (300 dpi) from saved run files. |
| `build_report.py` | Phase-3 report (.docx) generated from the saved tables and figures. |
| `single_link_dg.ipynb` | Colab orchestrator: setup, data checks, tests, quick mode, pilot, queue, analysis, export. |
| `cache/manifest.csv`, `cache/splits.csv` | One row per trial (exclusions kept with their reason) and the immutable split assignment. |
| `cache/features_meta.json`, `cache/data_provenance.json` | Locked pipeline settings + SHA-256 of the feature cache; SHA-256 and member counts of the raw zips. |
| `docs/diagnostics/` | Gate table, per-fit curves and before/after maps. |

Not in git: `dataset/` (raw data), `cache/features.npy` (feature cache), `results*/` (run outputs), `docs/*.pdf`.

## Configurations

| ID | Input | Consistency | Query augmentation | Role |
|---|---|---|---|---|
| C0 | phase | off | none | single-feature baseline |
| C1 | phase + Doppler | off | none | main matched baseline |
| C2 | phase + Doppler | off | curriculum | removes consistency from C5 |
| C3 | phase + Doppler | on | none | removes the curriculum from C5 |
| C4 | phase + Doppler | on | fixed level after warm-up | schedule control |
| C5 | phase + Doppler | on | curriculum | proposed |
| B1-B4 | phase + Doppler | - | - | ERM, Deep CORAL, DANN, Mixup on the same encoder with a linear head |

Queue: 10 configurations x 3 tasks x 3 seeds (42, 43, 44) = 90 trainings, plus C5 with lambda in {0.05, 0.3}
(seed 42, 3 tasks) = 96 trainings. The lambda = 0.1 sensitivity point is the main C5 seed-42 run.

## Reproducing

Local (Windows or Linux, CPU, Python 3.12, `pip install -r requirements.txt`):

```bash
# 1. data: put the three 20181130 zips in dataset/widar_r1/, then
python prepare_data.py --extract --provenance --audit
# 2. preprocessing gate (orientations 2 and 4 only)
python diagnose.py features
python diagnose.py fit --set single
python diagnose.py fit --set fused --epochs 45 --chosen <hampel>,<phase>,<doppler>
python diagnose.py maps
# 3. locked cache, manifest, splits
python prepare_data.py --preprocess
# 4. tests and smoke test
python train.py --unit_tests
python train.py --queue --run_kind quick --epochs 3 --data_fraction 0.1 --schedule compressed --tasks O1 --seeds 42 --no_sensitivity
```

GPU (Colab T4): open `single_link_dg.ipynb` with the VS Code Colab extension and run it top to bottom.
The notebook clones this private repository with a read-only token from Colab Secrets (`GH_TOKEN`), checks the
SHA-256 of the uploaded `features.npy`, runs the unit tests, quick mode, a timed pilot and the resumable
two-worker queue, builds all tables and figures and zips `results/` for download.

After downloading the results:

```bash
python analyze.py --results results/final --run_kind final --diag docs/diagnostics
python build_report.py --results results/final
```

## Result integrity

- No synthetic data path exists. A trial either yields features or is listed in the manifest with an exclusion reason.
- Final runs assert the full protocol (30 epochs, all data, full schedule). Quick and pilot runs are written to
  separate folders, tagged with `run_kind`, and `analyze.py` refuses to mix kinds or feature caches.
- Every table, figure and report sentence containing a number is generated from `metrics.json` / `preds.npz`.
- Published numbers of other papers appear only in the literature-context table and are labelled as reported values.
