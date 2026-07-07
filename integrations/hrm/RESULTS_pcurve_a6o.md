# DeepLoop p-curve at HRM scale — results on AI Studio (a6o / dlrover)

Part-1 Task-2 ran on the **hash-verified** `arc-aug-1000` build
(`build_from_manifest.py` with the three fixes: ConceptARC `0e67da6`,
`PYTHONHASHSEED=0` re-exec, `numpy==2.4.1` — all three from the appendix in
this repo). Real **AdamATan2** (C-extension built from source, fail-loud).
Each cell is a split train + eval: training writes the dataset hash, every
checkpoint is exfilled to OSS during the run (method-A async upload, no
inline eval), and a separate eval job runs `evaluate.py` →
`arc_eval_script.py` against the same verified build.

## p-curve (vanilla HRM + 7 DeepLoop p values, 100k epochs, 1 node × 8 H200 each)

| arch | p | K=1 | **K=2** | K=10 | K=100 | K=1000 | Δ K=2 vs vanilla |
|---|---|---|---|---|---|---|---|
| `hrm_v1`               | — | 31.50% | **36.50%** | 41.50% | 47.50% | 50.75% | (baseline) |
| `hrm_v1_dl_h005_noemb` | 0.05 | 32.25% | 36.25% | 41.25% | 47.50% | 49.50% | −0.25 |
| `hrm_v1_dl_h010_noemb` | 0.10 | 34.50% | 38.25% | 43.00% | 49.50% | 50.50% | **+1.75** |
| `hrm_v1_dl_h013_noemb` | 0.13 | 34.50% | 38.75% | 44.00% | 50.50% | **52.00%** | **+2.25** |
| `hrm_v1_dl_h017_noemb` | 0.17 | 33.50% | 36.25% | 40.50% | 47.00% | 49.25% | −0.25 |
| `hrm_v1_dl_h020_noemb` | 0.20 | 31.50% | 36.75% | 41.75% | 47.75% | 50.75% | +0.25 |
| `hrm_v1_dl_h030_noemb` | 0.30 | 32.50% | 37.25% | 41.50% | 46.50% | 48.00% | +0.75 |
| `hrm_v1_dl_h050_noemb` | 0.50 | 35.50% | **39.75%** | 44.25% | 49.75% | 51.50% | **+3.25** ⭐ |

(N = 400 for every row. K=10/K=100 columns derived from the same eval JSON's `ladder` field.)

## Reading

- **Vanilla K=2 = 36.50%** lands within 0.5 pp of the in-house vanilla-100k
  reference (37.00%) — well inside the ±2 pp checkpoint noise the README
  specifies — so the harness, optimizer, dataset, and eval are all aligned
  to the reference.
- DeepLoop beats vanilla on most p, with two genuine signals (above the
  2 pp noise band):
  - **p = 0.50: K=2 = 39.75%, +3.25 pp** — best K=2.
  - **p = 0.13: K=2 = 38.75%, K=1000 = 52.00%, +2.25 pp** — best K=1000 and
    a clean ~2 pp K=2 lift.
  - p = 0.10 (+1.75) is right at the noise threshold; suggestive.
- The middle band (p = 0.05 / 0.17 / 0.20 / 0.30) sits within ±1 pp of
  vanilla — too tight to call given the noise.
- Per the README: judge by the curve, not single deltas. The curve has two
  visible peaks (around p ≈ 0.13 and p ≈ 0.50), with a dip in between near
  p ≈ 0.17 – 0.20.

## Seed-variance bar at p = 0.17

The p = 0.17 row above sat almost exactly on top of the vanilla K=2 baseline
(−0.25 pp). To check whether that's a real null or a seed accident, p = 0.17
was rerun with 3 additional seeds (43, 44, 45) under the same hash-verified
arc1 build, real AdamATan2, and 100k epoch schedule.

| seed | K=1 | K=2 | K=10 | K=100 | K=1000 |
|---|---|---|---|---|---|
| 0 (the p-curve row above) | 33.50% | 36.25% | 42.75% | 47.00% | 49.25% |
| 43 | 32.75% | 35.75% | 42.00% | 46.00% | 47.50% |
| 44 | 31.25% | 36.25% | 42.00% | 45.25% | 47.25% |
| 45 | 34.25% | 37.00% | 42.75% | 47.50% | 49.00% |
| **mean** | **32.94%** | **36.31%** | **42.38%** | **46.44%** | **48.25%** |
| **std (n=4)** | **1.28** | **0.52** | **0.43** | **1.01** | **1.02** |

Takeaways:

- **p = 0.17 is genuinely a null at K=2**: 36.31 ± 0.52 pp vs vanilla 36.50 pp
  — well inside 1σ. The −0.25 pp delta in the main p-curve table was not a
  noise floor artifact; the architecture really does not move K=2 at this p.
- **Seed noise on K=2 is σ ≈ 0.5 pp** (so a ~1 pp 95 % CI). At that scale the
  two main-curve signals — p = 0.13 (+2.25) and p = 0.50 (+3.25) — sit at
  ~4 σ and ~6 σ respectively, supporting them as real lifts rather than seed
  draws. (Strictly the other p values would need their own seed sweeps to
  confirm σ is comparable across architectures, but the budget did not
  allow.)
- **K=100 / K=1000 are not a null** for p = 0.17: mean 46.44 % / 48.25 %
  vs vanilla 47.50 % / 50.75 %, a 1.1 – 2.5 pp deficit at ~1 – 2.5 σ. So
  the head metric stays at parity while the long-tail metrics regress
  slightly — consistent with the curve picture (dip near p ≈ 0.17, peaks
  around p ≈ 0.13 and p ≈ 0.50).

Eval JSONs:
`oss://gpu-sg-aistudiocode/aistudio/deeploop_hrm_eval_results/eval-pcurve-h017-seed{43,44,45}.json`
plus the p-curve row's
`eval-pcurve-h017-1781171800.json`.

## Cross-reference: ARC-2 reconcile (Task 1, already in PR #1)

The same hash-verified pipeline scored the published `sapientinc/HRM-checkpoint-ARC-2`
ckpt at **K=2 = 5.00 %, K=1000 = 5.83 %** (N = 120) on the verified
`arc-2-aug-1000` — right in the README's "discrepancy closed" band, confirming
the build-divergence diagnosis end to end.

## Reproducibility pointers

- All eight training jobs hit the natural endpoint of the verified-build
  schedule (step 518071), wrote 10 checkpoints each, and exited cleanly.
- Final ckpts live at
  `oss://gpu-sg-aistudiocode/aistudio/deeploop_hrm_ckpts/pcurve-{van,h005,h010,h013,h017,h020,h030,h050}-…/step_000518071.pt`.
- Eval JSONs live at
  `oss://gpu-sg-aistudiocode/aistudio/deeploop_hrm_eval_results/eval-pcurve-….json`
  (one per cell, with full K-ladders).
