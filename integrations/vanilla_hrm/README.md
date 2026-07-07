# Vanilla HRM reproduction package

Four self-contained experiments to definitively answer **"can we reproduce
the HRM paper numbers (32 % K=1 / 40.3 % K=2 on ARC-AGI-1; 5.0 % K=2 on
ARC-AGI-2) on our hardware with our env?"**

The four scripts are designed so that **if all four pass, our setup is
known-correct end-to-end.** If any one fails, the failure isolates where
the divergence is, so we know what to fix next.

## TL;DR — order of operations

| # | Script | Hardware | Wall time | What it validates |
|---|---|---|---|---|
| 1 | `1_validate_eval.sh` | 4 GPU | ~1 h | eval pipeline correctness, by running our eval on a paper-published checkpoint |
| 2 | `2_smoke_train_arc1.sh` | 8 GPU | ~3 h | training pipeline + env (short 8 k epoch vanilla run) |
| 3 | `3_full_train_arc1.sh` | 8 GPU | ~22 h | **THE answer**: full paper-recipe reproduction on ARC-AGI-1 |
| 4 | `4_full_train_arc2.sh` | 8 GPU | ~22 h | independent cross-check: same recipe on ARC-AGI-2 |
| 5 | `5_rebuild_arc2.sh` | CPU, 120 GB RAM | ~20 min | **byte-identical** ARC-2 dataset rebuild (manifest-pinned order + md5 gate) |
| 6 | `6_reeval_arc2.sh` | 4 GPU | ~45 min | corrected experiment 1: published-checkpoint eval on the hash-verified dataset |

Run 1-4 in order; stop and investigate if any one is far from its expected
result before running the next. **If your experiment-1 K=2 came out far below
5 % (e.g. ~1.7 %), run 5 then 6 — your dataset build almost certainly diverged
(see the dataset-reproducibility warning below).**

## Why we don't trust our previous reproduction

A previous in-house attempt (the DeepLoop+HRM exploration, including a
patched arch `hrm_v1_dl`) trained for 100 k epoch on ARC-AGI-1 and got:

* per-example exact_accuracy = 27.3 % (vs paper's 32 %)
* voted K=2 = 21.5 % (raw majority) or **15.3 %** (paper protocol) (vs paper 40.3 %)

After auditing the entire codebase against pristine upstream
`sapientinc/HRM @ ac15626`, exactly **one** Python file differed —
`pretrain.py`, which had a try/except fallback that silently used
`torch.optim.AdamW` whenever `from adam_atan2 import AdamATan2` raised
`ModuleNotFoundError`. That import fails on most environments because:

1. The PyPI package `adam_atan2==0.0.3` has a broken metadata version
   ("0.0.0" inside, "0.0.3" filename) so `pip install adam_atan2` always
   fails on pip's version-consistency check.
2. The Python wrapper imports cleanly even when its C-extension
   `adam_atan2_backend` failed to compile, so the import error only
   appears at first use of the optimizer — which our fallback hid.

Result: every previous training run silently used AdamW, not AdamATan2.
That's the most plausible reason for the gap. **These four experiments
all assume the C-extension is properly built and verified at preflight,
so the silent fallback can never happen.**

## Prerequisites

### Environment setup (once, per machine)

```bash
# 1. Clone vanilla HRM at exactly the commit our audit was done against:
git clone https://github.com/sapientinc/HRM.git
cd HRM
git checkout ac15626
git submodule update --init --recursive   # raw ARC-AGI / ARC-AGI-2 / ConceptARC data

# 2. Python 3.12 venv with torch + HRM requirements.
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # torch, einops, pydantic, omegaconf, hydra-core, huggingface_hub, ...
pip install numba                  # arc_eval_script.py uses @njit

# 3. adam_atan2 with working C-extension — THE CRITICAL STEP.
module load cudatoolkit/12.6   # use whatever matches your torch's torch.version.cuda
cd /tmp && rm -rf adam_atan2-0.0.3*
curl -sL -O https://files.pythonhosted.org/packages/06/a2/8ef8f0bb022f39d87edda9e04e77458195eb0011d13561c4e7a85e6c1c22/adam_atan2-0.0.3.tar.gz
tar xzf adam_atan2-0.0.3.tar.gz
cd adam_atan2-0.0.3
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.3 \
    python -m pip install --no-cache-dir --no-build-isolation .
# Verify (preflight will also do this):
python -c "import adam_atan2_backend; print(adam_atan2_backend.__file__)"

# 4. Build the ARC datasets — USE THE MANIFEST DRIVER, not the upstream
# script directly. Upstream build_arc_dataset.py feeds filesystem-ordered
# (unsorted!) file lists into a seeded RNG, so the same script + same seed
# produces a DIFFERENT dataset on a different machine (two same-seed builds
# were measured to share only 0.14 % of augmentations). The driver pins the
# enumeration order via a recorded manifest and md5-verifies every output
# file, so the build is byte-identical everywhere or fails loudly.
cd $HRM_DIR
python repro_vanilla/build_from_manifest.py --hrm-dir $HRM_DIR --dataset arc1  # → data/arc-aug-1000
python repro_vanilla/build_from_manifest.py --hrm-dir $HRM_DIR --dataset arc2  # → data/arc-2-aug-1000  (~120 GB RAM)
# Both print "ALL HASHES MATCH" on success. NOTE: do NOT use
# num_puzzle_identifiers as your alignment check — it is shape-based and
# stays 1,045,829 even for builds with ~disjoint augmentation content.
```

### Drop these scripts into HRM

```bash
mkdir -p $HRM_DIR/repro_vanilla/manifests
cp 1_validate_eval.{sh,slurm} 2_smoke_train_arc1.{sh,slurm} \
   3_full_train_arc1.{sh,slurm} 4_full_train_arc2.{sh,slurm} \
   5_rebuild_arc2.{sh,slurm} 6_reeval_arc2.{sh,slurm} \
   _preflight.sh arc_eval_script.py \
   build_from_manifest.py reference_hashes.json \
   $HRM_DIR/repro_vanilla/
cp manifests/*.json $HRM_DIR/repro_vanilla/manifests/
```

### Submit (export env first)

```bash
export HRM_DIR=/path/to/HRM
export PYTHON=$HRM_DIR/.venv/bin/python
cd $HRM_DIR
sbatch repro_vanilla/1_validate_eval.slurm
# wait for it to land, inspect the result, then:
sbatch repro_vanilla/2_smoke_train_arc1.slurm
# and so on.
```

## Expected results and decision rules

### Experiment 1 — eval-pipeline validation

Runs HRM's `evaluate.py` + paper-protocol voting on
`sapientinc/HRM-checkpoint-ARC-2` (paper-published ARC-AGI-2 checkpoint).

| K=2 result | Interpretation | Action |
|---|---|---|
| 4-6 % | matches paper's 5.0 % within ±1 pp | eval pipeline correct; proceed |
| < 3 % or > 7 % | pipeline divergence | stop, inspect: which crop / hash / voting step gives different numbers from the notebook |

### Experiment 2 — training pipeline smoke

8 k epoch vanilla ARC-AGI-1 run. The 8 k-epoch baseline number is known
internally to be ~9.4 % per-example exact_accuracy.

| exact_accuracy after 8 k | Interpretation | Action |
|---|---|---|
| 8-12 % | training pipeline produces the expected 8 k baseline | proceed |
| < 5 % | training is broken (most likely silent AdamW fallback) | confirm `import adam_atan2_backend` works, check wandb loss curve |
| > 15 % | also unexpected, may indicate eval split leakage | inspect data prep, re-run `build_arc_dataset.py` |

### Experiment 3 — full ARC-AGI-1 reproduction

100 k epoch vanilla ARC-AGI-1, exactly the README command
(`torchrun --nproc-per-node 8 pretrain.py`, no overrides).

| voted K=2 | Interpretation | Action |
|---|---|---|
| 38-42 % | full reproduction confirmed (paper: 40.3 %) | use this checkpoint as the canonical vanilla baseline; compare any custom recipe against it |
| 30-38 % | close but not quite, possible seed variance | re-run with `seed=43` to check variance bar |
| < 30 % | systematic gap | inspect: did wandb show AdamATan2's expected loss curve? was global_batch_size 768? |

### Experiment 4 — ARC-AGI-2 cross-check (optional but recommended)

Same recipe, ARC-AGI-2 data. Paper headline: 5.0 % K=2.

| voted K=2 | Interpretation |
|---|---|
| 4-6 % | reproduction holds on a second, independent dataset → very high confidence |
| disagrees with experiment 3 | look at which dataset's gap is bigger and chase that |

### Experiments 5 & 6 — dataset-reproducibility fix + corrected published-ckpt eval

**Why these exist (the 5.00 % vs 1.67 % incident):** two sites ran experiment 1
on the SAME published checkpoint and got K=2 = 5.00 % vs 1.67 %. Root cause:
upstream `build_arc_dataset.py` enumerates puzzle files with unsorted
`os.scandir()`/`glob()` and feeds that filesystem-dependent order into the
seeded RNG — the two sites' "same-seed" datasets shared only **0.14 %** of
their augmentations. The published checkpoint's per-augmentation `puzzle_emb`
(99.998 % non-zero rows, including test puzzles — HRM is transductive) is
welded to its training build, so a divergent build silently degrades K=2 while
K=1000 stays ~5 %. Full forensics: `h010nf_voting_recompute.md` (CLOSED
section) and `RESULTS_a6o.md`.

* **Experiment 5** rebuilds `data/arc-2-aug-1000` with the enumeration order
  pinned by `manifests/arc2_enum_manifest.json` and md5-verifies every output
  against `reference_hashes.json`. Outcome is binary: byte-identical to the
  reference build (the one that reproduces the paper's 5.0 %), or loud failure.
* **Experiment 6** re-runs the published-checkpoint eval, hard-gating on those
  hashes before spending GPU time.

| Exp 6 K=2 result | Interpretation |
|---|---|
| 4.2-5.8 % (5.00 % ± 1 puzzle) | aligned with reference/paper protocol — done |
| < 3.3 % | a gate was bypassed or the checkpoint file is wrong; do not trust |

Note on metrics: **K=1000 (~5 %) is the build-robust ARC-2 headline**; K=2 at
this noise floor carries ±1-puzzle cross-hardware jitter even on the verified
build.

## What this package does NOT do

* **Does not** include any DeepLoop / LoopDeepNorm patches. These are pure
  vanilla HRM runs — the baseline against which any modified recipe must
  be measured.
* **Does not** sweep seeds. ±2 pp seed variance on the 1k-aug ARC datasets
  is documented in the HRM README; if your numbers land just outside the
  expected interval, run a second seed before concluding anything.
* **Does not** try to reproduce HRM's Sudoku or Maze numbers. Same
  pipeline works for those (different `data_path` and a few CLI
  overrides per the HRM README), but the goal of this package is ARC-AGI
  paper reproduction.
