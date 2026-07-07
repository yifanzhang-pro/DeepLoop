# DeepLoop ↔ HRM integration

This directory contains drop-in patches that inject DeepLoop's residual
scaling (LoopDeepNorm + optional input-embedding RMSNorm) into a stock
clone of [sapientinc/HRM](https://github.com/sapientinc/HRM).

The patched model is a strict *additive* extension of HRM's
`models/hrm/hrm_act_v1.py` — when `use_loop_deepnorm=False` and
`input_embed_rmsnorm_placement=none` (defaults), it behaves byte-identically
to upstream. The new arch yamls flip the relevant flags on.

## Files

| File | Drops into |
|---|---|
| `hrm_act_v1_dl.py` | `<HRM_DIR>/models/hrm/hrm_act_v1_dl.py` |
| `hrm_v1_dl.yaml`              | `<HRM_DIR>/config/arch/`  (Phase 1: symmetric DN p=0.5, embed RMSNorm on) |
| `hrm_v1_dl_h010_noemb.yaml`   | `<HRM_DIR>/config/arch/`  (empirical winner: H-only DN p=0.10, no embed RMSNorm) |
| `hrm_v1_dl_h0{05,10,13,17,20,30,50}_noemb.yaml` | `<HRM_DIR>/config/arch/`  (the 7-point p curve: H-only DN, no embed RMSNorm) |
| **`psweep_train.sh` + `pst_h0{05,10,13,17,20,30,50}.slurm` + `pst_van.slurm`** | **`<HRM_DIR>/`  (TRAIN jobs, one per p-point + vanilla control; 8 GPU × 100000 epoch; preflight + ARC-1 content hash gate + checkpoint verification)** |
| **`psweep_eval.sh` + `psweval.slurm`** | **`<HRM_DIR>/`  (generic EVAL job, decoupled from training: 4 GPU, evaluate.py + count-first voting on any run/checkpoint)** |
| `_preflight.sh`         | `<HRM_DIR>/`  (AdamATan2 C-extension gate sourced by `psweep_train.sh` — catches the silent AdamW fallback that ruins reproduction) |
| `arc_eval_script.py`    | `<HRM_DIR>/`  (FIXED count-first voting + a1 dedup; supersedes the buggy `eval_k2.py`) |
| `vanilla_full.{sh,slurm}`, `h0{05,13,17,20,30,50}n_full.{sh,slurm}` | SUPERSEDED — train-only, no preflight/content-gate; the runs they produced were biased by the silent-AdamW bug. Use `pst_*` + `psweval` instead |
| `eval_k2.{py,slurm}`    | SUPERSEDED — grid-canonicalization bug floored all scores (~1%); use `arc_eval_script.py` |

## What's added

The patched model adds three independent knobs on top of upstream HRM:

1. **LoopDeepNorm residual scaling** — each block runs
   `hidden = rms_norm(α · hidden + sublayer(hidden))`,
   with α = (2 M)^p, β = (8 M)^-p, and
   M = `L_layers · H_cycles · L_cycles + H_layers · H_cycles`.
   For HRM ARC defaults (4 × 2 × 2 + 4 × 2): **M = 24**.
2. **Asymmetric application** via `loop_deepnorm_apply_to ∈
   {both, h, l, none}`: gates whether the scaling applies to the H module,
   the L module, both, or neither. Untargeted modules reset α=1 and skip
   the β-rescale.
3. **Input-embedding RMSNorm placement** via
   `input_embed_rmsnorm_placement ∈ {none, before_puzzle, after_puzzle}`:
   controls whether (and where) an RMSNorm is applied to the token-embedding
   table output relative to the puzzle-embedding concat.

The β-scaled weight init applies to `attn.o_proj.weight` and
`mlp.down_proj.weight` of every targeted block.

## Install into your HRM clone

```bash
cp integrations/hrm/hrm_act_v1_dl.py        $HRM_DIR/models/hrm/
cp integrations/hrm/hrm_v1_dl*.yaml         $HRM_DIR/config/arch/
cp integrations/hrm/train_p_sweep.slurm     $HRM_DIR/jobs/
```

Then either:

```bash
# Phase 1 default (symmetric, p=0.5, embed RMSNorm on):
cd $HRM_DIR && python pretrain.py arch=hrm_v1_dl epochs=27000 eval_interval=3000

# Empirical winner (H-only, p=0.10, no embed RMSNorm):
cd $HRM_DIR && python pretrain.py arch=hrm_v1_dl_h010_noemb epochs=27000 eval_interval=3000
```

Switch back to vanilla HRM by removing the `arch=...` override (defaults to
`hrm_v1`).

---

## Run the p-curve — corrected sweep, train and eval DECOUPLED (`pst_*` / `psweval`)

The original p-curve runs were biased by the silent-AdamW fallback and broken
voting; this sweep redoes them with every audit fix baked in, and with
**training and evaluation as separate jobs** (a 22 h train failure no longer
takes the eval down with it; any checkpoint can be re-evaluated in 45 min
without retraining).

Gates in `psweep_train.sh`:
- **AdamATan2 preflight** (`_preflight.sh`) — aborts before training if the
  `adam_atan2_backend` C-extension isn't importable.
- **ARC-1 dataset CONTENT gate** — md5 of `identifiers.json` must equal the
  reference build's (presence/count checks are NOT enough; a divergent build
  is not comparable to the in-house baselines). Rebuild deterministically with
  `integrations/vanilla_hrm/build_from_manifest.py --dataset arc1` if it fails.
- **Checkpoint saving verified**, not assumed: `checkpoint_every_eval=True` is
  passed explicitly; a watchdog warns if no checkpoint appeared after the
  first eval interval; after training the script asserts the expected
  checkpoint count, file size, and that the final checkpoint actually loads —
  then writes a run-dir marker for the eval job.

Install, then submit:

```bash
cp integrations/hrm/hrm_act_v1_dl.py   $HRM_DIR/models/hrm/
cp integrations/hrm/hrm_v1_dl*.yaml    $HRM_DIR/config/arch/
cp integrations/hrm/{_preflight.sh,arc_eval_script.py,psweep_train.sh,psweep_eval.sh,pst_*.slurm,psweval.slurm} $HRM_DIR/

cd $HRM_DIR
export PYTHON=/path/to/venv/python      # MUST have adam_atan2_backend importable

# TRAIN (8 GPU × 100000 epoch, ~22 h each; ailab cap 16 GPU → ≤2 concurrent):
sbatch pst_van.slurm    # vanilla control
sbatch pst_h005.slurm   # p=0.05
sbatch pst_h010.slurm   # p=0.10  (clean-AdamATan2 redo of the h010n anchor)
sbatch pst_h013.slurm   # p=0.13
sbatch pst_h017.slurm   # p=0.17
sbatch pst_h020.slurm   # p=0.20
sbatch pst_h030.slurm   # p=0.30
sbatch pst_h050.slurm   # p=0.50

# EVAL (4 GPU, ~45 min; after the corresponding train job lands):
sbatch --export=ALL,ARCH=hrm_v1 psweval.slurm
sbatch --export=ALL,ARCH=hrm_v1_dl_h013_noemb psweval.slurm
# ... or evaluate a specific checkpoint / explicit run dir:
sbatch --export=ALL,ARCH=hrm_v1_dl_h013_noemb,STEP=259035 psweval.slurm
sbatch --export=ALL,RUN_DIR="checkpoints/.../HierarchicalReasoningModel_ACTV1 <coolname>" psweval.slurm
```

Compare each K=2 against the in-harness vanilla-100k control (37.00%);
checkpoint-to-checkpoint noise is ~2 pp, so judge p-points by the curve.
