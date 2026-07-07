# Evaluation setup

The eval scripts in this directory depend on a project-adapted fork of
EleutherAI's `lm-evaluation-harness`. The fork is **vendored at
`DeepLoop/lm-evaluation-harness/`** — vanilla upstream
`lm-evaluation-harness` will *not* load DeepLoop checkpoints because
the `nanogpt-pro` model class is not registered with HF AutoConfig
upstream. The fork ships a `new_model.py` adapter slot in
`lm_eval/models/` that the wrapper script `test.sh` populates from
`DeepLoop/model/<model_type>.py` at run time, plus a small patch to
`lm_eval/models/huggingface.py` that imports `GPTConfig` / `GPT` from
`lm_eval.models.new_model` when the standard auto-detect fails.

## 1. Install the bundled harness

```bash
# from DeepLoop/ root
pip install -e lm-evaluation-harness
```

`lm-evaluation-harness` pulls in `accelerate`, `peft`, `sacrebleu`,
`rouge-score`, `pytablewriter`, etc. — first install takes ~2 min.

The base version is `v0.4.5`; the patches and `new_model.py` adapter
are layered on top.

## 2. Pre-fetch the benchmark datasets (required on offline clusters)

The eight downstream tasks are hosted on Hugging Face. Compute nodes
at many clusters (including Princeton della) have no network access,
so the datasets must be pre-fetched on a login node before any SLURM
job can run.

Run this on a machine with internet access; it populates
`$HF_DATASETS_CACHE` (default `~/.cache/huggingface/datasets`):

```python
from datasets import load_dataset

load_dataset("allenai/ai2_arc", "ARC-Easy",      split="validation")
load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
load_dataset("hellaswag",       split="validation")
load_dataset("openbookqa", "main",                split="validation")
load_dataset("piqa",            split="validation", trust_remote_code=True)
load_dataset("sciq",            split="validation")
load_dataset("social_i_qa",     split="validation", trust_remote_code=True)
load_dataset("winogrande", "winogrande_xl",       split="validation", trust_remote_code=True)
```

After this step, the eval sweep can run with `HF_DATASETS_OFFLINE=1`
(which `jobs/small_scale/eval_all.slurm` already sets).

## Two equivalent invocation styles

### A. `lm-evaluation-harness/test.sh` (CLI-style)

Mutates `lm-evaluation-harness/lm_eval/models/new_model.py` from the
requested `<MODEL_TYPE>.py` plus its dependencies, then runs `lm_eval`
with `--model hf --model_args pretrained=<ckpt>`.

```bash
cd DeepLoop/lm-evaluation-harness
bash test.sh \
    ../training_records/dlv1_l5s/checkpoint-best \
    gpt-mha-rope-loop \
    1 \
    results/dlv1_l5s_1shot \
    arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,social_iqa,winogrande
```

### B. `scripts/eval/run_lm_eval.py` (Python driver)

Loads `model.gpt-mha-rope-loop.GPT` directly, hands the
already-instantiated model to `HFLM(pretrained=<model object>)`, and
calls `simple_evaluate`. This avoids mutating the harness state at run
time, and is what `scripts/eval/run_all.sh` uses to sweep all 12 cells.

```bash
python scripts/eval/run_lm_eval.py \
    --checkpoint training_records/dlv1_l5s/checkpoint-best \
    --tasks arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,social_iqa,winogrande \
    --num_fewshot 1 --batch_size 32 --dtype bfloat16 \
    --output_dir eval_results --run_name dlv1_l5s
```

Both routes hit the same `acc` numbers up to numerical determinism
(±0.15 % depending on batch size). Path B is preferred for the
reproducible 12-cell sweep; path A is convenient for one-off runs
with `--batch_size auto` letting `lm_eval` find the largest fitting
batch (typically larger than 32).
