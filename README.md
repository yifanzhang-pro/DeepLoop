# DeepLoop: Depth Scaling for Looped Transformers

DeepLoop is a residual-scaling rule for **weight-tied looped Transformers**.
Looped models increase effective depth by revisiting the same physical blocks,
and the DeepNorm exponent that stabilizes *untied* depth ($p=1/4$) is no longer
the right one. DeepLoop makes the tied-depth effect explicit through a
visit-alignment analysis and keeps the Post-LN DeepNorm architecture with a
single loop-aware scaling rule:

$$
\mathbf{x}_{i+1} = \mathrm{Norm}\big(\alpha\,\mathbf{x}_i + f(\mathbf{x}_i)\big),
\qquad \alpha = (2N)^{1/2},\quad \beta = (8N)^{-1/2},
$$

where $N$ is the unrolled (effective) depth and $\beta$ is the initialization
gain of each residual-branch output projection. The exponent $p=1/2$ is the
aligned-visit threshold; it recovers DeepNorm's $1/4$ when visits decorrelate.
The same rule applies to hierarchical recurrent reasoners (HRM), where it
improves voted accuracy on ARC-AGI.

## Repository structure

```
model/                 model definitions (GPT-MHA-RoPE, looped variants, LoopDeepNorm scaling)
config/                training configurations (FineWeb-Edu LM and ARC-AGI)
utils/                 tokenizers, optimizer, checkpoint, distributed, reproducibility utilities
scripts/               training / evaluation launchers
  lib/runtime.sh       shared venv/env setup
  eval/                lm-evaluation-harness and ARC-AGI eval runners
integrations/
  hrm/                 DeepLoop applied to the Hierarchical Reasoning Model (ARC-AGI)
  vanilla_hrm/         reproducible vanilla-HRM baseline package (build + eval)
train_adam_finewebedu.py   FineWeb-Edu language-modeling entry point
train_adam_arc_agi.py      ARC-AGI entry point
dataloader.py, arc_agi_dataloader.py   token-stream / ARC data loaders
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .
```

## Usage

### Language modeling (FineWeb-Edu)

Multi-GPU training via `torchrun`; each experiment is a config file:

```bash
torchrun --standalone --nproc_per_node=4 \
    train_adam_finewebedu.py config/train_gpt_mha_rope_small_loop3_adam_50BT_ctx1024_80g4.py
```

The `base` baseline keeps $\alpha=\beta=1$; the DeepLoop variant enables the
$\alpha=(2N)^{1/2},\ \beta=(8N)^{-1/2}$ scaling on every block. Loop count $R$
is set in the config.

### Reasoning (ARC-AGI via HRM)

`integrations/hrm/` provides the DeepLoop residual-scaling drop-in for the
[Hierarchical Reasoning Model](https://github.com/sapientinc/HRM)
(`hrm_act_v1_dl.py` + `hrm_v1_dl*.yaml`) and the p-sweep launchers.
`integrations/vanilla_hrm/` provides a hash-verified vanilla-HRM baseline
package (dataset rebuild + evaluation). See the README in each directory for
step-by-step instructions.

## Citation

```bibtex
@article{deeploop2026,
  title  = {DeepLoop: Depth Scaling for Looped Transformers},
  author = {Li, Shuzhen and others},
  year   = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
