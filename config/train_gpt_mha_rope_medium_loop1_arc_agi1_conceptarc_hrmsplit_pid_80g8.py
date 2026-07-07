"""ARC-AGI ARC-AGI-1 + ConceptARC R=1 medium (350M) config — for the 32-experiment matrix.

Dataset: arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000.
Scale: medium (350M, 8× 80G). Method-flag (dlv1 vs base) is set on the slurm CLI.

Half the per-GPU batch (4 vs 8) of the small variant because block_size=1802
is 1.76× the FineWeb-Edu medium block, so memory pressure is higher.
"""

from train_adam_config import ArcAgiTrainConfig

_batch_size = 4

CONFIG = ArcAgiTrainConfig(
    wandb_log=True,
    wandb_project="nanogpt-pro",
    save_checkpoints=True,
    keep_last_checkpoints=3,
    dataset="arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000",
    block_size=1802,
    model_type="gpt-mha-rope-loop",
    num_hidden_layers=24,
    num_attention_heads=8,
    hidden_size=1024,
    head_dim=128,
    dropout=0.0,
    bias=False,
    using_groupnorm=False,
    use_qk_rmsnorm=True,
    embedding_init_std=0.02,
    hidden_init_std_factor=0.5,
    mup=True,
    hidden_size_base=1024,
    embedding_lr_multiplier=1.0,
    block_repeats=1,
    use_input_embed_rmsnorm=False,
    use_loop_deepnorm=False,
    use_shared_kv_activations=False,
    batch_size=_batch_size,
    global_batch_size=64,
    gradient_accumulation_steps=64 // (_batch_size * 8),
    max_iters=100000,
    lr_decay_iters=100000,
    eval_interval=5000,
    eval_iters=50,
    log_interval=200,
    optimizer_name="adamw",
    learning_rate_base=1e-3,
    min_lr_base=1e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=5000,
    schedule="cosine",
    compile=True,
)

globals().update(CONFIG.model_dump(exclude_unset=True))
