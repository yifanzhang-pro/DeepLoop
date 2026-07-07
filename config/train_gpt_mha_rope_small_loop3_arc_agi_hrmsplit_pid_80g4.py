"""ARC-AGI hrmsplit+puzzleid R=3 config — for the 14-cell sweep."""

from train_adam_config import ArcAgiTrainConfig

_batch_size = 8

CONFIG = ArcAgiTrainConfig(
    wandb_log=True,
    wandb_project="nanogpt-pro",
    save_checkpoints=True,
    keep_last_checkpoints=3,
    dataset="arc-agi-hrmsplit-puzzleid-aug1000",
    block_size=1802,
    model_type="gpt-mha-rope-loop",
    num_hidden_layers=12,
    num_attention_heads=6,
    hidden_size=768,
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
    block_repeats=3,
    use_input_embed_rmsnorm=False,
    use_loop_deepnorm=False,
    use_shared_kv_activations=False,
    batch_size=_batch_size,
    global_batch_size=64,
    gradient_accumulation_steps=64 // (_batch_size * 4),
    max_iters=12000,
    lr_decay_iters=12000,
    eval_interval=500,
    eval_iters=20,
    log_interval=50,
    optimizer_name="adamw",
    learning_rate_base=1e-3,
    min_lr_base=5e-5,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=1000,
    schedule="cosine",
    compile=True,
)

globals().update(CONFIG.model_dump(exclude_unset=True))
