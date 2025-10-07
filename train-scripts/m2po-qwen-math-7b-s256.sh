set -x

unset ROCR_VISIBLE_DEVICES

project_name=m2po-test
experiment_name=m2po_qwen-math-7b_s256

deepscaler_preview_train_path=./data/deepscaler_preview/train.parquet

train_files="['$deepscaler_preview_train_path']"

math500_test_path=./data/math500/test.parquet
amc23_test_path=./data/amc23/test.parquet
amc24_test_path=./data/amc24/test.parquet
aime2024_test_path=./data/aime2024x4/test.parquet
aime2025_test_path=./data/aime2025x4/test.parquet
gaokao_test_path=./data/gaokao/test.parquet
minerva_test_path=./data/minervamath/test.parquet
olympiad_test_path=./data/olympiadbench/test.parquet

test_files="['$math500_test_path', '$amc23_test_path', '$amc24_test_path', '$aime2024_test_path', '$gaokao_test_path', '$minerva_test_path', '$olympiad_test_path', '$aime2025_test_path']"

mkdir -p data-log/$project_name

# we set stale iteration to 64, which is total 64 * 4 = 256 model updates
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1536 \
    data.max_response_length=2560 \
    +data.stale_iteration=64 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    +actor_rollout_ref.actor.use_m2po_loss=True \
    +actor_rollout_ref.actor.M2_budget=0.04 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=30000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +trainer.max_steps=1201 \
    trainer.total_epochs=2000  2>&1 | tee data-log/$project_name/$experiment_name.log