set -x

export TORCHINDUCTOR_CACHE_DIR=/scratch/beidchen/torchinductor_cache
WANDB_DIR=/scratch/beidchen/projects/wandb
export WANDB_DIR
export WANDB_MODE=offline

project_name=verl_grpo_example_gsm8k
experiment_name=qwen3_8b_dapo_train_32trails_2node

gsm8k_train_path=/scratch/beidchen/projects/data/gsm8k/train.parquet
gsm8k_test_path=/scratch/beidchen/projects/data/gsm8k/test.parquet
math_train_path=/scratch/beidchen/projects/data/math/train.parquet
math_test_path=/scratch/beidchen/projects/data/math/test.parquet

dapo_math_train_path=/scratch/beidchen/projects/data/dapo_math/train.parquet
# openr1_train_path=/scratch/beidchen/projects/data/openr1_math/train.parquet

aime2024_test_path=/scratch/beidchen/projects/data/aime2024/test.parquet
amc_test_path=/scratch/beidchen/projects/data/amc/test.parquet
math500_test_path=/scratch/beidchen/projects/data/math500/test.parquet
minerva_test_path=/scratch/beidchen/projects/data/minervamath/test.parquet
olympiad_test_path=/scratch/beidchen/projects/data/olympiad/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/checkpoints/beidchen/checkpoints/checkpoints/qwen3_models/Qwen3-8B \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-32B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.default_local_dir="/checkpoints/beidchen/checkpoints/checkpoints/checkpoints/$project_name/$experiment_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@  2>&1 | tee "/scratch/beidchen/projects/multinode.log"
