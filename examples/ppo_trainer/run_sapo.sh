set -x

geo3k_train_path=path/geo3k/train.parquet
geo3k_test_path=path/geo3k/test.parquet

train_files="['$geo3k_train_path']"
test_files="['$geo3k_test_path']"

TOPK_PERCENT=30.0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=sapo \
    algorithm.lam=0.99 \
    algorithm.gamma=1.0 \
    +algorithm.sapo_topk_percent=$TOPK_PERCENT \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/path/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    +actor_rollout_ref.actor.ppo_actor_entropy_topk_percent=$TOPK_PERCENT \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.n=1 \
    critic.model.path=/path/Qwen2.5-VL-7B-Instruct \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=2e-6 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    +critic.ppo_critic_entropy_topk_percent=$TOPK_PERCENT \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=k1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name='verl_exp' \
    trainer.experiment_name='sapo_qwen2.5vl-7b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=4 \
    trainer.val_before_train=True \
    trainer.total_epochs=50 $@