set -x
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export RAY_OBJECT_STORE_MEMORY=64424509440
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/vllm-workspace/ray_spill"}}'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

swe_train_path=/data/hxy/swebench_verified_filtered-qwen3_4b/datasets/epoch_0/train.parquet
swe_test_path=/data/hxy/swebench_verified_filtered-qwen3_4b/datasets/epoch_0/test.parquet

python3 -m verl.trainer.main_offline_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$swe_train_path \
    data.val_files=$swe_test_path \
    data.train_batch_size=8000  \
    data.max_prompt_length=60000 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='agl_cc_onestepoffline' \
    trainer.experiment_name='qwen3_4b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10000 \
    trainer.total_epochs=1 2>&1 | tee /data/hxy/verl_offline.log


# trainer.logger='["console","wandb"]'
# actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
