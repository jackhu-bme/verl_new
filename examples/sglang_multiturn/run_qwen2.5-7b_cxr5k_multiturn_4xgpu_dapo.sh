# run on 4xH100
# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# TODO: check this parameter here: data.return_multi_modal_inputs

VERSION="7.0.3"
DATASET="cxr_5k_tool"
MODEL="qwen2.5-vl-7b-instruct"
DESCRIPTION="dapo_second_try_full_test"
HARDWARE="4xgpu"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='cxr1k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric="seq_final_reward_bool" \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode="ignore_strippable" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${DATASET}_${MODEL}_${VERSION}_${DESCRIPTION}" \
    trainer.experiment_name="${MODEL}_${VERSION}_${DESCRIPTION}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.log_val_generations=20 \
    trainer.rollout_data_dir="$PROJECT_DIR/generations/${DATASET}_${MODEL}_${VERSION}_${DESCRIPTION}/rollout" \
    trainer.validation_data_dir="$PROJECT_DIR/generations/${DATASET}_${MODEL}_${VERSION}_${DESCRIPTION}/val" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    data.train_files=$HOME/data/cxr_5k_tool/train_both_prompt.parquet \
    data.val_files=$HOME/data/cxr_5k_tool/test_both_prompt.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/cxr1k_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
    $@

python train_nv.py
