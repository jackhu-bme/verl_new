set -x
ENGINE=${1:-vllm}


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/aiscuser/vagen_crop/checkpoints/vagen_cxr_crop_grpo_try6/cxr_crop_grpo_try6/global_step_1500/actor \
    --target_dir /home/aiscuser/vagen_crop/checkpoints/vagen_cxr_crop_grpo_try6/cxr_crop_grpo_try6/global_step_1500/actor_merged_hf

DATA_PATH=$HOME/data/cxr_mini

MODEL_PATH=./checkpoints/verl_grpo_example_cxr_mini_1k_v2/qwen2_5_vl_7b_cxr_mini_1k_v2/global_step_555/actor_merged_hf

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=512 \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/test-output-8.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=2048 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536



