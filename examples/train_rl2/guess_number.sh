#!/bin/bash

# GEM Environment PPO Training Example for RL2
# Aligned with VeRL training configuration (only matching parameters)

# Configuration variables
n_gpus=8
batch_size=128
env=game:GuessTheNumber-v0

# Set logging to DEBUG mode for detailed vectorized environment tracking
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)/RL2"

# Run PPO training with GEM environment
torchrun \
    --nproc_per_node=$n_gpus \
    -m RL2.trainer.ppo \
    \
    trainer.project=gem \
    trainer.experiment_name=rl2-qwen3-1.7b-${env}-ppo-epoch-2 \
    trainer.n_epochs=300 \
    trainer.test_freq=9999999 \
    trainer.save_freq=9999999 \
    \
    data.responses_per_prompt=1 \
    data.prompts_per_rollout=1 \
    \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.lr=1e-6 \
    actor.max_length_per_device=8192 \
    actor.sp_size=2 \
    actor.update_per_rollout=2 \
    \
    adv.estimator=reinforce \
    adv.norm_var=true \
    adv.global_norm=true \
    adv.gamma=1.0 \
    \
    rollout.agent_class=gem_agent.GEMAgent \
    rollout.model_name=Qwen/Qwen3-1.7B-Base \
    rollout.tp_size=1 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    +rollout.env_id=${env} \
    +rollout.wrappers="concat" \
    +rollout.max_parallel_agents=16 \
    +rollout.prompt_template=qwen3_general \
    +rollout.apply_chat_template=false \
    +rollout.rollout_batch_size=${batch_size} \
    +rollout.max_model_len=12800