#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export NCCL_PROTO=Simple

# set CUDA devices
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPU_NUMS=${GPU_NUMS:-4}

# path to your Qwen3-VL-4B-Instruct
MODEL_PATH=${MODEL_PATH:-models/Qwen/Qwen3-VL-4B-Instruct}

# wandb and exp settings
PROJECT_NAME=${PROJECT_NAME:-weather-r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_weather_r1_dapo_locorft_500hpa_situation}

# reward weights and judge model settings
REWARD_WEIGHTS=${REWARD_WEIGHTS:-'{"format":0.1,"logic":0.3,"accuracy":0.6}'}
CLIENT_MODEL=${CLIENT_MODEL:-openai/gpt-oss-20b}

# set OpenAI API key and base for judge model (vLLM serving)
export WEATHER_R1_OPENAI_API_KEY=${WEATHER_R1_OPENAI_API_KEY:-EMPTY}
export WEATHER_R1_OPENAI_API_BASE=${WEATHER_R1_OPENAI_API_BASE:-http://0.0.0.0:50907/v1}

# data files and image dir
TRAIN_FILE=${TRAIN_FILE:-data/WeatherQA/train/split/train_500hpa_situation.json}
# TRAIN_FILE=data/WeatherQA/train/split/train_850hpa_situation.json
# TRAIN_FILE=data/WeatherQA/train/split/train_land_situation.json
# TRAIN_FILE=data/WeatherQA/train/split/train_max_temp.json
# TRAIN_FILE=data/WeatherQA/train/split/train_min_temp.json
# TRAIN_FILE=data/WeatherQA/train/split/train_phenomena.json
# TRAIN_FILE=data/WeatherQA/train/split/train_rain.json
VAL_FILE=${VAL_FILE:-data/WeatherQA/val.json}
IMAGE_DIR=${IMAGE_DIR:-data/WeatherQA/image}

# other settings
CONFIG_PATH=src/weather_r1/weather_r1_config.yaml
REWARD_PATH=src/weather_r1/weather_r1_reward.py
FORMAT_PROMPT=src/weather_r1/weather_r1_format.jinja
SAVE_CHECKPOINT_PATH=${SAVE_CHECKPOINT_PATH:-results/checkpoints/${PROJECT_NAME}/$(date +%Y%m%d_%H%M%S)-${EXPERIMENT_NAME}}
ROLLOUT_NUM=${ROLLOUT_NUM:-5}
MINI_ROLLOUT_BATCH_SIZE=${MINI_ROLLOUT_BATCH_SIZE:-128}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_DUAL=${CLIP_RATIO_DUAL:-10.0}
FILTER_KEY=${FILTER_KEY:-overall}
FILTER_LOW=${FILTER_LOW:-0.01}
FILTER_HIGH=${FILTER_HIGH:-0.99}

# Copy the config, this script, the reward function, and the prompt to the save path
mkdir -p ${SAVE_CHECKPOINT_PATH}
cp ${CONFIG_PATH} ${SAVE_CHECKPOINT_PATH}/config.yaml
cp $0 ${SAVE_CHECKPOINT_PATH}/run.sh
cp ${REWARD_PATH} ${SAVE_CHECKPOINT_PATH}/reward.py
cp ${FORMAT_PROMPT} ${SAVE_CHECKPOINT_PATH}/prompt.jinja

# launch training
python3 -m easyr1.verl.trainer.main \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    config=${CONFIG_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=false \
    worker.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    worker.actor.clip_ratio_dual=${CLIP_RATIO_DUAL} \
    worker.rollout.n=${ROLLOUT_NUM} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    worker.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    worker.reward.reward_function=${REWARD_PATH}:compute_score \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.online_filtering=true \
    algorithm.filter_key=${FILTER_KEY} \
    algorithm.filter_low=${FILTER_LOW} \
    algorithm.filter_high=${FILTER_HIGH} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${GPU_NUMS} \
    trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
    worker.reward.reward_function_kwargs='{"reward_weights":'"${REWARD_WEIGHTS}"',"client_model_name":"'"${CLIENT_MODEL}"'"}' \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.image_dir=${IMAGE_DIR} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    | tee ${SAVE_CHECKPOINT_PATH}/train.log
