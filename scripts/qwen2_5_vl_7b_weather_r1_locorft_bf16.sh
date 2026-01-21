#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export NCCL_PROTO=Simple

# set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_NUMS=4

# path to your Qwen2.5-VL-7B-Instruct
MODEL_PATH=models/Qwen/Qwen2.5-VL-7B-Instruct

# wandb and exp settings
PROJECT_NAME=weather-r1
EXPERIMENT_NAME=qwen2_5_7b_weather_r1_locorft_500hpa_situation

# reward weights and judge model settings
REWARD_WEIGHTS='{"format":0.1,"logic":0.3,"accuracy":0.6}'
# REWARD_WEIGHTS='{"format":0.1,"logic":0.0,"accuracy":0.9}' # diff of locorft is only the reward weights, we set logic to 0.0
CLIENT_MODEL="openai/gpt-oss-20b"

# set OpenAI API key and base for judge model (vLLM serving)
export WEATHER_R1_OPENAI_API_KEY="EMPTY"
export WEATHER_R1_OPENAI_API_BASE="http://0.0.0.0:50907/v1"

# data files and image dir
TRAIN_FILE=data/WeatherQA/train/split/train_500hpa_situation.json
# TRAIN_FILE=data/WeatherQA/train/split/train_850hpa_situation.json
# TRAIN_FILE=data/WeatherQA/train/split/train_land_situation.json
# TRAIN_FILE=data/WeatherQA/train/split/train_max_temp.json
# TRAIN_FILE=data/WeatherQA/train/split/train_min_temp.json
# TRAIN_FILE=data/WeatherQA/train/split/train_phenomena.json
# TRAIN_FILE=data/WeatherQA/train/split/train_rain.json
VAL_FILE=data/WeatherQA/val.json
IMAGE_DIR=data/WeatherQA/image

# other settings
CONFIG_PATH=src/weather_r1/weather_r1_config.yaml
REWARD_PATH=src/weather_r1/weather_r1_reward.py
FORMAT_PROMPT=src/weather_r1/weather_r1_format.jinja
SAVE_CHECKPOINT_PATH=results/checkpoints/${PROJECT_NAME}/$(date +%Y%m%d_%H%M%S)-${EXPERIMENT_NAME}
ROLLOUT_NUM=5

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
    worker.rollout.n=${ROLLOUT_NUM} \
    worker.reward.reward_function=${REWARD_PATH}:compute_score \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${GPU_NUMS} \
    trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
    worker.reward.reward_function_kwargs='{"reward_weights":'"${REWARD_WEIGHTS}"',"client_model_name":"'"${CLIENT_MODEL}"'"}' \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.image_dir=${IMAGE_DIR} \
    data.format_prompt=${FORMAT_PROMPT} \
    | tee ${SAVE_CHECKPOINT_PATH}/train.log
