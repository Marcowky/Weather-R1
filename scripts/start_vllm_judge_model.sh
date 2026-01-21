#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

PORT=50907

MODEL_BASE="models"

MODEL_NAME=openai/gpt-oss-20b
# MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
# MODEL_NAME=Qwen/Qwen3-32B

MODEL_PATH=${MODEL_BASE}/${MODEL_NAME}

# for gpt-oss-20b
vllm serve ${MODEL_PATH} --port ${PORT} --served-model-name ${MODEL_NAME}

# for Qwen3-4B-Instruct-2507
# vllm serve ${MODEL_PATH} --port ${PORT} --served-model-name ${MODEL_NAME} --max-model-len 240000

# for Qwen3-32B
# vllm serve ${MODEL_PATH} --port ${PORT} --served-model-name ${MODEL_NAME} --tensor_parallel_size 2