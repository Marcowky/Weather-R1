#!/bin/bash

########################################################################

######## 1. model type
model_name="qwen2.5_grpo_en_500hpa_seed_3_step_42"

######## 2. dataset
data_type="SQA_qcm_a"
# data_type="WCQ_en"

########################################################################

# prompt_type="direct"
prompt_type="weather-r1"

language="en"
# Select datafile and imagefolder based on data_type
if [ $data_type == "SQA_qcm_a" ]; then
    data_file="data/ScienceQA-Weather-R1/ScienceQA_dataset_qcm_a.json"
    image_folder="data/ScienceQA-Weather-R1/image"
elif [ $data_type == "WCQ_en" ]; then
    data_file="data/WeatherQA/all_split.json"
    image_folder="data/WeatherQA/image"
fi

########################################################################

# Parse CUDA_VISIBLE_DEVICES to get GPU count and IDs; default to all GPUs when unspecified
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "CUDA_VISIBLE_DEVICES is not set and nvidia-smi is unavailable to detect GPU count" >&2
        exit 1
    fi
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "No available GPU detected" >&2
        exit 1
    fi
    GPU_ARRAY=($(seq 0 $((GPU_COUNT-1))))
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    GPU_COUNT=${#GPU_ARRAY[@]}
fi

echo "Using $GPU_COUNT GPUs: ${GPU_ARRAY[*]}"

# Launch multiple parallel processes
for i in $(seq 0 $((GPU_COUNT-1))); do
    gpu_id=${GPU_ARRAY[$i]}
    echo "Starting evaluation on GPU $gpu_id (process $i/$GPU_COUNT)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python -m src.eval.gen_ans_multi_gpu \
        --data-file $data_file \
        --image-folder $image_folder \
        --answers-file "results/${data_type}/${model_name}_gpu${i}.jsonl" \
        --model-name "$model_name" \
        --temperature 0 \
        --language $language \
        --prompt-type $prompt_type \
        --gpu-id $i \
        --gpu_num $GPU_COUNT &
done

# Wait for all background processes
wait

# Merge all GPU result files
echo "Merging results from all GPUs..."
cat "results/${data_type}/${model_name}_gpu"*.jsonl > "results/${data_type}/${model_name}.jsonl"

# Clean up temporary files
rm "results/${data_type}/${model_name}_gpu"*.jsonl

echo "Evaluation completed. Results saved to results/${data_type}/${model_name}.jsonl"
